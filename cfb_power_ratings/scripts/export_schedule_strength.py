#!/usr/bin/env python
"""Ranks 2026 FBS teams by schedule difficulty, using the preseason power rating (outputs/
ratings/2026/week_00_ratings.csv) as each opponent's strength.

Since the season hasn't been played, this pulls the full 2026 schedule live from CFBD
(live_data.fetch_season_schedule) rather than reading the DB's games table (structurally empty
for 2026 -- see SQL Scripts/README.md). Non-FBS opponents get a fixed proxy rating -- the same
non-FBS pool-rating calibration srs.py's compute_srs uses internally, computed from season 2025's
actual completed games (the most recent real season), not invented.

Each scheduled game's "effective opponent strength" is the opponent's rating, site-adjusted:
a true road opponent is effectively harder (+hfa) than their rating suggests, a home opponent
effectively easier (-hfa), mirroring the same site-adjustment principle used everywhere else in
this project. A team's schedule-strength score is the mean effective opponent strength across
its full 2026 slate.

Usage: python scripts/export_schedule_strength.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cfb_power_ratings.cfbd_client import get_client
from cfb_power_ratings.database import get_engine, run_query
from cfb_power_ratings.live_data import fetch_fbs_teams, fetch_season_schedule
from cfb_power_ratings.srs import estimate_home_field_advantage, estimate_non_fbs_pool_rating, games_to_team_game_frame
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_RATINGS, OUTPUTS_SCHEDULE_STRENGTH, ensure_dirs

logger = get_logger(__name__)

SEASON = 2026
NON_FBS_PROXY_SOURCE_SEASON = 2025
TOP_N_FOR_RANKED_OPPONENT_COUNT = 25


def _load_preseason_ratings(season: int) -> pd.Series:
    path = OUTPUTS_RATINGS / str(season) / "week_00_ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/generate_preseason_ratings.py --season {season} first.")
    df = pd.read_csv(path)
    return df.set_index("team")["rating"]


def _estimate_non_fbs_proxy_rating(engine) -> tuple[float, float]:
    games = run_query(
        "SELECT * FROM games WHERE completed = 1 AND season = :season",
        params={"season": NON_FBS_PROXY_SOURCE_SEASON}, engine=engine,
    )
    if games.empty:
        raise RuntimeError(f"No completed games found for season={NON_FBS_PROXY_SOURCE_SEASON} -- can't calibrate the non-FBS proxy rating.")
    hfa = estimate_home_field_advantage(games)
    fbs_teams = set(games.loc[games["home_division"] == "fbs", "home_team"]) | set(games.loc[games["away_division"] == "fbs", "away_team"])
    team_games = games_to_team_game_frame(games)
    proxy = estimate_non_fbs_pool_rating(team_games, hfa, fbs_teams)
    logger.info(f"Non-FBS proxy rating (calibrated from season={NON_FBS_PROXY_SOURCE_SEASON}, hfa={hfa:.2f}): {proxy:.2f}")
    return proxy, hfa


def main() -> None:
    ensure_dirs()
    engine = get_engine()
    client = get_client()

    fbs_teams_2026 = fetch_fbs_teams(client, SEASON)
    logger.info(f"{len(fbs_teams_2026)} FBS teams for season={SEASON}")

    ratings = _load_preseason_ratings(SEASON)
    top25_teams = set(ratings.sort_values(ascending=False).head(TOP_N_FOR_RANKED_OPPONENT_COUNT).index)

    non_fbs_proxy, hfa = _estimate_non_fbs_proxy_rating(engine)

    schedule = fetch_season_schedule(client, SEASON)
    logger.info(f"Pulled {len(schedule)} scheduled {SEASON} FBS games")

    home = schedule.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(is_home=True)
    away = schedule.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(is_home=False)
    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games[team_games["team"].isin(fbs_teams_2026)]

    opponent_is_fbs = team_games["opponent"].isin(fbs_teams_2026)
    opponent_rating = np.where(opponent_is_fbs, team_games["opponent"].map(ratings), non_fbs_proxy)
    # Road opponent: effectively harder than a neutral-field rating suggests (+hfa). Home
    # opponent: effectively easier (-hfa). Neutral site: no adjustment.
    site_adjustment = np.where(
        team_games["neutral_site"], 0.0,
        np.where(team_games["is_home"], -hfa, hfa),
    )
    team_games = team_games.assign(
        opponent_rating=opponent_rating,
        effective_opponent_strength=opponent_rating + site_adjustment,
        opponent_is_fbs=opponent_is_fbs,
        opponent_is_ranked=team_games["opponent"].isin(top25_teams),
    )

    summary = team_games.groupby("team").agg(
        n_games=("opponent", "count"),
        n_fbs_games=("opponent_is_fbs", "sum"),
        n_road_games=("is_home", lambda s: int((~s).sum())),
        n_vs_2026_preseason_top25=("opponent_is_ranked", "sum"),
        sos_score=("effective_opponent_strength", "mean"),
        # Median alongside the mean: a team with one gauntlet game and eleven cupcakes can
        # produce the same mean as a team with a uniformly tough slate -- the median is far less
        # moved by a single extreme game (e.g. one road trip to the #1 team), so it answers a
        # different question ("how hard is a TYPICAL game on this schedule") than the mean does
        # ("how hard is the schedule on the whole").
        sos_median=("effective_opponent_strength", "median"),
    ).reset_index()

    summary = summary.sort_values("sos_score", ascending=False).reset_index(drop=True)
    summary["sos_rank"] = summary.index + 1
    summary = summary.sort_values("sos_median", ascending=False).reset_index(drop=True)
    summary["sos_median_rank"] = summary.index + 1
    summary = summary.sort_values("sos_rank").reset_index(drop=True)

    out_dir = OUTPUTS_SCHEDULE_STRENGTH / str(SEASON)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "schedule_strength.csv"
    summary.to_csv(out_path, index=False)

    print(f"Hardest {SEASON} schedules (top 10):")
    print(summary.head(10).to_string(index=False))
    print(f"\nEasiest {SEASON} schedules (bottom 10):")
    print(summary.tail(10).to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Weekly in-season update: blends the preseason prior with every completed game through week
N-1, writes the resulting power ratings, and scores every real matchup on week N's schedule
(predicted margin, favored team, home win probability).

Completed games/FBS-team-list are read from the MySQL DB first (fast, no rate limit); if the DB
has no rows yet for this season (e.g. a new season SQL Scripts/ingest_to_mysql.R hasn't been
re-run for), falls back to a live CFBD pull. The upcoming week's schedule is ALWAYS pulled live
-- games/betting_lines in the DB are completed-only by design (see SQL Scripts/README.md), so
there is no DB path for "who plays whom next week."

Usage: python scripts/update_ratings.py --season 2026 --week 4
    (blends games through week 3, scores week 4's real matchups)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_power_ratings.config import load_features_config, load_modeling_config
from cfb_power_ratings.database import get_engine, get_fbs_teams_by_season, run_query
from cfb_power_ratings.preseason import load_preseason_model_metadata, predict_preseason_ratings
from cfb_power_ratings.rating_engine import (
    fit_residual_std,
    historical_site_adjusted_residuals,
    implied_matchup,
    update_ratings,
    win_probability,
)
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_RATINGS

logger = get_logger(__name__)


def _completed_games_through(engine, season: int, week: int) -> pd.DataFrame:
    db_games = run_query(
        "SELECT * FROM games WHERE completed = 1 AND season = :season AND week < :week",
        params={"season": season, "week": week}, engine=engine,
    )
    if not db_games.empty:
        return db_games

    logger.warning(f"No completed games in the DB for season={season} week<{week} -- falling back to a live CFBD pull.")
    from cfb_power_ratings.cfbd_client import get_client
    from cfb_power_ratings.live_data import fetch_completed_games

    return fetch_completed_games(get_client(), season, list(range(1, week)))


def _fbs_teams(engine, season: int) -> set[str]:
    teams = get_fbs_teams_by_season(engine, season)
    if teams:
        return teams
    logger.warning(f"No FBS team list in the DB for season={season} -- falling back to a live CFBD pull.")
    from cfb_power_ratings.cfbd_client import get_client
    from cfb_power_ratings.live_data import fetch_fbs_teams

    return fetch_fbs_teams(get_client(), season)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True, help="Week to score; games through week-1 are blended in")
    parser.add_argument("--phantom-games", type=int, default=None, help="Override modeling.yaml's default_phantom_games")
    args = parser.parse_args()

    engine = get_engine()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    phantom_games = args.phantom_games or modeling_cfg.rating_engine.default_phantom_games

    metadata = load_preseason_model_metadata()
    hfa = float(metadata["hfa"])

    logger.info(f"Predicting preseason priors for season={args.season}")
    preseason_priors = predict_preseason_ratings(engine, args.season, features_cfg, modeling_cfg.srs_history_start_season)

    logger.info(f"Pulling completed games for season={args.season} through week {args.week - 1}")
    games_so_far = _completed_games_through(engine, args.season, args.week)
    fbs_teams = _fbs_teams(engine, args.season)

    blended = update_ratings(preseason_priors, games_so_far, hfa, fbs_teams, phantom_games=phantom_games)
    blended.insert(0, "rank", range(1, len(blended) + 1))

    out_dir = OUTPUTS_RATINGS / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = out_dir / f"week_{args.week:02d}_ratings.csv"
    blended.to_csv(ratings_path, index=False)
    print(f"Ratings through week {args.week - 1}, season {args.season} (top 25):")
    print(blended.head(25).to_string(index=False))
    print(f"\nWrote {ratings_path}")

    logger.info(f"Pulling week {args.week}'s real schedule live and scoring matchups")
    from cfb_power_ratings.cfbd_client import get_client
    from cfb_power_ratings.live_data import fetch_games

    schedule = fetch_games(get_client(), args.season, args.week)
    schedule = schedule[schedule["home_team"].isin(fbs_teams) | schedule["away_team"].isin(fbs_teams)]

    calibration_seasons = modeling_cfg.walk_forward_validation_seasons + [modeling_cfg.final_holdout_season]
    actual, predicted = historical_site_adjusted_residuals(engine, calibration_seasons, hfa)
    residual_std = fit_residual_std(actual, predicted) if len(actual) else 15.0

    ratings_lookup = blended.set_index("team")["rating"]
    matchup_rows = []
    for _, g in schedule.iterrows():
        home_rating = ratings_lookup.get(g["home_team"])
        away_rating = ratings_lookup.get(g["away_team"])
        if home_rating is None or away_rating is None:
            continue
        im = implied_matchup(home_rating, away_rating, hfa)
        prob = float(win_probability(im["predicted_margin"], residual_std))
        matchup_rows.append({
            "home_team": g["home_team"], "away_team": g["away_team"],
            "predicted_margin": im["predicted_margin"], "favored_team": im["favored_team"],
            "home_win_probability": prob,
        })

    matchups = pd.DataFrame(matchup_rows).sort_values("predicted_margin", ascending=False)
    matchups_path = out_dir / f"week_{args.week:02d}_matchups.csv"
    matchups.to_csv(matchups_path, index=False)
    print(f"\nWeek {args.week} matchups ({len(matchups)} games):")
    print(matchups.to_string(index=False))
    print(f"\nWrote {matchups_path}")


if __name__ == "__main__":
    main()

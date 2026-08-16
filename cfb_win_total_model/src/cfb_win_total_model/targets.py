"""Target construction: regular_season_wins per FBS team-season.

games.season_type is verified 100% 'regular' and completed is verified 100% TRUE across the
whole table (see docs/assumptions_and_limitations.md) -- no bowl/playoff filter is needed,
this is a confirmed fact about the data, not an assumption. A defensive
`home_points IS NOT NULL` guard is still applied since a future ingestion re-run could pull
in-progress games.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.database import run_query
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

NON_FEATURE_COLS = {"school", "season", "regular_season_wins", "regular_season_losses", "scheduled_games"}


def get_fbs_teams_by_season(engine: Engine, season: int) -> set[str]:
    """The per-season row universe every feature module and the final modeling table must
    join against: FBS teams are those where that team's OWN division='fbs' for the season --
    the opponent's division is irrelevant (an FBS team's games against FCS opponents still
    count toward its win total)."""
    sql = """
        SELECT DISTINCT school FROM (
            SELECT home_team AS school FROM games WHERE season = :season AND home_division = 'fbs'
            UNION
            SELECT away_team AS school FROM games WHERE season = :season AND away_division = 'fbs'
        ) t
    """
    df = run_query(sql, params={"season": season}, engine=engine)
    return set(df["school"])


def get_fbs_teams_by_seasons(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    """Same as get_fbs_teams_by_season but for multiple seasons at once, returned as a
    (school, season) DataFrame -- the LEFT side of every join in build_modeling_dataset.py."""
    frames = []
    for season in seasons:
        teams = get_fbs_teams_by_season(engine, season)
        frames.append(pd.DataFrame({"school": sorted(teams), "season": season}))
    return pd.concat(frames, ignore_index=True)


def build_target_table(engine: Engine, seasons: list[int] | None = None) -> pd.DataFrame:
    """Returns one row per (school, season) with regular_season_wins, regular_season_losses,
    scheduled_games -- for every FBS team-season in `seasons` (or all seasons if None)."""
    season_filter = ""
    params: dict = {}
    if seasons is not None:
        placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
        season_filter = f"AND season IN ({placeholders})"
        params = {f"s{i}": s for i, s in enumerate(seasons)}

    sql = f"""
        SELECT game_id, season, home_team, home_division, home_points,
               away_team, away_division, away_points
        FROM games
        WHERE (home_division = 'fbs' OR away_division = 'fbs')
          AND home_points IS NOT NULL AND away_points IS NOT NULL
          {season_filter}
    """
    games = run_query(sql, params=params, engine=engine)

    home_view = games.rename(
        columns={
            "home_team": "school",
            "home_division": "division",
            "home_points": "team_points",
            "away_points": "opp_points",
        }
    )[["game_id", "season", "school", "division", "team_points", "opp_points"]]

    away_view = games.rename(
        columns={
            "away_team": "school",
            "away_division": "division",
            "away_points": "team_points",
            "home_points": "opp_points",
        }
    )[["game_id", "season", "school", "division", "team_points", "opp_points"]]

    stacked = pd.concat([home_view, away_view], ignore_index=True)
    stacked = stacked[stacked["division"] == "fbs"]

    stacked["win"] = stacked["team_points"] > stacked["opp_points"]
    stacked["loss"] = stacked["team_points"] < stacked["opp_points"]
    n_ties = int((stacked["team_points"] == stacked["opp_points"]).sum())
    if n_ties:
        logger.warning(f"{n_ties} tied games found in {stacked['season'].nunique()} season(s) -- unexpected in this era.")

    grouped = (
        stacked.groupby(["school", "season"], as_index=False)
        .agg(
            regular_season_wins=("win", "sum"),
            regular_season_losses=("loss", "sum"),
            scheduled_games=("game_id", "count"),
        )
    )
    grouped["regular_season_wins"] = grouped["regular_season_wins"].astype(int)
    grouped["regular_season_losses"] = grouped["regular_season_losses"].astype(int)
    return grouped

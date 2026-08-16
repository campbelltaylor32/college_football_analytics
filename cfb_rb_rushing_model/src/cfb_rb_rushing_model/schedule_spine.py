"""Team-week schedule/opponent spine: one row per (team, game_id), built from `games`.

Deliberately NOT built from game_team_stats.opponent, even though that column exists and would
be simpler -- game_team_stats is a post-game-populated table with no row yet for an upcoming
(not-yet-played) game, which is exactly the case scripts/generate_week_predictions.py needs.
`games` has a row for a scheduled-but-not-yet-played game too, so this same function produces
the row universe for both historical training rows and live weekly inference -- no divergent
code path between the two modes.

Restricted to the team's OWN division='fbs' (opponent's division is irrelevant, same
convention as the sibling cfb_win_total_model project's targets.get_fbs_teams_by_season) --
an FBS team's game against an FCS opponent still produces a real rushing-yards row for its RBs.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

SPINE_COLUMNS = [
    "game_id", "season", "week", "start_date", "neutral_site", "conference_game", "completed",
    "team", "division", "conference", "opponent", "opponent_division", "opponent_conference", "home_away",
]


def _pull_games(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"""
        SELECT game_id, season, week, start_date, neutral_site, conference_game, completed,
               home_team, home_division, home_conference,
               away_team, away_division, away_conference
        FROM games
        WHERE season IN ({placeholders}) AND (home_division = 'fbs' OR away_division = 'fbs')
    """
    return run_query(sql, params=params, engine=engine)


def build_schedule_spine(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    """One row per (team, game_id) for every FBS team-game in `seasons`, including
    not-yet-played games (games.completed is carried through, not filtered on)."""
    if not seasons:
        return pd.DataFrame(columns=SPINE_COLUMNS)

    games = _pull_games(engine, seasons)
    if games.empty:
        return pd.DataFrame(columns=SPINE_COLUMNS)

    home = games.rename(
        columns={
            "home_team": "team", "home_division": "division", "home_conference": "conference",
            "away_team": "opponent", "away_division": "opponent_division", "away_conference": "opponent_conference",
        }
    ).assign(home_away="home")
    away = games.rename(
        columns={
            "away_team": "team", "away_division": "division", "away_conference": "conference",
            "home_team": "opponent", "home_division": "opponent_division", "home_conference": "opponent_conference",
        }
    ).assign(home_away="away")

    stacked = pd.concat([home[SPINE_COLUMNS], away[SPINE_COLUMNS]], ignore_index=True)
    stacked = stacked[stacked["division"] == "fbs"].copy()
    stacked["start_date"] = pd.to_datetime(stacked["start_date"])
    return stacked.sort_values(["team", "season", "start_date"]).reset_index(drop=True)


def attach_rest_days(spine: pd.DataFrame, default_rest_days: int) -> pd.DataFrame:
    """Adds `rest_days` = days since this team's previous game (any season), imputed to
    `default_rest_days` for a team's first recorded game (no prior game to diff against --
    typically a season opener, occasionally a true first game in the dataset)."""
    spine = spine.sort_values(["team", "start_date"]).copy()
    gap_days = spine.groupby("team")["start_date"].diff().dt.days
    spine["rest_days"] = gap_days.fillna(default_rest_days)
    return spine


def describe_features() -> list[dict]:
    return [
        {
            "feature_name": "rest_days",
            "description": "Days since this team's previous game (imputed to config default for a season opener / first recorded game)",
            "source_table": "games (start_date)",
            "source_season": "t (date/schedule structure only, never t results)",
            "transformation": "consecutive start_date diff, grouped by team",
            "known_before_kickoff": True,
            "missing_value_treatment": "imputed to features.yaml default_rest_days_season_opener",
            "expected_direction": "context",
            "category": "game_context",
        }
    ]

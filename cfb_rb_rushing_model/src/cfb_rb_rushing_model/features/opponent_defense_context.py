"""Team-week rolling run-defense-allowed features, keyed by (team, game_id) where `team` is
the DEFENSE's own team -- this table is joined onto a player-week row via that player's
OPPONENT for the target week (schedule_spine's `opponent` column), not via the player's own
team. This is what makes "explosive runs allowed by the upcoming opponent" and "opposing
team's time of possession" (both explicitly requested predictors) available to a player row.

Two sources, combined: game_team_stats' own `*_allowed` columns give rush-yield VOLUME stats
directly (that table already stores the defense's own row, no self-join needed); `plays`
grouped by `def_pos_team` gives the stats game_team_stats doesn't carry (success/explosive/EPA
allowed). `possession_time_minutes` is pulled again here (same source as
team_offense_context.py, from the defense's own perspective) rather than cross-joined from
that module's output -- keeps this module self-contained and independently testable, at the
cost of a few duplicated lines of query logic.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.config import FeaturesConfig
from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.features.rolling_utils import compute_rolling_and_lag
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

VALUE_COLS = [
    "rushing_yards_allowed", "yards_per_rush_attempt_allowed", "def_success_rate_allowed",
    "def_explosive_runs_allowed", "def_explosive_rate_allowed", "def_epa_allowed_per_rush",
    "def_stuffed_rate_forced", "def_possession_time_minutes",
]


def _pull_game_team_stats_allowed(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"""
        SELECT game_id, school AS team, season,
               rushing_yards_allowed, yards_per_rush_attempt_allowed,
               possession_time_minutes AS def_possession_time_minutes
        FROM game_team_stats
        WHERE season IN ({placeholders})
    """
    return run_query(sql, params=params, engine=engine)


def _pull_plays_allowed(engine: Engine, seasons: list[int], rush_play_types: list[str], explosive_run_yard_threshold: int) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    type_placeholders = ", ".join(f":t{i}" for i in range(len(rush_play_types)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    params.update({f"t{i}": t for i, t in enumerate(rush_play_types)})
    sql = f"""
        SELECT game_id, def_pos_team AS team, season, week, yards_gained, success, stuffed_run, epa
        FROM plays
        WHERE season IN ({season_placeholders}) AND play_type IN ({type_placeholders}) AND def_pos_team IS NOT NULL
    """
    raw = run_query(sql, params=params, engine=engine)
    if raw.empty:
        return raw
    raw["is_explosive"] = raw["yards_gained"] >= explosive_run_yard_threshold
    grouped = raw.groupby(["game_id", "team", "season"], as_index=False).agg(
        def_success_rate_allowed=("success", "mean"),
        def_explosive_runs_allowed=("is_explosive", "sum"),
        def_epa_allowed_per_rush=("epa", "mean"),
        def_stuffed_rate_forced=("stuffed_run", "mean"),
        def_carries_allowed=("yards_gained", "count"),
    )
    grouped["def_explosive_rate_allowed"] = grouped["def_explosive_runs_allowed"] / grouped["def_carries_allowed"]
    return grouped.drop(columns=["def_carries_allowed"])


def build_opponent_defense_context_features(engine: Engine, spine: pd.DataFrame, seasons: list[int], features_cfg: FeaturesConfig) -> pd.DataFrame:
    allowed_stats = _pull_game_team_stats_allowed(engine, seasons)
    plays_allowed = _pull_plays_allowed(engine, seasons, ["Rush", "Rushing Touchdown"], features_cfg.explosive_run_yard_threshold)

    base = spine[["team", "game_id", "season", "week", "start_date"]].drop_duplicates()
    merged = base.merge(allowed_stats, on=["team", "game_id", "season"], how="left")
    if not plays_allowed.empty:
        merged = merged.merge(plays_allowed, on=["team", "game_id", "season"], how="left")
    else:
        for col in ["def_success_rate_allowed", "def_explosive_runs_allowed", "def_explosive_rate_allowed", "def_epa_allowed_per_rush", "def_stuffed_rate_forced"]:
            merged[col] = pd.NA

    window = features_cfg.defense_rolling_windows[0]
    rolled = compute_rolling_and_lag(merged, group_cols=["team"], sort_col="start_date", value_cols=VALUE_COLS, window=window)

    lag1_cols = [c for c in rolled.columns if c.endswith("_lag1")]
    keep_cols = ["team", "game_id", "season", "week", "start_date"] + lag1_cols
    return rolled[keep_cols].drop_duplicates(subset=["team", "game_id"])


def describe_features() -> list[dict]:
    base = {
        "source_table": "game_team_stats (*_allowed cols) + plays (grouped by def_pos_team)",
        "source_season": "defense's own trailing games, strictly before the target game (two-step compute-then-lag)",
        "category": "opponent_defense_context",
        "known_before_kickoff": True,
        "missing_value_treatment": "NaN for a team's first recorded game of the window; median-imputed downstream",
    }
    descs = {
        "rushing_yards_allowed": "Rush yards allowed, this defense's game (volume)",
        "yards_per_rush_attempt_allowed": "Yards allowed per rush attempt faced",
        "def_success_rate_allowed": "Mean plays.success on rushes faced by this defense",
        "def_explosive_runs_allowed": "Count of rushes faced with yards_gained >= config threshold",
        "def_explosive_rate_allowed": "def_explosive_runs_allowed / rushes faced -- explicitly requested predictor",
        "def_epa_allowed_per_rush": "Mean EPA allowed per rush faced",
        "def_stuffed_rate_forced": "Mean plays.stuffed_run on rushes faced by this defense",
        "def_possession_time_minutes": "This team's own minutes of possession -- when this whole table is joined onto a player row via that player's OPPONENT, this becomes 'opposing team's time of possession', an explicitly requested predictor",
    }
    rows = []
    for col in VALUE_COLS:
        for suffix in ["avg3_lag1", "avg_all_lag1"]:
            rows.append({**base, "feature_name": f"{col}_{suffix}", "description": descs[col], "transformation": "rolling/expanding mean, shift(1)", "expected_direction": "context"})
    return rows

"""Team-week rolling offensive-context features (rush/pass mix, tempo, time of possession),
keyed by (team, game_id). Source: game_team_stats, LEFT JOINed onto `spine` (schedule_spine's
output, the row universe -- includes not-yet-played games, which naturally get NaN raw values
here and simply don't contribute to the rolling window until a real value exists).

`possession_time_minutes` is game_team_stats' own generated column (parsed from the raw
"MM:SS" string by the DB schema itself) -- no manual time parsing needed here.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.config import FeaturesConfig
from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.features.rolling_utils import compute_rolling_and_lag
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

VALUE_COLS = ["rush_pct", "pass_pct", "tempo_plays_per_minute", "possession_time_minutes"]


def _pull_game_team_stats(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"""
        SELECT game_id, school AS team, season, rushing_attempts, attempted_passes, possession_time_minutes
        FROM game_team_stats
        WHERE season IN ({placeholders})
    """
    return run_query(sql, params=params, engine=engine)


def build_team_offense_context_features(engine: Engine, spine: pd.DataFrame, seasons: list[int], features_cfg: FeaturesConfig) -> pd.DataFrame:
    stats = _pull_game_team_stats(engine, seasons)
    base = spine[["team", "game_id", "season", "week", "start_date"]].drop_duplicates()
    merged = base.merge(stats, on=["team", "game_id", "season"], how="left")

    total_plays = merged["rushing_attempts"] + merged["attempted_passes"]
    merged["rush_pct"] = merged["rushing_attempts"] / total_plays
    merged["pass_pct"] = 1 - merged["rush_pct"]
    merged["tempo_plays_per_minute"] = total_plays / merged["possession_time_minutes"]

    window = features_cfg.team_rolling_windows[0]
    rolled = compute_rolling_and_lag(merged, group_cols=["team"], sort_col="start_date", value_cols=VALUE_COLS, window=window)

    lag1_cols = [c for c in rolled.columns if c.endswith("_lag1")]
    keep_cols = ["team", "game_id", "season", "week", "start_date"] + lag1_cols
    return rolled[keep_cols].drop_duplicates(subset=["team", "game_id"])


def describe_features() -> list[dict]:
    base = {
        "source_table": "game_team_stats",
        "source_season": "team's own trailing games, strictly before the target game (two-step compute-then-lag)",
        "category": "team_offense_context",
        "known_before_kickoff": True,
        "missing_value_treatment": "NaN for a team's first recorded game of the window; median-imputed downstream",
    }
    descs = {
        "rush_pct": "Share of offensive plays that were rush attempts",
        "pass_pct": "1 - rush_pct",
        "tempo_plays_per_minute": "(rushing_attempts + attempted_passes) / possession_time_minutes -- plays-per-minute-of-own-possession, a tempo proxy (not raw seconds-per-play, which this DB does not aggregate at team-game grain)",
        "possession_time_minutes": "Minutes of possession (game_team_stats' own generated column, parsed from the raw MM:SS string)",
    }
    rows = []
    for col in VALUE_COLS:
        for suffix in ["avg3_lag1", "avg_all_lag1"]:
            rows.append({**base, "feature_name": f"{col}_{suffix}", "description": descs[col], "transformation": "rolling/expanding mean, shift(1)", "expected_direction": "context"})
    return rows

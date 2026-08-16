"""Returning-production features. Source: returning_production table, season **t itself**.

SANCTIONED EXCEPTION to the t-1 lag rule used everywhere else in features/: this table
already represents "how much of last season's production returns for season t" -- it is a
preseason-known-for-t measure by construction, published before season t's games. Using
`season = target_season` here is correct, not a leak. See docs/data_leakage_rules.md.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.database import run_query
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

PERCENT_COLS = ["percent_ppa", "percent_passing_ppa", "percent_receiving_ppa", "percent_rushing_ppa"]


def _source_season(target_season: int) -> int:
    """Returning_production is a sanctioned as-is exception: source season == target season."""
    return target_season


def build_returning_production_features(
    engine: Engine, target_season: int, winsorize_limits: tuple[float, float] = (-3.0, 3.0)
) -> pd.DataFrame:
    logger.info(f"Building returning_production features for target_season={target_season} (source season {target_season}, as-is)")
    sql = """
        SELECT season, team AS school, total_ppa, total_passing_ppa, total_receiving_ppa, total_rushing_ppa,
               percent_ppa, percent_passing_ppa, percent_receiving_ppa, percent_rushing_ppa,
               usage_pct, passing_usage, receiving_usage, rushing_usage
        FROM returning_production WHERE season = :season
    """
    df = run_query(sql, params={"season": target_season}, engine=engine)
    if df.empty:
        logger.warning(f"No returning_production rows for season={target_season}")
        return pd.DataFrame(columns=["school", "season", "returning_production_missing"])

    lower, upper = winsorize_limits
    for col in PERCENT_COLS:
        df[col] = df[col].clip(lower, upper)

    df = df.rename(columns={"percent_ppa": "returning_percent_ppa"})
    df["returning_production_missing"] = False
    return df


def describe_features() -> list[dict]:
    base = {
        "source_table": "returning_production",
        "source_season": "t (sanctioned exception -- preseason-known for t itself)",
        "category": "returning_production",
        "known_before_kickoff": True,
        "missing_value_treatment": "returning_production_missing flag; no rows before season 2014",
    }
    return [
        {**base, "feature_name": "total_ppa", "description": "Total predicted points added returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "total_passing_ppa", "description": "Total passing PPA returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "total_receiving_ppa", "description": "Total receiving PPA returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "total_rushing_ppa", "description": "Total rushing PPA returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "returning_percent_ppa", "description": "Share of total PPA returning", "transformation": "winsorize to [-3,3]", "expected_direction": "+"},
        {**base, "feature_name": "percent_passing_ppa", "description": "Share of passing PPA returning", "transformation": "winsorize to [-3,3]", "expected_direction": "+"},
        {**base, "feature_name": "percent_receiving_ppa", "description": "Share of receiving PPA returning", "transformation": "winsorize to [-3,3]", "expected_direction": "+"},
        {**base, "feature_name": "percent_rushing_ppa", "description": "Share of rushing PPA returning", "transformation": "winsorize to [-3,3]", "expected_direction": "+"},
        {**base, "feature_name": "usage_pct", "description": "Overall snap usage returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "passing_usage", "description": "Passing snap usage returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "receiving_usage", "description": "Receiving snap usage returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "rushing_usage", "description": "Rushing snap usage returning", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "returning_production_missing", "description": "True if no returning_production row exists (seasons < 2014)", "transformation": "isna flag", "expected_direction": "context"},
    ]

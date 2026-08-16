"""CSV loading plus introspection of the home_/away_ naming convention. This project reads
already-engineered features (see ../R Scripts/Merge_Predictors_CFB_Historical.R) rather than
querying a database, so this module replaces the sibling cfb_win_total_model project's
database.py -- its job is understanding column structure, not fetching rows.
"""

from __future__ import annotations

import re

import pandas as pd

from cfb_spread_model.config import DataConfig
from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)

_SIDES = ("home", "away")
_TEMPORAL_SUFFIXES = (("prev_week_", "prefix"), ("_avg_all", "suffix"), ("_avg3", "suffix"))


def load_raw_csv(cfg: DataConfig) -> pd.DataFrame:
    logger.info(f"Loading {cfg.source_csv}")
    df = pd.read_csv(cfg.source_csv, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")
    return df


def get_feature_columns(columns: list[str], cfg: DataConfig) -> list[str]:
    """Every modeling input column: everything except ids, the label, (by default) season/week,
    which exist only to drive walk-forward splitting (see modeling/splits.py), and any column
    matching one of cfg.excluded_column_patterns (e.g. prev_week_ -- see config/data.yaml for
    why). This is a hard exclusion applied before any column reaches Stage 1/2 feature
    selection, not a de-prioritization."""
    exclude = set(cfg.id_columns) | {cfg.label_column}
    if not cfg.include_split_columns_as_features:
        exclude |= set(cfg.split_only_columns)
    return [
        c
        for c in columns
        if c not in exclude and not any(pattern in c for pattern in cfg.excluded_column_patterns)
    ]


def build_feature_matrix(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    """The single choke point for feature representation. When
    cfg.feature_representation == "differential", the raw home_*/away_* avg_all/avg3/
    non-temporal columns are replaced with diff_*/trend_* columns (feature_engineering.py)
    before the usual id/label/split/excluded-pattern filtering runs -- every script that calls
    this function picks up the representation choice automatically."""
    working_df = df
    if cfg.feature_representation == "differential":
        from cfb_spread_model.feature_engineering import apply_differential_representation

        working_df = apply_differential_representation(df, cfg.id_columns)

    feature_cols = get_feature_columns(list(working_df.columns), cfg)
    return working_df[feature_cols], working_df[cfg.label_column]


def parse_side_and_metric(column: str) -> tuple[str, str, str] | None:
    """For a home_*/away_* feature column, return (side, temporal_transform, base_metric).
    temporal_transform is one of "prev_week", "avg_all", "avg3", "non_temporal". Returns None
    for columns that don't start with a home_/away_ prefix (ids, label, spread, etc)."""
    for side in _SIDES:
        prefix = f"{side}_"
        if not column.startswith(prefix):
            continue
        rest = column[len(prefix):]
        if rest.startswith("prev_week_"):
            return side, "prev_week", rest[len("prev_week_"):]
        if rest.endswith("_avg_all"):
            return side, "avg_all", rest[: -len("_avg_all")]
        if rest.endswith("_avg3"):
            return side, "avg3", rest[: -len("_avg3")]
        return side, "non_temporal", rest
    return None


def build_temporal_triplet_groups(columns: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    """Group home_*/away_* columns by (side, base_metric) -> {temporal_transform: column_name},
    for the 168-base-metric x 3-transform x 2-side structure verified in the source CSV. Used
    by feature_selection/correlation_pruning.py's Stage 1 (temporal-transform collapse)."""
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for col in columns:
        parsed = parse_side_and_metric(col)
        if parsed is None:
            continue
        side, transform, base = parsed
        if transform == "non_temporal":
            continue
        groups.setdefault((side, base), {})[transform] = col
    return groups


def non_temporal_side_columns(columns: list[str]) -> list[str]:
    """The 18-per-side non-temporal feature columns (talent, coaching, returning production)."""
    out = []
    for col in columns:
        parsed = parse_side_and_metric(col)
        if parsed is not None and parsed[1] == "non_temporal":
            out.append(col)
    return out


_ALLOWED_SUFFIX = re.compile(r"^(.*)_allowed$")


def offense_defense_mirror_pairs(base_metric_names: list[str]) -> list[tuple[str, str]]:
    """Among a set of base metric names (already stripped of temporal prefix/suffix and
    home_/away_ prefix), find (offense, defense) pairs where one is exactly `<x>_allowed` --
    verified to cover 41 of the 168 base metrics. Used by correlation_pruning.py's general
    redundancy pass to preferentially compare known mirror pairs."""
    name_set = set(base_metric_names)
    pairs = []
    for name in base_metric_names:
        match = _ALLOWED_SUFFIX.match(name)
        if match and match.group(1) in name_set:
            pairs.append((match.group(1), name))
    return pairs

"""Post-merge cleaning applied to the assembled modeling table: winsorization, imputation,
and a final sanity pass for inf/extreme values. Every imputation is logged. This is a coarse
first pass -- modeling/preprocessing.py's SimpleImputer is a defensive final layer in case a
future feature addition skips this module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_win_total_model.config import FeaturesConfig
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

RETURNING_PRODUCTION_PERCENT_COLS = ["returning_percent_ppa", "percent_passing_ppa", "percent_receiving_ppa", "percent_rushing_ppa"]

NON_FEATURE_COLS = {"school", "season", "regular_season_wins", "regular_season_losses", "scheduled_games"}


def apply_winsorization(df: pd.DataFrame, features_cfg: FeaturesConfig) -> pd.DataFrame:
    df = df.copy()
    lower, upper = features_cfg.winsorize_percent_ppa_limits
    for col in RETURNING_PRODUCTION_PERCENT_COLS:
        if col in df.columns:
            n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_clipped:
                logger.info(f"Winsorized {n_clipped} values in {col} to [{lower}, {upper}]")
            df[col] = df[col].clip(lower, upper)
    return df


def impute_missing(df: pd.DataFrame, zero_fill_cols: list[str] | None = None) -> pd.DataFrame:
    """Column-group-specific imputation, not blanket mean-imputation:
      - zero_fill_cols (count-like columns, e.g. n_transferred_out): NaN -> 0.
      - every other numeric column: NaN -> within-column median (global across the frame).
    Boolean `_missing` flag columns are left untouched -- they are the explicit missingness
    signal this imputation pairs with, not something to impute over.
    """
    df = df.copy()
    zero_fill_cols = zero_fill_cols or []

    for col in zero_fill_cols:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                logger.info(f"Imputed {n_missing} values in {col} using zero-fill")
            df[col] = df[col].fillna(0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in zero_fill_cols or col in NON_FEATURE_COLS or col.endswith("_missing"):
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            logger.info(f"Imputed {n_missing} values in {col} using median ({median:.4g})")
            df[col] = df[col].fillna(median)

    return df


def validate_no_inf_or_extreme(df: pd.DataFrame, max_abs_zscore: float = 6.0) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
    if inf_cols:
        raise ValueError(f"Columns contain inf/-inf values after cleaning: {inf_cols}")

    for zscore_col in [c for c in df.columns if c.endswith("_zscore")]:
        extreme = df[zscore_col].abs() > max_abs_zscore
        if extreme.any():
            raise ValueError(f"{zscore_col} has {int(extreme.sum())} values with |z| > {max_abs_zscore}")

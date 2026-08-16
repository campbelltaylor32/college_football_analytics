"""Post-merge cleaning applied to the assembled modeling table: imputation and a final
sanity pass for inf/extreme values. Every imputation is logged. This is a coarse first pass --
modeling/preprocessing.py's SimpleImputer is a defensive final layer in case a future feature
addition skips this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_rb_rushing_model.dataset import NON_FEATURE_COLS
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)


def impute_missing(df: pd.DataFrame, zero_fill_cols: list[str] | None = None) -> pd.DataFrame:
    """Column-group-specific imputation, not blanket mean-imputation:
      - zero_fill_cols (count-like columns where NaN genuinely means "none observed yet"):
        NaN -> 0.
      - every other numeric feature column: NaN -> within-column median (global across the
        frame). Median, not zero, is the default -- a missing rolling average (cold start,
        or a feature table with no row for this team/game yet) should not be silently treated
        as "worst possible value," which zero-fill would imply for most of these columns.
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
        if col in zero_fill_cols or col in NON_FEATURE_COLS:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            logger.info(f"Imputed {n_missing} values in {col} using median ({median:.4g})")
            df[col] = df[col].fillna(median)

    return df


def validate_no_inf_or_extreme(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
    if inf_cols:
        raise ValueError(f"Columns contain inf/-inf values after cleaning: {inf_cols}")

"""Defensive cleaning pass. The source CSV's upstream R pipeline already applies na.omit()
after lagging (Merge_Predictors_CFB_Historical.R:70), so this project should see near-zero NA
in practice -- data_validation.check_no_within_week_completeness_gap-equivalent tests document
that guarantee. This module is a safety net, not the primary NA-handling mechanism (that's
modeling/preprocessing.py's SimpleImputer, applied inside each fold's training data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)


def coerce_numeric_feature_dtypes(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_no_inf(df: pd.DataFrame, feature_cols: list[str]) -> None:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
    if inf_cols:
        raise ValueError(f"Columns contain inf/-inf values: {inf_cols}")


def report_residual_missingness(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Logs (does not fill) any NA remaining after the upstream R na.omit(). Non-empty output
    here is a signal something changed upstream -- imputation of any residual NA happens inside
    modeling/preprocessing.py's per-fold-fit SimpleImputer, never globally in this module."""
    na_counts = df[feature_cols].isna().sum()
    na_counts = na_counts[na_counts > 0]
    if not na_counts.empty:
        logger.warning(f"{len(na_counts)} feature columns have residual NA values (unexpected given upstream na.omit()): {na_counts.to_dict()}")
    return na_counts

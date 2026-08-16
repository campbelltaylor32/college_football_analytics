"""Sanity checks run once, right after building the modeling frame, so a silent data
problem doesn't propagate quietly into feature selection or model training."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    pass


def validate_modeling_frame(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    if frame["game_id"].duplicated().any():
        raise DataValidationError("Duplicate game_id rows in modeling frame.")

    if frame["home_covered"].isna().any():
        raise DataValidationError(
            "home_covered has NaNs after push filtering - pushes should already be dropped."
        )
    if not set(frame["home_covered"].unique()) <= {0, 1}:
        raise DataValidationError("home_covered contains values other than 0/1.")

    missing = frame[feature_columns].isna().sum()
    if missing.any():
        bad = missing[missing > 0]
        raise DataValidationError(f"NaNs remain in feature columns: {bad.to_dict()}")

    non_numeric = [
        c for c in feature_columns if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if non_numeric:
        raise DataValidationError(f"Non-numeric candidate feature columns: {non_numeric}")

    non_finite = {}
    for c in feature_columns:
        col = frame[c].to_numpy(dtype=float, copy=False)
        if not np.isfinite(col).all():
            non_finite[c] = int((~np.isfinite(col)).sum())
    if non_finite:
        raise DataValidationError(f"Non-finite values in feature columns: {non_finite}")

    if frame["week"].min() < 1 or frame["week"].max() > 20:
        raise DataValidationError(f"Suspicious week range: {frame['week'].min()}-{frame['week'].max()}")


def summarize(frame: pd.DataFrame, feature_columns: list[str]) -> dict:
    return {
        "n_rows": len(frame),
        "n_features": len(feature_columns),
        "seasons": sorted(frame["season"].unique().tolist()),
        "rows_per_season": frame["season"].value_counts().sort_index().to_dict(),
        "home_covered_rate": float(frame["home_covered"].mean()),
        "n_rows_dropped_for_na": int(frame.attrs.get("n_rows_dropped_for_na", 0)),
    }

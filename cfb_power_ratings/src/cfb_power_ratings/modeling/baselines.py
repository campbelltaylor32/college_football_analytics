"""Zero-parameter baselines, evaluated the same walk-forward way as the real candidates
(modeling/evaluate.py) so a fitted model's MAE improvement over these is honestly earned, not
assumed."""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_overall_mean(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> np.ndarray:
    mean_srs = train_df["target_srs"].mean()
    return np.full(len(predict_df), mean_srs)


def predict_prev_season_srs(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> np.ndarray:
    """Predicts last season's actual SRS unchanged -- the naive "nothing changes" baseline.
    Falls back to the training mean for any team with no srs_lag1 (first FBS season, etc.)."""
    fallback = train_df["target_srs"].mean()
    return predict_df["srs_lag1"].fillna(fallback).to_numpy()


BASELINES = {
    "overall_mean": predict_overall_mean,
    "prev_season_srs": predict_prev_season_srs,
}

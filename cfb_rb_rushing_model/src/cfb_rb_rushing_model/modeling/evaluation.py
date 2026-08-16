"""Evaluation metrics. MAE is the primary metric everywhere (the target is directly in
rushing yards). median_ae is reported alongside it specifically because it's more robust to
the zero-carry blowout/injury-game noise documented in targets.py -- a small number of true
zero-yard games (workload-eligible RB who didn't actually get carries) pulling MAE up without
reflecting a systematic model error. Ported near-verbatim from the sibling cfb_win_total_model
project -- every function here is already target-agnostic regression code."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    return {
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
        "mean_bias": float(np.mean(y_pred - y_true)),
        "pct_within_10": float(np.mean(abs_err <= 10)),
        "pct_within_20": float(np.mean(abs_err <= 20)),
    }


def calibration_by_predicted_bucket(y_true: np.ndarray, y_pred: np.ndarray, bucket_width: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    max_bucket = int(np.ceil(max(df["y_pred"].max(), 1) / bucket_width) * bucket_width) + bucket_width
    bins = list(range(0, max_bucket, bucket_width))
    df["bucket"] = pd.cut(df["y_pred"], bins=bins, right=False)
    out = df.groupby("bucket", observed=True).agg(
        n=("y_true", "count"), mean_predicted=("y_pred", "mean"), mean_actual=("y_true", "mean")
    ).reset_index()
    return out


def calibration_by_actual_bucket(y_true: np.ndarray, y_pred: np.ndarray, bucket_width: int = 20) -> pd.DataFrame:
    """Mirror of calibration_by_predicted_bucket, bucketed on y_true -- surfaces compression
    (a model that never predicts outside a narrow band) in a way predicted-bucketing can hide."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    max_bucket = int(np.ceil(max(df["y_true"].max(), 1) / bucket_width) * bucket_width) + bucket_width
    bins = list(range(0, max_bucket, bucket_width))
    df["bucket"] = pd.cut(df["y_true"], bins=bins, right=False)
    out = df.groupby("bucket", observed=True).agg(
        n=("y_true", "count"), mean_predicted=("y_pred", "mean"), mean_actual=("y_true", "mean")
    ).reset_index()
    return out


def std_range_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    std_actual = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")
    std_pred = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else float("nan")
    return {
        "n": len(y_true),
        "std_actual": std_actual,
        "std_pred": std_pred,
        "std_ratio_pred_over_actual": std_pred / std_actual if std_actual else float("nan"),
        "min_actual": float(np.min(y_true)),
        "max_actual": float(np.max(y_true)),
        "min_pred": float(np.min(y_pred)),
        "max_pred": float(np.max(y_pred)),
    }


def walk_forward_results(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model_name, fold_validation_season) metrics from a long OOF-predictions frame
    with columns model_name, fold_validation_season, y_true, y_pred."""
    rows = []
    for (model_name, val_season), group in oof_df.groupby(["model_name", "fold_validation_season"]):
        metrics = evaluate_predictions(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        metrics["model_name"] = model_name
        metrics["fold_validation_season"] = val_season
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_by_breakdown(df_with_preds: pd.DataFrame, breakdown_col: str, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    rows = []
    for value, group in df_with_preds.groupby(breakdown_col, observed=True):
        metrics = evaluate_predictions(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        metrics[breakdown_col] = value
        rows.append(metrics)
    return pd.DataFrame(rows)


def out_of_fold_residuals(oof_df: pd.DataFrame, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> np.ndarray:
    return (oof_df[y_true_col] - oof_df[y_pred_col]).to_numpy()


def prediction_interval_from_residuals(point_prediction: np.ndarray, residuals: np.ndarray, levels: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold residual quantile method: for a new prediction y_hat, the interval is
    [y_hat + quantile(resid, lo), y_hat + quantile(resid, hi)] using the pooled OOF residual
    distribution of the selected model."""
    lo_q, hi_q = np.quantile(residuals, levels[0]), np.quantile(residuals, levels[1])
    return point_prediction + lo_q, point_prediction + hi_q

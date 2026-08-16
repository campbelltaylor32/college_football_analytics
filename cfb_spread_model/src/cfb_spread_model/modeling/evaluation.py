"""Evaluation metrics. Precision is the primary metric everywhere (this model exists to place
bets, and a false positive is a bad bet) -- recall, coverage, and ROC-AUC are reported as
secondary context, never as a selection objective (see feature_selection/precision_scoring.py,
modeling/tuning.py, modeling/threshold_selection.py, all of which optimize precision)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cfb_spread_model.feature_selection.precision_scoring import coverage


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    preds = (y_score >= threshold).astype(int)
    return {
        "n": len(y_true),
        "threshold": threshold,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "coverage": coverage(y_score, threshold),
        "n_flagged": int(preds.sum()),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def probabilistic_fit_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Threshold-free metrics of how well predicted probabilities fit the data -- log_loss in
    particular is the classic overfitting diagnostic: a model that has memorized its training
    set produces training-set probabilities close to 0/1 (near-zero log_loss) that don't
    generalize, while a model that hasn't overfit shows a much smaller train-vs-holdout log_loss
    gap even if both numbers are mediocre. average_precision (area under the precision-recall
    curve) is reported alongside as a threshold-free companion to the single-threshold
    precision/recall numbers evaluate_predictions() reports."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    eps = 1e-15
    y_score_clipped = np.clip(y_score, eps, 1 - eps)
    return {
        "log_loss": log_loss(y_true, y_score_clipped) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def generalization_gap(reference_metrics: dict, comparison_metrics: dict, keys: tuple[str, ...]) -> dict:
    """reference_metrics MINUS comparison_metrics for each key -- e.g. train metrics minus
    holdout metrics. For precision/recall/roc_auc/average_precision, a large POSITIVE gap means
    the reference split (usually training) looks much better than the comparison split, the
    signature of overfitting. For log_loss, a large NEGATIVE gap (reference much lower than
    comparison) means the same thing, since lower log_loss is better -- flip the sign so every
    key in this function's output follows the same "positive = looks better on reference"
    convention regardless of whether the underlying metric is higher-is-better or lower-is-better.
    """
    gaps = {}
    for key in keys:
        if key not in reference_metrics or key not in comparison_metrics:
            continue
        diff = reference_metrics[key] - comparison_metrics[key]
        gaps[f"{key}_gap"] = -diff if key == "log_loss" else diff
    return gaps


def precision_recall_curve_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # sklearn's precision_recall_curve returns len(thresholds) == len(precision) - 1
    return pd.DataFrame({"threshold": np.append(thresholds, np.nan), "precision": precision, "recall": recall})


def calibration_by_predicted_bucket(y_true: np.ndarray, y_score: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df["bucket"] = pd.qcut(df["y_score"], q=n_buckets, duplicates="drop")
    return (
        df.groupby("bucket", observed=True)
        .agg(n=("y_true", "count"), mean_predicted=("y_score", "mean"), actual_cover_rate=("y_true", "mean"))
        .reset_index()
    )


def bucket_rank_monotonicity(bucket_df: pd.DataFrame) -> float:
    """Spearman correlation between bucket order (as returned by calibration_by_predicted_bucket
    -- ascending by predicted probability) and actual_cover_rate. +1 means the ranking is
    perfectly monotonic end to end (every bucket's actual rate is higher than the bucket below
    it, not just the extremes); near 0 means predicted probability carries no rank information;
    negative means the ranking is inverted."""
    if len(bucket_df) < 2:
        return float("nan")
    ranks = np.arange(len(bucket_df))
    corr, _ = spearmanr(ranks, bucket_df["actual_cover_rate"].to_numpy())
    return float(corr) if np.isfinite(corr) else float("nan")


def top_vs_bottom_summary(bucket_df: pd.DataFrame, y_true_overall: np.ndarray) -> dict:
    """Direct answer to "are the highest-predicted-probability games actually winners, and do
    the lowest-predicted-probability games correctly call the other side (home does NOT
    cover)." bucket_df is calibration_by_predicted_bucket's output (ascending order); the top
    bucket is its last row, the bottom bucket its first. Every rate is reported alongside its
    lift over the base rate, since a bucket "succeeding" at e.g. 50% means nothing on its own
    if the overall base rate is already ~49%."""
    y_true_overall = np.asarray(y_true_overall)
    base_rate = float(y_true_overall.mean())
    top = bucket_df.iloc[-1]
    bottom = bucket_df.iloc[0]
    top_rate = float(top["actual_cover_rate"])
    bottom_actual_cover_rate = float(bottom["actual_cover_rate"])
    bottom_other_side_rate = 1.0 - bottom_actual_cover_rate
    return {
        "n_buckets": len(bucket_df),
        "overall_base_rate": base_rate,
        "top_bucket_n": int(top["n"]),
        "top_bucket_mean_predicted": float(top["mean_predicted"]),
        "top_bucket_actual_rate": top_rate,
        "top_bucket_lift_vs_base_rate": top_rate - base_rate,
        "bottom_bucket_n": int(bottom["n"]),
        "bottom_bucket_mean_predicted": float(bottom["mean_predicted"]),
        "bottom_bucket_actual_cover_rate": bottom_actual_cover_rate,
        "bottom_bucket_other_side_rate": bottom_other_side_rate,
        "bottom_bucket_other_side_lift_vs_base_rate": bottom_other_side_rate - (1.0 - base_rate),
        "monotonicity": bucket_rank_monotonicity(bucket_df),
    }


def walk_forward_results(oof_df: pd.DataFrame, threshold_by_model: dict[str, float]) -> pd.DataFrame:
    """Per (model_name, fold_validation_season) metrics from a long OOF-predictions frame with
    columns model_name, fold_validation_season, y_true, y_score, evaluated at each model's own
    chosen threshold (modeling/threshold_selection.py's per-model selection)."""
    rows = []
    for (model_name, val_season), group in oof_df.groupby(["model_name", "fold_validation_season"]):
        threshold = threshold_by_model.get(model_name, 0.5)
        metrics = evaluate_predictions(group["y_true"].to_numpy(), group["y_score"].to_numpy(), threshold)
        metrics["model_name"] = model_name
        metrics["fold_validation_season"] = val_season
        rows.append(metrics)
    return pd.DataFrame(rows)

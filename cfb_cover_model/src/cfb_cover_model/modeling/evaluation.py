"""Shared scoring functions: precision-at-coverage-floor (the project's primary metric,
used for threshold selection, transform ablation, and final model comparison), plus
calibration diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score


def precision_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> tuple[float | None, float]:
    flagged = y_proba >= threshold
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return None, 0.0
    precision = float((np.asarray(y_true)[flagged] == 1).mean())
    coverage = n_flagged / len(y_true)
    return precision, coverage


def best_precision_at_coverage_floor(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold_grid: list[float],
    min_coverage_floor: float,
) -> dict:
    """Pick the threshold (from threshold_grid) with the highest precision among
    thresholds whose coverage clears min_coverage_floor. Falls back to the
    highest-coverage threshold if none clears the floor (reported honestly, not hidden)."""
    candidates = []
    for t in threshold_grid:
        precision, coverage = precision_at_threshold(y_true, y_proba, t)
        if precision is None:
            continue
        candidates.append((t, precision, coverage))

    meeting_floor = [c for c in candidates if c[2] >= min_coverage_floor]
    pool = meeting_floor if meeting_floor else candidates
    if not pool:
        return {"threshold": 0.5, "precision": None, "coverage": 0.0, "met_floor": False}

    best = max(pool, key=lambda c: c[1])
    return {
        "threshold": best[0],
        "precision": best[1],
        "coverage": best[2],
        "met_floor": bool(meeting_floor),
    }


def pooled_precision_at_threshold(
    fold_y_true: list[np.ndarray], fold_y_proba: list[np.ndarray], threshold: float
) -> dict:
    """Pool predictions across folds before computing one precision number, so fold size
    doesn't distort the result (a 60-game fold and a 260-game fold don't get equal weight
    if simply averaged)."""
    y_true = np.concatenate(fold_y_true)
    y_proba = np.concatenate(fold_y_proba)
    precision, coverage = precision_at_threshold(y_true, y_proba, threshold)
    return {"precision": precision, "coverage": coverage, "n": len(y_true)}


def calibration_report(y_true: np.ndarray, y_proba: np.ndarray, n_buckets: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    brier = float(brier_score_loss(y_true, y_proba))
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")

    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p").reset_index(drop=True)
    n_buckets = min(n_buckets, df["p"].nunique()) or 1
    df["bucket"] = pd.qcut(df["p"], q=n_buckets, labels=False, duplicates="drop")
    bucket_stats = (
        df.groupby("bucket")
        .agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    if len(bucket_stats) >= 2:
        monotonicity = float(
            spearmanr(bucket_stats["bucket"], bucket_stats["actual_rate"]).correlation
        )
    else:
        monotonicity = float("nan")

    return {
        "brier_score": brier,
        "roc_auc": auc,
        "rank_monotonicity": monotonicity,
        "buckets": bucket_stats.to_dict(orient="records"),
    }

"""Threshold selection: grid over config/modeling.yaml's precision_objective.candidate_thresholds
and pick the threshold that maximizes MEAN precision across walk-forward validation folds,
subject to a hard constraint -- coverage must clear min_coverage_floor in EVERY individual
fold, not just on average. A combo that meets the floor on average but drops to near-zero
coverage in one bad fold is a real "some seasons this model refuses to bet" failure mode a
bettor would notice, which an average-only constraint would hide.

Feature-count selection already happened per-fold in Stage 2 (feature_selection/selection.py),
scored by precision_at_coverage_floor. This module's job is choosing the single best operating
threshold given each model's out-of-fold predicted probabilities -- not re-running feature
selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from cfb_spread_model.feature_selection.precision_scoring import coverage
from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)


def fold_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    preds = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "coverage": coverage(y_score, threshold),
        "n_flagged": int(preds.sum()),
        "n_total": len(y_true),
    }


def evaluate_threshold_grid(oof_df: pd.DataFrame, candidate_thresholds: list[float]) -> pd.DataFrame:
    """oof_df: long frame with columns model_name, fold_validation_season, y_true, y_score.
    Returns one row per (model_name, fold_validation_season, threshold)."""
    rows = []
    for (model_name, season), group in oof_df.groupby(["model_name", "fold_validation_season"]):
        for threshold in candidate_thresholds:
            metrics = fold_metrics_at_threshold(group["y_true"].to_numpy(), group["y_score"].to_numpy(), threshold)
            metrics["model_name"] = model_name
            metrics["fold_validation_season"] = season
            rows.append(metrics)
    return pd.DataFrame(rows)


def select_best_threshold_per_model(grid_df: pd.DataFrame, min_coverage_floor: float) -> pd.DataFrame:
    """For each model, choose the threshold maximizing mean precision across folds subject to
    coverage >= min_coverage_floor in every individual fold. If no threshold clears the floor
    in every fold, falls back to whichever clears it in the most folds (tie-broken by mean
    precision) and marks meets_floor_every_fold=False so the shortfall is visible, not hidden.
    """
    results = []
    for model_name, model_group in grid_df.groupby("model_name"):
        summary = (
            model_group.groupby("threshold")
            .agg(
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_coverage=("coverage", "mean"),
                min_coverage=("coverage", "min"),
                n_folds_meeting_floor=("coverage", lambda s: int((s >= min_coverage_floor).sum())),
                n_folds=("coverage", "count"),
            )
            .reset_index()
        )
        summary["meets_floor_every_fold"] = summary["min_coverage"] >= min_coverage_floor

        qualifying = summary[summary["meets_floor_every_fold"]]
        if not qualifying.empty:
            best = qualifying.loc[qualifying["mean_precision"].idxmax()]
        else:
            logger.warning(
                f"{model_name}: no threshold met the coverage floor ({min_coverage_floor}) in "
                f"every fold; falling back to whichever threshold meets it in the most folds"
            )
            summary = summary.sort_values(["n_folds_meeting_floor", "mean_precision"], ascending=False)
            best = summary.iloc[0]

        results.append({"model_name": model_name, **best.to_dict()})

    return pd.DataFrame(results)

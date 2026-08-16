#!/usr/bin/env python
"""Overfitting vs. insufficient-signal diagnostic for the current production model: score it
on its OWN training rows, on pooled walk-forward out-of-fold predictions (5 seasons never seen
by that fold's fit), and on the true 2025 holdout (never seen at all), then compare.

A large train-vs-OOF/holdout gap (much better on training) is the signature of overfitting: the
model memorized training-set patterns that don't generalize. A SMALL gap where training itself
is only mediocre is the signature of insufficient signal: the model isn't even fitting its own
training data well, so there's nothing to "overfit" -- the features/model just don't carry much
predictive information for this problem at this sample size.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from cfb_spread_model.artifacts import load_latest_production_artifact
from cfb_spread_model.config import load_data_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.modeling import evaluation
from cfb_spread_model.modeling.holdout import load_holdout_frame
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
GAP_KEYS = ("precision", "recall", "roc_auc", "average_precision", "log_loss")


def score_split(model, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    y_score = model.predict_proba(X)[:, 1]
    metrics = evaluation.evaluate_predictions(y.to_numpy(), y_score, threshold)
    metrics.update(evaluation.probabilistic_fit_metrics(y.to_numpy(), y_score))
    return metrics


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    modeling_cfg = load_modeling_config()

    model, features, metadata = load_latest_production_artifact()
    model_name = metadata["model_name"]
    threshold = metadata["threshold"]
    train_seasons = metadata["trained_on_seasons"]

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)

    # --- Training split: the exact rows the saved pipeline was fit on ---
    train_df = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    X_train_full, y_train = build_feature_matrix(train_df, data_cfg)
    train_metrics = score_split(model, X_train_full[features], y_train, threshold)
    train_metrics["split"] = "train"

    # --- Walk-forward OOF, pooled across all 5 folds -- never seen by the fold that scored it ---
    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} missing -- run scripts/train_models.py first")
    oof_df = pd.read_csv(oof_path)
    model_oof = oof_df[oof_df["model_name"] == model_name]
    if model_oof.empty:
        raise ValueError(f"No OOF predictions found for production model '{model_name}' in {oof_path}")
    oof_metrics = evaluation.evaluate_predictions(
        model_oof["y_true"].to_numpy(), model_oof["y_score"].to_numpy(), threshold
    )
    oof_metrics.update(evaluation.probabilistic_fit_metrics(model_oof["y_true"].to_numpy(), model_oof["y_score"].to_numpy()))
    oof_metrics["split"] = "walk_forward_oof_pooled"

    # --- True 2025 holdout -- never seen at all ---
    val_df = load_holdout_frame(modeling_cfg)
    X_val_full, y_val = build_feature_matrix(val_df, data_cfg)
    holdout_metrics = score_split(model, X_val_full[features], y_val, threshold)
    holdout_metrics["split"] = "holdout_2025"

    comparison = pd.DataFrame([train_metrics, oof_metrics, holdout_metrics]).set_index("split")
    comparison = comparison[["n", "precision", "recall", "roc_auc", "average_precision", "log_loss", "coverage"]]

    gap_vs_oof = evaluation.generalization_gap(train_metrics, oof_metrics, GAP_KEYS)
    gap_vs_holdout = evaluation.generalization_gap(train_metrics, holdout_metrics, GAP_KEYS)

    today = date.today().strftime("%Y%m%d")
    out_path = OUTPUTS_MODEL_COMPARISON / f"train_vs_generalization_{today}.csv"
    comparison.to_csv(out_path)

    logger.info(f"{model_name} ({len(features)} features, threshold={threshold}):\n{comparison.to_string()}")
    logger.info(f"Train MINUS walk-forward-OOF gaps (positive = train looks better): {gap_vs_oof}")
    logger.info(f"Train MINUS 2025-holdout gaps (positive = train looks better): {gap_vs_holdout}")

    # Two independent axes, not a single threshold: CEILING (how good is the model even on the
    # data it was fit on -- a low ceiling means there isn't much signal to overfit in the first
    # place) and GAP (how much that ceiling erodes out of sample -- a real gap on top of a low
    # ceiling means what little signal exists doesn't transfer, i.e. some overfitting is still
    # happening even though it isn't "memorizing a strong pattern," just noise).
    train_roc_auc = train_metrics["roc_auc"]
    oof_roc_auc = oof_metrics["roc_auc"]
    holdout_roc_auc = holdout_metrics["roc_auc"]
    roc_auc_gap_oof = train_roc_auc - oof_roc_auc
    roc_auc_gap_holdout = train_roc_auc - holdout_roc_auc
    max_gap = max(roc_auc_gap_oof, roc_auc_gap_holdout)

    trivial_train_log_loss = log_loss(y_train, np.full(len(y_train), float(y_train.mean())))
    logger.info(
        f"ROC-AUC: train={train_roc_auc:.3f}, walk-forward OOF={oof_roc_auc:.3f} "
        f"(gap {roc_auc_gap_oof:+.3f}), 2025 holdout={holdout_roc_auc:.3f} (gap {roc_auc_gap_holdout:+.3f})"
    )
    logger.info(
        f"Train log_loss={train_metrics['log_loss']:.3f} vs. trivial constant-rate baseline="
        f"{trivial_train_log_loss:.3f} -- how much better than 'always predict the base rate' "
        f"the model is on the data it was FIT ON (a small gap here means the model barely "
        f"improves on the base rate even in-sample)"
    )

    ceiling_weak = train_roc_auc < 0.65
    gap_present = max_gap > 0.05
    if ceiling_weak and gap_present:
        verdict = (
            "MOSTLY INSUFFICIENT SIGNAL, with a modest secondary generalization gap: training "
            "performance itself is weak (ROC-AUC well under what a genuinely predictive model "
            "would show), so there isn't much real signal to overfit -- but what little "
            "in-sample signal exists doesn't fully survive out of sample either, eroding an "
            "already-weak fit down toward chance."
        )
    elif ceiling_weak:
        verdict = (
            "INSUFFICIENT SIGNAL: training and out-of-sample performance are both weak and "
            "similar -- the model isn't overfitting in any meaningful sense, it simply isn't "
            "finding much predictive signal in these features at this sample size."
        )
    elif gap_present:
        verdict = "OVERFITTING: training fit is strong but doesn't hold up out of sample -- the classic memorization signature."
    else:
        verdict = "Reasonable generalization: in-sample and out-of-sample performance are both solid and close to each other."
    logger.info(f"Verdict: {verdict}")

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_to_plot = ["precision", "recall", "roc_auc", "average_precision"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, split_name in enumerate(comparison.index):
        ax.bar(x + i * width, comparison.loc[split_name, metrics_to_plot], width, label=split_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_ylabel("Score")
    ax.set_title(f"Train vs. out-of-sample metrics -- {model_name} ({today})")
    ax.legend()
    fig.tight_layout()
    chart_path = OUTPUTS_MODEL_COMPARISON / "train_vs_generalization.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    logger.info(f"Wrote {out_path.name}, {chart_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

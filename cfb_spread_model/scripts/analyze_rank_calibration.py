#!/usr/bin/env python
"""Rank-calibration analysis for the current production model: sort games by predicted
probability of home_covered and ask (a) do the highest-probability games actually cover at a
high rate (the model's top picks are real winners), and (b) do the lowest-probability games
correctly call the OTHER side (home does NOT cover) at a high rate. Two views: pooled
walk-forward out-of-fold predictions (5 seasons, the honest cross-validated signal, already
produced by scripts/train_models.py) and the true 2025 final-holdout season (smaller n, zero
cross-validation reuse, scored with the already-fit saved pipeline -- no retraining).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cfb_spread_model.artifacts import load_latest_production_artifact
from cfb_spread_model.config import load_data_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.modeling import evaluation
from cfb_spread_model.modeling.holdout import load_holdout_frame
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import OUTPUTS_CALIBRATION, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)


def plot_buckets(bucket_df: pd.DataFrame, base_rate: float, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(1, len(bucket_df) + 1)
    ax.bar(x, bucket_df["actual_cover_rate"], color="steelblue")
    ax.axhline(base_rate, color="black", linestyle="--", linewidth=1, label=f"base rate ({base_rate:.2f})")
    ax.set_xticks(list(x))
    ax.set_xlabel("Bucket (1 = lowest predicted probability, N = highest)")
    ax.set_ylabel("Actual home_covered rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def log_summary(label: str, summary: dict) -> None:
    logger.info(
        f"[{label}] base rate={summary['overall_base_rate']:.3f}, monotonicity(Spearman)={summary['monotonicity']:.3f} | "
        f"TOP bucket (n={summary['top_bucket_n']}, mean predicted={summary['top_bucket_mean_predicted']:.3f}): "
        f"actual cover rate={summary['top_bucket_actual_rate']:.3f} "
        f"(lift {summary['top_bucket_lift_vs_base_rate']:+.3f} vs base rate) | "
        f"BOTTOM bucket (n={summary['bottom_bucket_n']}, mean predicted={summary['bottom_bucket_mean_predicted']:.3f}): "
        f"other-side (home does NOT cover) rate={summary['bottom_bucket_other_side_rate']:.3f} "
        f"(lift {summary['bottom_bucket_other_side_lift_vs_base_rate']:+.3f} vs base rate)"
    )


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    modeling_cfg = load_modeling_config()

    model, features, metadata = load_latest_production_artifact()
    model_name = metadata["model_name"]

    # --- View 1: pooled walk-forward out-of-fold predictions (5 seasons, honest OOS signal) ---
    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} missing -- run scripts/train_models.py first")
    oof_df = pd.read_csv(oof_path)
    model_oof = oof_df[oof_df["model_name"] == model_name]
    if model_oof.empty:
        raise ValueError(f"No OOF predictions found for production model '{model_name}' in {oof_path}")

    wf_buckets = evaluation.calibration_by_predicted_bucket(
        model_oof["y_true"].to_numpy(), model_oof["y_score"].to_numpy(), n_buckets=10
    )
    wf_summary = evaluation.top_vs_bottom_summary(wf_buckets, model_oof["y_true"].to_numpy())
    wf_summary["model_name"] = model_name
    wf_summary["view"] = "walk_forward_oof_pooled"

    wf_buckets.to_csv(OUTPUTS_CALIBRATION / "walk_forward_buckets.csv", index=False)
    with open(OUTPUTS_CALIBRATION / "walk_forward_summary.json", "w") as f:
        json.dump(wf_summary, f, indent=2)
    plot_buckets(
        wf_buckets,
        wf_summary["overall_base_rate"],
        f"Walk-forward OOF calibration -- {model_name} ({len(model_oof)} games, 5 seasons)",
        OUTPUTS_CALIBRATION / "walk_forward_calibration.png",
    )
    log_summary("walk-forward OOF (pooled, 5 seasons)", wf_summary)

    # --- View 2: true 2025 final-holdout season, scored with the already-fit saved pipeline ---
    val_df = load_holdout_frame(modeling_cfg)
    X_val_full, y_val = build_feature_matrix(val_df, data_cfg)
    X_val = X_val_full[features]
    y_score_holdout = model.predict_proba(X_val)[:, 1]

    holdout_buckets = evaluation.calibration_by_predicted_bucket(y_val.to_numpy(), y_score_holdout, n_buckets=5)
    holdout_summary = evaluation.top_vs_bottom_summary(holdout_buckets, y_val.to_numpy())
    holdout_summary["model_name"] = model_name
    holdout_summary["view"] = "final_holdout_2025"

    holdout_buckets.to_csv(OUTPUTS_CALIBRATION / "holdout_buckets.csv", index=False)
    with open(OUTPUTS_CALIBRATION / "holdout_summary.json", "w") as f:
        json.dump(holdout_summary, f, indent=2)
    plot_buckets(
        holdout_buckets,
        holdout_summary["overall_base_rate"],
        f"2025 final-holdout calibration -- {model_name} ({len(y_val)} games)",
        OUTPUTS_CALIBRATION / "holdout_calibration.png",
    )
    log_summary("2025 final holdout", holdout_summary)

    logger.info(f"Wrote bucket tables, summaries, and charts -> {OUTPUTS_CALIBRATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

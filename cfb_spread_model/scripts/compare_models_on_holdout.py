#!/usr/bin/env python
"""Is the model scripts/evaluate_models.py picked (by highest walk-forward mean precision)
actually the best model on the true 2025 holdout? evaluate_models.py only refits and scores the
WINNER on holdout -- this script refits and scores EVERY baseline + candidate model on the same
holdout season, using each model's own walk-forward-selected feature set and threshold
(outputs/threshold_selection/chosen_threshold_per_model.csv), so the walk-forward ranking and
the true out-of-sample ranking can be compared directly rather than assumed to agree.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import train_models as train_models_script

from cfb_spread_model.config import load_data_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.modeling import evaluation, tuning
from cfb_spread_model.modeling.fitting import fit_model
from cfb_spread_model.modeling.holdout import load_holdout_frame
from cfb_spread_model.modeling.splits import final_holdout_fold
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_THRESHOLD_SELECTION,
    ensure_dirs,
)

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    modeling_cfg = load_modeling_config()

    thresholds_path = OUTPUTS_THRESHOLD_SELECTION / "chosen_threshold_per_model.csv"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"{thresholds_path} missing -- run scripts/evaluate_models.py first")
    chosen = pd.read_csv(thresholds_path).set_index("model_name")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)
    holdout = final_holdout_fold(modeling_cfg)
    train_df = df[df["season"].isin(holdout.train_seasons)].reset_index(drop=True)
    val_df = load_holdout_frame(modeling_cfg)

    X_train_full, y_train = build_feature_matrix(train_df, data_cfg)
    X_val_full, y_val = build_feature_matrix(val_df, data_cfg)
    selected_features = train_models_script.load_selected_features(holdout.validation_season)
    cv_splits = tuning.build_inner_season_cv(train_df)

    all_model_names = [(name, True) for name in modeling_cfg.baseline_models] + [
        (name, False) for name in modeling_cfg.candidate_models
    ]

    rows = []
    for model_name, is_baseline in all_model_names:
        if model_name not in chosen.index:
            logger.warning(f"{model_name}: no chosen threshold found (walk-forward stage skipped it?); skipping")
            continue
        threshold = float(chosen.loc[model_name, "threshold"])
        wf_mean_precision = float(chosen.loc[model_name, "mean_precision"])

        feats = train_models_script.feature_set_for_model(model_name, selected_features, list(X_train_full.columns))
        X_train, X_val = X_train_full[feats], X_val_full[feats]

        try:
            fitted = fit_model(model_name, is_baseline, X_train, y_train, modeling_cfg, cv_splits)
        except Exception:
            logger.exception(f"{model_name} failed to fit on the final holdout training seasons; skipping")
            continue
        y_score = fitted.predict_proba(X_val)[:, 1]

        metrics = evaluation.evaluate_predictions(y_val.to_numpy(), y_score, threshold)
        metrics.update(evaluation.probabilistic_fit_metrics(y_val.to_numpy(), y_score))
        metrics["model_name"] = model_name
        metrics["is_baseline"] = is_baseline
        metrics["n_features"] = len(feats)
        metrics["walk_forward_mean_precision"] = wf_mean_precision
        rows.append(metrics)
        logger.info(
            f"  {model_name}: holdout precision={metrics['precision']:.3f} (walk-forward mean was {wf_mean_precision:.3f}), "
            f"holdout roc_auc={metrics['roc_auc']:.3f}, n_features={len(feats)}"
        )

    comparison = pd.DataFrame(rows)
    comparison["walk_forward_rank"] = comparison["walk_forward_mean_precision"].rank(ascending=False, method="min")
    comparison["holdout_rank"] = comparison["precision"].rank(ascending=False, method="min")
    comparison = comparison.sort_values("precision", ascending=False)

    today = date.today().strftime("%Y%m%d")
    out_path = OUTPUTS_MODEL_COMPARISON / f"holdout_model_comparison_{today}.csv"
    comparison.to_csv(out_path, index=False)

    cols = [
        "model_name",
        "is_baseline",
        "n_features",
        "walk_forward_mean_precision",
        "walk_forward_rank",
        "precision",
        "holdout_rank",
        "recall",
        "roc_auc",
        "log_loss",
    ]
    logger.info(f"Walk-forward vs. true 2025 holdout, ranked by holdout precision:\n{comparison[cols].to_string(index=False)}")

    holdout_winner = comparison.iloc[0]
    wf_winner_row = comparison.loc[comparison["walk_forward_rank"] == 1].iloc[0]
    if holdout_winner["model_name"] == wf_winner_row["model_name"]:
        logger.info(
            f"AGREEMENT: '{holdout_winner['model_name']}' wins on both walk-forward mean precision "
            f"and the true 2025 holdout (holdout precision={holdout_winner['precision']:.3f})."
        )
    else:
        logger.info(
            f"DISAGREEMENT: walk-forward selection picked '{wf_winner_row['model_name']}' "
            f"(walk-forward mean precision={wf_winner_row['walk_forward_mean_precision']:.3f}, "
            f"but only {wf_winner_row['precision']:.3f} on the true 2025 holdout), while "
            f"'{holdout_winner['model_name']}' actually scored highest on the true holdout "
            f"({holdout_winner['precision']:.3f})."
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["gray" if b else "steelblue" for b in comparison["is_baseline"]]
    ax.barh(comparison["model_name"], comparison["precision"], color=colors)
    ax.axvline(float(y_val.mean()), color="black", linestyle="--", linewidth=1, label="2025 base rate")
    ax.set_xlabel("2025 holdout precision")
    ax.set_title(f"Every model's TRUE 2025 holdout precision ({today})\n(gray = baseline, blue = candidate)")
    ax.legend()
    fig.tight_layout()
    chart_path = OUTPUTS_MODEL_COMPARISON / "holdout_model_comparison.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    logger.info(f"Wrote {out_path.name}, {chart_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

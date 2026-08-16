#!/usr/bin/env python
"""Everything so far (variance_aware_retune.py, penalty_weight_sweep.py, penalty_weight_single_check.py)
tested penalty_weight on walk-forward OOF folds only. This is the step promised in the report's
recommendations: refit under a given penalty_weight (default 0.5) on the TRUE final-holdout
training set (2015-2019, 2021-2024, same as scripts/evaluate_models.py's production refit) and
evaluate on the actual 2025 holdout season -- the same evaluation the shipped model is judged on
-- for all 3 contrast models, and compare directly against the real production
outputs/model_comparison/holdout_2025_predictions.csv.

EVALUATE-ONLY: writes its own final_model-equivalent artifacts under
outputs/diagnostics_compression/experiments/ -- does NOT touch
outputs/model_comparison/selected_model.json, outputs/models/final_model.joblib, or
outputs/predictions/predicted_win_totals_2025.csv.

Usage:
    python scripts/diagnostics/pw_final_holdout_check.py [penalty_weight]
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.dataset import NON_FEATURE_COLS
from cfb_win_total_model.modeling.evaluation import (
    evaluate_predictions,
    make_variance_aware_scorer,
    regression_slope_intercept,
    std_range_summary,
)
from cfb_win_total_model.modeling.models import get_candidate_models
from cfb_win_total_model.modeling.splits import final_holdout_fold
from cfb_win_total_model.modeling.train import TARGET_COL, fit_candidate_on_fold, get_feature_columns, predict_with_pipeline
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS,
    OUTPUTS_MODEL_COMPARISON,
    ensure_dirs,
)
from variance_aware_retune import TARGET_MODELS, WIDENED_GRIDS

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
MIN_STD_RATIO = 0.85


def main() -> int:
    ensure_dirs()
    penalty_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

    modeling_cfg = load_modeling_config()
    df = pd.read_parquet(DATASET_PATH)
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)

    holdout = final_holdout_fold(modeling_cfg)
    train_df = df[df["season"].isin(holdout.train_seasons)]
    val_df = df[df["season"] == holdout.validation_season]
    logger.info(
        f"Final holdout: train_seasons={holdout.train_seasons}, "
        f"validation_season={holdout.validation_season}, train n={len(train_df)}, val n={len(val_df)}"
    )

    scorer = make_variance_aware_scorer(min_std_ratio=MIN_STD_RATIO, penalty_weight=penalty_weight)
    models_dir = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    prod_holdout_preds = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "holdout_2025_predictions.csv")
    prod_holdout_results = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "holdout_2025_results.csv")

    all_preds = []
    summary_rows = []

    # Production baseline row (the real shipped gradient_boosting refit) for direct comparison.
    prod_summary = std_range_summary(prod_holdout_preds["y_true"], prod_holdout_preds["y_pred"])
    prod_slope_a_on_p, _ = regression_slope_intercept(prod_holdout_preds["y_pred"], prod_holdout_preds["y_true"])
    prod_slope_p_on_a, _ = regression_slope_intercept(prod_holdout_preds["y_true"], prod_holdout_preds["y_pred"])
    summary_rows.append(
        {
            "model_name": "gradient_boosting (SHIPPED PRODUCTION)",
            "penalty_weight": None,
            "n": prod_summary["n"],
            "mae": prod_holdout_results["mae"].iloc[0],
            "r2": prod_holdout_results["r2"].iloc[0],
            "mean_bias": prod_holdout_results["mean_bias"].iloc[0],
            "std_actual": prod_summary["std_actual"],
            "std_pred": prod_summary["std_pred"],
            "std_ratio_pred_over_actual": prod_summary["std_ratio_pred_over_actual"],
            "slope_actual_on_pred": prod_slope_a_on_p,
            "slope_pred_on_actual": prod_slope_p_on_a,
        }
    )

    for model_name in TARGET_MODELS:
        estimator = get_candidate_models([model_name], modeling_cfg.random_seed)[model_name]
        pipeline = fit_candidate_on_fold(estimator, WIDENED_GRIDS[model_name], train_df, feature_cols, scoring=scorer)
        full_model_name = f"{model_name}__pw_{penalty_weight}_FINAL_HOLDOUT"
        joblib.dump(pipeline, models_dir / f"{full_model_name}.joblib")

        preds = predict_with_pipeline(pipeline, val_df, feature_cols)
        pred_df = val_df[["school", "season"]].copy()
        pred_df["y_true"] = val_df[TARGET_COL].to_numpy()
        pred_df["y_pred"] = preds
        pred_df["model_name"] = full_model_name
        all_preds.append(pred_df)

        metrics = evaluate_predictions(val_df[TARGET_COL], preds)
        summary = std_range_summary(val_df[TARGET_COL], preds)
        slope_a_on_p, _ = regression_slope_intercept(preds, val_df[TARGET_COL])
        slope_p_on_a, _ = regression_slope_intercept(val_df[TARGET_COL], preds)
        summary_rows.append(
            {
                "model_name": full_model_name,
                "penalty_weight": penalty_weight,
                "n": summary["n"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "mean_bias": metrics["mean_bias"],
                "std_actual": summary["std_actual"],
                "std_pred": summary["std_pred"],
                "std_ratio_pred_over_actual": summary["std_ratio_pred_over_actual"],
                "slope_actual_on_pred": slope_a_on_p,
                "slope_pred_on_actual": slope_p_on_a,
            }
        )
        logger.info(f"{full_model_name}: MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}, std_ratio={summary['std_ratio_pred_over_actual']:.3f}")

    preds_out = pd.concat(all_preds, ignore_index=True)
    preds_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / f"pw_{penalty_weight}_final_holdout_predictions.csv"
    preds_out.to_csv(preds_path, index=False)
    logger.info(f"Wrote holdout predictions -> {preds_path}")

    summary_out = pd.DataFrame(summary_rows)
    summary_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / f"pw_{penalty_weight}_final_holdout_results.csv"
    summary_out.to_csv(summary_path, index=False)
    logger.info(f"Final holdout (2025) results, pw={penalty_weight} vs. shipped production:\n{summary_out.to_string(index=False)}")
    logger.info(f"Wrote results -> {summary_path}")

    logger.info(
        "EVALUATE-ONLY: no production artifact (selected_model.json, final_model.joblib, "
        "predicted_win_totals_2025.csv) was modified by this script."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

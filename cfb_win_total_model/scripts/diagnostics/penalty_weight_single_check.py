#!/usr/bin/env python
"""Extension of penalty_weight_sweep.py: re-checks a single penalty_weight (passed as argv[1],
default 5.0) for the same 3-model contrast set and widened grids, and -- unlike the original
sweep, which only recorded OOF/val metrics -- also saves each fold's fitted pipeline so a
train-vs-val comparison can be run immediately after (the same underfitting/overfitting check
already done for pw=0.5, 1.0, and 5.0 in docs/project_story.md). Appends its rows to the
existing penalty_weight_sweep_results.csv rather than replacing it.

EVALUATE-ONLY: nothing here touches outputs/model_comparison/selected_model.json,
outputs/models/final_model.joblib, or outputs/predictions/predicted_win_totals_2025.csv.

Usage:
    python scripts/diagnostics/penalty_weight_single_check.py [penalty_weight]
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
    make_variance_aware_scorer,
    regression_slope_intercept,
    std_range_summary,
    walk_forward_results,
)
from cfb_win_total_model.modeling.models import get_candidate_models
from cfb_win_total_model.modeling.splits import generate_walk_forward_folds
from cfb_win_total_model.modeling.train import TARGET_COL, fit_candidate_on_fold, get_feature_columns, predict_with_pipeline
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS,
    OUTPUTS_MODEL_COMPARISON,
    ensure_dirs,
)
from variance_aware_retune import TARGET_MODELS, WIDENED_GRIDS, _oof_frame

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
MIN_STD_RATIO = 0.85
PENALTY_WEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0


def main() -> int:
    ensure_dirs()
    modeling_cfg = load_modeling_config()
    df = pd.read_parquet(DATASET_PATH)
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    folds = generate_walk_forward_folds(modeling_cfg)

    models_dir = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    scorer = make_variance_aware_scorer(min_std_ratio=MIN_STD_RATIO, penalty_weight=PENALTY_WEIGHT)
    oof_rows = []
    train_rows = []
    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)]
        val_df = df[df["season"] == fold.validation_season]
        for model_name in TARGET_MODELS:
            estimator = get_candidate_models([model_name], modeling_cfg.random_seed)[model_name]
            pipeline = fit_candidate_on_fold(estimator, WIDENED_GRIDS[model_name], train_df, feature_cols, scoring=scorer)
            full_model_name = f"{model_name}__pw_{PENALTY_WEIGHT}"
            joblib.dump(pipeline, models_dir / f"fold_{fold.validation_season}_{full_model_name}.joblib")

            val_preds = predict_with_pipeline(pipeline, val_df, feature_cols)
            oof_rows.append(_oof_frame(val_df, val_preds, full_model_name, fold.validation_season))

            train_preds = predict_with_pipeline(pipeline, train_df, feature_cols)
            train_summary = std_range_summary(train_df[TARGET_COL], train_preds)
            train_slope_p_on_a, _ = regression_slope_intercept(train_df[TARGET_COL], train_preds)
            train_rows.append(
                {
                    "model_name": model_name,
                    "fold_validation_season": fold.validation_season,
                    "train_n": train_summary["n"],
                    "train_std_ratio": train_summary["std_ratio_pred_over_actual"],
                    "train_slope_pred_on_actual": train_slope_p_on_a,
                }
            )
            logger.info(f"Fold {fold.validation_season}, {model_name}, pw={PENALTY_WEIGHT}: done")

    oof_df = pd.concat(oof_rows, ignore_index=True)
    oof_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / f"pw_{PENALTY_WEIGHT}_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)

    train_df_out = pd.DataFrame(train_rows)
    train_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / f"pw_{PENALTY_WEIGHT}_train_metrics.csv"
    train_df_out.to_csv(train_path, index=False)

    baseline_oof = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv")
    baseline_oof = baseline_oof[baseline_oof["model_name"].isin(TARGET_MODELS)]
    for model_name in TARGET_MODELS:
        n_baseline = len(baseline_oof[baseline_oof["model_name"] == model_name])
        n_variant = len(oof_df[oof_df["model_name"] == f"{model_name}__pw_{PENALTY_WEIGHT}"])
        assert n_baseline == n_variant, f"{model_name}: row count mismatch ({n_variant} vs {n_baseline})"
    logger.info("Sanity check PASSED: row counts match production baseline")

    combined = pd.concat([baseline_oof, oof_df], ignore_index=True)
    wf = walk_forward_results(combined)
    rows = []
    for model_name in TARGET_MODELS:
        for name, variant_label, weight in [
            (model_name, "baseline (production, MAE-only)", None),
            (f"{model_name}__pw_{PENALTY_WEIGHT}", f"variance_aware pw={PENALTY_WEIGHT}", PENALTY_WEIGHT),
        ]:
            model_oof = combined[combined["model_name"] == name]
            model_wf = wf[wf["model_name"] == name]
            summary = std_range_summary(model_oof["y_true"], model_oof["y_pred"])
            slope_a_on_p, _ = regression_slope_intercept(model_oof["y_pred"], model_oof["y_true"])
            slope_p_on_a, _ = regression_slope_intercept(model_oof["y_true"], model_oof["y_pred"])
            rows.append(
                {
                    "base_model": model_name,
                    "model_name": name,
                    "penalty_weight": weight,
                    "variant": variant_label,
                    "mean_fold_mae": model_wf["mae"].mean(),
                    "std_fold_mae": model_wf["mae"].std(),
                    "n_oof": summary["n"],
                    "std_actual": summary["std_actual"],
                    "std_pred": summary["std_pred"],
                    "std_ratio_pred_over_actual": summary["std_ratio_pred_over_actual"],
                    "slope_actual_on_pred": slope_a_on_p,
                    "slope_pred_on_actual": slope_p_on_a,
                }
            )
    result = pd.DataFrame(rows)
    result["mae_pct_change_vs_baseline"] = result.groupby("base_model")["mean_fold_mae"].transform(lambda s: (s / s.iloc[0] - 1) * 100)

    # Merge train-side ratios in for the pw=5.0 rows only (baseline's train numbers already
    # exist in outputs/diagnostics_compression/tables/train_vs_val_metrics_by_fold.csv).
    train_mean = train_df_out.groupby("model_name")[["train_std_ratio", "train_slope_pred_on_actual"]].mean().reset_index()
    train_mean["base_model"] = train_mean["model_name"]
    result = result.merge(train_mean[["base_model", "train_std_ratio", "train_slope_pred_on_actual"]], on="base_model", how="left")
    result.loc[result["variant"].str.contains("baseline"), ["train_std_ratio", "train_slope_pred_on_actual"]] = None

    result_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / f"pw_{PENALTY_WEIGHT}_vs_baseline_comparison.csv"
    result.to_csv(result_path, index=False)
    logger.info(f"pw={PENALTY_WEIGHT} results (val + train std_ratio):\n{result.to_string(index=False)}")

    # Append to the master sweep table for a single combined view.
    sweep_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "penalty_weight_sweep_results.csv"
    if sweep_path.exists():
        master = pd.read_csv(sweep_path)
        new_rows = result[result["penalty_weight"].notna()].drop(columns=["train_std_ratio", "train_slope_pred_on_actual"])
        master = pd.concat([master, new_rows], ignore_index=True)
        master.to_csv(sweep_path, index=False)
        logger.info(f"Appended pw={PENALTY_WEIGHT} rows to {sweep_path}")

    logger.info("EVALUATE-ONLY: no production artifact modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

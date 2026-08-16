#!/usr/bin/env python
"""Follow-up to compute_compression_diagnostics.py: tests whether (a) widening the
hyperparameter grids and/or (b) tuning against a variance-aware objective instead of plain MAE
actually reduces prediction compression, for the 3 models already used as the diagnostic
contrast set (gradient_boosting, ridge, elasticnet). Every candidate model in the production
grid was found pinned at its shrinkage-favoring extreme under plain neg_mean_absolute_error
scoring (regularization_grid_boundary_check.csv) -- this isolates whether that's a grid-range
problem, a scoring-objective problem, or both.

Three variants are compared per model, all using the SAME production feature set (132 columns,
no diagnostic feature additions -- this experiment is about tuning, not features) and the SAME
walk-forward folds as production:
  1. baseline            -- current production grid + neg_mean_absolute_error (read from the
                             existing outputs/model_comparison/oof_predictions.csv, not refit)
  2. widened_grid_mae    -- widened grid + neg_mean_absolute_error (control: does grid range
                             alone matter if the objective doesn't change?)
  3. widened_grid_variance_aware -- widened grid + modeling.evaluation.variance_aware_score

This is EVALUATE-ONLY: nothing here touches outputs/model_comparison/selected_model.json,
outputs/models/final_model.joblib, or outputs/predictions/predicted_win_totals_2025.csv. All
new artifacts go under outputs/diagnostics_compression/experiments/.

Usage:
    python scripts/diagnostics/variance_aware_retune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
TARGET_MODELS = ["gradient_boosting", "ridge", "elasticnet"]

# Widened relative to config/modeling.yaml's hyperparam_grids -- extended in BOTH directions
# (more and less regularized) so the variance-aware scorer has real alternatives to choose
# from, not just a wider ceiling on shrinkage.
WIDENED_GRIDS = {
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]},
    "elasticnet": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.95]},
    # Trimmed from an initial 60-combo grid (3x5x4) that took over 45 minutes per fold/variant
    # combination and never finished -- still meaningfully wider than production's 12-combo
    # grid in both directions (deeper trees + faster learning available), just not exhaustive.
    "gradient_boosting": {
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.03, 0.1, 0.2],
    },
}

_HYPERPARAM_ATTRS = {"ridge": ["alpha"], "elasticnet": ["alpha", "l1_ratio"], "gradient_boosting": ["n_estimators", "max_depth", "learning_rate"]}

SCORING_VARIANTS = {
    "widened_grid_mae": "neg_mean_absolute_error",
    "widened_grid_variance_aware": make_variance_aware_scorer(min_std_ratio=0.85, penalty_weight=1.0),
}


def _oof_frame(val_df: pd.DataFrame, preds, model_name: str, validation_season: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "school": val_df["school"].to_numpy(),
            "season": val_df["season"].to_numpy(),
            "fold_validation_season": validation_season,
            "model_name": model_name,
            "y_true": val_df[TARGET_COL].to_numpy(),
            "y_pred": preds,
        }
    )


def _selected_hyperparams(pipeline, model_name: str) -> dict:
    estimator = pipeline.named_steps["model"]
    return {attr: getattr(estimator, attr) for attr in _HYPERPARAM_ATTRS[model_name] if hasattr(estimator, attr)}


def run_retune(df: pd.DataFrame, modeling_cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    folds = generate_walk_forward_folds(modeling_cfg)
    candidates = get_candidate_models(TARGET_MODELS, modeling_cfg.random_seed)

    oof_rows = []
    hyperparam_rows = []
    models_dir = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)]
        val_df = df[df["season"] == fold.validation_season]
        for model_name in TARGET_MODELS:
            for variant_name, scoring in SCORING_VARIANTS.items():
                estimator = get_candidate_models([model_name], modeling_cfg.random_seed)[model_name]
                pipeline = fit_candidate_on_fold(estimator, WIDENED_GRIDS[model_name], train_df, feature_cols, scoring=scoring)
                preds = predict_with_pipeline(pipeline, val_df, feature_cols)
                full_model_name = f"{model_name}__{variant_name}"
                oof_rows.append(_oof_frame(val_df, preds, full_model_name, fold.validation_season))
                joblib.dump(pipeline, models_dir / f"fold_{fold.validation_season}_{full_model_name}.joblib")

                selected = _selected_hyperparams(pipeline, model_name)
                hyperparam_rows.append(
                    {"model_name": full_model_name, "fold_validation_season": fold.validation_season, **selected}
                )
                logger.info(f"Fold {fold.validation_season}, {full_model_name}: selected {selected}")

    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(hyperparam_rows)


def build_comparison_table(baseline_oof: pd.DataFrame, retuned_oof: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([baseline_oof, retuned_oof], ignore_index=True)
    wf = walk_forward_results(combined)
    rows = []
    for base_model in TARGET_MODELS:
        model_names = [base_model] + [f"{base_model}__{v}" for v in SCORING_VARIANTS]
        for model_name in model_names:
            model_oof = combined[combined["model_name"] == model_name]
            if model_oof.empty:
                continue
            model_wf = wf[wf["model_name"] == model_name]
            summary = std_range_summary(model_oof["y_true"], model_oof["y_pred"])
            slope_a_on_p, _ = regression_slope_intercept(model_oof["y_pred"], model_oof["y_true"])
            slope_p_on_a, _ = regression_slope_intercept(model_oof["y_true"], model_oof["y_pred"])
            rows.append(
                {
                    "base_model": base_model,
                    "model_name": model_name,
                    "variant": "baseline (production)" if model_name == base_model else model_name.split("__", 1)[1],
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
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found -- run scripts/build_modeling_dataset.py first")

    modeling_cfg = load_modeling_config()
    df = pd.read_parquet(DATASET_PATH)

    logger.info(f"Retuning {TARGET_MODELS} under widened grids x {list(SCORING_VARIANTS)} (evaluate-only, no promotion)...")
    retuned_oof, hyperparams_df = run_retune(df, modeling_cfg)

    retuned_oof_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "variance_aware_retune_oof_predictions.csv"
    retuned_oof.to_csv(retuned_oof_path, index=False)
    logger.info(f"Wrote {len(retuned_oof)} retuned OOF predictions -> {retuned_oof_path}")

    hyperparams_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "variance_aware_retune_selected_hyperparams.csv"
    hyperparams_df.to_csv(hyperparams_path, index=False)
    logger.info(f"Wrote selected-hyperparameters table -> {hyperparams_path}")

    baseline_oof = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv")
    baseline_oof = baseline_oof[baseline_oof["model_name"].isin(TARGET_MODELS)]

    for base_model in TARGET_MODELS:
        n_baseline = len(baseline_oof[baseline_oof["model_name"] == base_model])
        for variant_name in SCORING_VARIANTS:
            n_variant = len(retuned_oof[retuned_oof["model_name"] == f"{base_model}__{variant_name}"])
            assert n_baseline == n_variant, (
                f"{base_model}__{variant_name}: row count ({n_variant}) != baseline ({n_baseline}) -- "
                "retuning likely dropped or duplicated rows"
            )
    logger.info("Sanity check PASSED: every retuned variant's OOF row count matches the production baseline")

    comparison = build_comparison_table(baseline_oof, retuned_oof)
    comparison_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "variance_aware_retune_vs_baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Variance-aware retune comparison:\n{comparison.to_string(index=False)}")
    logger.info(f"Wrote comparison table -> {comparison_path}")
    logger.info(
        "EVALUATE-ONLY: no production artifact (selected_model.json, final_model.joblib, "
        "predicted_win_totals_2025.csv) was modified by this script."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

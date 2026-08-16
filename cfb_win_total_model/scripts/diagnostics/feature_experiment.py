#!/usr/bin/env python
"""Bounded diagnostic experiment: does adding a handful of new features/interaction terms
measurably reduce prediction compression? Re-runs the *identical* walk-forward CV protocol
(same folds, same fitting/tuning logic as the production pipeline) for gradient_boosting (the
shipped model) and elasticnet (a competitive linear contrast), with vs. without the new
columns from features/diagnostic_experiments.py, and compares OOF MAE/std/slope.

Writes only to outputs/diagnostics_compression/experiments/ -- never touches
outputs/model_comparison/ or outputs/models/. New model names are suffixed "__plus_interactions"
so they can never collide with a production model_name.

Usage:
    python scripts/diagnostics/feature_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.dataset import NON_FEATURE_COLS
from cfb_win_total_model.features.diagnostic_experiments import (
    DIAGNOSTIC_FEATURE_COLUMNS,
    augment_with_diagnostic_features,
)
from cfb_win_total_model.modeling.evaluation import (
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
EXPERIMENT_MODELS = ["gradient_boosting", "elasticnet"]
EXPERIMENT_SUFFIX = "__plus_interactions"


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


def run_experiment(df: pd.DataFrame, modeling_cfg) -> pd.DataFrame:
    augmented_df = augment_with_diagnostic_features(df)
    base_feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    feature_cols = base_feature_cols + DIAGNOSTIC_FEATURE_COLUMNS
    logger.info(f"Base features: {len(base_feature_cols)}; augmented with {len(DIAGNOSTIC_FEATURE_COLUMNS)} diagnostic columns")

    folds = generate_walk_forward_folds(modeling_cfg)
    candidates = get_candidate_models(EXPERIMENT_MODELS, modeling_cfg.random_seed)

    oof_rows = []
    models_dir = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        train_df = augmented_df[augmented_df["season"].isin(fold.train_seasons)]
        val_df = augmented_df[augmented_df["season"] == fold.validation_season]
        for name, estimator in candidates.items():
            param_grid = modeling_cfg.hyperparam_grids.get(name)
            pipeline = fit_candidate_on_fold(estimator, param_grid, train_df, feature_cols)
            preds = predict_with_pipeline(pipeline, val_df, feature_cols)
            oof_rows.append(_oof_frame(val_df, preds, f"{name}{EXPERIMENT_SUFFIX}", fold.validation_season))
            joblib.dump(pipeline, models_dir / f"fold_{fold.validation_season}_{name}{EXPERIMENT_SUFFIX}.joblib")
            logger.info(f"Fold {fold.validation_season}, {name}{EXPERIMENT_SUFFIX}: fit on {len(train_df)} rows, scored {len(val_df)} rows")

    return pd.concat(oof_rows, ignore_index=True)


def build_comparison_table(baseline_oof: pd.DataFrame, experiment_oof: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([baseline_oof, experiment_oof], ignore_index=True)
    wf = walk_forward_results(combined)
    rows = []
    for base_model in EXPERIMENT_MODELS:
        for model_name in (base_model, f"{base_model}{EXPERIMENT_SUFFIX}"):
            model_oof = combined[combined["model_name"] == model_name]
            model_wf = wf[wf["model_name"] == model_name]
            summary = std_range_summary(model_oof["y_true"], model_oof["y_pred"])
            slope_a_on_p, _ = regression_slope_intercept(model_oof["y_pred"], model_oof["y_true"])
            slope_p_on_a, _ = regression_slope_intercept(model_oof["y_true"], model_oof["y_pred"])
            rows.append(
                {
                    "base_model": base_model,
                    "model_name": model_name,
                    "has_diagnostic_features": model_name.endswith(EXPERIMENT_SUFFIX),
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

    logger.info("Running walk-forward CV with diagnostic features added (gradient_boosting, elasticnet)...")
    experiment_oof = run_experiment(df, modeling_cfg)
    experiment_oof_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "feature_experiment_oof_predictions.csv"
    experiment_oof.to_csv(experiment_oof_path, index=False)
    logger.info(f"Wrote {len(experiment_oof)} augmented-feature OOF predictions -> {experiment_oof_path}")

    baseline_oof = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv")
    baseline_oof = baseline_oof[baseline_oof["model_name"].isin(EXPERIMENT_MODELS)]

    for base_model in EXPERIMENT_MODELS:
        n_baseline = len(baseline_oof[baseline_oof["model_name"] == base_model])
        n_experiment = len(experiment_oof[experiment_oof["model_name"] == f"{base_model}{EXPERIMENT_SUFFIX}"])
        assert n_baseline == n_experiment, (
            f"{base_model}: baseline OOF row count ({n_baseline}) != augmented row count ({n_experiment}) "
            "-- the feature augmentation likely dropped or duplicated rows"
        )
        logger.info(f"Sanity check PASSED: {base_model} row count matches baseline ({n_baseline} rows)")

    comparison = build_comparison_table(baseline_oof, experiment_oof)
    comparison_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "feature_experiment_vs_baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Feature experiment comparison:\n{comparison.to_string(index=False)}")
    logger.info(f"Wrote comparison table -> {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

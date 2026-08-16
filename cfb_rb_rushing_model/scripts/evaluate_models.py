#!/usr/bin/env python
"""Step 5 of the pipeline: walk-forward evaluation, model selection, refit on all eligible
seasons, final-holdout evaluation + diagnostics, and production artifact save (consumed by
scripts/generate_week_predictions.py via modeling/artifacts.py's glob-latest lookup).
Reads outputs/model_comparison/oof_predictions.csv (written by scripts/train_models.py).

Usage:
    python scripts/evaluate_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_rb_rushing_model.config import load_modeling_config
from cfb_rb_rushing_model.dataset import NON_FEATURE_COLS
from cfb_rb_rushing_model.modeling import diagnostics
from cfb_rb_rushing_model.modeling.baselines import get_baselines
from cfb_rb_rushing_model.modeling.evaluation import (
    calibration_by_predicted_bucket,
    evaluate_by_breakdown,
    evaluate_predictions,
    walk_forward_results,
)
from cfb_rb_rushing_model.modeling.models import get_candidate_models
from cfb_rb_rushing_model.modeling.splits import final_holdout_fold
from cfb_rb_rushing_model.modeling.train import TARGET_COL, fit_candidate_on_fold, get_feature_columns, predict_with_pipeline
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_DIAGNOSTICS, OUTPUTS_MODEL_COMPARISON, OUTPUTS_MODELS, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

BREAKDOWN_COLS = {"season": "season", "played": "played"}


def _select_model(walk_forward: pd.DataFrame) -> dict:
    summary = walk_forward.groupby("model_name")["mae"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values(["mean", "std"], na_position="last")
    best = summary.iloc[0]
    return {
        "model_name": best["model_name"],
        "mean_mae": float(best["mean"]),
        "std_mae": float(best["std"]) if pd.notna(best["std"]) else None,
        "selection_rationale": "Lowest mean OOF MAE across walk-forward folds; ties broken by lowest MAE std (stability).",
        "all_models_ranked": summary.to_dict(orient="records"),
    }


def main() -> int:
    ensure_dirs()
    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} not found -- run scripts/train_models.py first")

    oof_df = pd.read_csv(oof_path)
    walk_forward = walk_forward_results(oof_df)
    walk_forward.to_csv(OUTPUTS_MODEL_COMPARISON / "walk_forward_results.csv", index=False)
    logger.info(f"Walk-forward MAE by model:\n{walk_forward.groupby('model_name')['mae'].mean().sort_values()}")

    selection = _select_model(walk_forward)
    (OUTPUTS_MODEL_COMPARISON / "selected_model.json").write_text(json.dumps(selection, indent=2, default=str))
    logger.info(f"Selected model: {selection['model_name']} (mean OOF MAE={selection['mean_mae']:.3f})")

    modeling_cfg = load_modeling_config()
    df = pd.read_parquet(DATASET_PATH)
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    holdout = final_holdout_fold(modeling_cfg)
    train_df = df[df["season"].isin(holdout.train_seasons)]
    val_df = df[df["season"] == holdout.validation_season]

    model_name = selection["model_name"]
    baselines = get_baselines([model_name])
    if model_name in baselines:
        final_model = baselines[model_name]
        final_model.fit(train_df)
        final_preds = final_model.predict(val_df)
        metadata_model_type = "baseline"
    else:
        candidates = get_candidate_models([model_name], modeling_cfg.random_seed)
        estimator = candidates[model_name]
        param_grid = modeling_cfg.hyperparam_grids.get(model_name)
        final_model = fit_candidate_on_fold(estimator, param_grid, train_df, feature_cols)
        final_preds = predict_with_pipeline(final_model, val_df, feature_cols)
        metadata_model_type = "sklearn_pipeline"

    today = date.today().isoformat()
    joblib.dump(final_model, OUTPUTS_MODELS / f"best_model_{today}.joblib")
    (OUTPUTS_MODELS / f"selected_features_{today}.json").write_text(json.dumps(feature_cols, indent=2))

    holdout_df = val_df[["athlete_id", "game_id", "team", "opponent", "season", "week"]].copy()
    holdout_df["y_true"] = val_df[TARGET_COL].to_numpy()
    holdout_df["y_pred"] = final_preds
    for col in ("played", "carries_avg3_asof", "rushing_yards_avg3_asof"):
        if col in val_df.columns:
            holdout_df[col] = val_df[col].to_numpy()

    holdout_metrics = evaluate_predictions(holdout_df["y_true"].to_numpy(), holdout_df["y_pred"].to_numpy())
    logger.info(f"Final holdout ({holdout.validation_season}) metrics: {holdout_metrics}")
    pd.DataFrame([holdout_metrics]).to_csv(OUTPUTS_MODEL_COMPARISON / "holdout_results.csv", index=False)
    holdout_df.to_csv(OUTPUTS_MODEL_COMPARISON / "holdout_predictions.csv", index=False)

    for label, col in BREAKDOWN_COLS.items():
        if col in holdout_df.columns:
            breakdown = evaluate_by_breakdown(holdout_df, col)
            breakdown.to_csv(OUTPUTS_MODEL_COMPARISON / f"holdout_breakdown_{label}.csv", index=False)

    model_oof = oof_df[oof_df["model_name"] == model_name]
    metadata = {
        "model_name": model_name,
        "model_type": metadata_model_type,
        "trained_on_seasons": holdout.train_seasons,
        "final_holdout_season": holdout.validation_season,
        "final_holdout_metrics": holdout_metrics,
        "prediction_interval_method": modeling_cfg.prediction_interval_method,
        "prediction_interval_levels": list(modeling_cfg.prediction_interval_levels),
        "clip_min_yards": modeling_cfg.clip_min_yards,
        "n_oof_residuals": int(len(model_oof)),
        "trained_date": today,
    }
    (OUTPUTS_MODELS / f"model_metadata_{today}.json").write_text(json.dumps(metadata, indent=2))

    diagnostics.plot_actual_vs_predicted(holdout_df, OUTPUTS_DIAGNOSTICS / "actual_vs_predicted.png")
    diagnostics.plot_residuals(holdout_df, by=None, path=OUTPUTS_DIAGNOSTICS / "residuals_distribution.png")
    calib = calibration_by_predicted_bucket(holdout_df["y_true"].to_numpy(), holdout_df["y_pred"].to_numpy())
    calib.to_csv(OUTPUTS_DIAGNOSTICS / "calibration.csv", index=False)
    diagnostics.plot_calibration(calib, OUTPUTS_DIAGNOSTICS / "calibration.png")

    misses = diagnostics.largest_misses(holdout_df, n=15)
    misses.to_csv(OUTPUTS_DIAGNOSTICS / "largest_misses.csv", index=False)

    if hasattr(final_model, "named_steps"):
        model_step = final_model.named_steps["model"]
        try:
            preprocessor = final_model.named_steps["preprocess"]
            feature_names_out = list(preprocessor.get_feature_names_out())
            fi = diagnostics.feature_importance(model_step, feature_names_out)
            fi.to_csv(OUTPUTS_DIAGNOSTICS / "feature_importance.csv", index=False)

            X_holdout = val_df[feature_cols]
            perm = diagnostics.permutation_importance_report(final_model, X_holdout, holdout_df["y_true"], feature_cols)
            perm.to_csv(OUTPUTS_DIAGNOSTICS / "permutation_importance.csv", index=False)
        except Exception as e:
            logger.warning(f"Feature/permutation importance failed: {e}")

    logger.info(f"Diagnostics written to {OUTPUTS_DIAGNOSTICS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

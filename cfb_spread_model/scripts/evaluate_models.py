#!/usr/bin/env python
"""Threshold selection (modeling/threshold_selection.py) on the walk-forward OOF predictions
scripts/train_models.py produced, selection of the final production (model, feature_set,
threshold), and a final refit + evaluation on the 2025 final-holdout season. Saves the
production artifact to outputs/models/ and a machine-readable comparison against the current
notebook's documented baseline (see CURRENT_NOTEBOOK_BASELINE below) to
outputs/model_comparison/final_summary.json -- the source for docs/project_story.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import json
from datetime import date

import joblib
import pandas as pd

import train_models as train_models_script

from cfb_spread_model.config import load_data_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.modeling import evaluation, threshold_selection, tuning
from cfb_spread_model.modeling.fitting import fit_model
from cfb_spread_model.modeling.splits import final_holdout_fold
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_MODELS,
    OUTPUTS_THRESHOLD_SELECTION,
    ensure_dirs,
)

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

# Documented baseline from the current live notebook pipeline
# (../Python Scripts/CFB_Gambling_Model.ipynb, ../Model Information/selected_features_best_model_20250915.json,
# 52 features), evaluated on a SINGLE FIXED 2023-2024 test split -- not walk-forward. See
# docs/project_story.md for the full writeup; verified via nbconvert during this project's planning.
CURRENT_NOTEBOOK_BASELINE = {
    "precision": 0.569,
    "recall": 0.298,
    "coverage": 0.25,
    "n_features": 52,
    "threshold": 0.60,
    "selection_metric": "roc_auc (not precision)",
    "evaluation": "single fixed split, train=2015-2022, test=2023-2024",
}


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    modeling_cfg = load_modeling_config()

    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} missing -- run scripts/train_models.py first")
    oof_df = pd.read_csv(oof_path)

    grid_df = threshold_selection.evaluate_threshold_grid(oof_df, modeling_cfg.precision_objective.candidate_thresholds)
    grid_df.to_csv(OUTPUTS_THRESHOLD_SELECTION / "grid_all_models.csv", index=False)
    for model_name, group in grid_df.groupby("model_name"):
        group.to_csv(OUTPUTS_THRESHOLD_SELECTION / f"grid_{model_name}.csv", index=False)

    best_df = threshold_selection.select_best_threshold_per_model(grid_df, modeling_cfg.precision_objective.min_coverage_floor)
    best_df.to_csv(OUTPUTS_THRESHOLD_SELECTION / "chosen_threshold_per_model.csv", index=False)
    logger.info(
        "Chosen thresholds:\n"
        + best_df[["model_name", "threshold", "mean_precision", "mean_coverage", "meets_floor_every_fold"]].to_string(index=False)
    )

    threshold_by_model = dict(zip(best_df["model_name"], best_df["threshold"]))
    wf_results = evaluation.walk_forward_results(oof_df, threshold_by_model)
    wf_results.to_csv(OUTPUTS_MODEL_COMPARISON / "walk_forward_results.csv", index=False)

    qualifying = best_df[best_df["meets_floor_every_fold"]]
    candidates_only = qualifying[qualifying["model_name"].isin(modeling_cfg.candidate_models)]
    pool = candidates_only if not candidates_only.empty else qualifying
    if pool.empty:
        pool = best_df
    winner = pool.loc[pool["mean_precision"].idxmax()]
    winner_name = str(winner["model_name"])
    winner_threshold = float(winner["threshold"])
    logger.info(f"Selected production model: {winner_name} @ threshold={winner_threshold:.2f} (mean_precision={winner['mean_precision']:.3f})")

    # Final refit on the final-holdout fold's training seasons, using the feature set Stage 2
    # selected for THAT fold (outputs/feature_analysis/selected_features_fold_<2025>.json).
    df = pd.read_parquet(DATASET_PATH)
    holdout = final_holdout_fold(modeling_cfg)
    train_df = df[df["season"].isin(holdout.train_seasons)].reset_index(drop=True)
    val_df = df[df["season"] == holdout.validation_season].reset_index(drop=True)
    X_train_full, y_train = build_feature_matrix(train_df, data_cfg)
    X_val_full, y_val = build_feature_matrix(val_df, data_cfg)

    selected_features = train_models_script.load_selected_features(holdout.validation_season)
    is_baseline = winner_name in modeling_cfg.baseline_models
    feats = train_models_script.feature_set_for_model(winner_name, selected_features, list(X_train_full.columns))
    X_train, X_val = X_train_full[feats], X_val_full[feats]
    cv_splits = tuning.build_inner_season_cv(train_df)

    final_pipeline = fit_model(winner_name, is_baseline, X_train, y_train, modeling_cfg, cv_splits)

    y_score_holdout = final_pipeline.predict_proba(X_val)[:, 1]
    holdout_metrics = evaluation.evaluate_predictions(y_val.to_numpy(), y_score_holdout, winner_threshold)
    logger.info(f"2025 final-holdout metrics for {winner_name}: {holdout_metrics}")

    today = date.today().strftime("%Y%m%d")
    model_path = OUTPUTS_MODELS / f"best_model_{today}.pkl"
    features_path = OUTPUTS_MODELS / f"selected_features_{today}.json"
    metadata_path = OUTPUTS_MODELS / f"model_metadata_{today}.json"
    joblib.dump(final_pipeline, model_path)
    with open(features_path, "w") as f:
        json.dump(feats, f, indent=2)

    metadata = {
        "model_name": winner_name,
        "threshold": winner_threshold,
        "n_features": len(feats),
        "trained_on_seasons": holdout.train_seasons,
        "walk_forward_mean_precision": float(winner["mean_precision"]),
        "walk_forward_mean_coverage": float(winner["mean_coverage"]),
        "walk_forward_meets_floor_every_fold": bool(winner["meets_floor_every_fold"]),
        "final_holdout_season": holdout.validation_season,
        "final_holdout_metrics": holdout_metrics,
        "current_notebook_baseline": CURRENT_NOTEBOOK_BASELINE,
        "beats_baseline_precision": float(winner["mean_precision"]) >= CURRENT_NOTEBOOK_BASELINE["precision"],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(OUTPUTS_MODEL_COMPARISON / "final_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved production artifact: {model_path.name}, {features_path.name}, {metadata_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

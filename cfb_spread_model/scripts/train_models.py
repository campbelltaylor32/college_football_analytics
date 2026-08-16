#!/usr/bin/env python
"""Walk-forward fit of every baseline + candidate model (config/modeling.yaml `models`), one
fold at a time, each fold using the feature set scripts/select_features.py chose for it
(outputs/feature_analysis/selected_features_fold_<season>.json). Hyperparameter tuning uses
season-ordered inner CV and a precision-focused scorer (modeling/tuning.py), never
scoring="roc_auc". Produces a long out-of-fold predictions frame (model_name,
fold_validation_season, game_id, y_true, y_score) consumed by scripts/evaluate_models.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import json

import pandas as pd

from cfb_spread_model.config import load_data_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.modeling import models as models_module
from cfb_spread_model.modeling import tuning
from cfb_spread_model.modeling.fitting import fit_model
from cfb_spread_model.modeling.splits import generate_walk_forward_folds
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_FEATURE_ANALYSIS, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def load_selected_features(validation_season: int) -> list[str]:
    path = OUTPUTS_FEATURE_ANALYSIS / f"selected_features_fold_{validation_season}.json"
    with open(path) as f:
        return json.load(f)["selected_features"]


def feature_set_for_model(model_name: str, selected_features: list[str], all_columns: list[str]) -> list[str]:
    if model_name in models_module.BYPASS_FEATURE_SELECTION:
        return list(all_columns)
    feats = list(selected_features)
    for col in models_module.REQUIRES_RAW_CONTEXT_COLUMNS.get(model_name, []):
        if col not in feats and col in all_columns:
            feats.append(col)
    return feats


def fit_and_score(model_name, is_baseline, X_train, y_train, X_val, modeling_cfg, cv_splits):
    fitted = fit_model(model_name, is_baseline, X_train, y_train, modeling_cfg, cv_splits)
    return fitted.predict_proba(X_val)[:, 1]


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    modeling_cfg = load_modeling_config()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)

    folds = generate_walk_forward_folds(modeling_cfg)
    all_model_names = [(name, True) for name in modeling_cfg.baseline_models] + [
        (name, False) for name in modeling_cfg.candidate_models
    ]

    oof_rows = []
    for fold in folds:
        logger.info(f"=== Training fold validation_season={fold.validation_season} ===")
        train_df = df[df["season"].isin(fold.train_seasons)].reset_index(drop=True)
        val_df = df[df["season"] == fold.validation_season].reset_index(drop=True)

        X_train_full, y_train = build_feature_matrix(train_df, data_cfg)
        X_val_full, y_val = build_feature_matrix(val_df, data_cfg)

        selected_features = load_selected_features(fold.validation_season)
        cv_splits = tuning.build_inner_season_cv(train_df)

        for model_name, is_baseline in all_model_names:
            feats = feature_set_for_model(model_name, selected_features, list(X_train_full.columns))
            X_train, X_val = X_train_full[feats], X_val_full[feats]
            try:
                y_score = fit_and_score(model_name, is_baseline, X_train, y_train, X_val, modeling_cfg, cv_splits)
            except Exception:
                logger.exception(f"{model_name} failed on fold {fold.validation_season}; skipping")
                continue

            for game_id, yt, ys in zip(val_df["game_id"], y_val, y_score):
                oof_rows.append(
                    {
                        "model_name": model_name,
                        "fold_validation_season": fold.validation_season,
                        "game_id": game_id,
                        "y_true": int(yt),
                        "y_score": float(ys),
                    }
                )
            logger.info(f"  {model_name}: fit on {len(X_train)} rows / {len(feats)} features, scored {len(X_val)} val rows")

    oof_df = pd.DataFrame(oof_rows)
    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Wrote {len(oof_df)} OOF prediction rows -> {oof_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Step 5 of the pipeline. Fits every baseline + candidate model across all walk-forward
folds (splits.generate_walk_forward_folds), producing out-of-fold predictions for every
(school, season, model) triple. Candidate model pipelines are cached to outputs/models/;
baselines are cheap and refit on demand rather than persisted (ConferenceAverageBaseline
holds a live DB engine reference, which isn't reliably picklable).

Usage:
    python scripts/train_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_win_total_model.config import load_features_config, load_modeling_config
from cfb_win_total_model.database import get_engine
from cfb_win_total_model.dataset import NON_FEATURE_COLS
from cfb_win_total_model.modeling.baselines import get_baselines
from cfb_win_total_model.modeling.models import get_candidate_models
from cfb_win_total_model.modeling.splits import generate_walk_forward_folds
from cfb_win_total_model.modeling.train import TARGET_COL, fit_candidate_on_fold, get_feature_columns, predict_with_pipeline
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_MODELS, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def main() -> int:
    ensure_dirs()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found -- run scripts/build_modeling_dataset.py first")

    modeling_cfg = load_modeling_config()
    features_cfg = load_features_config()
    engine = get_engine()

    df = pd.read_parquet(DATASET_PATH)
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    logger.info(f"Loaded modeling dataset {df.shape}, {len(feature_cols)} feature columns")

    folds = generate_walk_forward_folds(modeling_cfg)
    baselines = get_baselines(modeling_cfg.baseline_models, engine)
    candidates = get_candidate_models(modeling_cfg.candidate_models, modeling_cfg.random_seed)
    logger.info(f"{len(folds)} walk-forward folds; baselines={list(baselines)}; candidates={list(candidates)}")

    oof_rows = []
    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)]
        val_df = df[df["season"] == fold.validation_season]
        if train_df.empty or val_df.empty:
            logger.warning(f"Fold val_season={fold.validation_season} has empty train/val split; skipping")
            continue
        logger.info(f"Fold val_season={fold.validation_season}: train={len(train_df)} rows, val={len(val_df)} rows")

        for name, baseline in baselines.items():
            baseline.fit(train_df)
            preds = baseline.predict(val_df)
            oof_rows.append(_oof_frame(val_df, preds, name, fold.validation_season))

        for name, estimator in candidates.items():
            param_grid = modeling_cfg.hyperparam_grids.get(name)
            pipeline = fit_candidate_on_fold(estimator, param_grid, train_df, feature_cols)
            preds = predict_with_pipeline(pipeline, val_df, feature_cols)
            oof_rows.append(_oof_frame(val_df, preds, name, fold.validation_season))

            model_path = OUTPUTS_MODELS / f"fold_{fold.validation_season}_{name}.joblib"
            joblib.dump(pipeline, model_path)

    oof_df = pd.concat(oof_rows, ignore_index=True)
    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Wrote {len(oof_df)} OOF predictions -> {oof_path}")
    return 0


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


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Step 4 of the pipeline. Fits every baseline + candidate model across all walk-forward
folds (splits.generate_walk_forward_folds), producing out-of-fold predictions for every
(athlete_id, game_id, model) triple. Candidate model pipelines are cached to outputs/models/;
baselines are cheap and refit on demand rather than persisted.

Resumable by design: OOF predictions are appended to outputs/model_comparison/oof_predictions.csv
incrementally, one (fold, model) pair at a time, and any (fold, model) pair already present in
that file is skipped on a re-run. This makes an interrupted run (e.g. a killed process partway
through a long grid search) cheap to continue rather than needing a full restart -- pass
--rebuild to ignore existing progress and start clean.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_rb_rushing_model.config import load_modeling_config
from cfb_rb_rushing_model.dataset import NON_FEATURE_COLS
from cfb_rb_rushing_model.modeling.baselines import get_baselines
from cfb_rb_rushing_model.modeling.models import get_candidate_models
from cfb_rb_rushing_model.modeling.splits import generate_walk_forward_folds
from cfb_rb_rushing_model.modeling.train import TARGET_COL, fit_candidate_on_fold, get_feature_columns, predict_with_pipeline
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_MODELS, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
OOF_PATH = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"


def _load_completed_pairs(rebuild: bool) -> set[tuple[str, int]]:
    if rebuild and OOF_PATH.exists():
        OOF_PATH.unlink()
        return set()
    if not OOF_PATH.exists():
        return set()
    existing = pd.read_csv(OOF_PATH, usecols=["model_name", "fold_validation_season"])
    return set(zip(existing["model_name"], existing["fold_validation_season"]))


def _append_oof(df: pd.DataFrame) -> None:
    df.to_csv(OOF_PATH, mode="a", header=not OOF_PATH.exists(), index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing oof_predictions.csv progress and start clean")
    args = parser.parse_args()

    ensure_dirs()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found -- run scripts/build_modeling_dataset.py first")

    modeling_cfg = load_modeling_config()

    df = pd.read_parquet(DATASET_PATH)
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    logger.info(f"Loaded modeling dataset {df.shape}, {len(feature_cols)} feature columns")

    folds = generate_walk_forward_folds(modeling_cfg)
    baselines = get_baselines(modeling_cfg.baseline_models)
    candidates = get_candidate_models(modeling_cfg.candidate_models, modeling_cfg.random_seed)
    logger.info(f"{len(folds)} walk-forward folds; baselines={list(baselines)}; candidates={list(candidates)}")

    completed = _load_completed_pairs(args.rebuild)
    if completed:
        logger.info(f"Resuming: {len(completed)} (model, fold) pairs already recorded in {OOF_PATH}")

    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)]
        val_df = df[df["season"] == fold.validation_season]
        if train_df.empty or val_df.empty:
            logger.warning(f"Fold val_season={fold.validation_season} has empty train/val split; skipping")
            continue
        logger.info(f"Fold val_season={fold.validation_season}: train={len(train_df)} rows, val={len(val_df)} rows")

        for name, baseline in baselines.items():
            if (name, fold.validation_season) in completed:
                continue
            baseline.fit(train_df)
            preds = baseline.predict(val_df)
            _append_oof(_oof_frame(val_df, preds, name, fold.validation_season))
            logger.info(f"  [{fold.validation_season}] {name}: done")

        for name, estimator in candidates.items():
            if (name, fold.validation_season) in completed:
                continue
            param_grid = modeling_cfg.hyperparam_grids.get(name)
            pipeline = fit_candidate_on_fold(estimator, param_grid, train_df, feature_cols)
            preds = predict_with_pipeline(pipeline, val_df, feature_cols)
            _append_oof(_oof_frame(val_df, preds, name, fold.validation_season))

            model_path = OUTPUTS_MODELS / f"fold_{fold.validation_season}_{name}.joblib"
            joblib.dump(pipeline, model_path)
            logger.info(f"  [{fold.validation_season}] {name}: done")

    logger.info(f"OOF predictions complete -> {OOF_PATH}")
    return 0


def _oof_frame(val_df: pd.DataFrame, preds, model_name: str, validation_season: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "athlete_id": val_df["athlete_id"].to_numpy(),
            "game_id": val_df["game_id"].to_numpy(),
            "season": val_df["season"].to_numpy(),
            "fold_validation_season": validation_season,
            "model_name": model_name,
            "y_true": val_df[TARGET_COL].to_numpy(),
            "y_pred": preds,
        }
    )


if __name__ == "__main__":
    sys.exit(main())

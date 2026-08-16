#!/usr/bin/env python
"""Answers a specific question: is the notebook's reported 0.569 precision explained by its
feature set, or by its single fixed 2023-2024 test split? Takes the EXACT 52 features saved by
the current notebook pipeline (../Model Information/selected_features_best_model_<date>.json)
and scores them two ways, using this project's own leakage-safe fitting code
(modeling/fitting.py) for both so the fitting methodology is held constant:

  A) This project's standard walk-forward validation (5 seasons, COVID-excluded) -- the
     leakage-safe, multi-season estimate.
  B) A reproduction of the notebook's OWN single split (train=2015-2022 INCLUDING 2020, since
     the notebook never excluded it; test=2023-2024) -- as close to the notebook's exact
     evaluation conditions as this project's tooling can get.

If (B) lands close to 0.569 while (A) lands well below it, that's direct evidence the single
split -- not the feature set -- is what makes the notebook's number look better. If (B) is
ALSO well below 0.569, something else (hyperparameters, a code difference, random seed) is
doing the work instead, which would be a distinct and equally important finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import json
from datetime import date

import pandas as pd

from cfb_spread_model.config import load_modeling_config
from cfb_spread_model.modeling import evaluation, tuning
from cfb_spread_model.modeling.fitting import fit_model
from cfb_spread_model.modeling.splits import generate_walk_forward_folds
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_MODEL_COMPARISON, REPO_ROOT, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
NOTEBOOK_FEATURES_DIR = REPO_ROOT / "Model Information"
NOTEBOOK_THRESHOLD = 0.60
NOTEBOOK_REPORTED_PRECISION = 0.569
LABEL_COL = "home_covered"


def latest_notebook_features() -> tuple[list[str], Path]:
    matches = sorted(NOTEBOOK_FEATURES_DIR.glob("selected_features_best_model_*.json"))
    if not matches:
        raise FileNotFoundError(f"No selected_features_best_model_*.json found in {NOTEBOOK_FEATURES_DIR}")
    path = matches[-1]
    with open(path) as f:
        return json.load(f), path


def main() -> int:
    ensure_dirs()
    modeling_cfg = load_modeling_config()

    feats, feats_path = latest_notebook_features()
    n_prev_week = sum("prev_week_" in f for f in feats)
    logger.info(f"Loaded {len(feats)} features from {feats_path.name} ({n_prev_week} are prev_week_*)")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} notebook features not found in this project's dataset: {missing}")

    # --- A) This project's standard walk-forward validation, restricted to the notebook's
    # exact 52 features (no Stage 1/2 selection -- the feature list is fixed) ---
    folds = generate_walk_forward_folds(modeling_cfg)
    oof_rows = []
    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)].reset_index(drop=True)
        val_df = df[df["season"] == fold.validation_season].reset_index(drop=True)
        X_train, y_train = train_df[feats], train_df[LABEL_COL]
        X_val, y_val = val_df[feats], val_df[LABEL_COL]
        cv_splits = tuning.build_inner_season_cv(train_df)

        fitted = fit_model("xgboost", False, X_train, y_train, modeling_cfg, cv_splits)
        y_score = fitted.predict_proba(X_val)[:, 1]
        for yt, ys in zip(y_val, y_score):
            oof_rows.append({"fold_validation_season": fold.validation_season, "y_true": int(yt), "y_score": float(ys)})
        logger.info(f"  walk-forward fold {fold.validation_season}: fit on {len(X_train)} rows, scored {len(X_val)} rows")

    oof_df = pd.DataFrame(oof_rows)
    walk_forward_metrics = evaluation.evaluate_predictions(oof_df["y_true"].to_numpy(), oof_df["y_score"].to_numpy(), NOTEBOOK_THRESHOLD)
    walk_forward_metrics.update(
        evaluation.probabilistic_fit_metrics(oof_df["y_true"].to_numpy(), oof_df["y_score"].to_numpy())
    )

    # --- B) Reproduction of the notebook's OWN single split: train=2015-2022 (2020 INCLUDED,
    # since the notebook never excludes it), test=2023-2024 ---
    train_df = df[df["season"].between(2015, 2022)].reset_index(drop=True)
    test_df = df[df["season"].between(2023, 2024)].reset_index(drop=True)
    X_train, y_train = train_df[feats], train_df[LABEL_COL]
    X_test, y_test = test_df[feats], test_df[LABEL_COL]
    cv_splits = tuning.build_inner_season_cv(train_df)

    fitted = fit_model("xgboost", False, X_train, y_train, modeling_cfg, cv_splits)
    y_score_test = fitted.predict_proba(X_test)[:, 1]
    single_split_metrics = evaluation.evaluate_predictions(y_test.to_numpy(), y_score_test, NOTEBOOK_THRESHOLD)
    single_split_metrics.update(evaluation.probabilistic_fit_metrics(y_test.to_numpy(), y_score_test))
    logger.info(f"  single split: fit on {len(X_train)} rows (2015-2022, incl. 2020), scored {len(X_test)} rows (2023-2024)")

    comparison = pd.DataFrame(
        [
            {"evaluation": "notebook_reported", "n": 492, "precision": NOTEBOOK_REPORTED_PRECISION, "recall": 0.298, "roc_auc": float("nan"), "log_loss": float("nan")},
            {"evaluation": "reproduced_single_split_2023_2024", **{k: single_split_metrics[k] for k in ("n", "precision", "recall", "roc_auc", "log_loss")}},
            {"evaluation": "walk_forward_pooled_5_seasons", **{k: walk_forward_metrics[k] for k in ("n", "precision", "recall", "roc_auc", "log_loss")}},
        ]
    ).set_index("evaluation")

    today = date.today().strftime("%Y%m%d")
    out_path = OUTPUTS_MODEL_COMPARISON / f"replicate_notebook_features_{today}.csv"
    comparison.to_csv(out_path)

    logger.info(f"Same 52 features (notebook's), threshold={NOTEBOOK_THRESHOLD}, this project's fitting code for both rows below:\n{comparison.to_string()}")

    gap = single_split_metrics["precision"] - walk_forward_metrics["precision"]
    reproduction_gap = NOTEBOOK_REPORTED_PRECISION - single_split_metrics["precision"]
    logger.info(
        f"Reproduced single-split precision ({single_split_metrics['precision']:.3f}) vs. "
        f"walk-forward pooled precision ({walk_forward_metrics['precision']:.3f}): gap = {gap:+.3f}"
    )
    logger.info(
        f"Reproduced single-split precision ({single_split_metrics['precision']:.3f}) vs. "
        f"notebook's own reported precision ({NOTEBOOK_REPORTED_PRECISION:.3f}): gap = {reproduction_gap:+.3f} "
        f"(near zero means the split, not the notebook's specific code/hyperparameters, explains its number)"
    )
    logger.info(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

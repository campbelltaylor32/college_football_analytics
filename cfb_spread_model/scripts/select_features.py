#!/usr/bin/env python
"""Stage 1 + Stage 2 feature selection (src/cfb_spread_model/feature_selection/), run
independently per walk-forward fold plus the final holdout fold, on that fold's training
seasons only -- never fit on validation/holdout rows (see
tests/test_leakage.py::test_correlation_pruning_never_fit_on_validation_rows).

The final holdout fold's result is what scripts/evaluate_models.py uses as the PRODUCTION
feature set; the walk-forward folds' results feed scripts/train_models.py's out-of-fold
precision estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cfb_spread_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.feature_selection import correlation_pruning, selection
from cfb_spread_model.feature_selection.precision_scoring import precision_at_coverage_floor_scorer
from cfb_spread_model.modeling.splits import final_holdout_fold, generate_walk_forward_folds
from cfb_spread_model.modeling.tuning import build_inner_season_cv
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_FEATURE_ANALYSIS, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def select_features_for_fold(df, fold, data_cfg, features_cfg, modeling_cfg):
    train_df = df[df["season"].isin(fold.train_seasons)].reset_index(drop=True)
    X_full, y = build_feature_matrix(train_df, data_cfg)

    pruned_cols, prune_report = correlation_pruning.prune(X_full, y, features_cfg.correlation_pruning)
    X_pruned = X_full[pruned_cols]

    scorer = precision_at_coverage_floor_scorer(modeling_cfg.precision_objective.min_coverage_floor)
    cv_splits = build_inner_season_cv(train_df)

    result = {
        "validation_season": fold.validation_season,
        "train_seasons": fold.train_seasons,
        "n_train_rows": len(train_df),
        "n_columns_before_pruning": len(X_full.columns),
        "n_columns_after_pruning": len(pruned_cols),
    }

    if len(cv_splits) < 2:
        logger.warning(f"Fold {fold.validation_season}: fewer than 2 inner CV splits; using correlation-pruned columns as-is")
        result["selected_features"] = pruned_cols
        result["permutation_sweep_best_n"] = None
        result["rfecv_features"] = None
        return result, prune_report, pd.DataFrame()

    train_idx, val_idx = cv_splits[-1]
    X_tr, y_tr = X_pruned.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X_pruned.iloc[val_idx], y.iloc[val_idx]

    importances = selection.compute_permutation_importance(
        X_tr, y_tr, X_val, y_val, scorer, features_cfg.permutation_importance_n_repeats, modeling_cfg.random_seed
    )
    sweep_result = selection.sweep_feature_counts(
        X_tr, y_tr, X_val, y_val, importances, features_cfg.candidate_feature_counts, scorer, modeling_cfg.random_seed
    )
    rfecv_features = selection.run_rfecv(X_pruned, y, cv_splits, scorer, modeling_cfg.random_seed, min_features_to_select=10)

    result["selected_features"] = sweep_result.selected_features
    result["permutation_sweep_best_n"] = sweep_result.best_n_features
    result["rfecv_features"] = rfecv_features
    result["rfecv_n_features"] = len(rfecv_features)

    return result, prune_report, sweep_result.sweep_table


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)

    folds = generate_walk_forward_folds(modeling_cfg) + [final_holdout_fold(modeling_cfg)]
    all_results = []
    for fold in folds:
        logger.info(f"=== Selecting features for fold validation_season={fold.validation_season} ===")
        result, prune_report, sweep_table = select_features_for_fold(df, fold, data_cfg, features_cfg, modeling_cfg)
        all_results.append(result)

        prune_report.to_csv(OUTPUTS_FEATURE_ANALYSIS / f"correlation_pruning_report_fold_{fold.validation_season}.csv", index=False)
        if not sweep_table.empty:
            sweep_table.to_csv(OUTPUTS_FEATURE_ANALYSIS / f"permutation_sweep_fold_{fold.validation_season}.csv", index=False)
        with open(OUTPUTS_FEATURE_ANALYSIS / f"selected_features_fold_{fold.validation_season}.json", "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"Fold {fold.validation_season}: {result['n_columns_before_pruning']} -> "
            f"{result['n_columns_after_pruning']} (pruning) -> {len(result['selected_features'])} (selected)"
        )

    with open(OUTPUTS_FEATURE_ANALYSIS / "feature_selection_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Feature selection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

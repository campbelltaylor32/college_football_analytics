#!/usr/bin/env python
"""Stage 2/3/4/5: pick the winning temporal-transform + home/away-representation
combination (walk-forward only, holdout untouched), then compare reduction strategies
(no reduction / correlation+embedded / PCA) on that winning combination - also walk-forward
only. Writes the winning configuration to outputs/feature_analysis/ for train_models.py to
consume; does not fit or persist any single "final" feature list, since every walk-forward
fold and the eventual production fit each refit their own reduction per data_leakage_rules.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.selection import apply_feature_set, fit_feature_set
from cfb_cover_model.feature_selection.transform_ablation import run_transform_ablation
from cfb_cover_model.modeling.evaluation import best_precision_at_coverage_floor
from cfb_cover_model.modeling.splits import get_holdout_split, walk_forward_folds

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis"


def compare_reduction_strategies(
    train_pool: pd.DataFrame,
    feature_columns: list[str],
    features_cfg: dict,
    modeling_cfg: dict,
    random_state: int = 42,
) -> list[dict]:
    """Probe-model walk-forward comparison of the three feature_set modes, on the winning
    transform/representation combo already baked into `train_pool`/`feature_columns`."""
    label = train_pool["home_covered"]
    folds = walk_forward_folds(train_pool, modeling_cfg)
    results = []

    for mode in ("deterministic_pruned_only", "reduced", "pca_reduced"):
        fold_true, fold_proba, fold_n_features = [], [], []
        for fold in folds:
            X_tr = train_pool.loc[fold.train_idx, feature_columns]
            X_va = train_pool.loc[fold.val_idx, feature_columns]
            y_tr = label.loc[fold.train_idx]
            y_va = label.loc[fold.val_idx]
            season_tr = train_pool.loc[fold.train_idx, "season"]

            artifact, _report = fit_feature_set(
                X_tr, y_tr, season_tr, mode, features_cfg, random_state
            )
            X_tr_reduced = apply_feature_set(X_tr, mode, artifact)
            X_va_reduced = apply_feature_set(X_va, mode, artifact)
            fold_n_features.append(X_tr_reduced.shape[1])

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_reduced)
            X_va_s = scaler.transform(X_va_reduced)

            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=3000, random_state=random_state)
            clf.fit(X_tr_s, y_tr)
            proba = clf.predict_proba(X_va_s)[:, 1]

            fold_true.append(y_va.to_numpy())
            fold_proba.append(proba)

        y_true_pooled = np.concatenate(fold_true)
        y_proba_pooled = np.concatenate(fold_proba)
        pooled = best_precision_at_coverage_floor(
            y_true_pooled,
            y_proba_pooled,
            modeling_cfg["evaluation"]["threshold_grid"],
            modeling_cfg["evaluation"]["min_coverage_floor"],
        )
        results.append(
            {
                "feature_set_mode": mode,
                "mean_n_features": float(np.mean(fold_n_features)),
                "pooled": pooled,
            }
        )

    results.sort(key=lambda r: (r["pooled"]["precision"] if r["pooled"]["precision"] is not None else -1.0), reverse=True)
    return results


def main() -> None:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    train_pool, _holdout = get_holdout_split(frame, data_cfg)

    print("Running temporal-transform / home-away-representation ablation (walk-forward only)...")
    winner, ablation_results = run_transform_ablation(
        train_pool, feature_columns, features_cfg, modeling_cfg, random_state=modeling_cfg["random_state"]
    )
    print(f"Winning combo: {winner['transform_name']} / {winner['representation']} "
          f"(pooled precision={winner['pooled']['precision']}, n_features={winner['n_features']})")

    winning_variant, winning_cols = build_transform_variant(
        train_pool, feature_columns, winner["transforms"]
    )
    winning_variant, winning_cols = apply_home_away_representation(winning_variant, winning_cols, winner["representation"])
    winning_variant = winning_variant.reset_index(drop=True)
    winning_variant.index = train_pool.index
    winning_variant["season"] = train_pool["season"]
    winning_variant["home_covered"] = train_pool["home_covered"]

    print("Comparing reduction strategies on the winning combo (walk-forward only)...")
    reduction_results = compare_reduction_strategies(
        winning_variant, winning_cols, features_cfg, modeling_cfg, random_state=modeling_cfg["random_state"]
    )
    for r in reduction_results:
        print(f"  {r['feature_set_mode']}: mean_n_features={r['mean_n_features']:.0f}, "
              f"pooled_precision={r['pooled']['precision']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "transform_ablation_results.json").write_text(
        json.dumps(ablation_results, indent=2, default=str)
    )
    (OUT_DIR / "reduction_strategy_comparison.json").write_text(
        json.dumps(reduction_results, indent=2, default=str)
    )
    winning_config = {
        "transform_name": winner["transform_name"],
        "transforms": winner["transforms"],
        "representation": winner["representation"],
        "n_features_after_transform_representation": winner["n_features"],
        "best_reduction_strategy": reduction_results[0]["feature_set_mode"],
        "reduction_strategy_ranking": [r["feature_set_mode"] for r in reduction_results],
    }
    (OUT_DIR / "winning_feature_config.json").write_text(json.dumps(winning_config, indent=2))
    print(json.dumps(winning_config, indent=2))


if __name__ == "__main__":
    main()

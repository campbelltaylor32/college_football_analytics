"""Stage 2: which temporal transform (or transform pair) actually carries signal?

Every game-stat base column in the source CSV exists as prev_week_/avg_all/avg3 versions.
Rather than feeding a model all three (today's notebook) or assuming a differential/trend
representation is better (untested assumption in prior work on this data), this runs a
small, fast probe model (regularized logistic regression - cheap enough to run once per
walk-forward fold per candidate) across every (temporal transform x home/away
representation) combination and scores it by pooled walk-forward precision-at-coverage-
floor. Only walk-forward folds are used here - the final holdout is never touched during
this comparison, so the winning combination isn't chosen by peeking at 2025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cfb_cover_model.feature_engineering import (
    apply_home_away_representation,
    build_transform_variant,
)
from cfb_cover_model.modeling.evaluation import best_precision_at_coverage_floor
from cfb_cover_model.modeling.splits import walk_forward_folds

PROBE_PARAMS = {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5, "C": 0.1, "max_iter": 3000}


def _probe_walk_forward(
    variant_frame: pd.DataFrame,
    feature_cols: list[str],
    label: pd.Series,
    train_pool: pd.DataFrame,
    modeling_cfg: dict,
    random_state: int,
) -> dict:
    folds = walk_forward_folds(train_pool, modeling_cfg)
    fold_true, fold_proba, fold_precisions = [], [], []

    for fold in folds:
        X_tr, X_va = variant_frame.loc[fold.train_idx, feature_cols], variant_frame.loc[fold.val_idx, feature_cols]
        y_tr, y_va = label.loc[fold.train_idx], label.loc[fold.val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf = LogisticRegression(random_state=random_state, **PROBE_PARAMS)
        clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_va_s)[:, 1]

        fold_true.append(y_va.to_numpy())
        fold_proba.append(proba)

        result = best_precision_at_coverage_floor(
            y_va.to_numpy(),
            proba,
            modeling_cfg["evaluation"]["threshold_grid"],
            modeling_cfg["evaluation"]["min_coverage_floor"],
        )
        fold_precisions.append(
            {"val_season": int(fold.val_season), "n_features": len(feature_cols), **result}
        )

    y_true_pooled = np.concatenate(fold_true)
    y_proba_pooled = np.concatenate(fold_proba)
    pooled = best_precision_at_coverage_floor(
        y_true_pooled,
        y_proba_pooled,
        modeling_cfg["evaluation"]["threshold_grid"],
        modeling_cfg["evaluation"]["min_coverage_floor"],
    )
    return {
        "n_features": len(feature_cols),
        "pooled": pooled,
        "per_fold": fold_precisions,
        "mean_fold_precision": float(
            np.mean([f["precision"] for f in fold_precisions if f["precision"] is not None])
        ),
    }


def run_transform_ablation(
    train_pool: pd.DataFrame,
    feature_columns: list[str],
    features_cfg: dict,
    modeling_cfg: dict,
    random_state: int = 42,
) -> tuple[dict, list[dict]]:
    label = train_pool["home_covered"]
    results = []

    for transform_candidate in features_cfg["transform_ablation"]["candidates"]:
        variant_frame, variant_cols = build_transform_variant(
            train_pool, feature_columns, transform_candidate["transforms"]
        )
        for rep_mode in features_cfg["home_away_representation"]["candidates"]:
            rep_frame, rep_cols = apply_home_away_representation(variant_frame, variant_cols, rep_mode)
            rep_frame = rep_frame.reset_index(drop=True)
            rep_frame.index = train_pool.index  # keep alignment with train_pool for fold indexing

            scored = _probe_walk_forward(
                rep_frame, rep_cols, label, train_pool, modeling_cfg, random_state
            )
            results.append(
                {
                    "transform_name": transform_candidate["name"],
                    "transforms": transform_candidate["transforms"],
                    "representation": rep_mode,
                    **scored,
                }
            )

    results.sort(
        key=lambda r: (r["pooled"]["precision"] if r["pooled"]["precision"] is not None else -1.0),
        reverse=True,
    )
    winner = results[0]
    return winner, results

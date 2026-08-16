"""Dispatcher tying stages 3-5 together behind one interface, used both by
scripts/select_features.py's reduction-strategy comparison and by scripts/train_models.py
(every walk-forward fold refits its own feature set here - never reused across folds).

feature_set_mode:
  "deterministic_pruned_only" - no stage 3/4/5 reduction at all; the honest "use
      everything past the data-layer cleanup" anchor.
  "reduced"     - stage 3 (correlation-cluster pruning) + stage 4 (embedded elastic-net
                  selection), fit only on the fold's training rows.
  "pca_reduced" - stage 5 (per-stat-family PCA collapse), fit only on the fold's training
                  rows; compared head-to-head against "reduced", not chained after it.
"""
from __future__ import annotations

import pandas as pd

from cfb_cover_model.feature_selection.correlation_pruning import prune_correlated_features
from cfb_cover_model.feature_selection.embedded_selection import select_features_embedded
from cfb_cover_model.feature_selection.pca_reduction import apply_pca_reducer, fit_pca_reducer


def fit_feature_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    season_train: pd.Series,
    feature_set_mode: str,
    features_cfg: dict,
    random_state: int = 42,
) -> tuple[object, dict]:
    if feature_set_mode == "deterministic_pruned_only":
        return list(X_train.columns), {}

    if feature_set_mode == "reduced":
        pruned_cols, corr_report = prune_correlated_features(
            X_train, y_train, features_cfg["correlation_pruning"]["correlation_threshold"]
        )
        selected, embed_report = select_features_embedded(
            X_train[pruned_cols],
            y_train,
            season_train,
            features_cfg["embedded_selection"]["l1_ratio_grid"],
            features_cfg["embedded_selection"]["C_grid"],
            features_cfg["embedded_selection"]["inner_cv_folds"],
            features_cfg["embedded_selection"]["max_features"],
            random_state=random_state,
        )
        return selected, {"correlation_pruning": corr_report, "embedded_selection": embed_report}

    if feature_set_mode == "pca_reduced":
        reducer = fit_pca_reducer(
            X_train,
            features_cfg["pca_reduction"]["stat_family_prefixes"],
            features_cfg["pca_reduction"]["variance_retained"],
            random_state,
        )
        report = {
            "n_components": reducer["n_components"],
            "n_matched_cols": len(reducer["matched_cols"]),
        }
        return reducer, report

    raise ValueError(f"Unknown feature_set_mode: {feature_set_mode!r}")


def apply_feature_set(X: pd.DataFrame, feature_set_mode: str, artifact: object) -> pd.DataFrame:
    if feature_set_mode in ("deterministic_pruned_only", "reduced"):
        return X[artifact]
    if feature_set_mode == "pca_reduced":
        return apply_pca_reducer(artifact, X)
    raise ValueError(f"Unknown feature_set_mode: {feature_set_mode!r}")

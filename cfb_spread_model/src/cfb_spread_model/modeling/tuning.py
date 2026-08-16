"""Hyperparameter tuning via GridSearchCV, but with an explicit pre-built list of
(train_idx, val_idx) index pairs derived from an expanding, season-ordered inner split -- never
sklearn's default random/K-fold CV, which would shuffle seasons and leak future training data
into an "earlier" validation fold. `scoring` is always a precision-focused callable from
feature_selection/precision_scoring.py, never "roc_auc" -- the single most consequential change
relative to the current notebook's GridSearchCV calls (scoring="roc_auc" at every site).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from cfb_spread_model.feature_selection.precision_scoring import Scorer
from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)


def build_inner_season_cv(train_df: pd.DataFrame, min_inner_train_seasons: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window inner CV within a single outer fold's training seasons -- shared by
    hyperparameter tuning (this module) and Stage 2 feature selection
    (feature_selection/selection.py's RFECV) so both apply the same season-ordering discipline
    as the outer walk-forward folds (modeling/splits.py)."""
    seasons = train_df["season"].reset_index(drop=True)
    unique_seasons = sorted(seasons.unique())
    splits = []
    for i in range(min_inner_train_seasons, len(unique_seasons)):
        val_season = unique_seasons[i]
        train_seasons_inner = unique_seasons[:i]
        train_idx = np.where(seasons.isin(train_seasons_inner))[0]
        val_idx = np.where(seasons == val_season)[0]
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def tune_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    scoring: Scorer,
) -> Pipeline:
    """param_grid keys must be prefixed "model__" to target the pipeline's estimator step."""
    if not param_grid or len(cv_splits) < 2:
        logger.info("No hyperparameter grid or too few inner CV folds; fitting with default params")
        pipeline.fit(X_train, y_train)
        return pipeline

    search = GridSearchCV(pipeline, param_grid, cv=cv_splits, scoring=scoring, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    logger.info(f"GridSearchCV best params: {search.best_params_} (score={search.best_score_:.4f})")
    return search.best_estimator_

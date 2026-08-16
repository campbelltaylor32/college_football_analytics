"""Hyperparameter tuning via GridSearchCV, but with an explicit pre-built list of
(train_idx, val_idx) index pairs derived from an expanding, season-ordered inner split --
never sklearn's default random/K-fold CV, which would shuffle seasons and leak future
training data into an "earlier" validation fold. Grids are small (config/modeling.yaml
`hyperparam_grids`) since the whole training set is on the order of 1,000-1,300 rows --
exhaustive grid search is cheap, a randomized search is unnecessary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)


def build_inner_season_cv(train_df: pd.DataFrame, min_inner_train_seasons: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window inner CV within a single outer fold's training seasons, used only for
    hyperparameter selection -- each inner validation season is preceded only by strictly
    earlier inner-training seasons, same discipline as the outer walk-forward folds."""
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
    scoring: str = "neg_mean_absolute_error",
) -> Pipeline:
    """param_grid keys must be prefixed "model__" to target the pipeline's estimator step
    (see train_models.py, which builds the "model" step name)."""
    if not param_grid or len(cv_splits) < 2:
        logger.info("No hyperparameter grid or too few inner CV folds; fitting with default params")
        pipeline.fit(X_train, y_train)
        return pipeline

    search = GridSearchCV(pipeline, param_grid, cv=cv_splits, scoring=scoring, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    logger.info(f"GridSearchCV best params: {search.best_params_} (score={search.best_score_:.4f})")
    return search.best_estimator_

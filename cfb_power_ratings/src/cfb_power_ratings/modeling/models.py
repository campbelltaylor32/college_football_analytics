"""Candidate model registry. `gradient_boosting` uses HistGradientBoostingRegressor
specifically for its native NaN handling -- the feature set has systematic, structural NaNs
(srs_lag2/srs_lag3 for the earliest feature-eligible seasons, blue_chip_ratio for
low-confidence team-seasons, transfer/rating sums when a team has zero portal activity), not
occasional missingness to impute away. `ridge` gets an explicit imputation step since sklearn's
linear models don't accept NaN at all. `xgboost` is optional (try/except ImportError) so the
pipeline runs on a bare `pip install -e .` without the `boosting` extra.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from cfb_power_ratings.dataset import FEATURE_COLUMNS
from cfb_power_ratings.utils.logging import get_logger

logger = get_logger(__name__)

# A team with no prior-season FBS SRS at all (srs_lag1 NaN) has never had a full FBS season in
# this project's history window -- i.e. it just moved up into FBS (Sacramento State/North Dakota
# State entering 2026; historically Coastal Carolina 2017, Delaware/Missouri State 2025, etc.).
# For a team like that, EVERY missing feature is filled with the training data's worst (minimum)
# observed value, not the median -- median imputation would score a team we have essentially no
# real information about as an average FBS roster, which is wrong in the specific direction of
# overrating it (verified live: Sacramento State came back ranked #46 nationally before this
# fix). Every other team -- including one missing just srs_lag3 because it's only had 1-2 prior
# FBS seasons -- still gets ordinary median imputation, since occasional missingness there
# doesn't carry the same "we truly know nothing about this team" meaning.
TRANSITION_INDICATOR_COLUMN = "srs_lag1"


class TransitionTeamAwareImputer(BaseEstimator, TransformerMixin):
    """Median-imputes every column by default (like SimpleImputer(strategy="median")). For rows
    where `transition_indicator_col` is NaN -- a brand-new FBS transition team, per the module
    docstring above -- every missing value in that row is filled with the column's training-data
    MINIMUM instead. `feature_columns` must match the column order X will be fit/transformed
    with (this project always calls model.fit(df[FEATURE_COLUMNS], ...), so that's what's
    passed)."""

    def __init__(self, feature_columns: list[str], transition_indicator_col: str = TRANSITION_INDICATOR_COLUMN):
        self.feature_columns = feature_columns
        self.transition_indicator_col = transition_indicator_col

    def fit(self, X, y=None):
        self.indicator_idx_ = self.feature_columns.index(self.transition_indicator_col)
        X = np.asarray(X, dtype=float)
        self.median_ = np.nanmedian(X, axis=0)
        self.min_ = np.nanmin(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        is_transition_row = np.isnan(X[:, self.indicator_idx_])
        nan_mask = np.isnan(X)
        median_fill = np.broadcast_to(self.median_, X.shape)
        min_fill = np.broadcast_to(self.min_, X.shape)
        fill = np.where(is_transition_row[:, None], min_fill, median_fill)
        X[nan_mask] = fill[nan_mask]
        return X


def _ridge(seed: int):
    return Pipeline([
        ("impute", TransitionTeamAwareImputer(feature_columns=FEATURE_COLUMNS)),
        ("ridge", Ridge(alpha=1.0, random_state=seed)),
    ])


_CORE_FACTORIES = {
    "ridge": _ridge,
    "gradient_boosting": lambda seed: HistGradientBoostingRegressor(random_state=seed, max_depth=4),
}


def _optional_factories() -> dict[str, Any]:
    factories = {}
    try:
        import xgboost

        factories["xgboost"] = lambda seed: xgboost.XGBRegressor(random_state=seed, n_estimators=200, max_depth=3, learning_rate=0.05)
    except ImportError:
        logger.warning("xgboost not installed; skipping (pip install -e '.[boosting]')")
    return factories


def get_candidate_models(model_names: list[str], seed: int) -> dict[str, Any]:
    """Returns only the real, fittable candidates (excludes the two baseline names, which
    baselines.py handles separately since they aren't sklearn estimators over FEATURE_COLUMNS)."""
    factories = {**_CORE_FACTORIES, **_optional_factories()}
    models = {}
    for name in model_names:
        if name not in factories:
            continue
        models[name] = factories[name](seed)
    return models

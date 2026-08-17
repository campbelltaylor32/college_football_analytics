"""Candidate model registry. `gradient_boosting` uses HistGradientBoostingRegressor
specifically for its native NaN handling -- the feature set has systematic, structural NaNs
(srs_lag2/srs_lag3 for the earliest feature-eligible seasons, blue_chip_ratio for
low-confidence team-seasons, transfer/rating sums when a team has zero portal activity), not
occasional missingness to impute away. `ridge` gets an explicit SimpleImputer step since
sklearn's linear models don't accept NaN at all. `xgboost` is optional (try/except ImportError)
so the pipeline runs on a bare `pip install -e .` without the `boosting` extra.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from cfb_power_ratings.dataset import FEATURE_COLUMNS
from cfb_power_ratings.utils.logging import get_logger

logger = get_logger(__name__)


def _ridge(seed: int):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
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

"""Candidate classifier registry, plus baselines the current notebook lacks entirely.

Core candidates (logistic_regression, random_forest, gradient_boosting) are always available.
xgboost/lightgbm/catboost are added only if installed (try/except ImportError, logged warning,
never a hard failure) -- consistent with the sibling cfb_win_total_model project's policy that
the pipeline must run on a bare `pip install -e .` without the `boosting` extra, even though
feature_selection/selection.py's Stage 2 importance signal does default to XGBoost when present.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)


class AlwaysFavoriteClassifier(BaseEstimator, ClassifierMixin):
    """Baseline: predict home_covered=1 whenever the home team is favored (home_favored==1),
    0 otherwise. Tests whether any candidate model actually beats "just bet the favorite" --
    a sanity floor the current notebook never computes. Needs the raw home_favored column,
    not necessarily a Stage 1/2-selected feature subset -- see REQUIRES_RAW_CONTEXT_COLUMNS."""

    def __init__(self, favored_column: str = "home_favored"):
        self.favored_column = favored_column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p1 = X[self.favored_column].to_numpy(dtype=float)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (X[self.favored_column].to_numpy() >= 0.5).astype(int)


_CORE_MODEL_FACTORIES = {
    "logistic_regression": lambda seed: LogisticRegression(random_state=seed, max_iter=2000),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
    # Same estimator as logistic_regression -- the distinction is which feature set it's fit
    # on (see BYPASS_FEATURE_SELECTION below), not the model class.
    "logistic_no_selection": lambda seed: LogisticRegression(random_state=seed, max_iter=2000),
}

_BASELINE_FACTORIES = {
    "always_favorite": lambda seed: AlwaysFavoriteClassifier(),
    "majority_class": lambda seed: DummyClassifier(strategy="prior", random_state=seed),
}


def _optional_model_factories() -> dict[str, Any]:
    factories = {}
    try:
        import xgboost

        factories["xgboost"] = lambda seed: xgboost.XGBClassifier(random_state=seed, eval_metric="logloss")
    except ImportError:
        logger.warning("xgboost not installed; skipping XGBClassifier candidate (pip install -e '.[boosting]')")

    try:
        import lightgbm

        factories["lightgbm"] = lambda seed: lightgbm.LGBMClassifier(random_state=seed, verbosity=-1)
    except ImportError:
        logger.warning("lightgbm not installed; skipping LGBMClassifier candidate (pip install -e '.[boosting]')")

    try:
        import catboost

        factories["catboost"] = lambda seed: catboost.CatBoostClassifier(random_state=seed, verbose=False)
    except ImportError:
        logger.warning("catboost not installed; skipping CatBoostClassifier candidate (pip install -e '.[boosting]')")

    return factories


def get_candidate_models(model_names: list[str], seed: int) -> dict[str, Any]:
    factories = {**_CORE_MODEL_FACTORIES, **_optional_model_factories()}
    models = {}
    for name in model_names:
        if name not in factories:
            logger.warning(f"Requested model '{name}' has no factory (missing optional dependency?); skipping")
            continue
        models[name] = factories[name](seed)
    return models


def get_baseline_models(model_names: list[str], seed: int) -> dict[str, Any]:
    factories = {**_BASELINE_FACTORIES, **_CORE_MODEL_FACTORIES}
    models = {}
    for name in model_names:
        if name not in factories:
            logger.warning(f"Requested baseline '{name}' has no factory; skipping")
            continue
        models[name] = factories[name](seed)
    return models


# Model names fit on the FULL (Stage 1/2 selection-bypassed) feature set rather than the fold's
# selected feature subset -- exists specifically to test whether any precision gain comes from
# feature selection or just model family (see config/modeling.yaml's baseline rationale).
BYPASS_FEATURE_SELECTION = {"logistic_no_selection"}

# Baselines that need a specific raw column present regardless of what Stage 1/2 selected.
REQUIRES_RAW_CONTEXT_COLUMNS = {"always_favorite": ["home_favored"]}

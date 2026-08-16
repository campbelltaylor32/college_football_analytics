"""Candidate regressor registry. HistGradientBoostingRegressor is the safe-default primary
boosted model: no extra dependency, native NaN handling. XGBoost/LightGBM are added only if
installed (try/except ImportError, logged warning, never a hard failure) -- the pipeline must
run on a bare `pip install -e .` without the `boosting` extra. Ported near-verbatim from the
sibling cfb_win_total_model project's modeling/models.py."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

_CORE_MODEL_FACTORIES = {
    "ols": lambda seed: LinearRegression(),
    "ridge": lambda seed: Ridge(random_state=seed),
    "lasso": lambda seed: Lasso(random_state=seed),
    "elasticnet": lambda seed: ElasticNet(random_state=seed),
    # n_jobs=1, not -1: this estimator is fit INSIDE GridSearchCV(n_jobs=-1) (modeling/tuning.py)
    # -- nesting two n_jobs=-1 joblib/loky parallelism layers (outer GridSearchCV spawning a
    # worker pool, each worker's RandomForestRegressor trying to spawn its OWN pool) is a
    # known joblib/loky deadlock trigger, confirmed live: training hung indefinitely
    # immediately after the first random_forest fit completed, workers idle at 0% CPU with no
    # further progress. The outer GridSearchCV parallelism is what actually matters here (it
    # parallelizes across hyperparameter x inner-fold combinations); the forest's own internal
    # tree-building parallelism is redundant with that and not worth the deadlock risk.
    "random_forest": lambda seed: RandomForestRegressor(random_state=seed, n_jobs=1),
    "gradient_boosting": lambda seed: GradientBoostingRegressor(random_state=seed),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingRegressor(random_state=seed),
}


def _optional_model_factories() -> dict[str, Any]:
    factories = {}
    try:
        import xgboost

        # n_jobs=1 for the same nested-parallelism reason as random_forest above -- these
        # fit inside GridSearchCV(n_jobs=-1).
        factories["xgboost"] = lambda seed: xgboost.XGBRegressor(random_state=seed, n_jobs=1)
    except ImportError:
        logger.warning("xgboost not installed; skipping XGBRegressor candidate (pip install -e '.[boosting]')")

    try:
        import lightgbm

        factories["lightgbm"] = lambda seed: lightgbm.LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=1)
    except ImportError:
        logger.warning("lightgbm not installed; skipping LGBMRegressor candidate (pip install -e '.[boosting]')")

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

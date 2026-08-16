"""Candidate regressor registry. HistGradientBoostingRegressor is the designated safe-default
primary boosted model: no extra dependency, native NaN handling, and appropriate for a
training set on the order of 1,000-1,300 team-season rows. XGBoost/LightGBM are added only if
installed (try/except ImportError, logged warning, never a hard failure) -- the pipeline must
run on a bare `pip install -e .` without the `boosting` extra. PoissonRegressor is available
as an explicit opt-in secondary (not in the default candidate list): win totals are bounded by
schedule length, not classically Poisson-shaped, so it's a comparison point, not the primary
path.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, PoissonRegressor, Ridge

from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

_CORE_MODEL_FACTORIES = {
    "ols": lambda seed: LinearRegression(),
    "ridge": lambda seed: Ridge(random_state=seed),
    "lasso": lambda seed: Lasso(random_state=seed),
    "elasticnet": lambda seed: ElasticNet(random_state=seed),
    "random_forest": lambda seed: RandomForestRegressor(random_state=seed, n_jobs=-1),
    "gradient_boosting": lambda seed: GradientBoostingRegressor(random_state=seed),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingRegressor(random_state=seed),
    "poisson_secondary": lambda seed: PoissonRegressor(),
}


def _optional_model_factories() -> dict[str, Any]:
    factories = {}
    try:
        import xgboost

        factories["xgboost"] = lambda seed: xgboost.XGBRegressor(random_state=seed)
    except ImportError:
        logger.warning("xgboost not installed; skipping XGBRegressor candidate (pip install -e '.[boosting]')")

    try:
        import lightgbm

        factories["lightgbm"] = lambda seed: lightgbm.LGBMRegressor(random_state=seed, verbosity=-1)
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

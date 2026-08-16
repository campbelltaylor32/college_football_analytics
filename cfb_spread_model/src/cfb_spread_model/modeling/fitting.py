"""Shared "build a pipeline (or fit a bare baseline estimator directly) and fit it" logic, used
by scripts/train_models.py (per-fold fit), scripts/evaluate_models.py (final refit of the
selected winner), and scripts/compare_models_on_holdout.py (final refit of every model, to
check whether the winner selected by walk-forward mean precision actually holds up on the true
holdout). Centralizing this is what keeps a hyperparameter-tuning change or a new baseline model
type a one-file edit instead of a three-file one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cfb_spread_model.config import ModelingConfig
from cfb_spread_model.feature_selection.precision_scoring import precision_at_coverage_floor_scorer
from cfb_spread_model.modeling import models as models_module
from cfb_spread_model.modeling import preprocessing, tuning


def fit_model(
    model_name: str,
    is_baseline: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modeling_cfg: ModelingConfig,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
):
    """Returns a fitted estimator (AlwaysFavoriteClassifier, fit directly on raw columns -- see
    models.REQUIRES_RAW_CONTEXT_COLUMNS) or a fitted sklearn Pipeline (preprocess + model,
    hyperparameter-tuned via modeling/tuning.py's season-ordered inner CV and the same
    precision-at-coverage-floor scorer used everywhere else in this project)."""
    factory = models_module.get_baseline_models if is_baseline else models_module.get_candidate_models
    estimator = factory([model_name], modeling_cfg.random_seed)[model_name]

    if model_name == "always_favorite":
        estimator.fit(X_train, y_train)
        return estimator

    preprocess = preprocessing.build_preprocessing_pipeline(list(X_train.columns))
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
    param_grid = {f"model__{k}": v for k, v in modeling_cfg.hyperparam_grids.get(model_name, {}).items()}
    scorer = precision_at_coverage_floor_scorer(modeling_cfg.precision_objective.min_coverage_floor)
    return tuning.tune_model(pipeline, param_grid, X_train, y_train, cv_splits, scorer)

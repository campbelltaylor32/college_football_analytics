"""Shared fold-fitting logic reused by scripts/train_models.py, evaluate_models.py, and
generate_predictions.py -- one place that builds a preprocessing+model Pipeline, tunes it
(when a hyperparameter grid is configured) using the season-ordered inner CV from tuning.py,
and fits it on a fold's training rows.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from cfb_win_total_model.modeling.preprocessing import build_preprocessing_pipeline
from cfb_win_total_model.modeling.tuning import build_inner_season_cv, tune_model
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COL = "regular_season_wins"


def get_feature_columns(df: pd.DataFrame, non_feature_cols: set[str]) -> list[str]:
    return [c for c in df.columns if c not in non_feature_cols]


def fit_candidate_on_fold(
    estimator,
    param_grid: dict | None,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    scoring: str = "neg_mean_absolute_error",
) -> Pipeline:
    """scoring defaults to the production criterion (plain MAE) -- pass a different string or
    a scorer callable (e.g. modeling.evaluation.make_variance_aware_scorer()) to tune under a
    different objective without touching this function's default, production behavior."""
    preprocessor = build_preprocessing_pipeline(feature_cols)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    grid = {f"model__{k}": v for k, v in (param_grid or {}).items()}
    if grid:
        cv_splits = build_inner_season_cv(train_df)
        pipeline = tune_model(pipeline, grid, X_train, y_train, cv_splits, scoring=scoring)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def predict_with_pipeline(pipeline: Pipeline, df: pd.DataFrame, feature_cols: list[str]):
    return pipeline.predict(df[feature_cols])

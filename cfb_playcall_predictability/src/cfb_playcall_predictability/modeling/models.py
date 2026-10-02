"""Model registry: baselines and candidates, each exposing fit(X, y) / predict_proba(X).

Every model receives the full feature frame and selects its own columns, so the training
script can treat them uniformly."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from cfb_playcall_predictability.dataset import SITUATIONAL_FEATURES

# Situation variables whose effect on pass rate is strongly non-linear -> spline-expanded in
# the logistic models (the tree models find this on their own).
SPLINE_FEATURES = ["distance", "yards_to_goal", "half_secs_rem", "game_secs_rem", "score_diff"]
CATEGORICAL_FEATURES = ["down", "period"]


class ColumnSelector(BaseEstimator, ClassifierMixin):
    """Wrap an estimator so it only sees `columns` of the frame it is given."""

    def __init__(self, estimator, columns: list[str]):
        self.estimator = estimator
        self.columns = columns

    def fit(self, X: pd.DataFrame, y):
        self.estimator.fit(X[self.columns], y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.columns])


class TeamRateBaseline(BaseEstimator, ClassifierMixin):
    """Predict the team's season-to-date pass rate (shrunk toward last season) for every call,
    ignoring the situation entirely -- "they throw it 55% of the time"."""

    column = "std_pr_overall"

    def fit(self, X: pd.DataFrame, y):
        self.fallback_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[self.column].fillna(self.fallback_).clip(1e-4, 1 - 1e-4).to_numpy()
        return np.column_stack([1 - p, p])


def _logit_pipeline(features: list[str], seed: int) -> Pipeline:
    spline = [c for c in SPLINE_FEATURES if c in features]
    cat = [c for c in CATEGORICAL_FEATURES if c in features]
    rest = [c for c in features if c not in spline and c not in cat]
    transformers = [
        ("spline", make_pipeline(SimpleImputer(strategy="median"),
                                 SplineTransformer(n_knots=6, degree=3), StandardScaler()), spline),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ]
    if rest:
        transformers.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), rest))
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=seed)),
    ])


def build_model(name: str, features: list[str], hyperparams: dict, seed: int):
    if name == "situational_logit":
        cols = [c for c in SITUATIONAL_FEATURES if c in features]
        return ColumnSelector(_logit_pipeline(cols, seed), cols)
    if name == "team_prior_rate":
        return TeamRateBaseline()
    if name == "logit_full":
        return ColumnSelector(_logit_pipeline(features, seed), features)
    if name == "xgboost":
        clf = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=-1, **hyperparams.get("xgboost", {}),
        )
        return ColumnSelector(clf, features)
    if name == "lightgbm":
        clf = LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, verbose=-1,
                             **hyperparams.get("lightgbm", {}))
        return ColumnSelector(clf, features)
    raise ValueError(f"Unknown model {name!r}")

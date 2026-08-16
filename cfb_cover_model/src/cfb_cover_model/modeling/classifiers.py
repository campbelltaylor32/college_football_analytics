"""Track A: direct classification. A thin factory over sklearn/xgboost/lightgbm/catboost
plus two zero-parameter baselines, all exposed through the same fit/predict_proba shape so
train_models.py can treat every candidate identically."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MajorityClassBaseline:
    """Predicts the training fold's home_covered base rate for every row - the "no
    information beyond the prior" anchor."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class AlwaysFavoriteBaseline:
    """Predicts a fixed high probability when the home team is favored, a fixed low
    probability otherwise - calibrated to the fold's own conditional cover rates rather
    than a hardcoded constant, so it's a fair "just follow the market favorite" anchor.
    Requires a `home_favored` column passed alongside X via `fit`/`predict_proba`'s
    `home_favored` kwarg (not a model feature - see cleaning.py)."""

    def fit(self, X, y, home_favored: pd.Series):
        y = pd.Series(np.asarray(y), index=home_favored.index)
        self.rate_favored_ = float(y[home_favored == 1].mean()) if (home_favored == 1).any() else 0.5
        self.rate_underdog_ = float(y[home_favored == 0].mean()) if (home_favored == 0).any() else 0.5
        return self

    def predict_proba(self, X, home_favored: pd.Series):
        p = np.where(home_favored.to_numpy() == 1, self.rate_favored_, self.rate_underdog_)
        return np.column_stack([1 - p, p])


def build_classifier(kind: str, params: dict, random_state: int):
    params = dict(params)
    if kind == "logistic_regression":
        params.setdefault("random_state", random_state)
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])
    if kind == "random_forest":
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)
    if kind == "gradient_boosting":
        params.setdefault("random_state", random_state)
        return GradientBoostingClassifier(**params)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        params.setdefault("random_state", random_state)
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**params)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        params.setdefault("random_state", random_state)
        return LGBMClassifier(**params)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        params.setdefault("random_seed", random_state)
        return CatBoostClassifier(**params)
    raise ValueError(f"Unknown classifier kind: {kind!r}")

"""Track B: predict the continuous cover_margin (points beyond the spread), then convert
to a cover probability via the fitted training-fold residual distribution - a standard
regression-to-probability conversion, giving a continuous-signal alternative to Track A's
direct classifiers using the same reduced feature sets.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_point_regressor(kind: str, params: dict, random_state: int):
    params = dict(params)
    if kind == "elastic_net":
        params.setdefault("random_state", random_state)
        return Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet(**params))])
    if kind == "xgboost_regressor":
        from xgboost import XGBRegressor

        params.setdefault("random_state", random_state)
        return XGBRegressor(**params)
    raise ValueError(f"Unknown point regressor kind: {kind!r}")


class ResidualProbabilityRegressor:
    """Fits a point-prediction regressor on cover_margin, then models P(cover_margin > 0)
    as Phi(predicted_margin / residual_std), residual_std estimated once from the
    training fold's own fitted residuals (never from validation/holdout rows)."""

    def __init__(self, base_regressor):
        self.base_regressor = base_regressor

    def fit(self, X, cover_margin):
        self.base_regressor.fit(X, cover_margin)
        residuals = np.asarray(cover_margin) - self.base_regressor.predict(X)
        self.residual_std_ = float(np.std(residuals)) or 1.0
        return self

    def predict_margin(self, X):
        return self.base_regressor.predict(X)

    def predict_proba(self, X):
        pred_margin = self.predict_margin(X)
        p = norm.cdf(pred_margin / self.residual_std_)
        return np.column_stack([1 - p, p])


class QuantileProbabilityRegressor:
    """Fits independent quantile regressions (default 0.25/0.5/0.75), derives an implied
    std from the interquartile range (IQR / 1.349, the normal-distribution relationship),
    and converts the median prediction to a cover probability the same way as
    ResidualProbabilityRegressor - avoids assuming a single global residual variance when
    the spread of outcomes may itself vary by matchup."""

    def __init__(self, quantiles: list[float], params: dict, random_state: int):
        self.quantiles = quantiles
        self.params = dict(params)
        self.random_state = random_state

    def fit(self, X, cover_margin):
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)
        self.models_ = {}
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, **self.params)
            model.fit(X_scaled, cover_margin)
            self.models_[q] = model
        self.median_q_ = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return self

    def predict_margin(self, X):
        X_scaled = self.scaler_.transform(X)
        return self.models_[self.median_q_].predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        median_pred = self.models_[self.median_q_].predict(X_scaled)
        if 0.25 in self.models_ and 0.75 in self.models_:
            iqr = self.models_[0.75].predict(X_scaled) - self.models_[0.25].predict(X_scaled)
            std = np.clip(iqr / 1.349, 1e-3, None)
        else:
            std = np.full_like(median_pred, 10.0)
        p = norm.cdf(median_pred / std)
        return np.column_stack([1 - p, p])

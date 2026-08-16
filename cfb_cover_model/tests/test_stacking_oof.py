import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.modeling.stacking import (
    fit_stacking_ensemble,
    generate_oof_base_predictions,
    safe_inner_min_seasons,
)


class MaxSeasonFingerprintModel:
    """Records the maximum season value present in its training data and reports it back
    as the "probability" - lets a test check exactly what training data a base model saw
    for each out-of-fold prediction, without relying on real signal in the data."""

    def fit(self, X, y):
        self.max_season_ = float(X["season_feature"].max())
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.max_season_)
        return np.column_stack([np.zeros_like(p), p])


def _synthetic_stacking_inputs(rng):
    n = 250
    seasons = np.repeat([2015, 2016, 2017, 2018, 2019], n // 5)
    X = pd.DataFrame({"season_feature": seasons.astype(float), "noise": rng.normal(0, 1, n)})
    y = pd.Series(rng.integers(0, 2, n))
    cover_margin = pd.Series(rng.normal(0, 10, n))
    season_series = pd.Series(seasons)
    return X, y, cover_margin, season_series


def test_oof_predictions_never_reflect_own_or_future_season(rng):
    X, y, cover_margin, season_series = _synthetic_stacking_inputs(rng)
    specs = [{"name": "fingerprint", "kind": "classifier", "builder": MaxSeasonFingerprintModel}]

    oof = generate_oof_base_predictions(X, y, cover_margin, season_series, specs, min_inner_train_seasons=2)

    non_na = oof.dropna()
    assert len(non_na) > 0
    for idx in non_na.index:
        own_season = season_series.loc[idx]
        max_season_model_trained_on = non_na.loc[idx, "fingerprint"]
        assert max_season_model_trained_on < own_season, (
            f"row {idx} (season {own_season}) got an OOF prediction from a model trained "
            f"through season {max_season_model_trained_on} - not strictly earlier."
        )


def test_earliest_seasons_have_no_oof_prediction(rng):
    X, y, cover_margin, season_series = _synthetic_stacking_inputs(rng)
    specs = [{"name": "fingerprint", "kind": "classifier", "builder": MaxSeasonFingerprintModel}]

    oof = generate_oof_base_predictions(X, y, cover_margin, season_series, specs, min_inner_train_seasons=2)

    earliest_two_seasons_idx = season_series.index[season_series.isin([2015, 2016])]
    assert oof.loc[earliest_two_seasons_idx, "fingerprint"].isna().all()


def test_meta_learner_only_trains_on_rows_with_complete_oof(rng):
    X, y, cover_margin, season_series = _synthetic_stacking_inputs(rng)
    specs = [
        {"name": "a", "kind": "classifier", "builder": MaxSeasonFingerprintModel},
        {"name": "b", "kind": "classifier", "builder": MaxSeasonFingerprintModel},
    ]

    class MeanMetaLearner:
        def fit(self, X, y):
            self.rate_ = float(np.mean(y))
            return self

        def predict_proba(self, X):
            p = np.full(len(X), self.rate_)
            return np.column_stack([1 - p, p])

    X_val = X.iloc[:10]
    stacked_proba, report = fit_stacking_ensemble(
        X, y, cover_margin, season_series, X_val, specs, MeanMetaLearner, min_inner_train_seasons=2
    )
    assert report["n_meta_train_rows"] < len(X)
    assert 2015 not in report["meta_train_seasons"]
    assert len(stacked_proba) == len(X_val)


def test_safe_inner_min_seasons_clamps_to_leave_one_validation_season():
    season_train = pd.Series([2015, 2015, 2016, 2016, 2017, 2017])  # 3 unique seasons
    assert safe_inner_min_seasons(season_train, configured_min=3) == 2
    assert safe_inner_min_seasons(season_train, configured_min=1) == 1

    season_train_wide = pd.Series(list(range(2010, 2020)))  # 10 unique seasons
    assert safe_inner_min_seasons(season_train_wide, configured_min=3) == 3

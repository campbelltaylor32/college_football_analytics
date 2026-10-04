import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_power_ratings.modeling.models import TRANSITION_INDICATOR_COLUMN, TransitionTeamAwareImputer, get_candidate_models

COLS = ["talent_composite", TRANSITION_INDICATOR_COLUMN, "other_feature"]


def test_non_transition_row_gets_median_imputed():
    X = np.array([
        [100.0, 5.0, 1.0],
        [np.nan, 6.0, 2.0],  # non-transition row (indicator present) -- talent should get MEDIAN
        [300.0, 7.0, 3.0],
    ])
    imputer = TransitionTeamAwareImputer(feature_columns=COLS).fit(X)
    out = imputer.transform(X)
    assert out[1, 0] == pytest.approx(200.0)  # median of [100, 300], not the min (100)


def test_transition_row_gets_every_missing_value_min_imputed():
    X = np.array([
        [100.0, 5.0, 1.0],
        [np.nan, np.nan, np.nan],  # transition row: indicator itself AND every other col missing
        [300.0, 7.0, 9.0],
    ])
    imputer = TransitionTeamAwareImputer(feature_columns=COLS).fit(X)
    out = imputer.transform(X)
    # All three missing values in the transition row should be filled with each column's MIN,
    # not its median -- talent_composite, the indicator column itself, and "other_feature" alike.
    assert out[1, 0] == pytest.approx(100.0)  # min(100, 300), not median (200)
    assert out[1, 1] == pytest.approx(5.0)    # min(5, 7)
    assert out[1, 2] == pytest.approx(1.0)    # min(1, 9), not median (5)


def test_transition_detection_is_based_on_indicator_col_only():
    # A row missing "other_feature" but NOT the indicator column is not a transition row --
    # even though it has a missing value, that value should be median-imputed.
    X = np.array([
        [100.0, 5.0, 1.0],
        [200.0, 6.0, np.nan],
        [300.0, 7.0, 9.0],
    ])
    imputer = TransitionTeamAwareImputer(feature_columns=COLS).fit(X)
    out = imputer.transform(X)
    assert out[1, 2] == pytest.approx(5.0)  # median of [1, 9], not the min (1)


def test_ridge_pipeline_uses_transition_aware_imputer_end_to_end():
    from cfb_power_ratings.dataset import FEATURE_COLUMNS

    n = len(FEATURE_COLUMNS)
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(loc=500, scale=50, size=(30, n)), columns=FEATURE_COLUMNS)
    X["talent_composite"] = np.linspace(500.0, 900.0, 30)  # real spread, min far below median
    y = pd.Series(rng.normal(size=30))

    # Row 0: a transition team -- missing srs_lag1 AND talent_composite.
    X.loc[0, "srs_lag1"] = np.nan
    X.loc[0, "talent_composite"] = np.nan
    # Row 1: an established team missing only talent_composite (srs_lag1 present) -- should
    # still get median treatment for its one missing value.
    X.loc[1, "talent_composite"] = np.nan

    ridge_pipeline = get_candidate_models(["ridge"], seed=42)["ridge"]
    ridge_pipeline.fit(X, y)
    transformed = ridge_pipeline.named_steps["impute"].transform(X)

    talent_idx = FEATURE_COLUMNS.index("talent_composite")
    real_min = X["talent_composite"].dropna().min()
    real_median = X["talent_composite"].dropna().median()

    assert transformed[0, talent_idx] == pytest.approx(real_min)      # transition row -> min
    assert transformed[1, talent_idx] == pytest.approx(real_median)   # established row -> median
    assert not np.isnan(transformed).any()

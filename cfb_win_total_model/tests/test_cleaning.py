from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb_win_total_model.cleaning import apply_winsorization, impute_missing, validate_no_inf_or_extreme


def test_apply_winsorization_clips_extreme_ppa(features_cfg):
    df = pd.DataFrame({"returning_percent_ppa": [-129.5, 0.5, 4.0], "school": ["A", "B", "C"]})
    out = apply_winsorization(df, features_cfg)
    lower, upper = features_cfg.winsorize_percent_ppa_limits
    assert out["returning_percent_ppa"].min() == lower
    assert out["returning_percent_ppa"].max() <= upper


def test_impute_missing_zero_fill_vs_median():
    df = pd.DataFrame(
        {
            "school": ["A", "B", "C"],
            "season": [2022, 2022, 2022],
            "regular_season_wins": [5, 6, 7],
            "n_transferred_out": [2.0, np.nan, 4.0],
            "some_rate": [0.1, np.nan, 0.3],
        }
    )
    out = impute_missing(df, zero_fill_cols=["n_transferred_out"])
    assert out["n_transferred_out"].isna().sum() == 0
    assert out.loc[1, "n_transferred_out"] == 0
    assert out["some_rate"].isna().sum() == 0
    assert out.loc[1, "some_rate"] == pytest.approx(0.2)  # median of [0.1, 0.3]


def test_impute_missing_leaves_missing_flags_alone():
    df = pd.DataFrame({"school": ["A"], "season": [2022], "regular_season_wins": [5], "talent_missing": [True]})
    out = impute_missing(df)
    assert out["talent_missing"].iloc[0] == True  # noqa: E712


def test_validate_no_inf_or_extreme_raises_on_inf():
    df = pd.DataFrame({"x": [1.0, np.inf]})
    with pytest.raises(ValueError):
        validate_no_inf_or_extreme(df)


def test_validate_no_inf_or_extreme_raises_on_extreme_zscore():
    df = pd.DataFrame({"talent_zscore": [0.5, 10.0]})
    with pytest.raises(ValueError):
        validate_no_inf_or_extreme(df)


def test_validate_no_inf_or_extreme_passes_clean_frame():
    df = pd.DataFrame({"x": [1.0, 2.0], "talent_zscore": [0.5, -0.5]})
    validate_no_inf_or_extreme(df)  # should not raise

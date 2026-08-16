"""Independently recomputes rolling values from raw rows and asserts equality against
features/rolling_utils.py's output -- the shared two-step compute-then-lag engine used
identically by rushing_workload.py, team_offense_context.py, and opponent_defense_context.py.
Testing this ONE shared function is what makes the leakage guarantee auditable across all
three modules at once (see docs/data_leakage_rules.md)."""

from __future__ import annotations

import pandas as pd

from cfb_rb_rushing_model.features.rolling_utils import attach_games_played_lag1, compute_rolling_and_lag


def _df():
    return pd.DataFrame(
        {
            "entity": ["a", "a", "a", "a", "b", "b"],
            "start_date": pd.to_datetime(
                ["2023-09-01", "2023-09-08", "2023-09-15", "2023-09-22", "2023-09-01", "2023-09-08"]
            ),
            "carries": [10, 20, 30, 40, 5, 15],
        }
    )


def test_first_row_per_entity_has_nan_lag_columns():
    result = compute_rolling_and_lag(_df(), group_cols=["entity"], sort_col="start_date", value_cols=["carries"], window=3)
    first_a = result[result["entity"] == "a"].iloc[0]
    assert pd.isna(first_a["carries_avg3_lag1"])
    assert pd.isna(first_a["carries_avg_all_lag1"])


def test_avg3_lag1_equals_independently_recomputed_trailing_window():
    result = compute_rolling_and_lag(_df(), group_cols=["entity"], sort_col="start_date", value_cols=["carries"], window=3)
    a = result[result["entity"] == "a"].reset_index(drop=True)

    # Row 3 (4th game, carries=40): trailing-3 average through the PRIOR game only = mean(10,20,30) = 20
    assert a.loc[3, "carries_avg3_lag1"] == 20.0
    # Row 2 (3rd game, carries=30): only 2 prior games exist -> mean(10,20) = 15 (min_periods=1 rolling)
    assert a.loc[2, "carries_avg3_lag1"] == 15.0
    # Row 1 (2nd game, carries=20): only 1 prior game -> 10
    assert a.loc[1, "carries_avg3_lag1"] == 10.0


def test_avg_all_lag1_equals_independently_recomputed_cumulative_mean():
    result = compute_rolling_and_lag(_df(), group_cols=["entity"], sort_col="start_date", value_cols=["carries"], window=3)
    a = result[result["entity"] == "a"].reset_index(drop=True)
    assert a.loc[3, "carries_avg_all_lag1"] == (10 + 20 + 30) / 3
    assert a.loc[2, "carries_avg_all_lag1"] == (10 + 20) / 2


def test_rolling_never_leaks_the_current_row_own_value():
    """The value at row i's _lag1 columns must be computable using ONLY rows before i -- this
    is the direct, mechanical no-lookahead assertion. Verified by checking that mutating row
    i's raw value does not change row i's own _lag1 columns."""
    df = _df()
    result_before = compute_rolling_and_lag(df, group_cols=["entity"], sort_col="start_date", value_cols=["carries"], window=3)

    df_mutated = df.copy()
    df_mutated.loc[df_mutated["entity"] == "a", "carries"] = df_mutated.loc[df_mutated["entity"] == "a", "carries"].iloc[0:1].tolist() * 0 + [999, 999, 999, 999]
    # Only mutate the LAST row's own value; every other row's raw value is unchanged.
    df_mutated2 = df.copy()
    a_idx = df_mutated2[df_mutated2["entity"] == "a"].index[-1]
    df_mutated2.loc[a_idx, "carries"] = 999999
    result_after = compute_rolling_and_lag(df_mutated2, group_cols=["entity"], sort_col="start_date", value_cols=["carries"], window=3)

    a_before = result_before[result_before["entity"] == "a"].reset_index(drop=True)
    a_after = result_after[result_after["entity"] == "a"].reset_index(drop=True)
    # The last row's OWN _lag1 columns must be unaffected by its OWN raw value change.
    assert a_before.loc[3, "carries_avg3_lag1"] == a_after.loc[3, "carries_avg3_lag1"]
    assert a_before.loc[3, "carries_avg_all_lag1"] == a_after.loc[3, "carries_avg_all_lag1"]


def test_games_played_lag1_is_zero_for_first_row():
    result = attach_games_played_lag1(_df(), group_cols=["entity"], sort_col="start_date")
    a = result[result["entity"] == "a"].reset_index(drop=True)
    assert a.loc[0, "games_played_lag1"] == 0
    assert a.loc[1, "games_played_lag1"] == 1
    assert a.loc[3, "games_played_lag1"] == 3

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.ingest.rolling_features import add_rolling_and_lag, add_rolling_averages


def _team_week(points):
    n = len(points)
    return pd.DataFrame(
        {
            "team": ["TeamA"] * n,
            "year": [2024] * n,
            "week": list(range(1, n + 1)),
            "points": points,
        }
    )


def test_add_rolling_averages_avg_all_is_cumulative_mean_including_current_row():
    df = _team_week([10, 20, 30, 40, 50])
    out = add_rolling_averages(df, ["points"], fill_value=None)
    assert out["points_avg_all"].tolist() == [10.0, 15.0, 20.0, 25.0, 30.0]


def test_add_rolling_averages_avg3_none_fill_leaves_nan_before_full_window():
    df = _team_week([10, 20, 30, 40, 50])
    out = add_rolling_averages(df, ["points"], fill_value=None)
    assert out["points_avg3"].isna().tolist() == [True, True, False, False, False]
    assert out["points_avg3"].dropna().tolist() == [20.0, 30.0, 40.0]


def test_add_rolling_averages_avg3_zero_fill_replaces_incomplete_window():
    df = _team_week([10, 20, 30, 40, 50])
    out = add_rolling_averages(df, ["points"], fill_value=0)
    assert out["points_avg3"].tolist() == [0.0, 0.0, 20.0, 30.0, 40.0]


def test_add_rolling_averages_groups_are_independent_per_team():
    df = pd.concat(
        [_team_week([10, 20, 30]), _team_week([100, 200, 300]).assign(team="TeamB")],
        ignore_index=True,
    )
    out = add_rolling_averages(df, ["points"], fill_value=0)
    a = out[out["team"] == "TeamA"]["points_avg_all"].tolist()
    b = out[out["team"] == "TeamB"]["points_avg_all"].tolist()
    assert a == [10.0, 15.0, 20.0]
    assert b == [100.0, 150.0, 200.0]


def test_add_rolling_and_lag_matches_hand_computed_values():
    """Verified by hand: team plays weeks 1-5 scoring [10,20,30,40,50] points.
    Historical path shifts avg_all/avg3/raw by exactly one row, so week N's
    prev_week_points/points_avg_all/points_avg3 describe games strictly before week N."""
    df = _team_week([10, 20, 30, 40, 50])
    out = add_rolling_and_lag(df, ["points"], fill_value=None, drop_incomplete=True)

    # weeks 1-3 dropped: avg3 has no full trailing window until the row that would be
    # shifted into week 4 (avg3 computed through week 3)
    assert sorted(out["week"].tolist()) == [4, 5]

    week4 = out[out["week"] == 4].iloc[0]
    assert week4["prev_week_points"] == 30
    assert week4["points_avg_all"] == 20.0
    assert week4["points_avg3"] == 20.0

    week5 = out[out["week"] == 5].iloc[0]
    assert week5["prev_week_points"] == 40
    assert week5["points_avg_all"] == 25.0
    assert week5["points_avg3"] == 30.0


def test_add_rolling_and_lag_no_lookahead_prev_week_never_equals_own_week_value():
    """Regression guard: prev_week_points for week N must equal the raw points value
    from week N-1, never week N's own value (that would be a lookahead leak)."""
    df = _team_week([10, 20, 30, 40, 50])
    out = add_rolling_and_lag(df, ["points"], fill_value=None, drop_incomplete=False)
    raw_by_week = dict(zip(df["week"], df["points"]))
    for _, row in out.dropna(subset=["prev_week_points"]).iterrows():
        assert row["prev_week_points"] == raw_by_week[row["week"] - 1]
        assert row["prev_week_points"] != raw_by_week.get(row["week"], object())


def test_add_rolling_and_lag_drop_incomplete_false_keeps_all_rows():
    df = _team_week([10, 20, 30, 40, 50])
    out = add_rolling_and_lag(df, ["points"], fill_value=None, drop_incomplete=False)
    assert len(out) == 5
    assert out[out["week"] == 1]["prev_week_points"].isna().all()

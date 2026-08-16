from __future__ import annotations

import pandas as pd

from cfb_spread_model.feature_engineering import (
    apply_differential_representation,
    build_differential_features,
    build_pythagorean_features,
    build_trend_features,
)

_ID_COLS = ["game_id", "home_team", "away_team"]


def _candidate_columns(df):
    return [c for c in df.columns if c not in _ID_COLS]


def _pythagorean_frame():
    """Small local frame (not the shared synthetic_df fixture) with the points_avg_all/avg3 /
    points_allowed_avg_all/avg3 columns build_pythagorean_features needs. home_points_avg3 is
    deliberately omitted to exercise the missing-pair case."""
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "home_team": ["Team A"] * 3,
            "away_team": ["Team B"] * 3,
            "home_points_avg_all": [30.0, 20.0, 0.0],
            "home_points_allowed_avg_all": [10.0, 20.0, 0.0],
            "away_points_avg_all": [14.0, 28.0, 21.0],
            "away_points_allowed_avg_all": [21.0, 14.0, 7.0],
            "away_points_avg3": [17.0, 24.0, 19.0],
            "away_points_allowed_avg3": [24.0, 17.0, 10.0],
        }
    )


def test_build_differential_features_matches_hand_calculation(synthetic_df):
    diffs = build_differential_features(synthetic_df, _candidate_columns(synthetic_df))

    expected = synthetic_df["home_total_yards_avg_all"] - synthetic_df["away_total_yards_avg_all"]
    assert (diffs["diff_avg_all_total_yards"] == expected).all()

    expected_avg3 = synthetic_df["home_total_yards_avg3"] - synthetic_df["away_total_yards_avg3"]
    assert (diffs["diff_avg3_total_yards"] == expected_avg3).all()

    expected_talent = synthetic_df["home_talent"] - synthetic_df["away_talent"]
    assert (diffs["diff_talent"] == expected_talent).all()


def test_build_differential_features_skips_unpaired_metrics(synthetic_df):
    """home_sacks_avg_all/avg3 exist but away_sacks_* does not in the fixture -- no diff should
    be created for a metric missing on one side."""
    diffs = build_differential_features(synthetic_df, _candidate_columns(synthetic_df))
    assert "diff_avg_all_sacks" not in diffs.columns
    assert "diff_avg3_sacks" not in diffs.columns


def test_build_differential_features_never_touches_prev_week(synthetic_df):
    diffs = build_differential_features(synthetic_df, _candidate_columns(synthetic_df))
    assert not any("prev_week" in c for c in diffs.columns)


def test_build_trend_features_matches_hand_calculation(synthetic_df):
    trends = build_trend_features(synthetic_df, _candidate_columns(synthetic_df))

    expected_home = synthetic_df["home_total_yards_avg3"] - synthetic_df["home_total_yards_avg_all"]
    assert (trends["trend_home_total_yards"] == expected_home).all()

    expected_away = synthetic_df["away_total_yards_avg3"] - synthetic_df["away_total_yards_avg_all"]
    assert (trends["trend_away_total_yards"] == expected_away).all()


def test_build_trend_features_requires_both_avg3_and_avg_all(synthetic_df):
    """home_sacks has avg_all and avg3 both present -- trend SHOULD be built here (unlike the
    diff case, trend only needs one side, not both sides paired)."""
    trends = build_trend_features(synthetic_df, _candidate_columns(synthetic_df))
    assert "trend_home_sacks" in trends.columns
    assert "trend_away_sacks" not in trends.columns  # away_sacks_* doesn't exist in the fixture


def test_apply_differential_representation_drops_consumed_raw_columns(synthetic_df):
    result = apply_differential_representation(synthetic_df, _ID_COLS)
    assert "home_total_yards_avg_all" not in result.columns
    assert "away_total_yards_avg_all" not in result.columns
    assert "home_total_yards_avg3" not in result.columns
    assert "home_talent" not in result.columns
    assert "away_talent" not in result.columns
    assert "diff_avg_all_total_yards" in result.columns
    assert "diff_talent" in result.columns
    assert "trend_home_total_yards" in result.columns


def test_apply_differential_representation_preserves_untouched_columns(synthetic_df):
    result = apply_differential_representation(synthetic_df, _ID_COLS)
    for col in ["game_id", "home_team", "away_team", "season", "week", "spread", "home_favored", "home_covered"]:
        assert col in result.columns
        assert (result[col] == synthetic_df[col]).all()
    # prev_week_* columns are untouched by this function (still present; excluded_column_patterns
    # is what removes them later, in get_feature_columns -- a separate, independently-tested step)
    assert "home_prev_week_total_yards" in result.columns
    assert "home_prev_week_sacks" in result.columns


def test_apply_differential_representation_excludes_id_columns_from_pairing(synthetic_df):
    """home_team/away_team must never be treated as a home_/away_ paired metric -- attempting to
    subtract two string columns would raise. This is the specific bug this test guards against."""
    result = apply_differential_representation(synthetic_df, _ID_COLS)
    assert "diff_team" not in result.columns
    assert (result["home_team"] == "Team A").all()
    assert (result["away_team"] == "Team B").all()


def test_apply_differential_representation_row_count_unchanged(synthetic_df):
    result = apply_differential_representation(synthetic_df, _ID_COLS)
    assert len(result) == len(synthetic_df)


def test_build_pythagorean_features_matches_hand_calculation():
    df = _pythagorean_frame()
    pyth = build_pythagorean_features(df, list(df.columns))

    expected_home_avg_all = df["home_points_avg_all"] ** 2 / (
        df["home_points_avg_all"] ** 2 + df["home_points_allowed_avg_all"] ** 2 + 1e-6
    )
    assert (pyth["home_pythagorean_win_pct_avg_all"] - expected_home_avg_all).abs().max() < 1e-9

    expected_away_avg3 = df["away_points_avg3"] ** 2 / (
        df["away_points_avg3"] ** 2 + df["away_points_allowed_avg3"] ** 2 + 1e-6
    )
    assert (pyth["away_pythagorean_win_pct_avg3"] - expected_away_avg3).abs().max() < 1e-9

    assert ((pyth >= 0) & (pyth <= 1)).all().all()


def test_build_pythagorean_features_handles_all_zero_points():
    """Row 3 in _pythagorean_frame has home PF=PA=0 -- must not divide by zero / produce NaN."""
    df = _pythagorean_frame()
    pyth = build_pythagorean_features(df, list(df.columns))
    assert not pyth["home_pythagorean_win_pct_avg_all"].isna().any()
    assert pyth["home_pythagorean_win_pct_avg_all"].iloc[2] == 0.0  # 0**2/(0**2+0**2+eps) = 0/eps = 0


def test_build_pythagorean_features_skips_unpaired_transform():
    """home_points_avg3/home_points_allowed_avg3 are absent from the fixture -- no
    home_pythagorean_win_pct_avg3 column should be produced."""
    df = _pythagorean_frame()
    pyth = build_pythagorean_features(df, list(df.columns))
    assert "home_pythagorean_win_pct_avg3" not in pyth.columns
    assert "home_pythagorean_win_pct_avg_all" in pyth.columns
    assert "away_pythagorean_win_pct_avg3" in pyth.columns


def test_pythagorean_features_flow_into_diff_and_trend():
    """Once home_/away_ pythagorean_win_pct_avg_all columns exist, build_differential_features
    and build_trend_features must pick them up automatically -- no new diff-specific code."""
    df = _pythagorean_frame()
    pyth = build_pythagorean_features(df, list(df.columns))
    combined = pd.concat([df, pyth], axis=1)
    candidate_cols = _candidate_columns(combined)

    diffs = build_differential_features(combined, candidate_cols)
    expected_diff = pyth["home_pythagorean_win_pct_avg_all"] - pyth["away_pythagorean_win_pct_avg_all"]
    assert (diffs["diff_avg_all_pythagorean_win_pct"] == expected_diff).all()

    trends = build_trend_features(combined, candidate_cols)
    expected_trend = pyth["away_pythagorean_win_pct_avg3"] - pyth["away_pythagorean_win_pct_avg_all"]
    assert (trends["trend_away_pythagorean_win_pct"] == expected_trend).all()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.engineered_features import (
    RETURNING_PRODUCTION_DROP,
    RETURNING_PRODUCTION_KEEP,
    apply_engineered_features,
    consolidate_returning_production,
    consolidate_special_teams,
)


def _minimal_special_teams_frame(n=5, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for prefix in ("home_", "away_"):
        for transform_cols in (
            [f"{prefix}prev_week_{s}" for s in [
                "kick_return_tds", "kick_return_tds_allowed", "kick_return_yards", "kick_return_yards_allowed",
                "kick_returns", "kick_returns_allowed", "kicking_points", "kicking_points_allowed",
                "punt_return_tds", "punt_return_tds_allowed", "punt_return_yards", "punt_return_yards_allowed",
                "punt_returns", "punt_returns_allowed",
            ]],
        ):
            for c in transform_cols:
                cols[c] = rng.integers(0, 10, n).astype(float)
    return pd.DataFrame(cols)


def test_returning_production_keeps_exactly_the_four_drivers():
    cols = {}
    for prefix in ("home_", "away_"):
        for stat in RETURNING_PRODUCTION_KEEP + RETURNING_PRODUCTION_DROP:
            cols[f"{prefix}{stat}"] = [1.0, 2.0]
    frame = pd.DataFrame(cols)
    feature_columns = list(cols.keys())

    out_frame, out_cols = consolidate_returning_production(frame, feature_columns)

    for prefix in ("home_", "away_"):
        for stat in RETURNING_PRODUCTION_KEEP:
            assert f"{prefix}{stat}" in out_cols
        for stat in RETURNING_PRODUCTION_DROP:
            assert f"{prefix}{stat}" not in out_cols
    assert len(out_cols) == 2 * len(RETURNING_PRODUCTION_KEEP)
    # frame itself is untouched (columns just excluded from feature_columns, not deleted)
    assert out_frame is frame


def test_special_teams_composite_matches_hand_computed_value():
    frame = pd.DataFrame(
        {
            "home_prev_week_kicking_points": [10.0],
            "home_prev_week_kick_return_yards": [34.0],
            "home_prev_week_kick_return_tds": [1.0],
            "home_prev_week_punt_return_yards": [17.0],
            "home_prev_week_punt_return_tds": [0.0],
            "home_prev_week_kicking_points_allowed": [3.0],
            "home_prev_week_kick_return_yards_allowed": [0.0],
            "home_prev_week_kick_return_tds_allowed": [0.0],
            "home_prev_week_punt_return_yards_allowed": [0.0],
            "home_prev_week_punt_return_tds_allowed": [0.0],
            "home_prev_week_kick_returns": [2.0],
            "home_prev_week_kick_returns_allowed": [1.0],
            "home_prev_week_punt_returns": [1.0],
            "home_prev_week_punt_returns_allowed": [0.0],
        }
    )
    feature_columns = list(frame.columns)

    out_frame, out_cols = consolidate_special_teams(frame, feature_columns)

    # hand-computed: kicking_points(10) + kick_return_yards/17(34/17=2) + kick_return_tds*6(6)
    #                + punt_return_yards/17(17/17=1) + punt_return_tds*6(0) = 19
    # allowed: kicking_points_allowed(3) + 0 + 0 + 0 + 0 = 3
    # net = 19 - 3 = 16
    assert "home_special_teams_net_score_prev_week" in out_cols
    assert out_frame["home_special_teams_net_score_prev_week"].iloc[0] == 16.0
    # every raw special-teams column should have been dropped from the feature list
    for stat in ("kicking_points", "kick_return_yards", "kick_returns"):
        assert f"home_prev_week_{stat}" not in out_cols


def test_special_teams_composite_no_nans_introduced():
    frame = _minimal_special_teams_frame()
    out_frame, out_cols = consolidate_special_teams(frame, list(frame.columns))
    assert out_frame[out_cols].isna().sum().sum() == 0


def test_fourth_down_zero_attempts_does_not_produce_nan():
    """Regression test for a real bug: fourth_down_attempts is frequently 0 (single-game
    prev_week transform especially), and a naive ratio would NaN out ~20-40% of rows once
    dropped downstream. The engineered feature must treat 'no attempts' as neutral (0.0),
    not propagate NaN."""
    n = 20
    frame = pd.DataFrame(
        {
            "home_prev_week_fourth_down_conversion": [0.0] * n,
            "home_prev_week_fourth_down_attempts": [0.0] * n,  # zero attempts for every row
            "away_prev_week_fourth_down_conversion_allowed": [0.0] * n,
            "away_prev_week_fourth_down_attempts_allowed": [0.0] * n,
            "away_prev_week_fourth_down_conversion": [1.0] * n,
            "away_prev_week_fourth_down_attempts": [2.0] * n,
            "home_prev_week_fourth_down_conversion_allowed": [1.0] * n,
            "home_prev_week_fourth_down_attempts_allowed": [2.0] * n,
        }
    )
    from cfb_cover_model.engineered_features import add_opponent_adjusted_features

    out_frame, out_cols = add_opponent_adjusted_features(frame, list(frame.columns))
    fourth_down_cols = [c for c in out_cols if "fourth_down_rate" in c]
    assert len(fourth_down_cols) > 0
    assert out_frame[fourth_down_cols].isna().sum().sum() == 0


def test_matchup_adjustment_is_additive_not_replacing():
    frame = pd.DataFrame(
        {
            "home_prev_week_Offense_Success_Rate": [0.5],
            "away_prev_week_Defense_Success_Rate": [0.4],
            "away_prev_week_Offense_Success_Rate": [0.45],
            "home_prev_week_Defense_Success_Rate": [0.35],
        }
    )
    from cfb_cover_model.engineered_features import add_opponent_adjusted_features

    out_frame, out_cols = add_opponent_adjusted_features(frame, list(frame.columns))
    # raw columns must still be present - opponent-adjustment adds, never replaces (see
    # module docstring and docs/data_leakage_rules.md's stability rationale)
    for raw_col in frame.columns:
        assert raw_col in out_cols
    assert "home_matchup_adj_Offense_Success_Rate_prev_week" in out_cols
    expected = 0.5 - 0.4
    assert abs(out_frame["home_matchup_adj_Offense_Success_Rate_prev_week"].iloc[0] - expected) < 1e-9


def test_apply_engineered_features_end_to_end_no_nans():
    frame = _minimal_special_teams_frame(n=10, seed=1)
    for prefix in ("home_", "away_"):
        for stat in ["rushing_usage", "receiving_usage", "percent_rushing_ppa", "total_rushing_ppa", "usage", "total_ppa"]:
            frame[f"{prefix}{stat}"] = np.random.default_rng(2).uniform(0, 1, 10)
        for off_stat, def_stat in [("Offense_Success_Rate", "Defense_Success_Rate")]:
            frame[f"{prefix}prev_week_{off_stat}"] = np.random.default_rng(3).uniform(0, 1, 10)
            frame[f"{prefix}prev_week_{def_stat}"] = np.random.default_rng(4).uniform(0, 1, 10)

    out_frame, out_cols = apply_engineered_features(frame, list(frame.columns))
    assert out_frame[out_cols].isna().sum().sum() == 0
    assert len(out_cols) > 0

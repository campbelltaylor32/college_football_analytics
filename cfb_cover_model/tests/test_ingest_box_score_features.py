import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.ingest.box_score_features import (
    add_allowed_columns,
    add_engineered_ratios,
    clean_box_score,
    consensus_or_average_spread,
)


def test_add_allowed_columns_attaches_opponent_stats():
    stats = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team": ["Home", "Away"],
            "home_away": ["home", "away"],
            "conference": ["X", "Y"],
            "week": [1, 1],
            "year": [2024, 2024],
            "total_yards": [400, 250],
        }
    )
    out = add_allowed_columns(stats)
    home_row = out[out["team"] == "Home"].iloc[0]
    away_row = out[out["team"] == "Away"].iloc[0]
    assert home_row["total_yards_allowed"] == 250
    assert away_row["total_yards_allowed"] == 400


def test_add_allowed_columns_skips_incomplete_games():
    stats = pd.DataFrame(
        {
            "game_id": [1],
            "team": ["Home"],
            "home_away": ["home"],
            "conference": ["X"],
            "week": [1],
            "year": [2024],
            "total_yards": [400],
        }
    )
    out = add_allowed_columns(stats)
    assert out.empty


def test_clean_box_score_splits_eff_fields_and_parses_possession_time():
    raw = pd.DataFrame(
        {
            "game_id": [1],
            "team": ["Home"],
            "home_away": ["home"],
            "conference": ["X"],
            "week": [1],
            "year": [2024],
            "third_down_eff": ["4-10"],
            "third_down_eff_allowed": ["2-8"],
            "possession_time": ["32:15"],
            "possession_time_allowed": ["27:45"],
        }
    )
    out = clean_box_score(raw)
    assert out["third_down_conversion"].iloc[0] == 4
    assert out["third_down_attempts"].iloc[0] == 10
    assert out["third_down_conversion_allowed"].iloc[0] == 2
    assert out["third_down_attempts_allowed"].iloc[0] == 8
    assert abs(out["possession_time"].iloc[0] - (32 + 15 / 60)) < 1e-9
    assert abs(out["possession_time_allowed"].iloc[0] - (27 + 45 / 60)) < 1e-9


def test_clean_box_score_fills_missing_with_zero():
    raw = pd.DataFrame(
        {
            "game_id": [1],
            "team": ["Home"],
            "home_away": ["home"],
            "conference": ["X"],
            "week": [1],
            "year": [2024],
            "total_yards": [np.nan],
        }
    )
    out = clean_box_score(raw)
    assert out["total_yards"].iloc[0] == 0


def test_add_engineered_ratios_hand_computed():
    df = pd.DataFrame(
        {
            "third_down_conversion": [4.0], "third_down_attempts": [10.0],
            "fourth_down_conversion": [1.0], "fourth_down_attempts": [2.0],
            "qb_hurries": [3.0], "sacks": [2.0],
            "completion_attempts_against": [30.0],
            "qb_hurries_allowed": [1.0], "sacks_allowed": [0.0],
            "attempted_passes": [25.0],
            "passes_intercepted": [1.0],
            "interceptions": [2.0],
            "points": [28.0], "points_allowed": [14.0],
            "possession_time": [32.0], "possession_time_allowed": [28.0],
            "turnovers": [1.0], "turnovers_allowed": [2.0],
            "penalty_yards": [50.0], "penalty_yards_allowed": [30.0],
            "rushing_attempts": [35.0], "rushing_attempts_allowed": [20.0],
            "total_yards": [450.0], "total_yards_allowed": [300.0],
        }
    )
    out = add_engineered_ratios(df)
    assert abs(out["third_down_percentage_offense"].iloc[0] - 0.4) < 1e-9
    assert abs(out["fourth_down_percentage_offense"].iloc[0] - 0.5) < 1e-9
    assert abs(out["pressure_percentage"].iloc[0] - 3 / 30) < 1e-9
    assert abs(out["sack_percentage"].iloc[0] - 2 / 30) < 1e-9
    assert abs(out["pressure_percentage_allowed"].iloc[0] - 1 / 25) < 1e-9
    assert abs(out["interception_rate_offense"].iloc[0] - 1 / 25) < 1e-9
    assert abs(out["intercetpion_rate_defense"].iloc[0] - 2 / 30) < 1e-9
    assert out["point_differential"].iloc[0] == 14
    assert out["possession_time_difference"].iloc[0] == 4
    assert out["turnover_margin"].iloc[0] == -1
    assert out["penalty_yard_margin"].iloc[0] == 20
    assert out["total_plays"].iloc[0] == 60
    assert abs(out["rush_percentage"].iloc[0] - 35 / 60) < 1e-9
    assert abs(out["yards_per_play"].iloc[0] - 450 / 60) < 1e-9
    assert out["total_plays_against"].iloc[0] == 50
    assert abs(out["rush_percentage_against"].iloc[0] - 20 / 50) < 1e-9


def test_add_engineered_ratios_zero_denominator_produces_nan_not_error():
    df = pd.DataFrame(
        {
            "third_down_conversion": [0.0], "third_down_attempts": [0.0],
            "fourth_down_conversion": [0.0], "fourth_down_attempts": [0.0],
            "qb_hurries": [0.0], "sacks": [0.0],
            "completion_attempts_against": [0.0],
            "qb_hurries_allowed": [0.0], "sacks_allowed": [0.0],
            "attempted_passes": [0.0],
            "passes_intercepted": [0.0],
            "interceptions": [0.0],
            "points": [0.0], "points_allowed": [0.0],
            "possession_time": [0.0], "possession_time_allowed": [0.0],
            "turnovers": [0.0], "turnovers_allowed": [0.0],
            "penalty_yards": [0.0], "penalty_yards_allowed": [0.0],
            "rushing_attempts": [0.0], "rushing_attempts_allowed": [0.0],
            "total_yards": [0.0], "total_yards_allowed": [0.0],
        }
    )
    out = add_engineered_ratios(df)
    assert pd.isna(out["third_down_percentage_offense"].iloc[0])
    assert pd.isna(out["rush_percentage"].iloc[0])


def test_consensus_spread_prefers_consensus_provider():
    lines = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "provider": ["consensus", "DraftKings", "ESPN Bet"],
            "spread": [-3.5, -3.0, -4.0],
            "formatted_spread": ["A -3.5", "A -3.0", "A -4.0"],
            "over_under": [50.0, 51.0, 49.5],
        }
    )
    out = consensus_or_average_spread(lines)
    assert len(out) == 1
    assert out["spread"].iloc[0] == -3.5


def test_consensus_spread_falls_back_to_average_when_no_consensus():
    lines = pd.DataFrame(
        {
            "game_id": [2, 2],
            "home_team": ["C", "C"],
            "away_team": ["D", "D"],
            "provider": ["DraftKings", "ESPN Bet"],
            "spread": [-3.0, -5.0],
            "formatted_spread": ["C -3.0", "C -5.0"],
            "over_under": [50.0, 52.0],
        }
    )
    out = consensus_or_average_spread(lines)
    assert len(out) == 1
    assert abs(out["spread"].iloc[0] - (-4.0)) < 1e-9


def test_consensus_spread_mixed_games_each_use_own_policy():
    lines = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "home_team": ["A", "A", "C", "C"],
            "away_team": ["B", "B", "D", "D"],
            "provider": ["consensus", "DraftKings", "DraftKings", "ESPN Bet"],
            "spread": [-3.5, -3.0, -3.0, -5.0],
            "formatted_spread": ["A -3.5", "A -3.0", "C -3.0", "C -5.0"],
            "over_under": [50.0, 51.0, 50.0, 52.0],
        }
    )
    out = consensus_or_average_spread(lines).set_index("game_id")
    assert out.loc[1, "spread"] == -3.5
    assert abs(out.loc[2, "spread"] - (-4.0)) < 1e-9

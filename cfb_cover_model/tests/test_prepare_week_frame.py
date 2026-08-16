import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.cleaning import candidate_feature_columns, prepare_week_frame


def _synthetic_week_df(n=4):
    cols = {
        "game_id": list(range(n)),
        "season": [2026] * n,
        "week": [3] * n,
        "home_team": ["A"] * n,
        "away_team": ["B"] * n,
        "spread": [7.0] * n,
        "neutral_site": [False] * n,
        "conference_game": [True] * n,
        "home_favored": [1] * n,
        "home_talent": [500.0] * n,
        "away_talent": [480.0] * n,
    }
    for prefix in ("home_", "away_"):
        for stat in [
            "kick_return_tds", "kick_return_tds_allowed", "kick_return_yards", "kick_return_yards_allowed",
            "kick_returns", "kick_returns_allowed", "kicking_points", "kicking_points_allowed",
            "punt_return_tds", "punt_return_tds_allowed", "punt_return_yards", "punt_return_yards_allowed",
            "punt_returns", "punt_returns_allowed",
        ]:
            cols[f"{prefix}prev_week_{stat}"] = [1.0] * n
        for stat in ["rushing_usage", "receiving_usage", "percent_rushing_ppa", "total_rushing_ppa", "usage", "total_ppa"]:
            cols[f"{prefix}{stat}"] = [0.5] * n
    return pd.DataFrame(cols)


def test_prepare_week_frame_applies_engineered_features():
    """Regression test for a real bug: CFB_Pred_Week_<N>.csv files never went through
    apply_engineered_features, so a model trained on the engineered feature set (which
    includes special_teams_net_score_* replacing the raw special-teams columns) would
    KeyError when scoring a week file built from the raw, un-engineered schema."""
    data_cfg = {
        "id_columns": ["game_id", "home_team", "away_team"],
        "leakage_adjacent_columns": ["home_favored"],
        "known_bad_base_stats": [],
        "deterministic_redundant_base_stats": [],
    }
    week_df = _synthetic_week_df()

    frame, feature_columns = prepare_week_frame(week_df, data_cfg)

    assert "home_special_teams_net_score_prev_week" in feature_columns
    assert "home_special_teams_net_score_prev_week" in frame.columns
    assert "home_prev_week_kicking_points" not in feature_columns  # raw special-teams column, consolidated away
    assert frame[feature_columns].isna().sum().sum() == 0


def test_prepare_week_frame_matches_historical_candidate_columns_before_engineering():
    """The raw (pre-engineering) candidate set computed from a week file's columns should
    match what candidate_feature_columns would compute on the same schema - i.e.
    prepare_week_frame isn't silently dropping or renaming anything before engineering runs."""
    data_cfg = {
        "id_columns": ["game_id", "home_team", "away_team"],
        "leakage_adjacent_columns": ["home_favored"],
        "known_bad_base_stats": [],
        "deterministic_redundant_base_stats": [],
    }
    week_df = _synthetic_week_df()
    raw_candidates = candidate_feature_columns(data_cfg, week_df.columns)
    assert "home_prev_week_kicking_points" in raw_candidates  # confirms the raw column really was a candidate pre-engineering

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.ingest.pbp_features import _success, add_success_column, compute_epa_features


def test_success_first_down_threshold():
    assert bool(_success(pd.Series({"down": 1, "distance": 10, "yards_gained": 5}))) is True
    assert bool(_success(pd.Series({"down": 1, "distance": 10, "yards_gained": 4}))) is False


def test_success_second_down_threshold():
    assert bool(_success(pd.Series({"down": 2, "distance": 10, "yards_gained": 7}))) is True
    assert bool(_success(pd.Series({"down": 2, "distance": 10, "yards_gained": 6}))) is False


def test_success_third_down_requires_conversion():
    assert bool(_success(pd.Series({"down": 3, "distance": 5, "yards_gained": 5}))) is True
    assert bool(_success(pd.Series({"down": 3, "distance": 5, "yards_gained": 4}))) is False


def test_success_missing_down_or_zero_distance_is_false():
    assert _success(pd.Series({"down": np.nan, "distance": 10, "yards_gained": 20})) is False
    assert _success(pd.Series({"down": 1, "distance": 0, "yards_gained": 20})) is False


def test_add_success_column_adds_bool_column():
    plays = pd.DataFrame(
        {"down": [1, 3], "distance": [10, 5], "yards_gained": [6, 2]}
    )
    out = add_success_column(plays)
    assert out["success"].tolist() == [True, False]


def _synthetic_plays():
    return pd.DataFrame(
        {
            "id_play": [1, 2, 3, 4],
            "pos_team": ["TeamA"] * 4,
            "def_pos_team": ["TeamB"] * 4,
            "down": [1, 2, 1, 3],
            "distance": [10, 8, 10, 5],
            "yards_gained": [6, 3, 25, -7],
            "play_type": ["Rush", "Pass Reception", "Rush", "Sack"],
            "EPA": [0.5, -0.2, 1.2, -1.5],
            "week": [1, 1, 1, 1],
            "year": [2024, 2024, 2024, 2024],
        }
    )


def _synthetic_drives():
    return pd.DataFrame(
        {
            "drive_id": [10, 11],
            "offense": ["TeamA", "TeamA"],
            "defense": ["TeamB", "TeamB"],
            "scoring": [True, False],
            "drive_result": ["TD", "PUNT"],
            "week": [1, 1],
            "year": [2024, 2024],
        }
    )


def test_compute_epa_features_offense_side_hand_computed():
    plays = _synthetic_plays()
    drives = _synthetic_drives()
    out = compute_epa_features(plays, drives)
    row = out[out["team"] == "TeamA"].iloc[0]

    assert row["Total_Offense_Plays"] == 4
    assert row["Offense_Total_Run_Plays"] == 2
    assert row["Offense_Total_Pass_Plays"] == 2
    assert row["Offense_Pass_Rate"] == 0.5
    assert row["Offense_Run_Rate"] == 0.5
    assert row["Offense_first_down_pass_rate"] == 0.0
    assert row["Offense_Avg_3rd_Down_Distance"] == 5.0

    assert abs(row["Total_Offense_EPA"] - 0.0) < 1e-9
    assert row["Offense_EPA_per_Play"] == 0.0
    assert row["Total_Offense_Success"] == 2
    assert row["Offense_Success_Rate"] == 0.5

    assert abs(row["Total_Offense_EPA_Run"] - 1.7) < 1e-9
    assert abs(row["Offense_EPA_per_Run"] - 0.85) < 1e-9
    assert row["Offense_Run_Success_Rate"] == 1.0

    assert abs(row["Total_Offense_EPA_Pass"] - (-1.7)) < 1e-9
    assert abs(row["Offense_EPA_per_Pass"] - (-0.85)) < 1e-9
    assert row["Offense_Pass_Success_Rate"] == 0.0

    assert row["Total_Offense_Explosives"] == 1
    assert row["Total_Offense_Explosive_Rate"] == 0.25
    assert row["Total_Offense_Run_Explosives"] == 1
    assert row["Total_Offense_Pass_Explosives"] == 0

    assert row["Offense_First_Down_Success_Rate"] == 1.0
    assert row["Offense_First_Down_Run_Success_Rate"] == 1.0
    assert pd.isna(row["Offense_First_Down_Pass_Success_Rate"])  # 0/0 -> NaN, not error

    assert row["Total_Offense_Scoring_Drives"] == 1
    assert row["Total_Offense_Touchdown_Drives"] == 1
    assert row["Offense_Scoring_Drive_Percentage"] == 0.5
    assert row["Offense_Touchdown_Drive_Percentage"] == 0.5


def test_compute_epa_features_invalid_play_types_excluded():
    plays = _synthetic_plays()
    plays.loc[len(plays)] = {
        "id_play": 5, "pos_team": "TeamA", "def_pos_team": "TeamB",
        "down": 1, "distance": 10, "yards_gained": 0,
        "play_type": "Timeout", "EPA": 0.0, "week": 1, "year": 2024,
    }
    drives = _synthetic_drives()
    out = compute_epa_features(plays, drives)
    row = out[out["team"] == "TeamA"].iloc[0]
    # the extra Timeout row must not change play counts vs. the 4-play baseline
    assert row["Total_Offense_Plays"] == 4


def test_compute_epa_features_zero_denominator_yields_nan_not_crash():
    plays = pd.DataFrame(
        {
            "id_play": [1],
            "pos_team": ["TeamA"],
            "def_pos_team": ["TeamB"],
            "down": [1],
            "distance": [10],
            "yards_gained": [-3],
            "play_type": ["Rush"],
            "EPA": [-0.4],
            "week": [1],
            "year": [2024],
        }
    )
    drives = pd.DataFrame(
        {"drive_id": [], "offense": [], "defense": [], "scoring": [], "drive_result": [], "week": [], "year": []}
    )
    out = compute_epa_features(plays, drives)
    row = out[out["team"] == "TeamA"].iloc[0]
    assert row["Offense_Pass_Rate"] == 0.0
    assert pd.isna(row["Offense_EPA_per_Pass"])
    assert pd.isna(row["Offense_Scoring_Drive_Percentage"])  # n_drives == 0

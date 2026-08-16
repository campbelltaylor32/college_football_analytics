import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.ingest.talent_coach_returning import (
    compute_blue_chip_ratio,
    compute_coach_cumulative_record,
    lag_coach_by_year,
    merge_talent,
    merge_talent_coach_returning,
)


def test_compute_blue_chip_ratio_hand_computed():
    roster = pd.DataFrame(
        {
            "athlete_id": [1, 2, 3],
            "team": ["TeamA", "TeamA", "TeamA"],
            "year": [2024, 2024, 2024],
            "position": ["QB", "RB", "WR"],
        }
    )
    recruits = pd.DataFrame(
        {
            "athlete_id": [1, 2],
            "recruit_year": [2022, 2021],
            "stars": [5, 3],
            "rating": [0.99, 0.85],
        }
    )
    out = compute_blue_chip_ratio(roster, recruits).set_index(["team", "year"])
    row = out.loc[("TeamA", 2024)]
    assert abs(row["blue_chip_ratio"] - 1 / 3) < 1e-9
    assert abs(row["avg_player_rating"] - 0.92) < 1e-9


def test_compute_blue_chip_ratio_no_recruits_at_all_is_zero_not_nan():
    roster = pd.DataFrame(
        {"athlete_id": [1], "team": ["TeamB"], "year": [2024], "position": ["QB"]}
    )
    recruits = pd.DataFrame({"athlete_id": [], "recruit_year": [], "stars": [], "rating": []})
    out = compute_blue_chip_ratio(roster, recruits)
    assert out["blue_chip_ratio"].iloc[0] == 0.0
    assert out["avg_player_rating"].iloc[0] == 0.0


def test_compute_blue_chip_ratio_joins_on_athlete_id_only_not_year():
    """Intentional (non-)behavior: a recruit's stars/rating attach to every season that
    athlete appears on a roster, not just their recruiting class year - see the module
    docstring. This locks in that behavior so a future "fix" doesn't silently change it."""
    roster = pd.DataFrame(
        {"athlete_id": [1, 1], "team": ["TeamA", "TeamA"], "year": [2022, 2024], "position": ["QB", "QB"]}
    )
    recruits = pd.DataFrame({"athlete_id": [1], "recruit_year": [2022], "stars": [5], "rating": [0.99]})
    out = compute_blue_chip_ratio(roster, recruits).set_index("year")
    assert out.loc[2022, "blue_chip_ratio"] == 1.0
    assert out.loc[2024, "blue_chip_ratio"] == 1.0


def test_merge_talent_fills_missing_blue_chip_with_zero():
    talent = pd.DataFrame({"year": [2024], "team": ["TeamC"], "talent": [800.0], "Scaled_Talent": [1.2]})
    blue_chip = pd.DataFrame({"team": ["TeamA"], "year": [2024], "blue_chip_ratio": [0.5], "avg_player_rating": [0.9]})
    out = merge_talent(talent, blue_chip)
    assert out["blue_chip_ratio"].iloc[0] == 0.0
    assert out["avg_player_rating"].iloc[0] == 0.0


def test_compute_coach_cumulative_record_hand_computed():
    coaches = pd.DataFrame(
        {
            "Name": ["Coach X", "Coach X", "Coach X"],
            "team": ["TeamA", "TeamA", "TeamA"],
            "year": [2021, 2022, 2023],
            "games": [12, 13, 14],
            "wins": [10, 11, 9],
        }
    )
    out = compute_coach_cumulative_record(coaches).set_index("year")
    assert out.loc[2021, "Total_Games_Coached"] == 12
    assert abs(out.loc[2021, "Winning_Percentage"] - 10 / 12) < 1e-9
    assert out.loc[2022, "Total_Games_Coached"] == 25
    assert abs(out.loc[2022, "Winning_Percentage"] - 21 / 25) < 1e-9
    assert out.loc[2023, "Total_Games_Coached"] == 39
    assert abs(out.loc[2023, "Winning_Percentage"] - 30 / 39) < 1e-9


def test_compute_coach_cumulative_record_separate_per_coach():
    coaches = pd.DataFrame(
        {
            "Name": ["Coach X", "Coach Y"],
            "team": ["TeamA", "TeamB"],
            "year": [2024, 2024],
            "games": [12, 8],
            "wins": [10, 3],
        }
    )
    out = compute_coach_cumulative_record(coaches).set_index("Name")
    assert out.loc["Coach X", "Total_Games_Coached"] == 12
    assert out.loc["Coach Y", "Total_Games_Coached"] == 8


def test_lag_coach_by_year_no_lookahead():
    """Verified against a real-world case (Kirby Smart, 2022-2024): a team's season-Y coach
    features must reflect the coach's cumulative record accumulated through season Y-1,
    never including season Y itself, and the most recent season (which has nothing to lag
    into) must be dropped entirely."""
    coach_record = pd.DataFrame(
        {
            "Name": ["Coach X", "Coach X", "Coach X"],
            "year": [2021, 2022, 2023],
            "team": ["TeamA", "TeamA", "TeamA"],
            "Total_Games_Coached": [12, 25, 39],
            "Winning_Percentage": [10 / 12, 21 / 25, 30 / 39],
        }
    )
    out = lag_coach_by_year(coach_record).set_index("year")
    # 2023's row now carries 2022's cumulative stats (through 2022, not through 2023)
    assert out.loc[2023, "Total_Games_Coached"] == 25
    assert abs(out.loc[2023, "Winning_Percentage"] - 21 / 25) < 1e-9
    # 2022's row now carries 2021's cumulative stats
    assert out.loc[2022, "Total_Games_Coached"] == 12
    assert abs(out.loc[2022, "Winning_Percentage"] - 10 / 12) < 1e-9
    # the original 2021 row has nothing to lag into and must be dropped
    assert 2021 not in out.index
    assert len(out) == 2


def test_merge_talent_coach_returning_filters_min_year_and_outer_joins():
    talent = pd.DataFrame(
        {"year": [2022, 2023], "team": ["TeamA", "TeamA"], "talent": [700.0, 750.0], "Scaled_Talent": [0.1, 0.2]}
    )
    coach_lagged = pd.DataFrame(
        {"Name": ["Coach X", "Coach X"], "year": [2022, 2023], "team": ["TeamA", "TeamA"],
         "Total_Games_Coached": [12, 25], "Winning_Percentage": [0.8, 0.84]}
    )
    returning = pd.DataFrame({"season": [2023], "team": ["TeamA"], "total_ppa": [55.0]})

    out = merge_talent_coach_returning(talent, coach_lagged, returning, min_year=2023)
    assert set(out["year"]) == {2023}
    row = out.iloc[0]
    assert row["talent"] == 750.0
    assert row["Total_Games_Coached"] == 25
    assert row["total_ppa"] == 55.0


def test_merge_talent_coach_returning_drops_duplicate_team_year_rows():
    talent = pd.DataFrame(
        {"year": [2023, 2023], "team": ["TeamA", "TeamA"], "talent": [750.0, 750.0], "Scaled_Talent": [0.2, 0.2]}
    )
    coach_lagged = pd.DataFrame(
        {"Name": [], "year": [], "team": [], "Total_Games_Coached": [], "Winning_Percentage": []}
    )
    returning = pd.DataFrame({"season": [], "team": [], "total_ppa": []})
    out = merge_talent_coach_returning(talent, coach_lagged, returning, min_year=2020)
    assert len(out[(out["year"] == 2023) & (out["team"] == "TeamA")]) == 1

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.schedule_features import attach_bye_flags, compute_bye_flags


def _schedule():
    # Team A: 2024 weeks 1 (home), 2 (away), 4 (home) - gap between week 2 and week 4.
    # Team B/C round out the home/away slots. Team A reappears in 2025 week 1 - a new
    # season, so the 2024 week-4 finish must not carry over as a "prior week".
    return pd.DataFrame(
        [
            {"season": 2024, "week": 1, "home_team": "Team A", "away_team": "Team B"},
            {"season": 2024, "week": 2, "home_team": "Team B", "away_team": "Team A"},
            {"season": 2024, "week": 4, "home_team": "Team A", "away_team": "Team C"},
            {"season": 2025, "week": 1, "home_team": "Team A", "away_team": "Team B"},
        ]
    )


def test_first_appearance_of_a_season_is_never_off_bye():
    flags = compute_bye_flags(_schedule())
    row = flags[(flags["team"] == "Team A") & (flags["season"] == 2024) & (flags["week"] == 1)]
    assert row["off_bye"].iloc[0] == False  # noqa: E712


def test_consecutive_played_weeks_is_not_off_bye():
    flags = compute_bye_flags(_schedule())
    row = flags[(flags["team"] == "Team A") & (flags["season"] == 2024) & (flags["week"] == 2)]
    assert row["off_bye"].iloc[0] == False  # noqa: E712


def test_gap_in_played_weeks_is_off_bye():
    flags = compute_bye_flags(_schedule())
    row = flags[(flags["team"] == "Team A") & (flags["season"] == 2024) & (flags["week"] == 4)]
    assert row["off_bye"].iloc[0] == True  # noqa: E712


def test_bye_detection_resets_across_seasons():
    """Team A's 2024 season ended at week 4; 2025 week 1 must be treated as a fresh start,
    not a multi-week gap since week 4 of the prior season."""
    flags = compute_bye_flags(_schedule())
    row = flags[(flags["team"] == "Team A") & (flags["season"] == 2025) & (flags["week"] == 1)]
    assert row["off_bye"].iloc[0] == False  # noqa: E712


def test_compute_bye_flags_one_row_per_team_week_appearance():
    flags = compute_bye_flags(_schedule())
    # Team A appears as home or away in every one of the 4 games - one row per appearance,
    # deduplicated (home/away frames concatenated then drop_duplicates()).
    team_a_rows = flags[flags["team"] == "Team A"]
    assert len(team_a_rows) == 4
    assert set(zip(team_a_rows["season"], team_a_rows["week"])) == {(2024, 1), (2024, 2), (2024, 4), (2025, 1)}


def test_attach_bye_flags_joins_onto_games_by_season_week_team():
    schedule = _schedule()
    games = schedule.copy()
    games["game_id"] = range(len(games))

    result = attach_bye_flags(games, schedule)

    week4_row = result[(result["season"] == 2024) & (result["week"] == 4)].iloc[0]
    assert week4_row["home_off_bye"] == 1  # Team A, home, off a bye
    assert week4_row["away_off_bye"] == 0  # Team C, first appearance

    week1_row = result[(result["season"] == 2024) & (result["week"] == 1)].iloc[0]
    assert week1_row["home_off_bye"] == 0
    assert week1_row["away_off_bye"] == 0


def test_attach_bye_flags_defaults_unmatched_rows_to_zero_int():
    schedule = _schedule()
    # A game for a team/season/week combination absent from `schedule` entirely.
    games = pd.DataFrame(
        [{"season": 2099, "week": 1, "home_team": "Team Z", "away_team": "Team Y", "game_id": 0}]
    )

    result = attach_bye_flags(games, schedule)

    assert result["home_off_bye"].iloc[0] == 0
    assert result["away_off_bye"].iloc[0] == 0
    assert result["home_off_bye"].dtype == int
    assert result["away_off_bye"].dtype == int

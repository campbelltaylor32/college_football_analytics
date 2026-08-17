import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_power_ratings.features import roster_experience


def _roster_row(season, team, athlete_id, year):
    return {"season": season, "team": team, "athlete_id": athlete_id, "year": year}


def test_class_filter_excludes_out_of_range_and_season_matching_values(monkeypatch):
    # athlete 1: valid class (JR=3). athlete 2: out-of-range (0). athlete 3: the corruption
    # pattern (year == season). Only athlete 1 should count toward class_avg.
    roster = pd.DataFrame([
        _roster_row(2024, "A", "1", 3),
        _roster_row(2024, "A", "2", 0),
        _roster_row(2024, "A", "3", 2024),
    ])

    def fake_run_query(sql, params=None, engine=None):
        return roster[roster["season"].isin(params["seasons"])]

    monkeypatch.setattr(roster_experience, "run_query", fake_run_query)

    out = roster_experience._build_class_features(engine=None, seasons=[2024], min_valid_class_rows=1)
    row = out[(out["team"] == "A") & (out["season"] == 2024)].iloc[0]

    assert row["class_avg"] == pytest.approx(3.0)
    assert row["class_valid_row_share"] == pytest.approx(1 / 3)


def test_class_avg_is_nan_below_min_valid_rows_gate(monkeypatch):
    roster = pd.DataFrame([_roster_row(2024, "A", "1", 3), _roster_row(2024, "A", "2", 4)])

    def fake_run_query(sql, params=None, engine=None):
        return roster[roster["season"].isin(params["seasons"])]

    monkeypatch.setattr(roster_experience, "run_query", fake_run_query)

    out = roster_experience._build_class_features(engine=None, seasons=[2024], min_valid_class_rows=5)
    row = out[(out["team"] == "A") & (out["season"] == 2024)].iloc[0]

    assert pd.isna(row["class_avg"])  # only 2 valid rows, gate requires 5
    assert row["class_valid_row_share"] == pytest.approx(1.0)  # share is still reported


def test_tenure_zero_for_true_freshman_first_appearance(monkeypatch):
    # Player "1" appears only in 2024 -- a true freshman that season, tenure must be 0.
    all_rosters = pd.DataFrame([_roster_dict_no_year(2024, "A", "1")])

    def fake_run_query(sql, params=None, engine=None):
        return all_rosters[(all_rosters["season"] >= params["start"]) & (all_rosters["season"] <= params["end"])]

    monkeypatch.setattr(roster_experience, "run_query", fake_run_query)

    out = roster_experience._build_tenure_features(engine=None, seasons=[2024], tenure_lookback_seasons=5)
    row = out[(out["team"] == "A") & (out["season"] == 2024)].iloc[0]
    assert row["avg_roster_experience"] == pytest.approx(0.0)


def test_tenure_counts_prior_seasons_at_any_team_transfer_included(monkeypatch):
    # Player "1": on team B in 2021 and 2022, transfers to team A in 2024 -- both prior
    # appearances (any team) should count toward tenure entering 2024: tenure = 2.
    all_rosters = pd.DataFrame([
        _roster_dict_no_year(2021, "B", "1"),
        _roster_dict_no_year(2022, "B", "1"),
        _roster_dict_no_year(2024, "A", "1"),
    ])

    def fake_run_query(sql, params=None, engine=None):
        return all_rosters[(all_rosters["season"] >= params["start"]) & (all_rosters["season"] <= params["end"])]

    monkeypatch.setattr(roster_experience, "run_query", fake_run_query)

    out = roster_experience._build_tenure_features(engine=None, seasons=[2024], tenure_lookback_seasons=5)
    row = out[(out["team"] == "A") & (out["season"] == 2024)].iloc[0]
    assert row["avg_roster_experience"] == pytest.approx(2.0)


def test_tenure_ignores_appearances_outside_lookback_window(monkeypatch):
    # Player "1" appeared in 2015 (10 years before 2025) -- outside a 5-year lookback window,
    # must not count. Only the 2022 appearance (3 years back) counts -> tenure = 1.
    all_rosters = pd.DataFrame([
        _roster_dict_no_year(2015, "A", "1"),
        _roster_dict_no_year(2022, "A", "1"),
        _roster_dict_no_year(2025, "A", "1"),
    ])

    captured_bounds = {}

    def fake_run_query(sql, params=None, engine=None):
        captured_bounds.update(params)
        return all_rosters[(all_rosters["season"] >= params["start"]) & (all_rosters["season"] <= params["end"])]

    monkeypatch.setattr(roster_experience, "run_query", fake_run_query)

    out = roster_experience._build_tenure_features(engine=None, seasons=[2025], tenure_lookback_seasons=5)
    # The lookback query itself should never even ask for 2015's data (start = 2025-5 = 2020),
    # confirming the 2015 row is excluded at the query level, not just post-hoc.
    assert captured_bounds["start"] == 2020

    row = out[(out["team"] == "A") & (out["season"] == 2025)].iloc[0]
    assert row["avg_roster_experience"] == pytest.approx(1.0)


def test_veteran_roster_share_threshold(monkeypatch):
    # Two players entering 2025: one with 3 prior seasons (veteran), one with 1 (not).
    all_rosters = pd.DataFrame([
        _roster_dict_no_year(2022, "A", "1"), _roster_dict_no_year(2023, "A", "1"), _roster_dict_no_year(2024, "A", "1"),
        _roster_dict_no_year(2024, "A", "2"),
        _roster_dict_no_year(2025, "A", "1"), _roster_dict_no_year(2025, "A", "2"),
    ])

    def fake_run_query(sql, params=None, engine=None):
        return all_rosters[(all_rosters["season"] >= params["start"]) & (all_rosters["season"] <= params["end"])]

    monkeypatch.setattr(roster_experience, "run_query", fake_run_query)

    out = roster_experience._build_tenure_features(engine=None, seasons=[2025], tenure_lookback_seasons=5)
    row = out[(out["team"] == "A") & (out["season"] == 2025)].iloc[0]
    assert row["veteran_roster_share"] == pytest.approx(0.5)  # 1 of 2 players has tenure >= 3


def _roster_dict_no_year(season, team, athlete_id):
    return {"season": season, "team": team, "athlete_id": athlete_id}

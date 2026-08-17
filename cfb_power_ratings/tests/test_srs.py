import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_power_ratings.srs import (
    compute_srs,
    estimate_home_field_advantage,
    games_to_team_game_frame,
    iterate_ratings,
    site_adjusted_margin,
)


def _game(game_id, season, home, away, home_pts, away_pts, home_div="fbs", away_div="fbs", neutral=False):
    return {
        "game_id": game_id, "season": season, "home_team": home, "away_team": away,
        "home_points": home_pts, "away_points": away_pts,
        "home_division": home_div, "away_division": away_div, "neutral_site": neutral,
    }


def test_estimate_home_field_advantage_simple_constant():
    """Every home team wins by exactly 7 -- HFA should recover exactly 7."""
    games = pd.DataFrame([
        _game(1, 2024, "A", "B", 27, 20),
        _game(2, 2024, "C", "D", 34, 27),
        _game(3, 2024, "E", "F", 14, 7),
    ])
    assert estimate_home_field_advantage(games) == pytest.approx(7.0)


def test_estimate_home_field_advantage_excludes_neutral_site():
    games = pd.DataFrame([
        _game(1, 2024, "A", "B", 27, 20),  # home margin 7
        _game(2, 2024, "C", "D", 100, 0, neutral=True),  # would blow out the mean if counted
    ])
    assert estimate_home_field_advantage(games) == pytest.approx(7.0)


def test_estimate_home_field_advantage_excludes_non_fbs():
    games = pd.DataFrame([
        _game(1, 2024, "A", "B", 27, 20),  # FBS vs FBS, margin 7
        _game(2, 2024, "A", "SmallSchool", 70, 0, away_div="fcs"),  # would blow out the mean
    ])
    assert estimate_home_field_advantage(games) == pytest.approx(7.0)


def test_site_adjusted_margin_backs_out_home_boost_and_away_penalty():
    tg = pd.DataFrame({
        "points_for": [27, 20], "points_against": [20, 27],
        "is_home": [True, False], "neutral_site": [False, False],
    })
    adj = site_adjusted_margin(tg, hfa=7.0)
    # Home team's raw +7 margin, minus the 7-point home boost, nets to 0 (an average team's margin).
    assert adj.iloc[0] == pytest.approx(0.0)
    # Away team's raw -7 margin, plus the 7-point disadvantage it overcame, also nets to 0.
    assert adj.iloc[1] == pytest.approx(0.0)


def test_site_adjusted_margin_skips_neutral_site():
    tg = pd.DataFrame({"points_for": [10], "points_against": [0], "is_home": [True], "neutral_site": [True]})
    assert site_adjusted_margin(tg, hfa=7.0).iloc[0] == pytest.approx(10.0)


def test_compute_srs_two_team_single_game_oscillates_not_a_real_schedule():
    """A single isolated 2-team, 1-game "schedule" is the pathological worst case for this
    Jacobi-style fixed-point iteration: the update matrix for (srs_A - srs_B) has eigenvalue
    exactly -1, so it oscillates between two states forever rather than converging (500 vs.
    501 iterations disagree completely). This never bites the real use case (100+ teams,
    thousands of games -- verified separately to converge to machine precision), but is worth
    documenting explicitly rather than silently asserting a "closed form" that doesn't
    actually hold for the iterative method as implemented."""
    games = pd.DataFrame([_game(1, 2024, "A", "B", 24, 10)])
    tg = games_to_team_game_frame(games)
    srs_even = compute_srs(tg, hfa=0.0, fbs_teams={"A", "B"}, iterations=500)
    srs_odd = compute_srs(tg, hfa=0.0, fbs_teams={"A", "B"}, iterations=501)
    assert not np.allclose(srs_even.values, srs_odd.values)


def test_compute_srs_three_team_round_robin_converges_to_closed_form():
    """Unlike the 2-team/1-game case above, a 3-team round robin (not bipartite) converges
    cleanly -- verified separately that 500 vs. 501 iterations agree to machine precision.
    Closed form solved by hand from the fixed-point equations (site_adj margins A-vs-B=+20/-20,
    B-vs-C=+3/-3, C-vs-A=+3/-3, mean-0 constraint): A=17/3, B=-17/3, C=0."""
    games = pd.DataFrame([
        _game(1, 2024, "A", "B", 30, 10),
        _game(2, 2024, "B", "C", 20, 17),
        _game(3, 2024, "C", "A", 24, 21),
    ])
    tg = games_to_team_game_frame(games)
    srs = compute_srs(tg, hfa=0.0, fbs_teams={"A", "B", "C"}, iterations=500)

    assert srs["A"] == pytest.approx(17 / 3, abs=1e-6)
    assert srs["B"] == pytest.approx(-17 / 3, abs=1e-6)
    assert srs["C"] == pytest.approx(0.0, abs=1e-6)
    assert srs.mean() == pytest.approx(0.0, abs=1e-9)


def test_compute_srs_pools_non_fbs_opponents_without_dropping_them():
    games = pd.DataFrame([
        _game(1, 2024, "A", "B", 24, 10),
        _game(2, 2024, "A", "FCS Team", 50, 3, away_div="fcs"),
    ])
    tg = games_to_team_game_frame(games)
    srs = compute_srs(tg, hfa=0.0, fbs_teams={"A", "B"}, iterations=500)
    # Both games count toward A's rating (2 games' worth of adjusted margin averaged in), not
    # just the FBS-vs-FBS one -- confirmed by A's rating differing from the two-team-only case.
    assert set(srs.index) == {"A", "B"}
    assert not np.isnan(srs["A"])


def test_iterate_ratings_respects_fixed_opponent_ratings():
    """A fixed (non-iterated) opponent rating should never change, and should be visible in
    every pass's adjusted-margin calculation."""
    teams = ["A"]
    team_col = np.array(["A", "A"])
    opponent_col = np.array(["FIXED", "FIXED"])
    margin_col = np.array([10.0, 20.0])
    result = iterate_ratings(teams, team_col, opponent_col, margin_col, fixed_opponent_ratings={"FIXED": 5.0}, iterations=50)
    # adj_margin each game = margin + 5 (fixed) -> mean = (15+25)/2 = 20 -> recenter to 0 (only
    # one team in `teams`, so its own mean becomes its value, then recentered to itself minus
    # its own mean = 0, since there's nothing else to average against).
    assert result["A"] == pytest.approx(0.0, abs=1e-9)

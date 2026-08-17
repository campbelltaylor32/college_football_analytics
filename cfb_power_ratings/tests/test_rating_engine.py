import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_power_ratings.rating_engine import fit_residual_std, implied_matchup, update_ratings, win_probability


def _empty_games():
    return pd.DataFrame(columns=["game_id", "season", "week", "home_team", "away_team", "home_points", "away_points", "home_division", "away_division", "neutral_site"])


def _game(game_id, home, away, home_pts, away_pts, neutral=False):
    return {
        "game_id": game_id, "season": 2024, "week": 1, "home_team": home, "away_team": away,
        "home_points": home_pts, "away_points": away_pts,
        "home_division": "fbs", "away_division": "fbs", "neutral_site": neutral,
    }


def test_zero_games_played_returns_prior_recentered_by_its_own_mean():
    """No completed games at all -> every team's rating is its raw prior, shifted by exactly
    the mean of all priors (the mean-0-across-the-field recentering every SRS pass applies --
    see rating_engine.py's module docstring). Not an exact pass-through unless the priors
    already average to 0."""
    priors = pd.Series({"A": 10.0, "B": 0.0, "C": -4.0})
    fbs_teams = {"A", "B", "C"}
    result = update_ratings(priors, _empty_games(), hfa=0.0, fbs_teams=fbs_teams, phantom_games=5)

    prior_mean = priors.mean()
    expected = (priors - prior_mean).sort_values(ascending=False)
    got = result.set_index("team")["rating"].reindex(expected.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-6)
    assert (result["games_played"] == 0).all()
    assert (result["effective_prior_weight"] == 1.0).all()


def test_rating_column_is_always_float_even_with_zero_games():
    """Regression test for a real bug caught during development: an empty games_so_far frame
    has no dtype info of its own, and concatenating it with the phantom-game rows silently
    upcast the rating column to object dtype, which crashed win_probability's norm.cdf call
    downstream."""
    priors = pd.Series({"A": 1.0, "B": -1.0})
    result = update_ratings(priors, _empty_games(), hfa=0.0, fbs_teams={"A", "B"}, phantom_games=5)
    assert result["rating"].dtype == np.float64
    # Also confirm the downstream consumer that originally broke on this now works.
    win_probability(result["rating"].to_numpy(), residual_std=10.0)


def test_effective_prior_weight_matches_phantom_game_formula():
    priors = pd.Series({"A": 0.0, "B": 0.0})
    games = pd.DataFrame([_game(1, "A", "B", 30, 10)])
    result = update_ratings(priors, games, hfa=0.0, fbs_teams={"A", "B"}, phantom_games=5)
    row = result.set_index("team").loc["A"]
    assert row["games_played"] == 1
    assert row["effective_prior_weight"] == pytest.approx(5 / (5 + 1))


def test_more_games_played_fades_prior_weight_monotonically():
    priors = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    one_game = pd.DataFrame([_game(1, "A", "B", 30, 10)])
    three_games = pd.DataFrame([
        _game(1, "A", "B", 30, 10), _game(2, "A", "C", 20, 17), _game(3, "B", "C", 14, 10),
    ])
    r1 = update_ratings(priors, one_game, hfa=0.0, fbs_teams={"A", "B", "C"}, phantom_games=5)
    r3 = update_ratings(priors, three_games, hfa=0.0, fbs_teams={"A", "B", "C"}, phantom_games=5)
    weight_after_1 = r1.set_index("team").loc["A", "effective_prior_weight"]
    weight_after_2 = r3.set_index("team").loc["A", "effective_prior_weight"]
    assert weight_after_2 < weight_after_1


def test_implied_matchup_favors_higher_rated_home_team():
    result = implied_matchup(rating_home=10.0, rating_away=3.0, hfa=2.0)
    assert result["predicted_margin"] == pytest.approx(9.0)
    assert result["favored_team"] == "home"


def test_implied_matchup_favors_away_when_rating_overcomes_hfa():
    result = implied_matchup(rating_home=1.0, rating_away=10.0, hfa=2.0)
    assert result["predicted_margin"] == pytest.approx(-7.0)
    assert result["favored_team"] == "away"


def test_win_probability_is_half_at_zero_margin():
    assert win_probability(0.0, residual_std=14.0) == pytest.approx(0.5)


def test_win_probability_monotonic_in_rating_diff():
    low = win_probability(-10.0, residual_std=14.0)
    mid = win_probability(0.0, residual_std=14.0)
    high = win_probability(10.0, residual_std=14.0)
    assert low < mid < high


def test_fit_residual_std_recovers_known_spread():
    rng = np.random.default_rng(0)
    predicted = rng.normal(0, 5, 1000)
    noise = rng.normal(0, 12, 1000)
    actual = predicted + noise
    assert fit_residual_std(actual, predicted) == pytest.approx(12.0, rel=0.1)


def test_fit_residual_std_never_returns_zero():
    assert fit_residual_std(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 1.0

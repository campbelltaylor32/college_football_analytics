import numpy as np
import pandas as pd
import pytest

from cfb_power_ratings.efficiency import (
    EfficiencyCalibration,
    efficiency_margins,
    opponent_adjusted_off_def,
    team_game_frame,
)


def _eff_rows():
    # game 1: A (off) vs B (def) and B (off) vs A (def); game 2: A vs C
    return pd.DataFrame([
        {"season": 2024, "week": 1, "game_id": 1, "pos_team": "A", "def_pos_team": "B", "plays": 60, "epa": 0.30, "sr": 0.50},
        {"season": 2024, "week": 1, "game_id": 1, "pos_team": "B", "def_pos_team": "A", "plays": 55, "epa": -0.10, "sr": 0.35},
        {"season": 2024, "week": 2, "game_id": 2, "pos_team": "A", "def_pos_team": "C", "plays": 65, "epa": 0.10, "sr": 0.45},
        {"season": 2024, "week": 2, "game_id": 2, "pos_team": "C", "def_pos_team": "A", "plays": 70, "epa": 0.05, "sr": 0.42},
    ])


def test_def_allowed_is_opponents_offense():
    tg = team_game_frame(_eff_rows()).set_index(["game_id", "team"])
    assert tg.loc[(1, "A"), "def_epa_allowed"] == pytest.approx(-0.10)
    assert tg.loc[(1, "B"), "def_epa_allowed"] == pytest.approx(0.30)
    assert tg.loc[(1, "A"), "def_sr_allowed"] == pytest.approx(0.35)
    assert tg.loc[(1, "A"), "def_plays"] == 55


def test_net_efficiency_is_antisymmetric_within_a_game():
    tg = team_game_frame(_eff_rows()).set_index(["game_id", "team"])
    assert tg.loc[(1, "A"), "epa_net"] == pytest.approx(-tg.loc[(1, "B"), "epa_net"])
    assert tg.loc[(2, "A"), "sr_net"] == pytest.approx(-tg.loc[(2, "C"), "sr_net"])


def test_game_missing_one_side_is_dropped():
    rows = _eff_rows()
    rows = rows[~((rows["game_id"] == 2) & (rows["pos_team"] == "C"))]
    tg = team_game_frame(rows)
    assert set(tg["game_id"]) == {1}


def test_efficiency_margins_applies_calibration():
    calib = EfficiencyCalibration(coef_epa=30.0, coef_sr=20.0, r2=0.0, mae=0.0, n_team_games=0)
    tg = team_game_frame(_eff_rows())
    out = efficiency_margins(tg, calib).set_index(["game_id", "team"])["eff_margin"]
    assert out.loc[(1, "A")] == pytest.approx(30.0 * 0.40 + 20.0 * 0.15)
    assert out.loc[(1, "B")] == pytest.approx(-out.loc[(1, "A")])


def test_calibration_least_squares_recovers_known_coefficients(rng):
    """fit_efficiency_to_points is a no-intercept lstsq on [epa_net, sr_net]; check the same
    solve recovers known coefficients from noisy synthetic data."""
    X = np.column_stack([rng.normal(0, 0.35, 5000), rng.normal(0, 0.2, 5000)])
    y = X @ np.array([35.0, 28.0]) + rng.normal(0, 8, 5000)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert coef == pytest.approx([35.0, 28.0], rel=0.05)


def test_opponent_adjusted_off_def_credits_offense_vs_strong_defense():
    """Neutral sites: D's defense allows little to everyone. A has worse raw
    EPA than B only because A faced D and B didn't -- adjusted, their offenses are equal."""
    offense = {"A": 0.1, "B": 0.1, "C": 0.0, "D": 0.0}
    defense_allowed = {"A": 0.0, "B": 0.0, "C": 0.0, "D": -0.3}
    matchups = [(1, "A", "D"), (2, "B", "C"), (3, "C", "D"), (4, "A", "C"), (5, "B", "A")]
    eff, games = [], []
    for gid, h, a in matchups:
        for o, d in [(h, a), (a, h)]:
            eff.append({"season": 2024, "week": 1, "game_id": gid, "pos_team": o, "def_pos_team": d,
                        "plays": 60, "epa": offense[o] + defense_allowed[d], "sr": 0.4})
        games.append({"game_id": gid, "season": 2024, "week": 1, "home_team": h, "away_team": a,
                      "home_points": 0, "away_points": 0, "home_division": "fbs", "away_division": "fbs",
                      "neutral_site": True})
    tg = team_game_frame(pd.DataFrame(eff))
    adj = opponent_adjusted_off_def(tg, pd.DataFrame(games), {"A", "B", "C", "D"}, ridge_alpha=1e-6).set_index("team")
    assert adj.loc["D", "adj_def_epa"] < adj.loc["C", "adj_def_epa"]
    assert adj.loc["A", "adj_off_epa"] == pytest.approx(adj.loc["B", "adj_off_epa"], abs=1e-3)
    raw_a = tg[tg["team"] == "A"]["off_epa"].mean()
    raw_b = tg[tg["team"] == "B"]["off_epa"].mean()
    assert raw_a < raw_b  # raw says B's offense is better; adjustment corrects for A facing D

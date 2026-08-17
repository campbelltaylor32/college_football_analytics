import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_power_ratings.features import pythagorean


def test_pythagorean_win_pct_equal_scoring_is_half():
    result = pythagorean.pythagorean_win_pct(pd.Series([30.0]), pd.Series([30.0]))
    assert result.iloc[0] == pytest.approx(0.5)


def test_pythagorean_win_pct_double_the_scoring_margin():
    # PF = 2x PA -> 2^2 / (2^2 + 1^2) = 4/5 = 0.8
    result = pythagorean.pythagorean_win_pct(pd.Series([20.0]), pd.Series([10.0]))
    assert result.iloc[0] == pytest.approx(0.8)


def _game(game_id, season, home, away, home_pts, away_pts):
    return {
        "game_id": game_id, "season": season, "home_team": home, "away_team": away,
        "home_points": home_pts, "away_points": away_pts,
        "home_division": "fbs", "away_division": "fbs", "neutral_site": False,
    }


def test_build_pythagorean_features_lag_boundary_and_gap_formula(monkeypatch):
    # Team A in 2024: game 1 home vs B, wins 40-10; game 2 away at B, loses 20-30 -> record
    # 1-1 (actual_win_pct=0.5), PF=60 PA=40 (dominant scoring margin despite the split record,
    # so Pythagorean expects a much better record than 0.5 -> negative gap, "underperformed").
    # Team B is the exact mirror: also 1-1, PF=40 PA=60 -> positive gap, "overperformed" its
    # scoring margin by going .500 despite being outscored.
    games_2024 = pd.DataFrame([
        _game(1, 2024, "A", "B", 40, 10),
        _game(2, 2024, "B", "A", 30, 20),
    ])
    # 2025 games present too, to confirm they're never used for the season=2025 feature row
    # (which must reflect ONLY 2024).
    games_2025 = pd.DataFrame([_game(3, 2025, "A", "B", 100, 0)])
    all_games = pd.concat([games_2024, games_2025], ignore_index=True)

    def fake_run_query(sql, params=None, engine=None):
        seasons = params["seasons"]
        return all_games[all_games["season"].isin(seasons)]

    monkeypatch.setattr(pythagorean, "run_query", fake_run_query)

    out = pythagorean.build_pythagorean_features(engine=None, seasons=[2025])

    assert set(out["season"]) == {2025}  # labeled as the feature season, not the source season

    row_a = out[out["team"] == "A"].iloc[0]
    # Team A 2024: PF = 40 + 20 = 60, PA = 10 + 30 = 40, record 1-1 -> actual_win_pct = 0.5
    expected_pyth_a = pythagorean.pythagorean_win_pct(pd.Series([60.0]), pd.Series([40.0])).iloc[0]
    assert row_a["pythagorean_win_pct_lag1"] == pytest.approx(expected_pyth_a)
    assert row_a["win_pct_over_pythagorean_lag1"] == pytest.approx(0.5 - expected_pyth_a)
    assert row_a["win_pct_over_pythagorean_lag1"] < 0  # dominant scoring margin, only .500 record

    row_b = out[out["team"] == "B"].iloc[0]
    # Team B 2024: PF = 10 + 30 = 40, PA = 40 + 20 = 60, record 1-1 -> actual_win_pct = 0.5
    expected_pyth_b = pythagorean.pythagorean_win_pct(pd.Series([40.0]), pd.Series([60.0])).iloc[0]
    assert row_b["pythagorean_win_pct_lag1"] == pytest.approx(expected_pyth_b)
    assert row_b["win_pct_over_pythagorean_lag1"] == pytest.approx(0.5 - expected_pyth_b)
    assert row_b["win_pct_over_pythagorean_lag1"] > 0  # outscored overall, still .500 record

    # The lopsided 2025 game (A 100-0 over B) must have zero influence on the 2025 feature row --
    # if it leaked in, team A's PF would be far higher than the 60 computed above.
    assert row_a["pythagorean_win_pct_lag1"] < 0.99


def test_build_pythagorean_features_empty_games_returns_empty_frame(monkeypatch):
    def fake_run_query(sql, params=None, engine=None):
        return pd.DataFrame(columns=["game_id", "season", "home_team", "away_team", "home_points", "away_points", "home_division", "away_division", "neutral_site"])

    monkeypatch.setattr(pythagorean, "run_query", fake_run_query)
    out = pythagorean.build_pythagorean_features(engine=None, seasons=[2025])
    assert out.empty
    assert list(out.columns) == ["team", "season", "pythagorean_win_pct_lag1", "win_pct_over_pythagorean_lag1"]

import numpy as np
import pandas as pd

from cfb_playcall_predictability.predictability import bootstrap_predictability, team_scores


def _frame(p, y, team="A"):
    return pd.DataFrame({"pos_team": team, "game_id": np.arange(len(y)) // 10, "is_pass": y, "p": p})


def test_constant_own_rate_scores_zero():
    y = np.array([1] * 40 + [0] * 60)
    s = team_scores(_frame(np.full(100, 0.4), y), "p")
    assert abs(s["predictability"].iloc[0]) < 1e-9


def test_near_perfect_predictions_score_near_one():
    y = np.array([1, 0] * 50)
    s = team_scores(_frame(np.where(y == 1, 0.999, 0.001), y), "p")
    assert s["predictability"].iloc[0] > 0.99
    assert s["auc"].iloc[0] == 1.0


def test_wrong_confident_predictions_score_negative():
    y = np.array([1, 0] * 50)
    s = team_scores(_frame(np.where(y == 1, 0.2, 0.8), y), "p")
    assert s["predictability"].iloc[0] < 0


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.15, 300), 0.01, 0.99)
    df = _frame(p, y)
    point = team_scores(df, "p")["predictability"].iloc[0]
    ci = bootstrap_predictability(df, "p", reps=500, level=0.9, seed=0).iloc[0]
    assert ci["predictability_lo"] < point < ci["predictability_hi"]

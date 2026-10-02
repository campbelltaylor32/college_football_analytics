"""A feature row may only depend on information available before that snap."""

import numpy as np
import pandas as pd

from cfb_playcall_predictability.dataset import feature_columns
from cfb_playcall_predictability.features.in_game import in_game_features
from cfb_playcall_predictability.features.rates import prior_and_season_to_date

OUTCOME_COLUMNS = ["is_pass", "success", "EPA", "yards_gained"]


def test_in_game_features_ignore_current_and_future_plays(calls):
    base = in_game_features(calls)
    k = 25
    perturbed = calls.copy()
    perturbed.loc[k:, "is_pass"] = 1 - perturbed.loc[k:, "is_pass"]
    perturbed.loc[k:, "success"] = 1 - perturbed.loc[k:, "success"]
    perturbed.loc[k:, "EPA"] += 3.0
    perturbed.loc[k:, "yards_gained"] += 10
    after = in_game_features(perturbed)
    # Rows up to and including k only see plays before them, all unchanged.
    pd.testing.assert_frame_equal(base.loc[:k], after.loc[:k])
    # ...and later rows of the same team do move, so the test has teeth.
    same_team_later = (calls.index > k) & (calls["pos_team"] == calls.loc[k, "pos_team"]) & (calls["game_id"] == calls.loc[k, "game_id"])
    assert not base.loc[same_team_later].equals(after.loc[same_team_later])


def test_in_game_first_call_has_no_history(calls):
    feats = in_game_features(calls)
    first = calls.groupby(["game_id", "pos_team"]).head(1).index
    assert (feats.loc[first, "g_calls_so_far"] == 0).all()
    assert (feats.loc[first, "prev_call_pass"] == -1).all()
    assert (feats.loc[first, "g_passes_so_far"] == 0).all()


def _team_games():
    rows = []
    for season in (2023, 2024):
        for gnum in range(1, 6):
            for team in ("A", "B"):
                n = 60
                rows.append({"game_id": season * 100 + gnum, "team": team, "season": season,
                             "game_number": gnum, "n_x": n, "sum_x": n * (0.4 if team == "A" else 0.6)})
    return pd.DataFrame(rows)


def test_season_to_date_uses_only_earlier_games():
    tg = _team_games()
    base = prior_and_season_to_date(tg, ["x"], k_prior=40, k_std=60)
    g = 3
    perturbed = tg.copy()
    hit = (perturbed["season"] == 2024) & (perturbed["game_number"] >= g)
    perturbed.loc[hit, "sum_x"] = perturbed.loc[hit, "n_x"]   # pass on every call from game g on
    after = prior_and_season_to_date(perturbed, ["x"], k_prior=40, k_std=60)
    key = ["game_id", "team"]
    b, a = base.set_index(key).sort_index(), after.set_index(key).sort_index()
    upto = b.index.get_level_values("game_id") <= 2024 * 100 + g
    pd.testing.assert_frame_equal(b[upto], a[upto])
    assert not np.allclose(b.loc[~upto, "std_x"], a.loc[~upto, "std_x"])


def test_prior_season_features_ignore_current_season():
    tg = _team_games()
    base = prior_and_season_to_date(tg, ["x"], k_prior=40, k_std=60)
    perturbed = tg.copy()
    perturbed.loc[perturbed["season"] == 2024, "sum_x"] = 0
    after = prior_and_season_to_date(perturbed, ["x"], k_prior=40, k_std=60)
    in_2024 = base["season"] == 2024
    np.testing.assert_allclose(base.loc[in_2024, "prior_x"], after.loc[in_2024, "prior_x"])
    # Team A passed 40% on 300 calls in 2023; shrunk with k=40 toward the 50% league rate.
    a_prior = base.loc[in_2024 & (base["team"] == "A"), "prior_x"].iloc[0]
    assert np.isclose(a_prior, (0.4 * 300 + 40 * 0.5) / (300 + 40))


def test_outcome_columns_are_never_features():
    df = pd.DataFrame(columns=OUTCOME_COLUMNS + ["play_type", "play_text", "down", "std_pr_overall", "g_pass_rate"])
    feats = feature_columns(df)
    assert not set(feats) & set(OUTCOME_COLUMNS + ["play_type", "play_text"])
    assert {"down", "std_pr_overall", "g_pass_rate"} <= set(feats)

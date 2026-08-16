import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.targets import add_push_and_targets, drop_pushes


def test_exact_push_is_flagged_and_excluded():
    df = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "home_minus_away": [7, -7, 3, 0],
            "signed_spread": [-7.0, 7.0, -3.0, 3.0],
            # game 1: home won by 7, favored by 7 -> exact push (7 + -7 = 0)
            # game 2: home lost by 7, home was underdog +7 -> exact push (-7 + 7 = 0)
            # game 3: home won by 3, favored by 3 -> exact push (3 + -3 = 0)
            # game 4: home tied 0, underdog +3 -> home covers (0 + 3 = 3 > 0)
        }
    )
    out = add_push_and_targets(df)
    assert out["is_push"].tolist() == [True, True, True, False]
    assert out.loc[3, "home_covered"] == 1

    filtered = drop_pushes(out)
    assert len(filtered) == 1
    assert filtered.iloc[0]["game_id"] == 4


def test_cover_margin_sign_matches_home_covered(rng):
    n = 200
    home_minus_away = rng.integers(-40, 40, n)
    signed_spread = rng.integers(-20, 20, n).astype(float)
    df = pd.DataFrame(
        {
            "game_id": np.arange(n),
            "home_minus_away": home_minus_away,
            "signed_spread": signed_spread,
        }
    )
    out = add_push_and_targets(df)
    non_push = out.loc[~out["is_push"]]
    expected_covered = (non_push["cover_margin"] > 0).to_numpy()
    actual_covered = (non_push["home_covered"] == 1).to_numpy()
    assert (expected_covered == actual_covered).all()
    assert ((non_push["cover_margin"] > 0) | (non_push["cover_margin"] < 0)).all()


def test_no_pushes_survive_drop_pushes(rng):
    n = 500
    home_minus_away = rng.integers(-40, 40, n)
    signed_spread = rng.integers(-20, 20, n).astype(float)
    df = pd.DataFrame(
        {
            "game_id": np.arange(n),
            "home_minus_away": home_minus_away,
            "signed_spread": signed_spread,
        }
    )
    filtered = drop_pushes(add_push_and_targets(df))
    assert (filtered["cover_margin"] != 0).all()
    assert filtered["home_covered"].isin([0, 1]).all()
    assert filtered["home_covered"].notna().all()

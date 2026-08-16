import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_results_csv(tmp_path, rng):
    """A small, hand-checkable results CSV: 40 games across 4 seasons, with a few games
    engineered to be exact pushes so push-handling tests have known ground truth."""
    n = 40
    seasons = np.repeat([2020, 2021, 2022, 2023], 10)
    game_id = np.arange(1000, 1000 + n)
    home_points = rng.integers(10, 45, n)
    away_points = rng.integers(10, 45, n)
    home_minus_away = home_points - away_points
    signed_spread = rng.integers(-14, 14, n).astype(float)

    # Force games 0-2 to be exact pushes: home_minus_away == -signed_spread
    for i in range(3):
        signed_spread[i] = -float(home_minus_away[i])

    df = pd.DataFrame(
        {
            "game_id": game_id,
            "home_points": home_points,
            "away_points": away_points,
            "home_minus_away": home_minus_away,
            "spread": signed_spread,
        }
    )
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path, df, seasons


@pytest.fixture
def synthetic_modeling_frame(rng):
    """A minimal frame shaped like the real modeling frame (season/week/home_covered/
    feature columns) for split and leakage tests - no CSV I/O involved."""
    seasons = np.repeat([2015, 2016, 2017, 2018, 2019], 20)
    n = len(seasons)
    df = pd.DataFrame(
        {
            "game_id": np.arange(n),
            "season": seasons,
            "week": rng.integers(4, 12, n),
            "home_covered": rng.integers(0, 2, n),
            "cover_margin": rng.normal(0, 10, n),
            "home_favored": rng.integers(0, 2, n),
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
        }
    ).reset_index(drop=True)
    return df

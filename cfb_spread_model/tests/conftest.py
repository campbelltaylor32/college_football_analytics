"""Shared pytest fixtures. Config-based fixtures are cheap; `real_dataset` loads the actual
../Data/CFB_Gambling_Predictors_Final_PBP.csv (real data, not mocked -- consistent with this
project's scope of reading the existing CSV pipeline directly) and is session-scoped so it's
loaded once. `synthetic_df` is a small hand-built frame for tests of pure column-parsing/
pruning logic that don't need the real 1054-column CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_spread_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_spread_model.data import load_raw_csv


@pytest.fixture(scope="session")
def data_cfg():
    return load_data_config()


@pytest.fixture(scope="session")
def features_cfg():
    return load_features_config()


@pytest.fixture(scope="session")
def modeling_cfg():
    return load_modeling_config()


@pytest.fixture(scope="session")
def real_dataset(data_cfg):
    return load_raw_csv(data_cfg)


@pytest.fixture
def synthetic_df():
    """Exercises the home_/away_ prefix + prev_week/avg_all/avg3 naming convention at small
    scale: one near-perfectly-correlated temporal triplet (should collapse), one independent
    triplet (should not), an offense/defense-mirror-shaped pair, and non-temporal columns."""
    rng = np.random.RandomState(0)
    n = 200
    base = rng.normal(size=n)
    label = (base + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "game_id": range(n),
            "home_team": ["Team A"] * n,
            "away_team": ["Team B"] * n,
            "season": rng.choice([2021, 2022, 2023], size=n),
            "week": rng.randint(4, 12, size=n),
            "neutral_site": 0,
            "conference_game": 0,
            "spread": rng.uniform(0, 20, size=n),
            "home_favored": rng.randint(0, 2, size=n),
            "home_covered": label,
            "home_prev_week_total_yards": base + rng.normal(scale=0.01, size=n),
            "home_total_yards_avg_all": base + rng.normal(scale=0.01, size=n),
            "home_total_yards_avg3": base + rng.normal(scale=0.01, size=n),
            "home_prev_week_sacks": rng.normal(size=n),
            "home_sacks_avg_all": rng.normal(size=n),
            "home_sacks_avg3": rng.normal(size=n),
            "home_prev_week_total_yards_allowed": rng.normal(size=n),
            "home_talent": rng.normal(size=n),
            "away_prev_week_total_yards": rng.normal(size=n),
            "away_total_yards_avg_all": rng.normal(size=n),
            "away_total_yards_avg3": rng.normal(size=n),
            "away_talent": rng.normal(size=n),
        }
    )

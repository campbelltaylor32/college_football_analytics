"""Shared fixtures. Unlike the sibling projects' DB integration tests, these run on small
synthetic play-by-play frames, so the leakage tests can perturb individual plays/games and
check exactly which rows move."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_playcall_predictability.config import load_data_config, load_features_config, load_modeling_config


@pytest.fixture(scope="session")
def data_cfg():
    return load_data_config()


@pytest.fixture(scope="session")
def features_cfg():
    return load_features_config()


@pytest.fixture(scope="session")
def modeling_cfg():
    return load_modeling_config()


def make_calls(n_games: int = 2, plays_per_team: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic run/pass calls already in snap order, two teams alternating drives."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for i in range(2 * plays_per_team):
            drive = i // 5
            offense, defense = ("A", "B") if drive % 2 == 0 else ("B", "A")
            rows.append({
                "game_id": 1000 + g, "season": 2024, "week": g + 1, "pos_team": offense,
                "def_pos_team": defense, "home": "A", "drive_number": drive,
                "down": int(rng.integers(1, 5)), "distance": float(rng.integers(1, 15)),
                "yards_to_goal": float(rng.integers(1, 99)), "period": 1 + i * 4 // (2 * plays_per_team),
                "pos_score_diff_start": float(rng.integers(-14, 15)),
                "is_pass": int(rng.integers(0, 2)), "success": int(rng.integers(0, 2)),
                "EPA": float(rng.normal()), "yards_gained": float(rng.integers(-5, 20)),
                "garbage_time": False,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def calls() -> pd.DataFrame:
    return make_calls()

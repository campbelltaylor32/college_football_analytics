"""Shared pytest fixtures. These are integration tests against the live local MySQL
database (cfb_football, already populated) -- not mocked -- consistent with the project's
"already built, already populated" ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_win_total_model.config import load_features_config, load_modeling_config
from cfb_win_total_model.database import get_engine
from cfb_win_total_model.dataset import build_modeling_dataset


@pytest.fixture(scope="session")
def engine():
    return get_engine()


@pytest.fixture(scope="session")
def features_cfg():
    return load_features_config()


@pytest.fixture(scope="session")
def modeling_cfg():
    return load_modeling_config()


@pytest.fixture(scope="session")
def small_target_seasons() -> list[int]:
    return [2022, 2023]


@pytest.fixture(scope="session")
def full_modeling_df(engine, features_cfg):
    """Built once per test session and reused across test files to avoid rebuilding the
    (fairly expensive, many-query) modeling dataset repeatedly."""
    return build_modeling_dataset(engine, target_seasons=[2019, 2022, 2025], features_cfg=features_cfg)

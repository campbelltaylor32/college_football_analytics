"""Shared pytest fixtures. These are integration tests against the live local MySQL database
(cfb_football, already populated) -- not mocked -- consistent with the sibling projects'
"already built, already populated" ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_rb_rushing_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_rb_rushing_model.database import get_engine
from cfb_rb_rushing_model.dataset import build_modeling_dataset
from cfb_rb_rushing_model.schedule_spine import attach_rest_days, build_schedule_spine


@pytest.fixture(scope="session")
def engine():
    return get_engine()


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
def small_seasons() -> list[int]:
    return [2023]


@pytest.fixture(scope="session")
def small_spine(engine, features_cfg, small_seasons):
    spine = build_schedule_spine(engine, small_seasons)
    return attach_rest_days(spine, features_cfg.default_rest_days_season_opener)


@pytest.fixture(scope="session")
def small_modeling_df(engine, data_cfg, features_cfg, small_seasons):
    """Built once per test session and reused across test files to avoid rebuilding the
    (fairly expensive, many-query) modeling dataset repeatedly."""
    return build_modeling_dataset(engine, target_seasons=small_seasons, data_cfg=data_cfg, features_cfg=features_cfg)

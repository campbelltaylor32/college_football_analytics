"""YAML config loaders. Every ${VAR}-style token in a config file is resolved against the
process environment (after loading .env via python-dotenv) -- database credentials are never
hardcoded in a tracked file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from cfb_power_ratings.utils.paths import CONFIG_DIR, PROJECT_ROOT

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_dotenv_loaded = False


def _load_env_once() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(PROJECT_ROOT.parent / ".env")  # repo-root .env holds CFBD_API_KEY
        _dotenv_loaded = True


def _resolve_env_tokens(value):
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value.strip())
        if match:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise ValueError(
                    f"Config references ${{{var_name}}} but it is not set in the environment. "
                    f"Copy .env.example to .env and fill in real values."
                )
            return os.environ[var_name]
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_tokens(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_tokens(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    _load_env_once()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _resolve_env_tokens(raw)


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def sqlalchemy_url(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class FeaturesConfig:
    feature_groups: dict[str, bool]
    min_matched_recruits_for_blue_chip_ratio: int
    transfer_portal_start_season: int
    program_history_trailing_seasons: int
    min_valid_class_rows: int
    tenure_lookback_seasons: int


@dataclass
class SRSConfig:
    iterations: int
    non_fbs_pool_name: str
    hfa_override: float | None


@dataclass
class RatingEngineConfig:
    default_phantom_games: int
    phantom_games_sweep: list[int]


@dataclass
class ModelingConfig:
    srs_history_start_season: int
    full_feature_start_season: int
    excluded_seasons: list[int]
    min_train_seasons: int
    first_validation_season: int
    walk_forward_validation_seasons: list[int]
    final_holdout_season: int
    random_seed: int
    candidate_models: list[str]
    srs: SRSConfig
    rating_engine: RatingEngineConfig


def load_database_config(path: Path = CONFIG_DIR / "database.yaml") -> DatabaseConfig:
    raw = _load_yaml(path)
    return DatabaseConfig(
        host=raw["host"], port=int(raw["port"]), user=raw["user"],
        password=raw["password"], database=raw["database"],
    )


def load_features_config(path: Path = CONFIG_DIR / "features.yaml") -> FeaturesConfig:
    raw = _load_yaml(path)
    return FeaturesConfig(
        feature_groups=dict(raw["feature_groups"]),
        min_matched_recruits_for_blue_chip_ratio=raw["min_matched_recruits_for_blue_chip_ratio"],
        transfer_portal_start_season=raw["transfer_portal_start_season"],
        program_history_trailing_seasons=raw["program_history_trailing_seasons"],
        min_valid_class_rows=raw["min_valid_class_rows"],
        tenure_lookback_seasons=raw["tenure_lookback_seasons"],
    )


def load_modeling_config(path: Path = CONFIG_DIR / "modeling.yaml") -> ModelingConfig:
    raw = _load_yaml(path)
    srs_raw = raw["srs"]
    engine_raw = raw["rating_engine"]
    return ModelingConfig(
        srs_history_start_season=raw["srs_history_start_season"],
        full_feature_start_season=raw["full_feature_start_season"],
        excluded_seasons=list(raw["excluded_seasons"]),
        min_train_seasons=raw["min_train_seasons"],
        first_validation_season=raw["first_validation_season"],
        walk_forward_validation_seasons=list(raw["walk_forward_validation_seasons"]),
        final_holdout_season=raw["final_holdout_season"],
        random_seed=raw["random_seed"],
        candidate_models=list(raw["candidate_models"]),
        srs=SRSConfig(
            iterations=srs_raw["iterations"],
            non_fbs_pool_name=srs_raw["non_fbs_pool_name"],
            hfa_override=srs_raw.get("hfa_override"),
        ),
        rating_engine=RatingEngineConfig(
            default_phantom_games=engine_raw["default_phantom_games"],
            phantom_games_sweep=list(engine_raw["phantom_games_sweep"]),
        ),
    )

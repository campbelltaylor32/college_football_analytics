"""YAML config loaders. Every ${VAR}-style token in a config file is resolved against the
process environment (after loading .env via python-dotenv) -- database credentials are never
hardcoded in a tracked file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from cfb_rb_rushing_model.utils.paths import CONFIG_DIR, PROJECT_ROOT

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_dotenv_loaded = False


def _load_env_once() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(PROJECT_ROOT / ".env")
        _dotenv_loaded = True


def _resolve_env_tokens(value):
    """Recursively resolve ${VAR} tokens in strings/dicts/lists loaded from YAML."""
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
class DataConfig:
    positions: list[str]
    rush_play_types: list[str]
    name_suffixes_to_strip: list[str]
    include_betting_context: bool
    expected_min_rush_plays_with_name: int


@dataclass
class EligibilityConfig:
    min_trailing3_avg_carries: float
    min_season_to_date_carries: float
    min_games_played_for_avg3: int


@dataclass
class FeaturesConfig:
    player_rolling_windows: list[int]
    team_rolling_windows: list[int]
    defense_rolling_windows: list[int]
    explosive_run_yard_threshold: int
    eligibility: EligibilityConfig
    default_rest_days_season_opener: int


@dataclass
class ModelingConfig:
    target_season: int
    full_feature_start_season: int
    excluded_seasons: list[int]
    min_train_seasons: int
    first_validation_season: int
    walk_forward_validation_seasons: list[int]
    final_holdout_season: int
    final_holdout_max_week: int | None
    random_seed: int
    clip_min_yards: int
    prediction_interval_method: str
    prediction_interval_levels: tuple[float, float]
    baseline_models: list[str]
    candidate_models: list[str]
    hyperparam_grids: dict = field(default_factory=dict)


def load_database_config(path: Path = CONFIG_DIR / "database.yaml") -> DatabaseConfig:
    raw = _load_yaml(path)
    return DatabaseConfig(
        host=raw["host"],
        port=int(raw["port"]),
        user=raw["user"],
        password=raw["password"],
        database=raw["database"],
    )


def load_data_config(path: Path = CONFIG_DIR / "data.yaml") -> DataConfig:
    raw = _load_yaml(path)
    return DataConfig(
        positions=list(raw["positions"]),
        rush_play_types=list(raw["rush_play_types"]),
        name_suffixes_to_strip=list(raw["name_suffixes_to_strip"]),
        include_betting_context=bool(raw["include_betting_context"]),
        expected_min_rush_plays_with_name=int(raw["expected_min_rush_plays_with_name"]),
    )


def load_features_config(path: Path = CONFIG_DIR / "features.yaml") -> FeaturesConfig:
    raw = _load_yaml(path)
    elig = raw["eligibility"]
    return FeaturesConfig(
        player_rolling_windows=list(raw["player_rolling_windows"]),
        team_rolling_windows=list(raw["team_rolling_windows"]),
        defense_rolling_windows=list(raw["defense_rolling_windows"]),
        explosive_run_yard_threshold=int(raw["explosive_run_yard_threshold"]),
        eligibility=EligibilityConfig(
            min_trailing3_avg_carries=float(elig["min_trailing3_avg_carries"]),
            min_season_to_date_carries=float(elig["min_season_to_date_carries"]),
            min_games_played_for_avg3=int(elig["min_games_played_for_avg3"]),
        ),
        default_rest_days_season_opener=int(raw["default_rest_days_season_opener"]),
    )


def load_modeling_config(path: Path = CONFIG_DIR / "modeling.yaml") -> ModelingConfig:
    raw = _load_yaml(path)
    clip = raw["clip_predictions"]
    models = raw["models"]
    return ModelingConfig(
        target_season=raw["target_season"],
        full_feature_start_season=raw["full_feature_start_season"],
        excluded_seasons=list(raw["excluded_seasons"]),
        min_train_seasons=raw["min_train_seasons"],
        first_validation_season=raw["first_validation_season"],
        walk_forward_validation_seasons=list(raw["walk_forward_validation_seasons"]),
        final_holdout_season=raw["final_holdout_season"],
        final_holdout_max_week=raw.get("final_holdout_max_week"),
        random_seed=raw["random_seed"],
        clip_min_yards=clip["min_yards"],
        prediction_interval_method=raw["prediction_interval_method"],
        prediction_interval_levels=tuple(raw["prediction_interval_levels"]),
        baseline_models=list(models["baselines"]),
        candidate_models=list(models["candidates"]),
        hyperparam_grids=raw.get("hyperparam_grids", {}),
    )

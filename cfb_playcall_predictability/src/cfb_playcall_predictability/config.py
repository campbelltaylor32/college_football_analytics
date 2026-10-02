"""YAML config loaders. Every ${VAR}-style token in a config file is resolved against the
process environment (after loading .env via python-dotenv) -- database credentials are never
hardcoded in a tracked file. Ported from cfb_rb_rushing_model/config.py."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from cfb_playcall_predictability.utils.paths import CONFIG_DIR, PROJECT_ROOT

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_dotenv_loaded = False


def _load_env_once() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(PROJECT_ROOT.parent / ".env")  # repo-root .env, for anything not set above
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
        return _resolve_env_tokens(yaml.safe_load(f))


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
    first_season: int
    current_season: int
    pass_play_types: list[str]
    run_play_types: list[str]
    flag_labeled_play_types: list[str]
    exclude_text_pattern: str
    garbage_time_margin: dict[int, int]
    max_period: int
    line_providers: list[str]


@dataclass
class FeaturesConfig:
    shrinkage: dict[str, float]
    tendency_buckets: list[str]
    long_distance: int
    short_distance: int
    neutral_margin: int
    lead_margin: int


@dataclass
class ModelingConfig:
    full_feature_start_season: int
    excluded_seasons: list[int]
    min_train_seasons: int
    walk_forward_validation_seasons: list[int]
    final_holdout_season: int
    target_season: int
    random_seed: int
    min_team_plays: int
    bootstrap_reps: int
    ci_level: float
    baseline_models: list[str]
    candidate_models: list[str]
    hyperparams: dict


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
        first_season=int(raw["first_season"]),
        current_season=int(raw["current_season"]),
        pass_play_types=list(raw["pass_play_types"]),
        run_play_types=list(raw["run_play_types"]),
        flag_labeled_play_types=list(raw["flag_labeled_play_types"]),
        exclude_text_pattern=raw["exclude_text_pattern"],
        garbage_time_margin={int(k): int(v) for k, v in raw["garbage_time_margin"].items()},
        max_period=int(raw["max_period"]),
        line_providers=list(raw["line_providers"]),
    )


def load_features_config(path: Path = CONFIG_DIR / "features.yaml") -> FeaturesConfig:
    raw = _load_yaml(path)
    return FeaturesConfig(
        shrinkage={k: float(v) for k, v in raw["shrinkage"].items()},
        tendency_buckets=list(raw["tendency_buckets"]),
        long_distance=int(raw["long_distance"]),
        short_distance=int(raw["short_distance"]),
        neutral_margin=int(raw["neutral_margin"]),
        lead_margin=int(raw["lead_margin"]),
    )


def load_modeling_config(path: Path = CONFIG_DIR / "modeling.yaml") -> ModelingConfig:
    raw = _load_yaml(path)
    return ModelingConfig(
        full_feature_start_season=int(raw["full_feature_start_season"]),
        excluded_seasons=list(raw["excluded_seasons"]),
        min_train_seasons=int(raw["min_train_seasons"]),
        walk_forward_validation_seasons=list(raw["walk_forward_validation_seasons"]),
        final_holdout_season=int(raw["final_holdout_season"]),
        target_season=int(raw["target_season"]),
        random_seed=int(raw["random_seed"]),
        min_team_plays=int(raw["min_team_plays"]),
        bootstrap_reps=int(raw["bootstrap_reps"]),
        ci_level=float(raw["ci_level"]),
        baseline_models=list(raw["models"]["baselines"]),
        candidate_models=list(raw["models"]["candidates"]),
        hyperparams=raw.get("hyperparams", {}),
    )

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

from cfb_win_total_model.utils.paths import CONFIG_DIR, PROJECT_ROOT

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
class ModelingConfig:
    target_season: int
    full_feature_start_season: int
    excluded_seasons: list[int]
    min_train_seasons: int
    first_validation_season: int
    walk_forward_validation_seasons: list[int]
    final_holdout_season: int
    random_seed: int
    clip_min_wins: int
    prediction_interval_method: str
    prediction_interval_levels: tuple[float, float]
    baseline_models: list[str]
    candidate_models: list[str]
    hyperparam_grids: dict = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    feature_groups: dict[str, bool]
    power_conferences_default: list[str]
    power_conferences_by_season: dict[int, list[str]]
    independent_power_overrides: list[str]
    use_coach_preseason_rank: bool
    rolling_windows: list[int]
    winsorize_percent_ppa_limits: tuple[float, float]
    explosiveness_epa_threshold: float
    early_season_max_week: int
    late_season_min_week: int
    positional_talent_groups: dict[str, list[str]]

    def power_conferences_for_season(self, season: int) -> set[str]:
        confs = self.power_conferences_by_season.get(season, self.power_conferences_default)
        return set(confs)

    def is_power_conference_opponent(self, conference: str | None, season: int, school: str | None = None) -> bool:
        if school in self.independent_power_overrides:
            return True
        if conference is None:
            return False
        return conference in self.power_conferences_for_season(season)


def load_database_config(path: Path = CONFIG_DIR / "database.yaml") -> DatabaseConfig:
    raw = _load_yaml(path)
    return DatabaseConfig(
        host=raw["host"],
        port=int(raw["port"]),
        user=raw["user"],
        password=raw["password"],
        database=raw["database"],
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
        random_seed=raw["random_seed"],
        clip_min_wins=clip["min_wins"],
        prediction_interval_method=raw["prediction_interval_method"],
        prediction_interval_levels=tuple(raw["prediction_interval_levels"]),
        baseline_models=list(models["baselines"]),
        candidate_models=list(models["candidates"]),
        hyperparam_grids=raw.get("hyperparam_grids", {}),
    )


def load_features_config(path: Path = CONFIG_DIR / "features.yaml") -> FeaturesConfig:
    raw = _load_yaml(path)
    power = raw["power_conferences"]
    return FeaturesConfig(
        feature_groups=dict(raw["feature_groups"]),
        power_conferences_default=list(power["default"]),
        power_conferences_by_season={int(k): list(v) for k, v in power["by_season"].items()},
        independent_power_overrides=list(power["independent_overrides"]["power"]),
        use_coach_preseason_rank=raw["use_coach_preseason_rank"],
        rolling_windows=list(raw["rolling_windows"]),
        winsorize_percent_ppa_limits=tuple(raw["winsorize_limits"]["percent_ppa_columns"]),
        explosiveness_epa_threshold=raw["explosiveness_epa_threshold"],
        early_season_max_week=raw["early_season_max_week"],
        late_season_min_week=raw["late_season_min_week"],
        positional_talent_groups={k: list(v) for k, v in raw["positional_talent_groups"].items()},
    )

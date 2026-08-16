"""YAML config loaders. Unlike the sibling cfb_win_total_model project, no database
credentials are involved -- this project only reads a CSV -- so there is no ${VAR}-style env
token resolution here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cfb_spread_model.utils.paths import CONFIG_DIR, PROJECT_ROOT


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class DataConfig:
    source_csv: Path
    id_columns: list[str]
    label_column: str
    split_only_columns: list[str]
    include_split_columns_as_features: bool
    retained_context_columns: list[str]
    excluded_column_patterns: list[str]
    feature_representation: str
    expected_row_count_min: int
    expected_column_count: int
    week_min: int
    week_max: int


@dataclass
class CorrelationPruningConfig:
    temporal_collapse_corr_threshold: float
    general_corr_threshold: float
    univariate_association_metric: str


@dataclass
class FeaturesConfig:
    correlation_pruning: CorrelationPruningConfig
    candidate_feature_counts: list[int]
    permutation_importance_n_repeats: int
    selection_methods: list[str]


@dataclass
class PrecisionObjectiveConfig:
    min_coverage_floor: float
    candidate_thresholds: list[float]


@dataclass
class ModelingConfig:
    full_feature_start_season: int
    excluded_seasons: list[int]
    min_train_seasons: int
    walk_forward_validation_seasons: list[int]
    final_holdout_season: int
    random_seed: int
    precision_objective: PrecisionObjectiveConfig
    baseline_models: list[str]
    candidate_models: list[str]
    hyperparam_grids: dict = field(default_factory=dict)


_VALID_FEATURE_REPRESENTATIONS = {"raw_dual", "differential"}


def load_data_config(path: Path = CONFIG_DIR / "data.yaml") -> DataConfig:
    raw = _load_yaml(path)
    week_range = raw["week_range"]
    source_csv = (PROJECT_ROOT / raw["source_csv"]).resolve()
    feature_representation = raw.get("feature_representation", "raw_dual")
    if feature_representation not in _VALID_FEATURE_REPRESENTATIONS:
        raise ValueError(
            f"config/data.yaml feature_representation={feature_representation!r} not in {_VALID_FEATURE_REPRESENTATIONS}"
        )
    return DataConfig(
        source_csv=source_csv,
        id_columns=list(raw["id_columns"]),
        label_column=raw["label_column"],
        split_only_columns=list(raw["split_only_columns"]),
        include_split_columns_as_features=bool(raw["include_split_columns_as_features"]),
        retained_context_columns=list(raw["retained_context_columns"]),
        excluded_column_patterns=list(raw.get("excluded_column_patterns", [])),
        feature_representation=feature_representation,
        expected_row_count_min=int(raw["expected_row_count_min"]),
        expected_column_count=int(raw["expected_column_count"]),
        week_min=int(week_range["min"]),
        week_max=int(week_range["max"]),
    )


def load_features_config(path: Path = CONFIG_DIR / "features.yaml") -> FeaturesConfig:
    raw = _load_yaml(path)
    corr = raw["correlation_pruning"]
    sel = raw["feature_selection"]
    return FeaturesConfig(
        correlation_pruning=CorrelationPruningConfig(
            temporal_collapse_corr_threshold=float(corr["temporal_collapse_corr_threshold"]),
            general_corr_threshold=float(corr["general_corr_threshold"]),
            univariate_association_metric=corr["univariate_association_metric"],
        ),
        candidate_feature_counts=list(sel["candidate_feature_counts"]),
        permutation_importance_n_repeats=int(sel["permutation_importance_n_repeats"]),
        selection_methods=list(sel["selection_methods"]),
    )


def load_modeling_config(path: Path = CONFIG_DIR / "modeling.yaml") -> ModelingConfig:
    raw = _load_yaml(path)
    precision = raw["precision_objective"]
    models = raw["models"]
    return ModelingConfig(
        full_feature_start_season=raw["full_feature_start_season"],
        excluded_seasons=list(raw["excluded_seasons"]),
        min_train_seasons=raw["min_train_seasons"],
        walk_forward_validation_seasons=list(raw["walk_forward_validation_seasons"]),
        final_holdout_season=raw["final_holdout_season"],
        random_seed=raw["random_seed"],
        precision_objective=PrecisionObjectiveConfig(
            min_coverage_floor=float(precision["min_coverage_floor"]),
            candidate_thresholds=list(precision["candidate_thresholds"]),
        ),
        baseline_models=list(models["baselines"]),
        candidate_models=list(models["candidates"]),
        hyperparam_grids=raw.get("hyperparam_grids", {}),
    )

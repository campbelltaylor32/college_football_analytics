"""Shared production-artifact lookup, used by scripts/generate_week_predictions.py and
scripts/explain_model.py -- both need "whichever model/feature-list/metadata files
scripts/evaluate_models.py most recently saved," and previously duplicated this lookup."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from cfb_spread_model.utils.paths import OUTPUTS_MODELS


def latest_artifact(pattern: str) -> Path:
    matches = sorted(OUTPUTS_MODELS.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {OUTPUTS_MODELS} -- run scripts/evaluate_models.py first")
    return matches[-1]


def load_latest_production_artifact() -> tuple[object, list[str], dict]:
    """Returns (fitted_pipeline_or_estimator, selected_features, metadata) for whichever
    production model scripts/evaluate_models.py most recently saved."""
    model_path = latest_artifact("best_model_*.pkl")
    features_path = latest_artifact("selected_features_*.json")
    metadata_path = latest_artifact("model_metadata_*.json")

    model = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return model, features, metadata

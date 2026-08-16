"""Load and expose the YAML configs in config/ as plain dicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


def load_data_config() -> dict[str, Any]:
    return _load_yaml("data.yaml")


def load_features_config() -> dict[str, Any]:
    return _load_yaml("features.yaml")


def load_modeling_config() -> dict[str, Any]:
    return _load_yaml("modeling.yaml")


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config (relative to the project root) to an absolute Path."""
    return (PROJECT_ROOT / relative_path).resolve()

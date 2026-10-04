"""Redirects pipeline output (and dataset) paths into an isolated outputs/experiments/<tag>/
and data/processed/experiments/<tag>/ subtree when CFB_EXPERIMENT is set, so a
feature-engineering experiment can run the full select_features -> train_models ->
evaluate_models -> analyze_* chain without touching the confirmed-baseline files those
scripts otherwise overwrite unconditionally (fixed, non-timestamped paths - see
docs/final_writeup_2026.md's confirmed baseline results).

Unset (the default): every path below resolves to exactly what it did before this module
existed - CFB_EXPERIMENT must be explicitly set for any output location, or any
under-development engineered feature, to be included. Set it once in the shell before running
the pipeline scripts in sequence, e.g. `CFB_EXPERIMENT=fatigue_bye_v1 python scripts/select_features.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def experiment_tag() -> str | None:
    return os.environ.get("CFB_EXPERIMENT") or None


def output_path(*parts: str) -> Path:
    """Resolve a path under outputs/ - e.g. output_path("model_comparison", "oof_predictions.csv").
    Redirected under outputs/experiments/<tag>/ when CFB_EXPERIMENT is set."""
    tag = experiment_tag()
    base = REPO_ROOT / "outputs" / "experiments" / tag if tag else REPO_ROOT / "outputs"
    return base.joinpath(*parts)


def dataset_path() -> Path:
    """data/processed/modeling_dataset.parquet, or a per-experiment copy when CFB_EXPERIMENT
    is set - so an experiment's feature changes never overwrite the shared training set the
    confirmed baseline (and every script that omits CFB_EXPERIMENT) reads by default."""
    tag = experiment_tag()
    if tag:
        return REPO_ROOT / "data" / "processed" / "experiments" / tag / "modeling_dataset.parquet"
    return REPO_ROOT / "data" / "processed" / "modeling_dataset.parquet"


def include_experimental_features() -> bool:
    """Whether apply_engineered_features should turn on features currently under test
    (fatigue/workload composite + bye-week flag). Tied to the same CFB_EXPERIMENT switch as
    the path redirection above, so a tagged run is self-consistent: new features in, results
    isolated; an untagged run is untouched on both counts."""
    return experiment_tag() is not None

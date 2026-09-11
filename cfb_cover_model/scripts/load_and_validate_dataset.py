#!/usr/bin/env python
"""Stage 1: load the two source CSVs, build the push-aware modeling frame, validate it,
and cache it to data/processed/modeling_dataset.parquet.

Set CFB_EXPERIMENT=<tag> to run this (and select_features.py/train_models.py/
evaluate_models.py/analyze_*.py downstream) as an isolated experiment: the dataset and every
output land under data/processed/experiments/<tag>/ and outputs/experiments/<tag>/ instead of
the shared default paths, and the fatigue/workload + bye-week features currently under test
get included. Unset (the default): identical to before CFB_EXPERIMENT existed - see
src/cfb_cover_model/experiment_paths.py."""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model import experiment_paths
from cfb_cover_model.cleaning import build_clean_modeling_frame, build_excluded_columns
from cfb_cover_model.config import load_data_config, resolve_path
from cfb_cover_model.data import load_raw_joined, load_schedule_df
from cfb_cover_model.data_validation import summarize, validate_modeling_frame
from cfb_cover_model.engineered_features import apply_engineered_features
from cfb_cover_model.schedule_features import attach_bye_flags
from cfb_cover_model.targets import add_push_and_targets, drop_pushes

OUT_PATH = experiment_paths.dataset_path()
INVENTORY_PATH = experiment_paths.output_path("data_inventory", "column_inventory.json")
FEATURE_COLUMNS_PATH = experiment_paths.output_path("data_inventory", "feature_columns.json")


def main() -> None:
    data_cfg = load_data_config()
    include_experimental = experiment_paths.include_experimental_features()

    raw = load_raw_joined(data_cfg)
    if include_experimental:
        schedule = load_schedule_df(data_cfg)
        raw = attach_bye_flags(raw, schedule)
    with_targets = add_push_and_targets(raw)
    n_pushes = int(with_targets["is_push"].sum())
    filtered = drop_pushes(with_targets)

    feature_engineering_fn = functools.partial(
        apply_engineered_features, include_fatigue_features=include_experimental
    )
    frame, feature_columns = build_clean_modeling_frame(
        filtered, data_cfg, feature_engineering_fn=feature_engineering_fn
    )
    validate_modeling_frame(frame, feature_columns)

    excludes = build_excluded_columns(data_cfg, filtered.columns)

    holdout_seasons = set(data_cfg["seasons"]["final_holdout"])
    exclude_seasons = set(data_cfg["seasons"]["exclude"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_tag": experiment_paths.experiment_tag(),
        "n_games_raw_joined": len(raw),
        "n_pushes_excluded": n_pushes,
        "n_games_after_push_filter": len(filtered),
        **summarize(frame, feature_columns),
        "n_candidate_features": len(feature_columns),
        "n_columns_excluded_id": len(excludes["id_columns"]),
        "n_columns_excluded_leakage_adjacent": len(excludes["leakage_adjacent_columns"]),
        "n_columns_excluded_known_bad": len(excludes["known_bad_columns"]),
        "n_columns_excluded_deterministic_redundant": len(
            excludes["deterministic_redundant_columns"]
        ),
        "known_bad_columns": excludes["known_bad_columns"],
        "deterministic_redundant_columns": excludes["deterministic_redundant_columns"],
        "holdout_seasons": sorted(holdout_seasons),
        "excluded_seasons": sorted(exclude_seasons),
        "output_path": str(OUT_PATH),
    }
    INVENTORY_PATH.write_text(json.dumps(report, indent=2, default=str))
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_columns, indent=2))

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

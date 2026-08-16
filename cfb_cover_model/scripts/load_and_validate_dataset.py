#!/usr/bin/env python
"""Stage 1: load the two source CSVs, build the push-aware modeling frame, validate it,
and cache it to data/processed/modeling_dataset.parquet."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.cleaning import build_clean_modeling_frame, build_excluded_columns
from cfb_cover_model.config import load_data_config, resolve_path
from cfb_cover_model.data import load_raw_joined
from cfb_cover_model.data_validation import summarize, validate_modeling_frame
from cfb_cover_model.engineered_features import apply_engineered_features
from cfb_cover_model.targets import add_push_and_targets, drop_pushes

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
INVENTORY_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "column_inventory.json"
)
FEATURE_COLUMNS_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
)


def main() -> None:
    data_cfg = load_data_config()

    raw = load_raw_joined(data_cfg)
    with_targets = add_push_and_targets(raw)
    n_pushes = int(with_targets["is_push"].sum())
    filtered = drop_pushes(with_targets)

    frame, feature_columns = build_clean_modeling_frame(
        filtered, data_cfg, feature_engineering_fn=apply_engineered_features
    )
    validate_modeling_frame(frame, feature_columns)

    excludes = build_excluded_columns(data_cfg, filtered.columns)

    holdout_seasons = set(data_cfg["seasons"]["final_holdout"])
    exclude_seasons = set(data_cfg["seasons"]["exclude"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
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

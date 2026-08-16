#!/usr/bin/env python
"""Step 2 of the pipeline. Builds the one-row-per-(school,season) modeling table and the
feature registry. Caches the result to data/processed/modeling_dataset.parquet, keyed by a
hash of the config contents + target_seasons list -- pass --rebuild to force a rebuild.

Usage:
    python scripts/build_modeling_dataset.py [--target-seasons 2015 2016 ... 2025] [--rebuild]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_win_total_model.cleaning import apply_winsorization, impute_missing, validate_no_inf_or_extreme
from cfb_win_total_model.config import CONFIG_DIR, load_features_config, load_modeling_config
from cfb_win_total_model.database import get_engine
from cfb_win_total_model.dataset import build_feature_registry, build_modeling_dataset
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_FEATURE_ANALYSIS, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
MANIFEST_PATH = DATA_PROCESSED_DIR / "modeling_dataset_manifest.json"

ZERO_FILL_COLS = [
    "n_returning_players", "n_departed_players", "n_transferred_out", "n_incoming_players", "n_transferred_in",
    "n_5_star", "n_4_star", "n_distinct_recruited_players",
    "n_power_opponents", "n_group_of_5_opponents", "n_sub_fbs_opponents", "n_opponents_above_own_talent",
    "n_power_opponents_early", "n_power_opponents_late", "bye_week_count", "short_rest_count", "back_to_back_road_count",
]


def _manifest_hash(target_seasons: list[int]) -> str:
    payload = {
        "target_seasons": target_seasons,
        "modeling_yaml": (CONFIG_DIR / "modeling.yaml").read_text(),
        "features_yaml": (CONFIG_DIR / "features.yaml").read_text(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-seasons", type=int, nargs="+", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    modeling_cfg = load_modeling_config()
    features_cfg = load_features_config()

    target_seasons = args.target_seasons or list(range(modeling_cfg.full_feature_start_season, modeling_cfg.target_season + 1))
    below_floor = [s for s in target_seasons if s < modeling_cfg.full_feature_start_season]
    if below_floor:
        raise ValueError(
            f"target_seasons {below_floor} are below full_feature_start_season="
            f"{modeling_cfg.full_feature_start_season} (team_talent/returning_production don't exist yet). "
            f"The per-module feature functions remain permissive for flexibility; this top-level "
            f"script gates the floor."
        )

    manifest_hash = _manifest_hash(target_seasons)
    if not args.rebuild and DATASET_PATH.exists() and MANIFEST_PATH.exists():
        cached_hash = json.loads(MANIFEST_PATH.read_text()).get("hash")
        if cached_hash == manifest_hash:
            logger.info(f"Cached modeling dataset at {DATASET_PATH} matches current config; skipping rebuild (--rebuild to force)")
            return 0

    engine = get_engine()
    df = build_modeling_dataset(engine, target_seasons=target_seasons, features_cfg=features_cfg)
    df = apply_winsorization(df, features_cfg)
    df = impute_missing(df, zero_fill_cols=ZERO_FILL_COLS)
    validate_no_inf_or_extreme(df)

    if df.duplicated(subset=["school", "season"]).any():
        raise AssertionError("Post-cleaning duplicate (school, season) rows in modeling dataset")

    df.to_parquet(DATASET_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps({"hash": manifest_hash, "target_seasons": target_seasons, "n_rows": len(df)}, indent=2))
    logger.info(f"Wrote modeling dataset: {df.shape} -> {DATASET_PATH}")

    registry = build_feature_registry(features_cfg)
    registry.to_csv(OUTPUTS_FEATURE_ANALYSIS / "feature_registry.csv", index=False)
    logger.info(f"Wrote feature registry: {len(registry)} features -> {OUTPUTS_FEATURE_ANALYSIS / 'feature_registry.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Step 2 of the pipeline. Builds the one-row-per-(athlete_id, game_id) modeling table and
the feature registry. Caches the result to data/processed/modeling_dataset.parquet, keyed by
a hash of the config contents + target_seasons list -- pass --rebuild to force a rebuild.

Usage:
    python scripts/build_modeling_dataset.py [--target-seasons 2014 2015 ... 2024] [--rebuild]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_rb_rushing_model.cleaning import impute_missing, validate_no_inf_or_extreme
from cfb_rb_rushing_model.config import CONFIG_DIR, load_data_config, load_features_config, load_modeling_config
from cfb_rb_rushing_model.database import get_engine
from cfb_rb_rushing_model.dataset import build_feature_registry, build_modeling_dataset
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_FEATURE_ANALYSIS, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
MANIFEST_PATH = DATA_PROCESSED_DIR / "modeling_dataset_manifest.json"


def _manifest_hash(target_seasons: list[int]) -> str:
    payload = {
        "target_seasons": target_seasons,
        "modeling_yaml": (CONFIG_DIR / "modeling.yaml").read_text(),
        "features_yaml": (CONFIG_DIR / "features.yaml").read_text(),
        "data_yaml": (CONFIG_DIR / "data.yaml").read_text(),
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
    data_cfg = load_data_config()

    target_seasons = args.target_seasons or [
        s for s in range(modeling_cfg.full_feature_start_season, modeling_cfg.target_season + 1)
        if s not in modeling_cfg.excluded_seasons
    ]
    below_floor = [s for s in target_seasons if s < modeling_cfg.full_feature_start_season]
    if below_floor:
        raise ValueError(
            f"target_seasons {below_floor} are below full_feature_start_season="
            f"{modeling_cfg.full_feature_start_season} (plays.rusher_player_name is unreliable "
            f"before this season -- see docs/assumptions_and_limitations.md)."
        )

    manifest_hash = _manifest_hash(target_seasons)
    if not args.rebuild and DATASET_PATH.exists() and MANIFEST_PATH.exists():
        cached_hash = json.loads(MANIFEST_PATH.read_text()).get("hash")
        if cached_hash == manifest_hash:
            logger.info(f"Cached modeling dataset at {DATASET_PATH} matches current config; skipping rebuild (--rebuild to force)")
            return 0

    engine = get_engine()
    df = build_modeling_dataset(engine, target_seasons=target_seasons, data_cfg=data_cfg, features_cfg=features_cfg)

    if modeling_cfg.final_holdout_max_week is not None:
        capped_season = modeling_cfg.final_holdout_season
        max_week = modeling_cfg.final_holdout_max_week
        drop_mask = (df["season"] == capped_season) & (df["week"] > max_week)
        n_dropped = int(drop_mask.sum())
        if n_dropped:
            logger.info(
                f"Dropping {n_dropped} rows for season={capped_season} week>{max_week} "
                f"(final_holdout_max_week) -- see docs/assumptions_and_limitations.md for why: "
                f"target/player-feature rows past this week would be silently zero-filled by "
                f"targets.py's LEFT JOIN due to the confirmed rusher-name data gap, "
                f"indistinguishable from a genuine zero-carry game."
            )
        df = df[~drop_mask]

    df = impute_missing(df)
    validate_no_inf_or_extreme(df)

    if df.duplicated(subset=["athlete_id", "game_id"]).any():
        raise AssertionError("Post-cleaning duplicate (athlete_id, game_id) rows in modeling dataset")

    df.to_parquet(DATASET_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps({"hash": manifest_hash, "target_seasons": target_seasons, "n_rows": len(df)}, indent=2))
    logger.info(f"Wrote modeling dataset: {df.shape} -> {DATASET_PATH}")

    registry = build_feature_registry(features_cfg, data_cfg)
    registry.to_csv(OUTPUTS_FEATURE_ANALYSIS / "feature_registry.csv", index=False)
    logger.info(f"Wrote feature registry: {len(registry)} features -> {OUTPUTS_FEATURE_ANALYSIS / 'feature_registry.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

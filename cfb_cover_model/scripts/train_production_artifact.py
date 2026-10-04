#!/usr/bin/env python
"""Retrain the dual-model ("agreement") production signal on all eligible history and persist
it as a new versioned artifact under outputs/models/production/ - this is the step that used
to not exist at all (see outputs/models/README.md's prior "no production model saved" note).
The FastAPI service (src/cfb_cover_model/api/) loads whatever this script last wrote via
latest.json rather than refitting per request; scripts/run_scheduler.py calls this indirectly
by hitting the API's own POST /admin/retrain (see docs/serving_and_monitoring.md for why the
API - not this script running standalone in a second process - owns all writes to
outputs/models/production/ in the Docker deployment).

Usage:
    python scripts/train_production_artifact.py --season 2026 --week 3
        (labels this artifact as "trained through season 2026 week 3" - purely descriptive,
        does not filter which history rows are used; that's whatever
        data/processed/modeling_dataset.parquet + extended_history.parquet already contain)
    python scripts/train_production_artifact.py --season 2026 --week 3 --dry-run
        (train and print a summary, but skip writing anything to disk)

Set CFB_EXPERIMENT=<tag> to train against an isolated experiment's dataset/feature-selection
outputs (see src/cfb_cover_model/experiment_paths.py) and persist the resulting artifact under
outputs/experiments/<tag>/models/production/ instead of the real outputs/models/production/ -
the confirmed production artifact and the FastAPI service that loads it are never touched by a
tagged run. Unset (the default): identical to before this existed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model import experiment_paths
from cfb_cover_model.serving.artifact import (
    ProductionArtifact,
    resolve_default_configs,
    save_artifact,
    train_production_artifact,
)


# Resolved once at import time from CFB_EXPERIMENT (must be set in the environment before this
# script runs, same convention as every other pipeline script's experiment_paths-derived
# constants) - redirected under outputs/experiments/<tag>/... when tagged, else identical to
# before this existed. See module docstring.
DATASET_PATH = experiment_paths.dataset_path()
FEATURE_COLUMNS_PATH = experiment_paths.output_path("data_inventory", "feature_columns.json")
WINNING_CONFIG_PATH = experiment_paths.output_path("feature_analysis", "winning_feature_config.json")
THRESHOLD_TABLE_PATH = experiment_paths.output_path("threshold_selection", "chosen_threshold_per_model.csv")
OUTPUT_DIR = experiment_paths.output_path("models", "production")


def _prune_old_versions(output_dir: Path, keep: int) -> list[Path]:
    """Delete all but the `keep` most-recently-trained version directories (by name, which
    sorts chronologically since versions are v{season}w{week}_{UTC timestamp}). Never deletes
    latest.json itself or anything that isn't a version directory this script recognizes."""
    version_dirs = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("v")),
        key=lambda p: p.name,
    )
    to_delete = version_dirs[:-keep] if keep > 0 else []
    for d in to_delete:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    return to_delete


def run(
    season: int,
    week: int,
    output_dir: Path = OUTPUT_DIR,
    *,
    dry_run: bool = False,
    prune: bool = True,
    retention: int | None = None,
) -> ProductionArtifact:
    """Trains and persists a new artifact. Also prunes old versions by default (so this
    behaves the same whether called from main() below or from the API's POST /admin/retrain
    handler - the scheduler's weekly job always goes through the latter, so pruning must not
    be main()-only or it would never run in the Docker deployment)."""
    data_cfg, features_cfg, modeling_cfg, random_state = resolve_default_configs()
    artifact = train_production_artifact(
        data_cfg, features_cfg, modeling_cfg, random_state,
        season_through=season, week_through=week,
        dataset_path=DATASET_PATH,
        feature_columns_path=FEATURE_COLUMNS_PATH,
        winning_config_path=WINNING_CONFIG_PATH,
        threshold_table_path=THRESHOLD_TABLE_PATH,
    )
    if not dry_run:
        save_artifact(artifact, output_dir=output_dir)
        if prune:
            keep = retention if retention is not None else int(os.environ.get("PRODUCTION_ARTIFACT_RETENTION", 8))
            _prune_old_versions(Path(output_dir), keep)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="Most recent completed season, label only")
    parser.add_argument("--week", type=int, required=True, help="Most recent completed week, label only")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Train but skip persisting to disk")
    parser.add_argument("--no-prune", action="store_true", help="Skip pruning old artifact versions")
    parser.add_argument(
        "--retention", type=int, default=int(os.environ.get("PRODUCTION_ARTIFACT_RETENTION", 8)),
        help="Number of most-recent artifact versions to keep (default: $PRODUCTION_ARTIFACT_RETENTION or 8)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    artifact = run(
        args.season, args.week, output_dir=output_dir,
        dry_run=args.dry_run, prune=not args.no_prune, retention=args.retention,
    )

    print(f"Trained artifact {artifact.version}")
    print(f"  season_through={artifact.season_through} week_through={artifact.week_through}")
    print(f"  training_row_count={artifact.training_row_count}")
    print(f"  selected_columns={len(artifact.selected_columns)}")
    print(f"  classifier_threshold={artifact.classifier_threshold} regressor_threshold={artifact.regressor_threshold}")
    print(f"  git_commit={artifact.git_commit}")

    if args.dry_run:
        print("\n--dry-run set - nothing written to disk.")
        return

    print(f"\nWrote {output_dir / artifact.version}")
    print(f"Updated {output_dir / 'latest.json'}")
    if not args.no_prune:
        print(f"Pruned old versions beyond retention={args.retention} (if any).")


if __name__ == "__main__":
    main()

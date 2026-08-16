#!/usr/bin/env python
"""Fetch newly-completed games directly from the CFBD API and append them to
data/processed/extended_history.parquet - the additive, live-pipeline counterpart to
modeling_dataset.parquet (which is built once from the R-generated historical CSVs by
load_and_validate_dataset.py and is never touched by this script).
generate_weekly_predictions.py unions the two before fitting the production models, so a
completed 2026 week becomes usable as training history for the next week's --live run
without waiting for a new R pull.

Runs the same push-filtering / target-construction / engineered-feature / validation steps
as load_and_validate_dataset.py (see cfb_cover_model.targets, cfb_cover_model.cleaning,
cfb_cover_model.engineered_features, cfb_cover_model.data_validation), just sourced from
ingest.pipeline.build_historical_rows() instead of the R-generated results/predictors CSVs.

Usage:
    python scripts/ingest_and_update_history.py --season 2026 --weeks 1 2 3
        (re-fetches and rebuilds rows for the given completed weeks; idempotent per-week
        thanks to raw_cache's parquet cache, and existing game_ids in
        extended_history.parquet are overwritten rather than duplicated - safe to re-run)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.cleaning import build_clean_modeling_frame
from cfb_cover_model.config import load_data_config
from cfb_cover_model.data_validation import validate_modeling_frame
from cfb_cover_model.engineered_features import apply_engineered_features
from cfb_cover_model.ingest import cfbd_client, pipeline
from cfb_cover_model.targets import add_push_and_targets, drop_pushes

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "extended_history.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--weeks", type=int, nargs="+", required=True,
        help="Completed week numbers to (re-)fetch and append, e.g. --weeks 1 2 3",
    )
    args = parser.parse_args()

    data_cfg = load_data_config()
    client = cfbd_client.get_client()

    raw = pipeline.build_historical_rows(client, args.season, args.weeks)
    if raw.empty:
        print(f"No completed games found for season={args.season} weeks={args.weeks}")
        return

    # build_historical_rows returns an *absolute* spread + home_favored (0/1), matching
    # ../Data/CFB_Gambling_Predictors_Final_PBP.csv's schema - targets.py needs the
    # *signed* spread instead (negative means home favored), matching what
    # data.load_results_df pulls from the R results CSV's own signed "spread" column.
    raw["signed_spread"] = raw["spread"].where(raw["home_favored"] == 0, -raw["spread"])

    with_targets = add_push_and_targets(raw)
    n_pushes = int(with_targets["is_push"].sum())
    filtered = drop_pushes(with_targets)

    frame, feature_columns = build_clean_modeling_frame(
        filtered, data_cfg, feature_engineering_fn=apply_engineered_features
    )
    validate_modeling_frame(frame, feature_columns)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    else:
        combined = frame
    combined = combined.reset_index(drop=True)
    combined.to_parquet(OUT_PATH, index=False)

    print(
        f"season={args.season} weeks={args.weeks}: {len(raw)} games fetched, "
        f"{n_pushes} pushes excluded, {len(frame)} rows added/updated -> {OUT_PATH} "
        f"({len(combined)} total rows)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generates "Week 0" preseason ratings for a season before any games are played -- pure
preseason-model output, no in-season blending (that starts with scripts/update_ratings.py once
games begin). This is what makes "publish 2026 rankings before kickoff" possible.

Prerequisite: season-<season> rows must already exist in team_talent/coaches/
returning_production/team_rosters/recruiting_players -- run SQL Scripts/ingest_to_mysql.R for
the new season first (this script only reads, it doesn't ingest).

Usage: python scripts/generate_preseason_ratings.py --season 2026

For an honest reconstruction of a PAST season's preseason ratings (a model trained only on
seasons strictly before it, never having seen its own outcome -- e.g. "what would 2025 preseason
rankings have looked like"), pass --out-of-sample:

    python scripts/generate_preseason_ratings.py --season 2025 --out-of-sample

See preseason.py's predict_out_of_sample_preseason_ratings docstring for why this differs from
the default (which uses the persisted production model -- appropriate for a genuinely future
season, not for backtest-honest reconstruction of a past one).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_power_ratings.config import load_features_config, load_modeling_config
from cfb_power_ratings.database import get_engine
from cfb_power_ratings.preseason import predict_out_of_sample_preseason_ratings, predict_preseason_ratings
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_RATINGS

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--out-of-sample", action="store_true",
        help="Train fresh on seasons strictly before --season instead of using the persisted "
             "production model -- an honest reconstruction of a past season's preseason ratings.",
    )
    args = parser.parse_args()

    engine = get_engine()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()

    if args.out_of_sample:
        ratings = predict_out_of_sample_preseason_ratings(engine, args.season, features_cfg, modeling_cfg)
    else:
        ratings = predict_preseason_ratings(engine, args.season, features_cfg, modeling_cfg.srs_history_start_season)
    out = ratings.sort_values(ascending=False).reset_index()
    out.columns = ["team", "rating"]
    out["rank"] = range(1, len(out) + 1)

    out_dir = OUTPUTS_RATINGS / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "week_00_ratings.csv"
    out[["rank", "team", "rating"]].to_csv(out_path, index=False)

    print(f"Preseason {args.season} ratings ({len(out)} FBS teams):")
    print(out[["rank", "team", "rating"]].head(25).to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

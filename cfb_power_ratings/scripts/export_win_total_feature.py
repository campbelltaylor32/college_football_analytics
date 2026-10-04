#!/usr/bin/env python
"""Exports an honest, out-of-sample preseason-power-rating series across every season this
project has enough training history for -- one row per (team, season), each rating from a model
trained ONLY on seasons strictly before that season (never that season's own outcome).

Built for cross-project use: cfb_win_total_model's diagnostic power_rating feature reads this
CSV to test whether cfb_power_ratings' forward-looking preseason projection adds signal beyond
cfb_win_total_model's own features (notably sp_overall_entering_t, a backward-looking third-party
rating). Not specific to that one use case -- this is the general "give me the honest preseason
rating for every season with enough history" export, reusable for any future backtest or
cross-project need.

A season is skipped (not written with a placeholder/zero row -- simply absent) whenever fewer
than modeling.yaml's min_train_seasons non-excluded seasons precede it -- predict_out_of_sample_
preseason_ratings would otherwise be asked to fit on a degenerate few-season sample. Given
full_feature_start_season=2016 and 2020 excluded, the first season with enough history is 2021
(seasons 2016-2019 = 4 seasons).

Usage: python scripts/export_win_total_feature.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_power_ratings.config import load_features_config, load_modeling_config
from cfb_power_ratings.database import get_engine
from cfb_power_ratings.preseason import predict_out_of_sample_preseason_ratings
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_RATINGS_HISTORY, ensure_dirs

logger = get_logger(__name__)


def main() -> None:
    ensure_dirs()
    engine = get_engine()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()

    seasons = list(range(modeling_cfg.full_feature_start_season, modeling_cfg.final_holdout_season + 1))
    rows = []
    for season in seasons:
        train_seasons = [
            s for s in range(modeling_cfg.full_feature_start_season, season)
            if s not in modeling_cfg.excluded_seasons
        ]
        if len(train_seasons) < modeling_cfg.min_train_seasons:
            logger.info(
                f"season={season}: only {len(train_seasons)} non-excluded training seasons "
                f"available (need {modeling_cfg.min_train_seasons}) -- skipping"
            )
            continue

        logger.info(f"season={season}: training on {train_seasons} ({len(train_seasons)} seasons)")
        ratings = predict_out_of_sample_preseason_ratings(engine, season, features_cfg, modeling_cfg)
        rows.append(pd.DataFrame({
            "team": ratings.index, "season": season, "preseason_power_rating": ratings.values,
        }))

    if not rows:
        raise RuntimeError("No season had enough training history to produce an honest out-of-sample rating")

    out = pd.concat(rows, ignore_index=True)
    out_path = OUTPUTS_RATINGS_HISTORY / "preseason_ratings_by_season.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out)} (team, season) rows across {out['season'].nunique()} seasons -> {out_path}")
    print(out.groupby("season").size().rename("n_teams").to_string())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

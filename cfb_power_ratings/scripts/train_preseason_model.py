#!/usr/bin/env python
"""Builds the full modeling dataset, walk-forward evaluates every baseline + candidate model
against the actual SRS target, runs the consensus-spread external sanity check on the winning
model's out-of-fold predictions, then fits the winner on all eligible history and persists it
to outputs/models/preseason_model.joblib (+ metadata.json).

Usage: python scripts/train_preseason_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from cfb_power_ratings.config import load_features_config, load_modeling_config
from cfb_power_ratings.database import get_engine
from cfb_power_ratings.dataset import FEATURE_COLUMNS, build_modeling_dataset
from cfb_power_ratings.modeling.baselines import BASELINES
from cfb_power_ratings.modeling.evaluate import evaluate_against_consensus_spread, pooled_mae, walk_forward_evaluate, walk_forward_predictions
from cfb_power_ratings.modeling.models import get_candidate_models
from cfb_power_ratings.srs import estimate_home_field_advantage
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_MODELS, ensure_dirs

logger = get_logger(__name__)


def main() -> None:
    ensure_dirs()
    engine = get_engine()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()

    seasons = list(range(modeling_cfg.full_feature_start_season, modeling_cfg.final_holdout_season + 1))
    logger.info(f"Building modeling dataset for seasons {seasons}")
    df = build_modeling_dataset(engine, seasons, features_cfg, modeling_cfg.srs_history_start_season)
    df = df.dropna(subset=["target_srs"])
    logger.info(f"Dataset: {len(df)} team-season rows")

    logger.info("Walk-forward evaluating candidates...")
    fold_results = walk_forward_evaluate(df, modeling_cfg)
    ranked = pooled_mae(fold_results)
    print("\nPooled walk-forward MAE (lower is better):")
    print(ranked.to_string())

    non_baseline = [m for m in ranked.index if m not in BASELINES]
    if not non_baseline:
        raise RuntimeError("No non-baseline candidate models were evaluated -- check modeling.yaml's candidate_models.")
    winner = ranked[non_baseline].idxmin()
    print(f"\nSelected model: {winner} (pooled MAE {ranked[winner]:.3f})")

    print("\nMarket-spread sanity check (winning model's OOF predictions vs. real betting-line averages):")
    oof = walk_forward_predictions(df, modeling_cfg, winner)
    spread_checks = []
    for season in modeling_cfg.walk_forward_validation_seasons:
        check = evaluate_against_consensus_spread(engine, oof, season, hfa=estimate_home_field_advantage(_all_games_cache(engine)))
        spread_checks.append(check)
        print(f"  {season}: n_games={check['n_games']}, mae={check['mae']}, correlation={check['correlation']}")

    logger.info("Fitting final model on all eligible history...")
    hfa = estimate_home_field_advantage(_all_games_cache(engine))
    final_model = get_candidate_models([winner], modeling_cfg.random_seed)[winner]
    train_df = df[~df["season"].isin(modeling_cfg.excluded_seasons)]
    final_model.fit(train_df[FEATURE_COLUMNS], train_df["target_srs"])

    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, OUTPUTS_MODELS / "preseason_model.joblib")
    metadata = {
        "model_name": winner,
        "feature_columns": FEATURE_COLUMNS,
        "pooled_walk_forward_mae": ranked.to_dict(),
        "consensus_spread_checks": spread_checks,
        "hfa": hfa,
        "training_seasons": sorted(train_df["season"].unique().tolist()),
        "training_row_count": len(train_df),
    }
    (OUTPUTS_MODELS / "preseason_model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nWrote {OUTPUTS_MODELS / 'preseason_model.joblib'}")
    print(f"Wrote {OUTPUTS_MODELS / 'preseason_model_metadata.json'}")


_GAMES_CACHE: pd.DataFrame | None = None


def _all_games_cache(engine) -> pd.DataFrame:
    global _GAMES_CACHE
    if _GAMES_CACHE is None:
        from cfb_power_ratings.database import run_query

        _GAMES_CACHE = run_query("SELECT * FROM games WHERE completed = 1", engine=engine)
    return _GAMES_CACHE


if __name__ == "__main__":
    main()

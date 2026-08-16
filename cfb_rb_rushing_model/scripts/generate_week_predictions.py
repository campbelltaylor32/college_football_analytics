#!/usr/bin/env python
"""Step 6 of the pipeline (weekly, run once per week during the season): scores workload-
eligible RBs for one upcoming week's games.

DB-driven, not CSV-driven -- unlike the sibling cfb_spread_model project's equivalent script,
which reads a pre-built CSV the R pipeline produces, this project has no such file. It reuses
dataset.build_modeling_dataset for the target season and filters to the target week, which
guarantees the exact same feature-building code path as training (no separate,
inference-only feature logic that could silently drift out of column-parity with the trained
model) -- schedule_spine/eligibility are designed to work directly against `games`' rows for a
scheduled-but-not-yet-played game, so this works for a genuine upcoming week.

Includes a hard, loud data-quality gate before scoring anything: if the target week's
plays.rusher_player_name NULL rate exceeds the configured floor, this script aborts rather
than silently producing predictions from a near-empty rushing population -- the direct
response to the 2025-week-9+ ingestion gap discovered during planning (see
docs/assumptions_and_limitations.md).

Usage:
    python scripts/generate_week_predictions.py --season 2024 --week 6
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_rb_rushing_model import data_validation as dv
from cfb_rb_rushing_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_rb_rushing_model.database import get_engine, run_query
from cfb_rb_rushing_model.dataset import build_modeling_dataset
from cfb_rb_rushing_model.modeling.artifacts import load_latest_production_artifact
from cfb_rb_rushing_model.modeling.evaluation import prediction_interval_from_residuals
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import OUTPUTS_PREDICTIONS, ensure_dirs

logger = get_logger(__name__)

OUTPUT_COLS = [
    "athlete_id", "player_name", "team", "opponent", "season", "week",
    "predicted_rushing_yards", "prediction_interval_low", "prediction_interval_high",
    "carries_avg3_asof", "model_name", "model_version", "data_cutoff_date",
]


def _player_names(engine, athlete_ids: list, season: int) -> pd.DataFrame:
    if not athlete_ids:
        return pd.DataFrame(columns=["athlete_id", "player_name"])
    placeholders = ", ".join(f":a{i}" for i in range(len(athlete_ids)))
    params = {f"a{i}": aid for i, aid in enumerate(athlete_ids)}
    params["season"] = season
    sql = f"""
        SELECT DISTINCT athlete_id, CONCAT(first_name, ' ', last_name) AS player_name
        FROM team_rosters WHERE season = :season AND athlete_id IN ({placeholders})
    """
    return run_query(sql, params=params, engine=engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--max-null-rate", type=float, default=None, help="Override data_validation.MAX_ACCEPTABLE_RUSHER_NAME_NULL_RATE")
    args = parser.parse_args()

    ensure_dirs()
    engine = get_engine()
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()

    completeness = dv.check_rusher_name_completeness(engine, season=args.season, week=args.week)
    null_rate_floor = args.max_null_rate if args.max_null_rate is not None else dv.MAX_ACCEPTABLE_RUSHER_NAME_NULL_RATE
    if not completeness.empty:
        row = completeness.iloc[0]
        if row["null_rate"] > null_rate_floor:
            raise RuntimeError(
                f"ABORTING: season={args.season} week={args.week} plays.rusher_player_name NULL "
                f"rate is {row['null_rate']:.1%}, above the {null_rate_floor:.0%} floor -- this "
                f"almost certainly means the upstream ingestion for this week is broken (the "
                f"same failure mode discovered for 2025 weeks 9+ during planning), not that "
                f"nobody ran the ball. Investigate SQL Scripts/ingest_to_mysql.R before trusting "
                f"any prediction built on this week's data."
            )

    model, feature_cols, metadata = load_latest_production_artifact()
    logger.info(f"Loaded production model '{metadata['model_name']}' (trained {metadata['trained_date']})")

    df = build_modeling_dataset(engine, [args.season], data_cfg, features_cfg)
    week_df = df[df["week"] == args.week].copy()
    if week_df.empty:
        logger.warning(f"No workload-eligible RB rows for season={args.season} week={args.week} -- nothing to predict")
        return 0

    missing = [c for c in feature_cols if c not in week_df.columns]
    if missing:
        raise ValueError(f"Week {args.week} predictor rows are missing {len(missing)} required columns: {missing[:10]}...")

    from cfb_rb_rushing_model.cleaning import impute_missing

    week_df_imputed = impute_missing(week_df)

    # Baselines' .predict(df) takes the FULL row (it reads named columns like
    # rushing_yards_avg3_asof directly); sklearn Pipelines' .predict(X) takes only the
    # feature-column subset in training column order. metadata["model_type"] (written by
    # scripts/evaluate_models.py) disambiguates which calling convention this artifact needs.
    if metadata.get("model_type") == "baseline":
        preds = model.predict(week_df_imputed)
    else:
        preds = model.predict(week_df_imputed[feature_cols])
    preds = pd.Series(preds).clip(lower=modeling_cfg.clip_min_yards).to_numpy()
    week_df["predicted_rushing_yards"] = preds

    if metadata.get("n_oof_residuals", 0) >= 2:
        oof_path = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison" / "oof_predictions.csv"
        oof_df = pd.read_csv(oof_path)
        model_oof = oof_df[oof_df["model_name"] == metadata["model_name"]]
        residuals = (model_oof["y_true"] - model_oof["y_pred"]).to_numpy()
        lo, hi = prediction_interval_from_residuals(week_df["predicted_rushing_yards"].to_numpy(), residuals, tuple(metadata["prediction_interval_levels"]))
        week_df["prediction_interval_low"] = lo.clip(min=modeling_cfg.clip_min_yards)
        week_df["prediction_interval_high"] = hi
    else:
        week_df["prediction_interval_low"] = None
        week_df["prediction_interval_high"] = None

    names = _player_names(engine, week_df["athlete_id"].unique().tolist(), args.season)
    week_df = week_df.merge(names, on="athlete_id", how="left")

    week_df["model_name"] = metadata["model_name"]
    week_df["model_version"] = f"{metadata['model_name']}_{metadata['trained_date']}"
    week_df["data_cutoff_date"] = date.today().isoformat()

    out = week_df[OUTPUT_COLS].sort_values("predicted_rushing_yards", ascending=False).reset_index(drop=True)
    out_path = OUTPUTS_PREDICTIONS / f"week_{args.season}_{args.week}_rb_rushing_predictions.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out)} predictions -> {out_path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

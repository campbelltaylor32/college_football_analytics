#!/usr/bin/env python
"""Loads ../Data/CFB_Gambling_Predictors_Final_PBP.csv, validates it against config/data.yaml's
expectations, derives the Pythagorean win%% columns (feature_engineering.build_pythagorean_features)
on top of the validated raw columns, and caches the result to
data/processed/modeling_dataset.parquet. Every other stage reads the parquet cache, never the
CSV directly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from cfb_spread_model.cleaning import report_residual_missingness, validate_no_inf
from cfb_spread_model.config import load_data_config
from cfb_spread_model.data import get_feature_columns, load_raw_csv
from cfb_spread_model.data_validation import check_covid_season_flag, validate_raw_dataset
from cfb_spread_model.feature_engineering import build_pythagorean_features
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force reload from CSV even if the parquet cache exists")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_data_config()

    if DATASET_PATH.exists() and not args.rebuild:
        logger.info(f"{DATASET_PATH} already exists; skipping (pass --rebuild to force)")
        return 0

    df = load_raw_csv(cfg)
    validate_raw_dataset(df, cfg)

    pyth_df = build_pythagorean_features(df, list(df.columns))
    df = pd.concat([df, pyth_df], axis=1)
    logger.info(f"Added Pythagorean win%% columns: {list(pyth_df.columns)}")

    covid_check = check_covid_season_flag(df)
    logger.info(f"COVID-season check: {covid_check}")

    feature_cols = get_feature_columns(list(df.columns), cfg)
    validate_no_inf(df, feature_cols)
    residual_na = report_residual_missingness(df, feature_cols)
    if residual_na.empty:
        logger.info("No residual NA in feature columns (matches upstream R na.omit() guarantee)")

    df.to_parquet(DATASET_PATH, index=False)
    logger.info(f"Cached validated dataset -> {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

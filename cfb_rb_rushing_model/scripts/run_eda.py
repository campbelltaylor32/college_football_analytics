#!/usr/bin/env python
"""Step 3 of the pipeline. Exploratory analysis of the assembled modeling table, including
the eligibility-threshold sensitivity sweep flagged as an open item in the approved plan
(features.yaml's min_trailing3_avg_carries=8 is a starting point, not validated-optimal).
Writes plots/tables to outputs/eda/.

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_rb_rushing_model.config import load_data_config, load_features_config
from cfb_rb_rushing_model.database import get_engine
from cfb_rb_rushing_model.dataset import NON_FEATURE_COLS
from cfb_rb_rushing_model.eligibility import build_eligibility_spine
from cfb_rb_rushing_model.schedule_spine import attach_rest_days, build_schedule_spine
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_EDA, ensure_dirs

logger = get_logger(__name__)
DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

ELIGIBILITY_THRESHOLD_SWEEP = [4, 6, 8, 10, 12, 15]


def _eligibility_sensitivity_sweep(engine, seasons: list[int]) -> pd.DataFrame:
    """Row-count / target-variance table across candidate min_trailing3_avg_carries
    thresholds -- flagged in docs/assumptions_and_limitations.md as the thing to run before
    treating features.yaml's default of 8 as final."""
    data_cfg = load_data_config()
    rows = []
    for threshold in ELIGIBILITY_THRESHOLD_SWEEP:
        features_cfg = load_features_config()
        features_cfg.eligibility.min_trailing3_avg_carries = threshold
        spine = build_schedule_spine(engine, seasons)
        spine = attach_rest_days(spine, features_cfg.default_rest_days_season_opener)
        elig = build_eligibility_spine(engine, spine, seasons, data_cfg, features_cfg)
        eligible = elig[elig["eligible"]]
        rows.append(
            {
                "min_trailing3_avg_carries": threshold,
                "n_eligible_rows": len(eligible),
                "n_unique_players": eligible["athlete_id"].nunique() if not eligible.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found -- run scripts/build_modeling_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"EDA on modeling dataset {df.shape}")

    df["rushing_yards"].describe().to_csv(OUTPUTS_EDA / "rushing_yards_distribution_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["rushing_yards"], bins=40)
    ax.set_xlabel("Rushing yards")
    ax.set_title("Distribution of RB rushing yards (eligible player-games)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "rushing_yards_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="rushing_yards", by="season", ax=ax, rot=0)
    ax.set_title("Rushing yards by season")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "rushing_yards_by_season.png")
    plt.close(fig)

    if "played" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        df.boxplot(column="rushing_yards", by="played", ax=ax)
        ax.set_title("Rushing yards: played vs. zero-carry eligible games")
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(OUTPUTS_EDA / "rushing_yards_by_played.png")
        plt.close(fig)

    players_by_season = df.groupby("season")["athlete_id"].nunique()
    players_by_season.to_csv(OUTPUTS_EDA / "unique_eligible_players_by_season.csv")

    missingness = df.isna().mean().sort_values(ascending=False)
    missingness.to_csv(OUTPUTS_EDA / "missingness_by_column.csv")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols + ["rushing_yards"]].corr()
    corr.to_csv(OUTPUTS_EDA / "correlation_matrix.csv")

    target_corr = corr["rushing_yards"].drop("rushing_yards").sort_values(key=abs, ascending=False)
    target_corr.head(25).to_csv(OUTPUTS_EDA / "top_feature_target_correlations.csv")

    engine = get_engine()
    sweep_seasons = sorted(df["season"].unique().tolist())[:2] or [df["season"].min()]
    sweep = _eligibility_sensitivity_sweep(engine, sweep_seasons)
    sweep.to_csv(OUTPUTS_EDA / "eligibility_threshold_sensitivity.csv", index=False)
    logger.info(f"Eligibility threshold sensitivity sweep (seasons={sweep_seasons}):\n{sweep}")

    logger.info(f"EDA outputs written to {OUTPUTS_EDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

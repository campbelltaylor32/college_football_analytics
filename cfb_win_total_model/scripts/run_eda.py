#!/usr/bin/env python
"""Step 4 of the pipeline. Exploratory analysis of the assembled modeling table, run from a
script (not only notebooks, per the project requirements). Writes plots/tables to
outputs/eda/.

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

from cfb_win_total_model.database import get_engine
from cfb_win_total_model.dataset import NON_FEATURE_COLS
from cfb_win_total_model.modeling.baselines import _team_conference_by_season
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_EDA, ensure_dirs

logger = get_logger(__name__)
DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

KEY_SCATTER_FEATURES = [
    ("prior_season_wins", "Prior-year wins vs next-year wins"),
    ("talent", "Talent composite vs wins"),
    ("returning_percent_ppa", "Returning production vs wins"),
    ("avg_opponent_prior_win_pct", "Schedule strength vs wins"),
    ("net_transfer_talent", "Transfer activity (net talent) vs wins"),
]


def main() -> int:
    ensure_dirs()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found -- run scripts/build_modeling_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"EDA on modeling dataset {df.shape}")

    df["regular_season_wins"].describe().to_csv(OUTPUTS_EDA / "wins_distribution_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["regular_season_wins"], bins=range(0, 16))
    ax.set_xlabel("Regular season wins")
    ax.set_title("Distribution of regular-season wins")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "wins_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="regular_season_wins", by="season", ax=ax, rot=90)
    ax.set_title("Wins by season")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "wins_by_season.png")
    plt.close(fig)

    engine = get_engine()
    conf_map = _team_conference_by_season(engine, sorted(df["season"].unique().tolist()))
    df_conf = df.merge(conf_map, on=["school", "season"], how="left")
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df_conf.groupby("conference")["regular_season_wins"].median().sort_values(ascending=False).index
    df_conf.boxplot(column="regular_season_wins", by="conference", ax=ax, rot=90)
    ax.set_title("Wins by conference")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "wins_by_conference.png")
    plt.close(fig)

    teams_by_season = df.groupby("season")["school"].nunique()
    teams_by_season.to_csv(OUTPUTS_EDA / "teams_by_season.csv")

    missingness = df.isna().mean().sort_values(ascending=False)
    missingness.to_csv(OUTPUTS_EDA / "missingness_by_column.csv")
    fig, ax = plt.subplots(figsize=(8, 10))
    top_missing = missingness.head(30)
    ax.barh(top_missing.index[::-1], top_missing.values[::-1])
    ax.set_title("Top 30 columns by missingness rate")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "missingness_top30.png")
    plt.close(fig)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols + ["regular_season_wins"]].corr()
    corr.to_csv(OUTPUTS_EDA / "correlation_matrix.csv")

    target_corr = corr["regular_season_wins"].drop("regular_season_wins").sort_values(key=abs, ascending=False)
    target_corr.head(25).to_csv(OUTPUTS_EDA / "top_feature_target_correlations.csv")

    fig, ax = plt.subplots(figsize=(8, 8))
    top20_cols = target_corr.head(20).index.tolist()
    im = ax.imshow(corr.loc[top20_cols, top20_cols], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top20_cols)))
    ax.set_xticklabels(top20_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top20_cols)))
    ax.set_yticklabels(top20_cols, fontsize=7)
    fig.colorbar(im)
    ax.set_title("Correlation matrix (top 20 features by |corr| with wins)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "correlation_heatmap_top20.png")
    plt.close(fig)

    for col, title in KEY_SCATTER_FEATURES:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[col], df["regular_season_wins"], alpha=0.4)
        ax.set_xlabel(col)
        ax.set_ylabel("regular_season_wins")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUTPUTS_EDA / f"scatter_{col}_vs_wins.png")
        plt.close(fig)

    if "coaching_change_indicator" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        df.boxplot(column="regular_season_wins", by="coaching_change_indicator", ax=ax)
        ax.set_title("Wins by coaching-change indicator")
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(OUTPUTS_EDA / "wins_by_coaching_change.png")
        plt.close(fig)

    outlier_cols = ["school", "season", "regular_season_wins"] + [c for c, _ in KEY_SCATTER_FEATURES if c in df.columns]
    z = (df["regular_season_wins"] - df["regular_season_wins"].mean()) / df["regular_season_wins"].std()
    outliers = df.loc[z.abs() > 2, outlier_cols].assign(win_zscore=z[z.abs() > 2])
    outliers.sort_values("win_zscore", key=abs, ascending=False).to_csv(OUTPUTS_EDA / "win_outliers.csv", index=False)

    stability_cols = ["talent", "off_epa_per_play", "points_per_game"]
    stability = df.groupby("season")[[c for c in stability_cols if c in df.columns]].mean()
    stability.to_csv(OUTPUTS_EDA / "temporal_stability_by_season.csv")

    coverage_by_year = df.groupby("season")[feature_cols].apply(lambda g: g.notna().mean())
    coverage_by_year.to_csv(OUTPUTS_EDA / "feature_coverage_by_season.csv")

    era_col = "talent"
    if era_col in df.columns:
        df["era"] = df["season"].apply(lambda s: "2015-2019" if s < 2020 else "2021-2025")
        era_compare = df.groupby("era")[["regular_season_wins"] + [c for c in stability_cols if c in df.columns]].mean()
        era_compare.to_csv(OUTPUTS_EDA / "era_comparison.csv")

    logger.info(f"EDA outputs written to {OUTPUTS_EDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

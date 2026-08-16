#!/usr/bin/env python
"""Class balance, missingness, per-season game counts, and a correlation heatmap of the
home-side feature columns (visualizing the verified 3x-temporal + offense/defense-mirror
redundancy Stage 1 correlation pruning targets) -> outputs/eda/."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cfb_spread_model.config import load_data_config
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_EDA, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def main() -> int:
    ensure_dirs()
    cfg = load_data_config()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)

    season_counts = df.groupby("season").size().rename("n_games")
    season_counts.to_csv(OUTPUTS_EDA / "games_per_season.csv")

    class_balance = df[cfg.label_column].value_counts(normalize=True).rename("fraction")
    class_balance.to_csv(OUTPUTS_EDA / "class_balance.csv")

    missingness = df.isna().sum()
    missingness = missingness[missingness > 0]
    missingness.to_csv(OUTPUTS_EDA / "missingness.csv")

    home_cols = [c for c in df.columns if c.startswith("home_") and pd.api.types.is_numeric_dtype(df[c])][:150]
    corr = df[home_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pairwise correlation, first 150 home_* feature columns\n(visualizes the 3x temporal-transform redundancy)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA / "home_columns_correlation_heatmap.png", dpi=120)
    plt.close(fig)

    logger.info(f"EDA outputs written -> {OUTPUTS_EDA}")
    logger.info(f"Season counts:\n{season_counts}")
    logger.info(f"Class balance:\n{class_balance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

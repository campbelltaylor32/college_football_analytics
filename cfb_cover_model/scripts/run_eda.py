#!/usr/bin/env python
"""Stage 1.5: lightweight EDA on the cleaned modeling frame - class balance, games per
season, and a correlation heatmap of a sample of feature columns (the full ~350-970
column matrix isn't legible as a single heatmap, so this samples down for readability)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "eda"


def main() -> None:
    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class_balance = frame["home_covered"].value_counts().rename_axis("home_covered").reset_index(name="n")
    class_balance.to_csv(OUT_DIR / "class_balance.csv", index=False)

    games_per_season = frame["season"].value_counts().sort_index().rename_axis("season").reset_index(name="n_games")
    games_per_season.to_csv(OUT_DIR / "games_per_season.csv", index=False)

    missingness = frame[feature_columns].isna().mean().rename("frac_missing").reset_index()
    missingness.columns = ["column", "frac_missing"]
    missingness.to_csv(OUT_DIR / "missingness.csv", index=False)

    sample_cols = feature_columns[:60] if len(feature_columns) > 60 else feature_columns
    corr = frame[sample_cols].corr(method="spearman")
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, xticklabels=False, yticklabels=False)
    plt.title(f"Spearman correlation, first {len(sample_cols)} candidate features")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sample_columns_correlation_heatmap.png", dpi=100)
    plt.close()

    print(f"class_balance:\n{class_balance}\n")
    print(f"games_per_season:\n{games_per_season}\n")
    print(f"Wrote EDA outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

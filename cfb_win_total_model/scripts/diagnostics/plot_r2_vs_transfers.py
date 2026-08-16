#!/usr/bin/env python
"""One-off plot: out-of-sample R^2 by season (gradient_boosting, walk-forward OOF for
2019-2024 + the true 2025 holdout) alongside average transfers per team by season, to visualize
the correlation between the accelerating transfer-portal era and declining OOF predictability
(see docs/diagnostics_compression_report.md's "what's different about 2025" discussion).

Two stacked panels sharing a season x-axis, NOT a dual-axis chart -- R^2 and transfer counts are
different scales/units, so they get separate panels rather than two y-axes on one plot.

Usage:
    python scripts/diagnostics/plot_r2_vs_transfers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS, ensure_dirs

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

# Out-of-sample R^2 for gradient_boosting: 2019-2024 from walk-forward OOF
# (outputs/diagnostics_compression/tables/season_breakdown_oof.csv), 2025 from the true final
# holdout (outputs/model_comparison/holdout_2025_results.csv) -- both genuinely out-of-sample,
# never trained on the season being scored.
SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
OOF_R2 = [0.575, 0.541, 0.510, 0.629, 0.325, 0.215]

BLUE = "#2a78d6"
ORANGE = "#eb6834"


def main() -> int:
    ensure_dirs()
    df = pd.read_parquet(DATASET_PATH)
    g = df[df["season"].isin(SEASONS)].groupby("season").apply(
        lambda s: (s["n_transferred_in"] + s["n_transferred_out"]).mean(), include_groups=False
    )
    avg_transfers = [g.loc[s] for s in SEASONS]

    x = range(len(SEASONS))
    labels = [str(s) for s in SEASONS]

    fig, (ax_r2, ax_transfers) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_r2.plot(x, OOF_R2, color=BLUE, linewidth=2, marker="o", markersize=8)
    ax_r2.set_ylabel("Out-of-sample R²")
    ax_r2.set_title("Out-of-sample R² by season (gradient_boosting)")
    ax_r2.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax_r2.spines[["top", "right"]].set_visible(False)
    for xi, yi, season in zip(x, OOF_R2, SEASONS):
        if season in (2024, 2025):
            ax_r2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=BLUE)

    ax_transfers.plot(x, avg_transfers, color=ORANGE, linewidth=2, marker="o", markersize=8)
    ax_transfers.set_ylabel("Avg. transfers per team")
    ax_transfers.set_title("Average transfers in + out per team, by season")
    ax_transfers.set_xlabel("Season (2020 excluded, COVID-shortened)")
    ax_transfers.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax_transfers.spines[["top", "right"]].set_visible(False)
    for xi, yi, season in zip(x, avg_transfers, SEASONS):
        if season in (2019, 2025):
            ax_transfers.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=ORANGE)

    ax_transfers.set_xticks(list(x))
    ax_transfers.set_xticklabels(labels)

    fig.suptitle("Predictability has declined as transfer-portal churn has risen", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "oof_r2_vs_transfers_by_season.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

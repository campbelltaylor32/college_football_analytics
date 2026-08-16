#!/usr/bin/env python
"""Renders the walk-forward fold table (train seasons, # train seasons, OOF R^2) as a PNG --
the same table used in the "out-of-sample R^2 is declining despite growing training data"
discussion in docs/project_story.md.

Usage:
    python scripts/diagnostics/plot_fold_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cfb_win_total_model.utils.paths import OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS, ensure_dirs

BLUE = "#2a78d6"
ORANGE = "#eb6834"
LIGHT_ORANGE_BG = "#fdece3"

ROWS = [
    ("2019", "2015-2018", "4", "0.575"),
    ("2021", "2015-2019", "5", "0.541"),
    ("2022", "2015-2019, 2021", "6", "0.510"),
    ("2023", "2015-2019, 2021-2022", "7", "0.629"),
    ("2024", "2015-2019, 2021-2023", "8", "0.325"),
    ("2025", "2015-2019, 2021-2024", "9", "0.215"),
]
HIGHLIGHT_FOLDS = {"2024", "2025"}
COLUMNS = ["Fold", "Train seasons", "# train seasons", "OOF R²"]


def main() -> int:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.axis("off")

    table = ax.table(
        cellText=ROWS,
        colLabels=COLUMNS,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    table.auto_set_column_width(col=list(range(len(COLUMNS))))

    n_cols = len(COLUMNS)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", weight="bold")
        else:
            fold_label = ROWS[row - 1][0]
            if fold_label in HIGHLIGHT_FOLDS:
                cell.set_facecolor(LIGHT_ORANGE_BG)
                if col == n_cols - 1:
                    cell.set_text_props(color=ORANGE, weight="bold")
            else:
                cell.set_facecolor("white")

    ax.set_title(
        "Walk-forward folds: more training data, worse out-of-sample R²",
        fontsize=12,
        pad=14,
    )

    out_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "walk_forward_fold_table.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

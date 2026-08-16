#!/usr/bin/env python
"""Static, shareable PNG version of the 2025 over/under-performer leaderboard -- reads the
CSV scripts/analyze_2025_over_under_performers.py already wrote (ranked by
avg_residual_per_game, yards over/under the model's per-game prediction), no re-computation.

Usage:
    python scripts/plot_2025_over_under_performers.py [--min-games 3] [--n 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import OUTPUTS_DIAGNOSTICS, ensure_dirs

logger = get_logger(__name__)

IN_PATH = OUTPUTS_DIAGNOSTICS / "rb_2025_over_under_performers.csv"
OUT_PATH = OUTPUTS_DIAGNOSTICS / "rb_2025_over_under_performers.png"

# Dark "scouting report" palette -- matches the companion interactive artifact's dark theme.
BG = "#12100c"
CARD = "#1b1913"
INK = "#f2efe6"
INK_SECONDARY = "#c3c0b3"
INK_MUTED = "#8f8c80"
HAIRLINE = "#322f26"
ACCENT = "#d9a94a"
OVER = "#3987e5"
UNDER = "#e66767"


def _rounded_bar(ax, y, x0, x1, height, color, radius=0.35):
    """A horizontal bar rounded at the data end, square at the baseline (x0)."""
    left = min(x0, x1)
    width = abs(x1 - x0)
    if width == 0:
        return
    box = FancyBboxPatch(
        (left, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, mutation_aspect=1,
    )
    ax.add_patch(box)
    # Square off the baseline-side corners by overpainting a thin rect at that edge.
    square_w = min(radius * 1.4, width)
    if x1 >= x0:
        ax.add_patch(plt.Rectangle((x0, y - height / 2), square_w, height, facecolor=color, linewidth=0))
    else:
        ax.add_patch(plt.Rectangle((x0 - square_w, y - height / 2), square_w, height, facecolor=color, linewidth=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--n", type=int, default=12, help="Rows per direction (over/under)")
    args = parser.parse_args()

    ensure_dirs()
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found -- run scripts/analyze_2025_over_under_performers.py first")

    df = pd.read_csv(IN_PATH)
    df = df[df["n_games"] >= args.min_games].sort_values("avg_residual_per_game", ascending=False)
    top = df.head(args.n)
    bottom = df.tail(args.n).sort_values("avg_residual_per_game", ascending=False)
    rows = pd.concat([top, bottom]).reset_index(drop=True)
    n_rows = len(rows)

    plt.rcParams["font.family"] = [f.name for f in fm.fontManager.ttflist if "DejaVu Sans" in f.name][:1] or ["sans-serif"]

    fig_w, fig_h = 10.8, 13.5  # 1080x1350 @ 100dpi -- Twitter-friendly portrait stat card
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100, facecolor=BG)

    # --- Header block ---
    header_h = 0.145
    ax_head = fig.add_axes([0, 1 - header_h, 1, header_h])
    ax_head.set_axis_off()
    ax_head.set_facecolor(BG)
    ax_head.text(0.055, 0.74, "2 0 2 5   R U N N I N G   B A C K S     ·     W E E K S   1 – 8", transform=ax_head.transAxes,
                 fontsize=11.5, fontweight="bold", color=ACCENT, ha="left", va="center", fontfamily="monospace")
    ax_head.text(0.055, 0.46, "Who beat the model — and who it won", transform=ax_head.transAxes,
                 fontsize=27, fontweight="bold", color=INK, ha="left", va="center")
    ax_head.text(0.055, 0.14, f"Rushing yards per game vs. prediction, min {args.min_games} qualifying games  ·  cfb_rb_rushing_model",
                 transform=ax_head.transAxes, fontsize=11.5, color=INK_SECONDARY, ha="left", va="center")

    # --- Chart ---
    chart_bottom, chart_top = 0.065, 1 - header_h - 0.012
    ax = fig.add_axes([0.055, chart_bottom, 0.90, chart_top - chart_bottom], facecolor=BG)

    max_abs = max(rows["avg_residual_per_game"].abs().max(), 10)
    max_abs = (int(max_abs / 10) + 1) * 10

    bar_h = 0.62
    gap_between_groups = 1.3

    y_positions = []
    y = 0
    for i in range(n_rows):
        if i == args.n:
            y += gap_between_groups
        y_positions.append(y)
        y += 1

    ax.set_xlim(-max_abs, max_abs)
    # Padding accounts for the section-label row above each group and the mid-group gap --
    # computed from the actual y_positions, not just n_rows, since the gap shifts everything
    # below it down by gap_between_groups.
    ax.set_ylim(y_positions[-1] + 0.9, y_positions[0] - 1.15)
    ax.set_axis_off()

    ax.axvline(0, color=HAIRLINE, linewidth=1.1, zorder=1)

    for i, (_, r) in enumerate(rows.iterrows()):
        yy = y_positions[i]
        val = r["avg_residual_per_game"]
        is_pos = val >= 0
        color = OVER if is_pos else UNDER
        _rounded_bar(ax, yy, 0, val, bar_h, color)

        name = r["player_name"]
        team = r["team"]
        ax.text(-max_abs * 0.012 if is_pos else max_abs * 0.012, yy - 0.135,
                name, fontsize=12.3, fontweight="bold", color=INK,
                ha="right" if is_pos else "left", va="center", zorder=3)
        ax.text(-max_abs * 0.012 if is_pos else max_abs * 0.012, yy + 0.155,
                f"{team}  ·  {int(r['n_games'])} gm", fontsize=9.6, color=INK_MUTED,
                ha="right" if is_pos else "left", va="center", zorder=3)

        val_txt = f"+{val:.1f}" if is_pos else f"−{abs(val):.1f}"
        pad = max_abs * 0.014
        ax.text(val + (pad if is_pos else -pad), yy, val_txt, fontsize=12,
                fontweight="bold", color=OVER if is_pos else UNDER,
                ha="left" if is_pos else "right", va="center", zorder=3)

    # Section labels -- one row's worth of clearance above each group's first bar
    ax.text(0, y_positions[0] - 0.85, "RAN AHEAD OF SCHEDULE", fontsize=10.5, fontweight="bold",
            color=OVER, ha="center", va="center", alpha=0.9)
    ax.text(0, y_positions[args.n] - 0.85, "FELL BEHIND SCHEDULE", fontsize=10.5,
            fontweight="bold", color=UNDER, ha="center", va="center", alpha=0.9)

    # --- Footer (figure-level, well clear of the last bar row) ---
    fig.text(0.055, 0.022, "Source: outputs/model_comparison/holdout_predictions.csv  ·  OLS model, walk-forward validated  ·  yds/game, actual − predicted",
              fontsize=8.4, color=INK_MUTED, ha="left", va="bottom")
    fig.text(0.055, 0.005, "⚠ low-actual misses often mean injury or role loss the model has no data for, not a bad prediction",
              fontsize=8.4, color=INK_MUTED, ha="left", va="bottom", style="italic")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(OUT_PATH, facecolor=BG, dpi=100)
    plt.close(fig)
    logger.info(f"Wrote {OUT_PATH} ({fig_w*100:.0f}x{fig_h*100:.0f}px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

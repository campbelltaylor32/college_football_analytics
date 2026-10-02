"""Charts for the predictability rankings -> outputs/figures/.

  <target>_extremes.png       15 least / 15 most predictable offenses, with 90% bootstrap CIs
  <target>_pass_rate_vs_predictability.png   is "unpredictable" just "balanced"? (it shouldn't be)
  stability.png               team score in one season vs the next

Usage: python scripts/plot_predictability.py   (after scripts/rank_predictability.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

from cfb_playcall_predictability.config import load_modeling_config
from cfb_playcall_predictability.utils.paths import OUTPUTS_FIGURES, OUTPUTS_RANKINGS, ensure_output_dirs

# Reference data-viz palette (light mode): slot 1 blue, slot 2 orange, recessive ink/grid.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LEAST, MOST = "#2a78d6", "#eb6834"
N_EXTREME = 15
N_SCATTER_LABELS = 6

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _label(row) -> str:
    return f"{row['team']} ({row['head_coach']})" if isinstance(row.get("head_coach"), str) else row["team"]


def plot_extremes(table: pd.DataFrame, season: int) -> None:
    least = table.head(N_EXTREME)
    most = table.tail(N_EXTREME).iloc[::-1]
    lo = min(table["predictability_lo"].min(), 0) - 0.02
    hi = table["predictability_hi"].max() + 0.02

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True)
    for ax, df, color, title in [
        (axes[0], least, LEAST, f"Least predictable ({N_EXTREME})"),
        (axes[1], most, MOST, f"Most predictable ({N_EXTREME})"),
    ]:
        y = range(len(df))[::-1]
        ax.hlines(y, df["predictability_lo"], df["predictability_hi"], color=color, linewidth=2, alpha=0.45)
        ax.scatter(df["predictability"], y, s=50, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_yticks(list(y), [_label(r) for _, r in df.iterrows()])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.axvline(table["predictability"].median(), color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlabel("Predictability score")
        ax.text(table["predictability"].median(), len(df) - 0.4, " FBS median", color=INK_2, fontsize=8, va="bottom")
    fig.text(0.01, 0.01, "Predictability = share of a team's run/pass uncertainty explained by situation, "
             "tendencies and game flow (1 - model log loss / log loss of guessing the team's own pass rate).",
             color=INK_2, fontsize=8)
    fig.suptitle(f"{season} FBS play-calling predictability, run vs pass "
                 f"(through week {int(table['through_week'].iat[0])}; bars = 90% CI)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUTPUTS_FIGURES / f"{season}_extremes.png", dpi=160)
    plt.close(fig)


def _place_labels(ax, labeled: pd.DataFrame, table: pd.DataFrame) -> None:
    """Greedy label placement: try right/left x above/below each point, keep the first spot whose
    (approximate, data-unit) box doesn't overlap a label already placed or run off the left edge."""
    x_span = table["pass_rate"].max() - table["pass_rate"].min()
    y_span = table["predictability"].max() - table["predictability"].min()
    char_w, box_h = 0.0075 * x_span, 0.045 * y_span
    x_min = table["pass_rate"].min()
    # Highlighted dots are obstacles too, so a label never sits on another team's point.
    pad_x, pad_y = 0.006 * x_span, 0.012 * y_span
    placed: list[tuple[float, float, float, float]] = [
        (px - pad_x, px + pad_x, py - pad_y, py + pad_y)
        for px, py in zip(labeled["pass_rate"], labeled["predictability"])
    ]
    for _, r in labeled.iterrows():
        x, y, w = r["pass_rate"], r["predictability"], len(r["team"]) * char_w
        for dx, dy, ha in [(6, 5, "left"), (6, -11, "left"), (-6, 5, "right"), (-6, -11, "right")]:
            x0 = x + 0.01 * x_span if ha == "left" else x - 0.01 * x_span - w
            y0 = y + (0.01 if dy > 0 else -0.05) * y_span
            box = (x0, x0 + w, y0, y0 + box_h)
            own_dot = (x - pad_x, x + pad_x, y - pad_y, y + pad_y)
            clash = any(not (box[1] < b[0] or box[0] > b[1] or box[3] < b[2] or box[2] > b[3])
                        for b in placed if b != own_dot)
            if not clash and box[0] >= x_min - 0.02 * x_span:
                break
        placed.append(box)
        ax.annotate(r["team"], (x, y), xytext=(dx, dy), ha=ha, textcoords="offset points", fontsize=8, color=INK)


def plot_pass_rate(table: pd.DataFrame, season: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(table["pass_rate"], table["predictability"], s=36, color=INK_2, alpha=0.35, edgecolor="none")
    for df, color in [(table.head(N_SCATTER_LABELS), LEAST), (table.tail(N_SCATTER_LABELS), MOST)]:
        ax.scatter(df["pass_rate"], df["predictability"], s=50, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    _place_labels(ax, pd.concat([table.head(N_SCATTER_LABELS), table.tail(N_SCATTER_LABELS)]), table)
    r = table["pass_rate"].corr(table["predictability"])
    ax.set_xlabel(f"{season} pass rate (non-garbage-time calls)")
    ax.set_ylabel("Predictability score")
    ax.set_title(f"Predictability vs run/pass mix (r = {r:.2f}). The score is relative to each team's own mix,\n"
                 f"so a 50/50 offense isn't automatically 'unpredictable'.", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUTS_FIGURES / f"{season}_pass_rate_vs_predictability.png", dpi=160)
    plt.close(fig)


def plot_stability(long: pd.DataFrame, stability: pd.DataFrame) -> None:
    pairs = stability.tail(2)
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.5), squeeze=False)
    wide = long.pivot(index="team", columns="season", values="predictability")
    for ax, (_, p) in zip(axes[0], pairs.iterrows()):
        a, b = int(p["season"]), int(p["next_season"])
        d = wide[[a, b]].dropna()
        ax.scatter(d[a], d[b], s=30, color=LEAST, alpha=0.7, edgecolor=SURFACE, linewidth=1)
        ax.set_xlabel(f"{a} predictability")
        ax.set_ylabel(f"{b} predictability")
        partial = " (partial season)" if p.get("next_season_partial", False) else ""
        ax.set_title(f"{a} -> {b}{partial}: r = {p['pearson']:.2f} ({int(p['teams'])} teams)", loc="left", fontsize=11)
    fig.suptitle("Is predictability a stable trait of an offense?", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUTS_FIGURES / "stability.png", dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = load_modeling_config()
    ensure_output_dirs()
    table = pd.read_csv(OUTPUTS_RANKINGS / f"{cfg.target_season}_predictability.csv")
    plot_extremes(table, cfg.target_season)
    plot_pass_rate(table, cfg.target_season)
    plot_stability(pd.read_csv(OUTPUTS_RANKINGS / "team_season_predictability.csv"),
                   pd.read_csv(OUTPUTS_RANKINGS / "stability.csv"))


if __name__ == "__main__":
    main()

"""Render a week's agreement-bet selections (both models flag home cover) as a PNG card.

Only games with a posted spread are shown. The prediction CSV stores |spread|, so the
signed line (negative = home favored) is recovered from the cached CFBD betting lines
under data/raw/betting_lines/ by averaging providers - the same averaging that produced
the prediction file's spread.

Usage:
    python scripts/plot_week_selections.py --season 2026 --week 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
LINES_DIR = ROOT / "data" / "raw" / "betting_lines"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e4e3df"
BAR = "#2a78d6"
BAR_TRACK = "#eeede9"


def load_selections(season: int, week: int) -> pd.DataFrame:
    preds = pd.read_csv(PRED_DIR / f"live_{season}_week_{week}_dual_model_predictions.csv")
    picks = preds[preds["agreement_bet"] & preds["spread"].notna()].copy()

    lines_path = LINES_DIR / f"{season}_{week:02d}.parquet"
    if lines_path.exists():
        lines = pd.read_parquet(lines_path)
        signed = lines.groupby("game_id")["spread"].mean().rename("signed_spread")
        picks = picks.merge(signed, on="game_id", how="left")
    else:
        picks["signed_spread"] = pd.NA
    return picks.sort_values("avg_probability", ascending=False).reset_index(drop=True)


def format_line(row: pd.Series) -> str:
    s = row["signed_spread"]
    if pd.isna(s):
        return f"{row['spread']:g}"
    # Home-team perspective: negative = home laying points
    return "PK" if s == 0 else f"{s:+g}"


def render(picks: pd.DataFrame, season: int, week: int, out_path: Path) -> None:
    n = len(picks)
    row_h = 0.62
    fig_h = 1.9 + n * row_h + 0.7
    fig = plt.figure(figsize=(10, fig_h), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.45
    ax.text(0.4, top, f"Week {week} Cover Picks · {season}", fontsize=19, fontweight="bold",
            color=TEXT_PRIMARY, va="top")
    ax.text(0.4, top - 0.52,
            "Home team to cover · both models agree (logistic regression + XGBoost regressor) · spread posted",
            fontsize=9.5, color=TEXT_SECONDARY, va="top")

    # Column layout
    x_pick, x_opp, x_line, x_lr, x_xgb, x_bar = 0.4, 2.45, 4.6, 5.6, 6.5, 7.35
    bar_w = 1.55
    header_y = top - 1.12
    for x, label, ha in [(x_pick, "PICK (HOME)", "left"), (x_opp, "OPPONENT", "left"), (x_line, "LINE", "center"),
                         (x_lr, "LOGIT", "center"), (x_xgb, "XGB", "center"),
                         (x_bar, "AVG COVER PROB.", "left")]:
        ax.text(x, header_y, label, fontsize=8, color=TEXT_MUTED, fontweight="bold", ha=ha, va="center")
    ax.plot([0.4, 9.6], [header_y - 0.22, header_y - 0.22], color=GRID, lw=1)

    for i, row in picks.iterrows():
        y = header_y - 0.62 - i * row_h
        ax.text(x_pick, y, row["home_team"], fontsize=12.5, fontweight="bold", color=TEXT_PRIMARY, va="center")
        ax.text(x_opp, y, row["away_team"],
                fontsize=10, color=TEXT_SECONDARY, va="center")
        ax.text(x_line, y, format_line(row), fontsize=12, fontweight="bold", color=TEXT_PRIMARY,
                ha="center", va="center")
        ax.text(x_lr, y, f"{row['logistic_regression_probability']:.0%}", fontsize=10.5,
                color=TEXT_SECONDARY, ha="center", va="center")
        ax.text(x_xgb, y, f"{row['xgboost_regressor_probability']:.0%}", fontsize=10.5,
                color=TEXT_SECONDARY, ha="center", va="center")

        p = row["avg_probability"]
        ax.add_patch(plt.Rectangle((x_bar, y - 0.09), bar_w, 0.18, color=BAR_TRACK, lw=0))
        ax.add_patch(plt.Rectangle((x_bar, y - 0.09), bar_w * p, 0.18, color=BAR, lw=0))
        ax.text(x_bar + bar_w + 0.1, y, f"{p:.1%}", fontsize=11, fontweight="bold",
                color=TEXT_PRIMARY, va="center")
        if i < n - 1:
            ax.plot([0.4, 9.6], [y - row_h / 2, y - row_h / 2], color=GRID, lw=0.6)

    version = picks["artifact_version"].iloc[0] if n else ""
    ax.text(0.4, 0.3, f"Line = consensus spread from home team's perspective (− = home favored).  Model artifact {version}",
            fontsize=7.5, color=TEXT_MUTED, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    picks = load_selections(args.season, args.week)
    out_path = PRED_DIR / f"live_{args.season}_week_{args.week}_selections.png"
    render(picks, args.season, args.week, out_path)
    print(picks[["home_team", "away_team", "signed_spread", "avg_probability"]].to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

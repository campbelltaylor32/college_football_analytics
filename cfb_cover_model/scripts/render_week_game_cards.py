"""Render every game in a week's dual-model prediction file as its own PNG card, and bundle
them into a single PDF (summary table page + one card per page).

Unlike plot_week_selections.py (agreement bets only), this covers the full slate, including
games without a posted spread. Signed lines / totals come from the cached CFBD betting lines
(averaged across providers) and kickoff/conference info from the cached CFBD games file.

Usage:
    python scripts/render_week_game_cards.py --season 2026 --week 5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
RAW_DIR = ROOT / "data" / "raw"
THRESHOLD_PATH = ROOT / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e4e3df"
BAR = "#2a78d6"
BAR_TRACK = "#eeede9"
BET = "#1f8a4c"
LEAN = "#c98a14"
PASS = "#8a8984"
AWAY = "#7c3aed"

MODELS = [("logistic_regression", "Logistic regression"), ("xgboost_regressor", "XGBoost regressor")]
TIER_RANK = {"BET": 0, "AWAY": 1, "LEAN": 2, "PASS": 3, "NO LINE": 4}


def load_thresholds() -> dict[str, float]:
    tdf = pd.read_csv(THRESHOLD_PATH).set_index("model_name")["threshold"]
    return {key: float(tdf[key]) for key, _ in MODELS}


def load_week(season: int, week: int) -> pd.DataFrame:
    preds = pd.read_csv(PRED_DIR / f"live_{season}_week_{week}_dual_model_predictions.csv")

    lines_path = RAW_DIR / "betting_lines" / f"{season}_{week:02d}.parquet"
    if lines_path.exists():
        lines = pd.read_parquet(lines_path)
        agg = lines.groupby("game_id").agg(signed_spread=("spread", "mean"), over_under=("over_under", "mean"),
                                           n_books=("provider", "nunique"))
        preds = preds.merge(agg, on="game_id", how="left")
    else:
        preds[["signed_spread", "over_under", "n_books"]] = pd.NA

    games_path = RAW_DIR / "games" / f"{season}_{week:02d}.parquet"
    if games_path.exists():
        games = pd.read_parquet(games_path)[["game_id", "start_date", "neutral_site", "home_conference",
                                             "away_conference"]]
        preds = preds.merge(games, on="game_id", how="left")
    else:
        preds[["start_date", "neutral_site", "home_conference", "away_conference"]] = pd.NA

    thresholds = load_thresholds()
    preds["tier"] = preds.apply(lambda r: tier_of(r, thresholds), axis=1)
    preds["tier_rank"] = preds["tier"].map(TIER_RANK)
    # Most confident first within each tier: highest home-cover prob, except AWAY where lowest is strongest
    preds["confidence"] = preds["avg_probability"].where(preds["tier"] != "AWAY", 1 - preds["avg_probability"])
    return preds.sort_values(["tier_rank", "confidence"], ascending=[True, False]).reset_index(drop=True)


def tier_of(row: pd.Series, thresholds: dict[str, float]) -> str:
    """BET = both models clear their home-cover threshold. AWAY = the mirror image: both models are at or
    below (1 - threshold), i.e. equally confident the home side does NOT cover, so the play is the away team."""
    if pd.isna(row["spread"]):
        return "NO LINE"
    if row["agreement_bet"]:
        return "BET"
    if row["logistic_regression_flag"] or row["xgboost_regressor_flag"]:
        return "LEAN"
    if all(row[f"{key}_probability"] <= 1 - thresholds[key] for key, _ in MODELS):
        return "AWAY"
    return "PASS"


def format_line(row: pd.Series) -> str:
    s = row["signed_spread"]
    if pd.isna(s):
        return "—" if pd.isna(row["spread"]) else f"{row['spread']:g}"
    # Home-team perspective: negative = home laying points
    return "PK" if s == 0 else f"{s:+g}"


def away_line(row: pd.Series) -> str:
    s = row["signed_spread"]
    if pd.isna(s):
        return "—" if pd.isna(row["spread"]) else f"{row['spread']:g}"
    return "PK" if s == 0 else f"{-s:+g}"


def format_kickoff(ts) -> str:
    if ts is None or pd.isna(ts):
        return "Kickoff TBD"
    et = pd.Timestamp(ts).tz_convert("America/New_York")
    return et.strftime("%a %b %-d · %-I:%M %p ET")


TIER_STYLE = {
    "BET": (BET, "BET · home covers", "Both models clear their threshold"),
    "AWAY": (AWAY, "AWAY · away covers", "Both models at or below the mirror of their threshold"),
    "LEAN": (LEAN, "LEAN · split models", "Only one model clears its threshold"),
    "PASS": (PASS, "NO PLAY", "Neither model clears its threshold"),
    "NO LINE": (PASS, "NO LINE", "No posted spread — not bettable"),
}


def render_card(row: pd.Series, thresholds: dict[str, float], season: int, week: int) -> plt.Figure:
    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.text(0.35, 4.18, f"{season} · WEEK {week}", fontsize=8, color=TEXT_MUTED, fontweight="bold", va="center")
    ax.text(7.65, 4.18, format_kickoff(row["start_date"]), fontsize=8, color=TEXT_MUTED, ha="right", va="center")

    sep = "vs" if row.get("neutral_site") is True else "@"
    ax.text(0.35, 3.72, f"{row['away_team']}  {sep}  {row['home_team']}", fontsize=17, fontweight="bold",
            color=TEXT_PRIMARY, va="center")
    confs = [c for c in (row.get("away_conference"), row.get("home_conference")) if isinstance(c, str)]
    sub = " vs ".join(confs) if confs else ""
    if row.get("neutral_site") is True:
        sub = (sub + " · neutral site").strip(" ·")
    ax.text(0.35, 3.38, sub, fontsize=9, color=TEXT_SECONDARY, va="center")

    # Verdict pill
    color, label, why = TIER_STYLE[row["tier"]]
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.35, 2.68), 2.6, 0.38, boxstyle="round,pad=0,rounding_size=0.12",
                                                   color=color, lw=0))
    ax.text(0.35 + 1.3, 2.87, label, fontsize=10.5, fontweight="bold", color="white", ha="center", va="center")
    ax.text(3.15, 2.87, why,
            fontsize=8.5, color=TEXT_SECONDARY, va="center")

    # Line / total / avg
    pick_line = format_line(row) if row["tier"] != "AWAY" else away_line(row)
    stats = [("HOME LINE" if row["tier"] != "AWAY" else f"PICK: {row['away_team'].upper()}", pick_line),
             ("TOTAL", "—" if pd.isna(row["over_under"]) else f"{row['over_under']:g}"),
             ("AVG HOME COVER PROB.", f"{row['avg_probability']:.1%}")]
    for (lab, val), x in zip(stats, (0.35, 2.35, 3.95)):
        ax.text(x, 2.32, lab, fontsize=7.5, color=TEXT_MUTED, fontweight="bold", va="center")
        ax.text(x, 2.0, val, fontsize=15, fontweight="bold", color=TEXT_PRIMARY, va="center")
    ax.plot([0.35, 7.65], [1.68, 1.68], color=GRID, lw=1)

    # Per-model bars with threshold tick
    bar_x, bar_w = 2.35, 4.4
    for i, (key, name) in enumerate(MODELS):
        y = 1.3 - i * 0.48
        p = row[f"{key}_probability"]
        t = thresholds[key]
        flagged = bool(row[f"{key}_flag"])
        faded = row["tier"] == "AWAY" and p <= 1 - t
        ax.text(0.35, y, name, fontsize=9.5, color=TEXT_SECONDARY, va="center")
        ax.add_patch(plt.Rectangle((bar_x, y - 0.09), bar_w, 0.18, color=BAR_TRACK, lw=0))
        ax.add_patch(plt.Rectangle((bar_x, y - 0.09), bar_w * p, 0.18, color=BET if flagged else AWAY if faded else BAR, lw=0))
        ticks = [(t, TEXT_PRIMARY)] + ([(1 - t, AWAY)] if row["tier"] == "AWAY" else [])
        for tx, tc in ticks:
            ax.plot([bar_x + bar_w * tx] * 2, [y - 0.16, y + 0.16], color=tc, lw=1.2)
            ax.text(bar_x + bar_w * tx, y + 0.2, f"{tx:.0%}", fontsize=6.5, color=tc, ha="center", va="bottom")
        ax.text(bar_x + bar_w + 0.12, y, f"{p:.1%}", fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY,
                va="center")

    ax.text(0.35, 0.2, f"Probability that {row['home_team']} covers. Line from home perspective (− = home favored). "
                       f"Tick = model threshold{' (purple = AWAY mirror)' if row['tier'] == 'AWAY' else ''}.  Artifact {row['artifact_version']}",
            fontsize=6.3, color=TEXT_MUTED, va="center")
    return fig


def render_summary(df: pd.DataFrame, season: int, week: int) -> plt.Figure:
    row_h = 0.205
    fig_h = 1.35 + len(df) * row_h + 0.35
    fig = plt.figure(figsize=(8.5, fig_h), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.35
    counts = df["tier"].value_counts()
    ax.text(0.4, top, f"Week {week} Cover Model · {season} — Full Slate", fontsize=15, fontweight="bold",
            color=TEXT_PRIMARY, va="top")
    ax.text(0.4, top - 0.36, f"{len(df)} games · {counts.get('BET', 0)} home bets · {counts.get('AWAY', 0)} away bets · {counts.get('LEAN', 0)} leans · "
                             f"{counts.get('PASS', 0)} no play · {counts.get('NO LINE', 0)} no line · sorted by tier, then model confidence",
            fontsize=8, color=TEXT_SECONDARY, va="top")

    cols = [(0.4, "AWAY @ HOME", "left"), (4.55, "LINE", "center"), (5.25, "LOGIT", "center"),
            (5.9, "XGB", "center"), (6.6, "AVG", "center"), (7.6, "TIER", "center")]
    hy = top - 0.78
    for x, lab, ha in cols:
        ax.text(x, hy, lab, fontsize=6.5, color=TEXT_MUTED, fontweight="bold", ha=ha, va="center")
    ax.plot([0.4, 8.1], [hy - 0.11, hy - 0.11], color=GRID, lw=0.8)

    for i, row in df.iterrows():
        y = hy - 0.3 - i * row_h
        color = TIER_STYLE[row["tier"]][0]
        weight = "bold" if row["tier"] in ("BET", "AWAY") else "normal"
        ax.text(0.4, y, f"{row['away_team']} @ {row['home_team']}", fontsize=7, color=TEXT_PRIMARY,
                fontweight=weight, va="center")
        ax.text(4.55, y, format_line(row), fontsize=7, color=TEXT_PRIMARY, ha="center", va="center")
        ax.text(5.25, y, f"{row['logistic_regression_probability']:.0%}", fontsize=7, color=TEXT_SECONDARY,
                ha="center", va="center")
        ax.text(5.9, y, f"{row['xgboost_regressor_probability']:.0%}", fontsize=7, color=TEXT_SECONDARY,
                ha="center", va="center")
        ax.text(6.6, y, f"{row['avg_probability']:.1%}", fontsize=7, color=TEXT_PRIMARY, fontweight="bold",
                ha="center", va="center")
        ax.text(7.6, y, row["tier"], fontsize=6.5, color=color, fontweight="bold", ha="center", va="center")
    return fig


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    thresholds = load_thresholds()

    df = load_week(args.season, args.week)
    png_dir = PRED_DIR / f"live_{args.season}_week_{args.week}_game_cards"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = PRED_DIR / f"live_{args.season}_week_{args.week}_full_predictions.pdf"

    with PdfPages(pdf_path) as pdf:
        fig = render_summary(df, args.season, args.week)
        fig.savefig(png_dir / "00_summary.png", facecolor=SURFACE)
        pdf.savefig(fig, facecolor=SURFACE)
        plt.close(fig)
        for i, row in df.iterrows():
            fig = render_card(row, thresholds, args.season, args.week)
            fig.savefig(png_dir / f"{i + 1:02d}_{slug(row['away_team'])}_at_{slug(row['home_team'])}.png",
                        facecolor=SURFACE)
            pdf.savefig(fig, facecolor=SURFACE)
            plt.close(fig)

    print(df["tier"].value_counts().to_string())
    print(f"Wrote {len(df)} game cards to {png_dir}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()

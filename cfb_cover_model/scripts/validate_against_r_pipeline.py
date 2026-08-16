#!/usr/bin/env python
"""Required before trusting the API-direct ingestion pipeline for anything real: computes
features via the new Python path for a week already covered by the R-generated historical
CSV, and diffs column-by-column against the real, already-trusted
../Data/CFB_Gambling_Predictors_Final_PBP.csv.

Uses build_current_week_rows(season, week) - which computes features from weeks 1..week-1
of history, exactly matching how the historical CSV's week==`week` rows were built (both
use the R live-pipeline's "current form heading into this game" snapshot) - so at a week
comfortably past the early-season fill-policy edge case (avg3 needs 3 prior games; week 8
gives every team 7 games of history), this is a fair, direct row-for-row comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.ingest import cfbd_client, pipeline

REAL_CSV = Path(__file__).resolve().parents[2] / "Data" / "CFB_Gambling_Predictors_Final_PBP.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "validation"


def main(season: int = 2024, week: int = 8, tolerance: float = 0.05) -> None:
    print(f"Validating season={season}, week={week} against {REAL_CSV}")

    real = pd.read_csv(REAL_CSV)
    real = real[(real["season"] == season) & (real["week"] == week)]
    print(f"Real CSV rows for this season/week: {len(real)}")

    with cfbd_client.get_client() as client:
        mine = pipeline.build_current_week_rows(client, season=season, week=week)
    print(f"Python-computed rows: {len(mine)}")

    merged = real.merge(mine, on="game_id", suffixes=("_real", "_mine"), how="inner")
    print(f"Matched games (by game_id): {len(merged)}")
    if merged.empty:
        print("No matching games - can't validate. Check game_id overlap and season/week choice.")
        return

    real_cols = set(real.columns) - {"game_id"}
    mine_cols = set(mine.columns) - {"game_id"}
    shared_cols = sorted(real_cols & mine_cols)
    only_real = sorted(real_cols - mine_cols)
    only_mine = sorted(mine_cols - real_cols)

    print(f"\nColumns in both: {len(shared_cols)}")
    print(f"Columns only in real CSV (missing from Python port): {len(only_real)}")
    print(f"Columns only in Python output (not in real CSV): {len(only_mine)}")

    rows = []
    for col in shared_cols:
        if col in ("home_team", "away_team", "season", "week"):
            continue
        # merge(..., suffixes=("_real","_mine")) only renames columns that collided
        # between the two frames - which, since shared_cols was computed from columns
        # present in *both* pre-merge frames, is every column reaching this loop.
        real_col, mine_col = merged.get(f"{col}_real"), merged.get(f"{col}_mine")
        if real_col is None or mine_col is None:
            continue
        real_vals = pd.to_numeric(real_col, errors="coerce") if real_col.dtype == object else real_col.astype(float) if real_col.dtype == bool else real_col
        mine_vals = pd.to_numeric(mine_col, errors="coerce") if mine_col.dtype == object else mine_col.astype(float) if mine_col.dtype == bool else mine_col
        if not pd.api.types.is_numeric_dtype(real_vals) or not pd.api.types.is_numeric_dtype(mine_vals):
            continue
        diff = (real_vals - mine_vals).abs()
        rows.append(
            {
                "column": col,
                "max_abs_diff": diff.max(),
                "mean_abs_diff": diff.mean(),
                "n_mismatched": int((diff > tolerance).sum()),
                "n_compared": int(diff.notna().sum()),
            }
        )
    report = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"validation_{season}_week{week}.csv"
    report.to_csv(out_path, index=False)

    pd.set_option("display.width", 140)
    print(f"\nTop 30 columns by max discrepancy (tolerance={tolerance}):")
    print(report.head(30).to_string(index=False))
    print(f"\nColumns with zero mismatches: {int((report['n_mismatched'] == 0).sum())} of {len(report)}")
    print(f"Only-in-real (first 30): {only_real[:30]}")
    print(f"Only-in-mine (first 30): {only_mine[:30]}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--week", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    main(args.season, args.week, args.tolerance)

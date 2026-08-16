#!/usr/bin/env python
"""Ad hoc analysis (not a pipeline stage, not wired into run_pipeline.py): ranks RBs by how
much their 2025 (weeks 1-8) actual rushing yards exceeded or fell short of the model's
prediction, using outputs/model_comparison/holdout_predictions.csv -- already the scored 2025
holdout, produced by scripts/evaluate_models.py. No re-scoring, no DB write.

Usage:
    python scripts/analyze_2025_over_under_performers.py [--min-games 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_rb_rushing_model.database import get_engine, run_query
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import OUTPUTS_DIAGNOSTICS, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)

HOLDOUT_PREDICTIONS_PATH = OUTPUTS_MODEL_COMPARISON / "holdout_predictions.csv"
OUT_PATH = OUTPUTS_DIAGNOSTICS / "rb_2025_over_under_performers.csv"


def _player_names(engine, athlete_ids: list, season: int) -> pd.DataFrame:
    """Same query pattern as scripts/generate_week_predictions.py's _player_names()."""
    if not athlete_ids:
        return pd.DataFrame(columns=["athlete_id", "player_name"])
    placeholders = ", ".join(f":a{i}" for i in range(len(athlete_ids)))
    params = {f"a{i}": aid for i, aid in enumerate(athlete_ids)}
    params["season"] = season
    sql = f"""
        SELECT DISTINCT athlete_id, CONCAT(first_name, ' ', last_name) AS player_name, team
        FROM team_rosters WHERE season = :season AND athlete_id IN ({placeholders})
    """
    return run_query(sql, params=params, engine=engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-games", type=int, default=3, help="Minimum 2025 holdout appearances to qualify for ranking")
    args = parser.parse_args()

    ensure_dirs()
    if not HOLDOUT_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"{HOLDOUT_PREDICTIONS_PATH} not found -- run scripts/evaluate_models.py first")

    preds = pd.read_csv(HOLDOUT_PREDICTIONS_PATH)
    preds = preds[preds["season"] == 2025]
    if preds.empty:
        raise ValueError("No season=2025 rows in holdout_predictions.csv -- is 2025 the configured final_holdout_season?")

    agg = preds.groupby(["athlete_id", "team"], as_index=False).agg(
        n_games=("game_id", "count"),
        total_actual=("y_true", "sum"),
        total_predicted=("y_pred", "sum"),
    )
    agg["total_residual"] = agg["total_actual"] - agg["total_predicted"]
    agg["avg_residual_per_game"] = agg["total_residual"] / agg["n_games"]

    engine = get_engine()
    names = _player_names(engine, agg["athlete_id"].unique().tolist(), season=2025)[["athlete_id", "player_name"]]
    agg["athlete_id"] = agg["athlete_id"].astype(str)
    names["athlete_id"] = names["athlete_id"].astype(str)
    agg = agg.merge(names, on="athlete_id", how="left")
    agg["player_name"] = agg["player_name"].fillna("Unknown Player")

    qualified = agg[agg["n_games"] >= args.min_games].copy()
    logger.info(f"{len(agg)} distinct RBs in 2025 holdout; {len(qualified)} qualify with >= {args.min_games} games")

    # Ranked by avg_residual_per_game (yards over/under expectation per qualifying game) --
    # normalizes for how many games a player had in this 8-week window, rather than letting a
    # high-carry back's total simply outrank a fewer-game back with a more extreme per-game gap.
    qualified = qualified.sort_values("avg_residual_per_game", ascending=False)
    cols = ["player_name", "team", "n_games", "total_actual", "total_predicted", "total_residual", "avg_residual_per_game"]
    qualified[cols].round(1).to_csv(OUT_PATH, index=False)
    logger.info(f"Wrote {len(qualified)} ranked RBs -> {OUT_PATH}")

    top_over = qualified.head(12)
    top_under = qualified.tail(12).sort_values("avg_residual_per_game")

    print("\n=== Top over-performers (actual >> predicted), 2025 weeks 1-8 ===")
    print(top_over[cols].round(1).to_string(index=False))
    print("\n=== Top under-performers (actual << predicted), 2025 weeks 1-8 ===")
    print(top_under[cols].round(1).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

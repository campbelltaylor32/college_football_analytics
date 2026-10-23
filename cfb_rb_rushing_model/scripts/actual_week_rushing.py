#!/usr/bin/env python
"""Grade one week's RB projections: realized carries/rushing yards for every player in
outputs/predictions/week_{season}_{week}_rb_rushing_predictions.csv.

Uses the same raw aggregation + name recovery + roster resolution as training
(features.rushing_workload.build_resolved_player_game_rushing), so a player is graded by the
same rules that built his features. A projected RB with no carries in a completed game gets
rushing_yards=0 / played=False, the same as the training target (targets.py). Games not yet
completed are left out.

Usage:
    python scripts/actual_week_rushing.py --season 2026 --week 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_rb_rushing_model.config import load_data_config
from cfb_rb_rushing_model.database import get_engine, run_query
from cfb_rb_rushing_model.features.rushing_workload import build_resolved_player_game_rushing
from cfb_rb_rushing_model.schedule_spine import build_schedule_spine
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import OUTPUTS_PREDICTIONS

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    pred_path = OUTPUTS_PREDICTIONS / f"week_{args.season}_{args.week}_rb_rushing_predictions.csv"
    if not pred_path.exists():
        logger.warning(f"No projections at {pred_path} -- nothing to grade")
        return 0
    preds = pd.read_csv(pred_path)
    if "game_id" not in preds.columns:
        logger.warning(f"{pred_path.name} predates the game_id column -- can't grade it")
        return 0

    engine = get_engine()
    completed = run_query(
        "SELECT game_id FROM games WHERE season = :s AND week = :w AND completed = 1",
        params={"s": args.season, "w": args.week}, engine=engine,
    )["game_id"]

    spine = build_schedule_spine(engine, [args.season])
    rushing = build_resolved_player_game_rushing(engine, spine, [args.season], load_data_config())
    rushing = rushing[rushing["week"] == args.week][["athlete_id", "game_id", "carries", "rushing_yards"]]
    # team_rosters.athlete_id is a string column; the predictions CSV reads back as int
    rushing["athlete_id"] = rushing["athlete_id"].astype(str)
    preds["athlete_id"] = preds["athlete_id"].astype(str)

    out = preds[preds["game_id"].isin(completed)][["athlete_id", "game_id"]].merge(rushing, on=["athlete_id", "game_id"], how="left")
    out["played"] = out["carries"].notna()
    out[["carries", "rushing_yards"]] = out[["carries", "rushing_yards"]].fillna(0).astype(int)

    out_path = OUTPUTS_PREDICTIONS / f"week_{args.season}_{args.week}_rb_rushing_actuals.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Graded {len(out)} of {len(preds)} projections ({(~out['played']).sum()} with no carries) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

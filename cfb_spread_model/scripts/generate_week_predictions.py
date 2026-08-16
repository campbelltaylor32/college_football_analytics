#!/usr/bin/env python
"""Replacement for ../Python Scripts/Week_Predictions.ipynb. Loads the production model
artifact (scripts/evaluate_models.py's output) and scores a week's games from
../Data/CFB_Pred_Week_<N>.csv. Uses utils/paths.py for path resolution -- does NOT reproduce
the stale hardcoded os.chdir() confirmed in the current notebook (it points at
/Users/campbelltaylor/College_Football_Gambling_Model/, not this machine's actual repo path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cfb_spread_model.artifacts import load_latest_production_artifact
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import OUTPUTS_PREDICTIONS, REPO_DATA_DIR, ensure_dirs

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, help="Week number, matches ../Data/CFB_Pred_Week_<N>.csv")
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved production threshold")
    args = parser.parse_args()

    ensure_dirs()
    model, features, metadata = load_latest_production_artifact()
    threshold = args.threshold if args.threshold is not None else metadata["threshold"]

    week_csv = REPO_DATA_DIR / f"CFB_Pred_Week_{args.week}.csv"
    if not week_csv.exists():
        raise FileNotFoundError(f"{week_csv} not found")
    data = pd.read_csv(week_csv, low_memory=False)

    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ValueError(f"Week {args.week} predictors file is missing {len(missing)} required columns: {missing[:10]}...")

    X_pred = data[features]
    proba = model.predict_proba(X_pred)[:, 1]
    data = data.assign(cover_prob=proba, cover_prediction=(proba >= threshold).astype(int))

    output_cols = [c for c in ["home_team", "away_team", "spread", "home_favored", "cover_prediction", "cover_prob"] if c in data.columns]
    predictions = (
        data.loc[data["cover_prediction"] == 1, output_cols].sort_values("cover_prob", ascending=False).reset_index(drop=True)
    )

    out_path = OUTPUTS_PREDICTIONS / f"week_{args.week}_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(f"Week {args.week}: {len(predictions)} of {len(data)} games flagged at threshold={threshold:.2f} -> {out_path}")
    print(predictions.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

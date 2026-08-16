"""Shared "load the final-holdout season's rows" helper, used by scripts/explain_model.py and
scripts/analyze_rank_calibration.py -- both need the true 2025 holdout rows to score an
already-fit production model against (no retraining), and previously duplicated this lookup.
"""

from __future__ import annotations

import pandas as pd

from cfb_spread_model.config import ModelingConfig
from cfb_spread_model.modeling.splits import final_holdout_fold
from cfb_spread_model.utils.paths import DATA_PROCESSED_DIR

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"


def load_holdout_frame(modeling_cfg: ModelingConfig) -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- run scripts/load_and_validate_dataset.py first")
    df = pd.read_parquet(DATASET_PATH)
    holdout = final_holdout_fold(modeling_cfg)
    return df[df["season"] == holdout.validation_season].reset_index(drop=True)

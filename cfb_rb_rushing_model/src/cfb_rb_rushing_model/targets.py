"""Target construction: rushing_yards per (athlete_id, game_id), for every workload-eligible
RB-game.

The row universe is `eligibility.build_eligibility_spine`'s ELIGIBLE rows -- not "games where
a realized carry exists." A workload-eligible RB who left a game early (blowout, in-game
injury) or was a surprise healthy scratch still produces a `rushing_yards=0` row (LEFT JOIN +
fillna(0)) instead of silently vanishing from the target table. This is a deliberate design
choice, not an oversight: training only on realized-carry games would systematically bias the
target upward by excluding exactly the workload-collapse games a real prop model needs to be
evaluated against. An in-game injury/benching genuinely cannot be predicted from any
pre-game-known feature in this DB (no injury-report table exists) -- these rows are correctly
labeled noise the model cannot learn to anticipate, not a bug. See
docs/assumptions_and_limitations.md.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.config import DataConfig, FeaturesConfig
from cfb_rb_rushing_model.eligibility import build_eligibility_spine
from cfb_rb_rushing_model.features.rushing_workload import build_resolved_player_game_rushing
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COL = "rushing_yards"
REALIZED_ZERO_FILL_COLS = [
    "carries", "rushing_yards", "explosive_runs", "red_zone_carries",
]


def merge_realized_onto_eligible(eligible_rows: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Pure, DB-independent core of target construction -- LEFT JOIN + fillna(0) only. Split
    out from build_target_table specifically so it can be unit-tested against small synthetic
    DataFrames (tests/test_target_construction.py) without a live database.

    eligible_rows: one row per (athlete_id, game_id) that cleared the eligibility gate.
    realized: one row per (athlete_id, game_id) with actual recorded carries -- NOT
    guaranteed to have a matching row for every eligible_rows row.
    """
    realized_cols = ["athlete_id", "game_id", "carries", "rushing_yards", "explosive_runs", "red_zone_carries", "success_rate", "yards_per_carry"]
    realized = realized[realized_cols].drop_duplicates(subset=["athlete_id", "game_id"])

    merged = eligible_rows.merge(realized, on=["athlete_id", "game_id"], how="left")
    merged["played"] = merged["carries"].notna()

    for col in REALIZED_ZERO_FILL_COLS:
        merged[col] = merged[col].fillna(0)
    merged["success_rate"] = merged["success_rate"].fillna(0)
    merged["yards_per_carry"] = merged["yards_per_carry"].fillna(0)

    n_zero = int((~merged["played"]).sum())
    if n_zero:
        logger.info(f"{n_zero}/{len(merged)} eligible RB-game rows had no realized carries (blowout/injury/scratch) -- rushing_yards set to 0, not dropped")

    return merged.reset_index(drop=True)


def build_target_table(engine: Engine, spine: pd.DataFrame, seasons: list[int], data_cfg: DataConfig, features_cfg: FeaturesConfig) -> pd.DataFrame:
    elig = build_eligibility_spine(engine, spine, seasons, data_cfg, features_cfg)
    if elig.empty:
        return pd.DataFrame()
    eligible_rows = elig[elig["eligible"]].copy()
    if eligible_rows.empty:
        logger.warning(f"No eligible RB-game rows for seasons={seasons} -- eligibility thresholds may be too strict for this sample")
        return eligible_rows

    realized = build_resolved_player_game_rushing(engine, spine, seasons, data_cfg)
    return merge_realized_onto_eligible(eligible_rows, realized)

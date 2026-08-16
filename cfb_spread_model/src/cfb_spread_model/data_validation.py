"""Schema/dtype/range assertions run once at load time (scripts/load_and_validate_dataset.py)
and re-exercised by tests/test_leakage.py's test_no_within_week_completeness_gap. These are a
regression safety net for assumptions this project's design depends on -- the actual
lookahead-bias prevention happened upstream in R (see docs/data_leakage_rules.md) and is not
re-implemented here.
"""

from __future__ import annotations

import pandas as pd

from cfb_spread_model.config import DataConfig
from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that must never appear in this CSV -- if the upstream R pipeline ever merges raw
# post-game box-score fields into the predictors file (they exist in an intermediate R frame,
# `fin_data` in Full_CFB_Game_Outcome_Historical.R, but are never merged into total_outcome
# today), this is the tripwire.
#
# EXACT match only for home_points/away_points/home_minus_away: this game's raw final score,
# the actual leak risk. A *substring* check on these would also flag legitimate, correctly-
# lagged columns like home_points_avg_all/home_points_allowed_avg3 (rolling averages of PRIOR
# games' scores, computed and lagged upstream in R -- not this game's outcome) -- verified via
# a real test failure against the live CSV during this project's build, which is exactly the
# false-positive this exact-match distinction exists to avoid.
POSTGAME_EXACT_DENYLIST = {"home_points", "away_points", "home_minus_away", "away_minus_home"}
POSTGAME_SUBSTRING_DENYLIST = ("_result", "final_score")


class DatasetValidationError(AssertionError):
    pass


def validate_raw_dataset(df: pd.DataFrame, cfg: DataConfig) -> None:
    if len(df) < cfg.expected_row_count_min:
        raise DatasetValidationError(
            f"Loaded {len(df)} rows, expected at least {cfg.expected_row_count_min}. "
            f"Has the upstream R pipeline (R Scripts/Merge_Predictors_CFB_Historical.R) changed?"
        )
    if len(df.columns) != cfg.expected_column_count:
        raise DatasetValidationError(
            f"Loaded {len(df.columns)} columns, expected exactly {cfg.expected_column_count}. "
            f"Column-grouping assumptions in feature_selection/correlation_pruning.py and the "
            f"leakage tests may no longer hold -- re-verify before trusting this run."
        )

    required = set(cfg.id_columns) | {cfg.label_column} | set(cfg.split_only_columns) | set(cfg.retained_context_columns)
    missing = required - set(df.columns)
    if missing:
        raise DatasetValidationError(f"Missing required columns: {sorted(missing)}")

    if df["game_id"].duplicated().any():
        n_dupes = int(df["game_id"].duplicated().sum())
        raise DatasetValidationError(f"{n_dupes} duplicate game_id rows found")

    bad_week = df[(df["week"] < cfg.week_min) | (df["week"] > cfg.week_max)]
    if not bad_week.empty:
        raise DatasetValidationError(
            f"{len(bad_week)} rows outside the documented week range "
            f"[{cfg.week_min}, {cfg.week_max}] (config/data.yaml week_range) -- upstream scope "
            f"may have changed (see R Scripts/Full_CFB_Game_Outcome_Historical.R week pull)."
        )

    label = df[cfg.label_column]
    if label.isna().any():
        raise DatasetValidationError(
            f"{int(label.isna().sum())} rows have a missing {cfg.label_column} label -- this "
            f"CSV is meant to be the fully-labeled historical training set, not an inference file."
        )
    bad_labels = set(label.unique()) - {0, 1}
    if bad_labels:
        raise DatasetValidationError(f"{cfg.label_column} has non-binary values: {bad_labels}")

    exact_hits = [c for c in df.columns if c in POSTGAME_EXACT_DENYLIST]
    if exact_hits:
        raise DatasetValidationError(
            f"Found post-game column(s): {exact_hits}. These must never be used as model "
            f"inputs -- see docs/data_leakage_rules.md."
        )
    for pattern in POSTGAME_SUBSTRING_DENYLIST:
        hits = [c for c in df.columns if pattern in c]
        if hits:
            raise DatasetValidationError(
                f"Found post-game-looking column(s) matching '{pattern}': {hits}. These must "
                f"never be used as model inputs -- see docs/data_leakage_rules.md."
            )

    logger.info(f"Validated dataset: {len(df):,} rows, {len(df.columns):,} columns, no post-game leakage columns present")


def check_covid_season_flag(df: pd.DataFrame, covid_season: int = 2020) -> dict:
    """Verified this session: 2020 has 91 games vs 183-268 in neighboring seasons. This function
    documents that check in code rather than only in a comment, so a future re-run of this
    project against a refreshed CSV can re-verify the exclusion is still justified."""
    counts = df.groupby("season").size()
    if covid_season not in counts.index:
        return {"covid_season": covid_season, "present": False}
    neighbors = counts.drop(index=covid_season, errors="ignore")
    return {
        "covid_season": covid_season,
        "present": True,
        "covid_season_games": int(counts[covid_season]),
        "median_other_season_games": float(neighbors.median()),
        "is_shortened": counts[covid_season] < 0.6 * neighbors.median(),
    }

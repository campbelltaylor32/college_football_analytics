"""Domain-informed feature engineering, grounded in docs/feature_importance_findings.md's
three highest-priority recommendations:

1. Opponent-adjusted matchup features for the two categories with the strongest empirical
   support (down_conversion, epa_success_rate) - a team's own rate isn't very meaningful
   without knowing what its *this-game* opponent typically allows.
2. Trim the 12-column returning_production family down to the 4 sub-metrics that actually
   showed up as fold-stable (rushing_usage, receiving_usage, percent_rushing_ppa,
   total_rushing_ppa), instead of feeding the model 8 more that don't.
3. Consolidate the 14-column special_teams family (the category with the sharpest
   fold-stability-vs-holdout-importance disagreement, a likely overfitting signature) into
   one net point-value-weighted composite per side per transform, rather than 14 raw counts.

All three operate on (frame, feature_columns) *after* build_clean_modeling_frame's base
candidate set is assembled but *before* the final NA-drop (see cleaning.py's
feature_engineering_fn hook), so nothing here needs its own NA handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREFIXES = ("home_", "away_")
OTHER_PREFIX = {"home_": "away_", "away_": "home_"}

# --- Recommendation 2: trim returning_production ------------------------------------------
# Kept (per side, no temporal transform - these are preseason snapshots): the 4 sub-metrics
# that were selected in most walk-forward folds under both transform configs in
# docs/feature_importance_findings.md. Dropped: the other 8 (passing/receiving/overall
# variants), which rarely showed up.
RETURNING_PRODUCTION_KEEP = ["rushing_usage", "receiving_usage", "percent_rushing_ppa", "total_rushing_ppa"]
RETURNING_PRODUCTION_DROP = [
    "passing_usage", "percent_passing_ppa", "percent_ppa", "percent_receiving_ppa",
    "total_passing_ppa", "total_ppa", "total_receiving_ppa", "usage",
]

# --- Recommendation 3: consolidate special_teams -------------------------------------------
# Rough point-value weights: kicking_points is already in points; a return TD is a 6-point
# score; return yards are converted to an approximate point value using a standard rough
# field-position heuristic of ~1 expected point per 17 yards of field position (this is a
# documented approximation, not a fitted parameter - deliberately, to keep this feature
# deterministic and leakage-free rather than requiring per-fold normalization). Raw return
# *counts* (kick_returns/punt_returns) are dropped without contributing to the score - they
# don't have an obvious point value independent of the yards/tds they produced.
_YARDS_PER_POINT = 17.0
SPECIAL_TEAMS_WEIGHTS = {
    "kicking_points": 1.0,
    "kick_return_yards": 1.0 / _YARDS_PER_POINT,
    "kick_return_tds": 6.0,
    "punt_return_yards": 1.0 / _YARDS_PER_POINT,
    "punt_return_tds": 6.0,
}
SPECIAL_TEAMS_ALL_BASE_STATS = [
    "kick_return_tds", "kick_return_tds_allowed", "kick_return_yards", "kick_return_yards_allowed",
    "kick_returns", "kick_returns_allowed", "kicking_points", "kicking_points_allowed",
    "punt_return_tds", "punt_return_tds_allowed", "punt_return_yards", "punt_return_yards_allowed",
    "punt_returns", "punt_returns_allowed",
]

# --- Recommendation 1: opponent-adjusted matchup features -----------------------------------
# (offense_stat, defense_allowed_stat) pairs already expressed as a rate/EPA value (no
# division needed) - the differential is home_offense - away_defense_allowed (and the
# mirror for away). "ratio" pairs need constructing a rate from a count/attempts pair first.
DOWN_CONVERSION_RATE_PAIRS = [
    ("Offense_First_Down_Success_Rate", "Defense_First_Down_Success_Rate"),
    ("Offense_First_Down_Pass_Success_Rate", "Defense_First_Down_Pass_Success_Rate"),
    ("Offense_First_Down_Run_Success_Rate", "Defense_First_Down_Run_Success_Rate"),
    ("Offense_first_down_pass_rate", "Defense_first_down_pass_rate"),
    ("Offense_Avg_3rd_Down_Distance", "Defense_Avg_3rd_Down_Distance"),
]
DOWN_CONVERSION_COUNT_DIFF_PAIRS = [("first_downs", "first_downs_allowed")]
DOWN_CONVERSION_RATIO_PAIRS = [
    # (offense_numerator, offense_denominator, defense_numerator, defense_denominator)
    ("third_down_conversion", "third_down_attempts", "third_down_conversion_allowed", "third_down_attempts_allowed"),
    ("fourth_down_conversion", "fourth_down_attempts", "fourth_down_conversion_allowed", "fourth_down_attempts_allowed"),
]

EPA_SUCCESS_RATE_PAIRS = [
    ("Offense_Success_Rate", "Defense_Success_Rate"),
    ("Offense_Pass_Success_Rate", "Defense_Pass_Success_Rate"),
    ("Offense_Run_Success_Rate", "Defense_Run_Success_Rate"),
    ("Total_Offense_EPA", "Total_Defense_EPA"),
    ("Total_Offense_EPA_Pass", "Total_Defense_EPA_Pass"),
    ("Total_Offense_EPA_Run", "Total_Defense_EPA_Run"),
    ("Total_Offense_Success", "Total_Defense_Success"),
    ("Total_Offense_Pass_Success", "Total_Defense_Pass_Success"),
    ("Total_Offense_Run_Success", "Total_Defense_Run_Success"),
]

TRANSFORMS = ("prev_week", "avg_all", "avg3")


def _col(prefix: str, transform: str, stat: str) -> str:
    if transform == "prev_week":
        return f"{prefix}prev_week_{stat}"
    if transform == "avg_all":
        return f"{prefix}{stat}_avg_all"
    if transform == "avg3":
        return f"{prefix}{stat}_avg3"
    raise ValueError(f"Unknown transform: {transform!r}")


def consolidate_returning_production(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = [f"{p}{stat}" for p in PREFIXES for stat in RETURNING_PRODUCTION_DROP]
    drop_cols = [c for c in drop_cols if c in feature_columns]
    remaining = [c for c in feature_columns if c not in drop_cols]
    return frame, remaining


def consolidate_special_teams(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    kept_cols = [c for c in feature_columns if not any(
        c == _col(p, t, stat) for p in PREFIXES for t in TRANSFORMS for stat in SPECIAL_TEAMS_ALL_BASE_STATS
    )]
    new_cols: dict[str, pd.Series] = {}
    new_col_names: list[str] = []

    for prefix in PREFIXES:
        for transform in TRANSFORMS:
            offense_cols = {stat: _col(prefix, transform, stat) for stat in SPECIAL_TEAMS_WEIGHTS}
            allowed_cols = {stat: _col(prefix, transform, f"{stat}_allowed") for stat in SPECIAL_TEAMS_WEIGHTS}
            if not all(c in frame.columns for c in list(offense_cols.values()) + list(allowed_cols.values())):
                continue  # this (prefix, transform) combination isn't part of the current candidate set

            offense_score = sum(weight * frame[offense_cols[stat]] for stat, weight in SPECIAL_TEAMS_WEIGHTS.items())
            allowed_score = sum(weight * frame[allowed_cols[stat]] for stat, weight in SPECIAL_TEAMS_WEIGHTS.items())
            new_col = f"{prefix}special_teams_net_score_{transform}"
            new_cols[new_col] = offense_score - allowed_score
            new_col_names.append(new_col)

    frame = pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)
    return frame, kept_cols + new_col_names


def _matchup_diff_series(offense_series: pd.Series, defense_allowed_series_other_side: pd.Series) -> pd.Series:
    return offense_series - defense_allowed_series_other_side


def add_opponent_adjusted_features(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    new_cols: dict[str, pd.Series] = {}
    new_col_names: list[str] = []

    def register(prefix: str, transform: str, new_name: str, offense_series: pd.Series, defense_allowed_series_other_side: pd.Series) -> None:
        col_name = f"{prefix}matchup_adj_{new_name}_{transform}"
        new_cols[col_name] = _matchup_diff_series(offense_series, defense_allowed_series_other_side)
        new_col_names.append(col_name)

    for transform in TRANSFORMS:
        for prefix in PREFIXES:
            other = OTHER_PREFIX[prefix]

            for off_stat, def_stat in DOWN_CONVERSION_RATE_PAIRS + EPA_SUCCESS_RATE_PAIRS + DOWN_CONVERSION_COUNT_DIFF_PAIRS:
                off_col = _col(prefix, transform, off_stat)
                def_col_other_side = _col(other, transform, def_stat)
                if off_col in frame.columns and def_col_other_side in frame.columns:
                    register(prefix, transform, off_stat, frame[off_col], frame[def_col_other_side])

            for off_num, off_den, def_num, def_den in DOWN_CONVERSION_RATIO_PAIRS:
                off_num_col, off_den_col = _col(prefix, transform, off_num), _col(prefix, transform, off_den)
                def_num_col_other, def_den_col_other = _col(other, transform, def_num), _col(other, transform, def_den)
                cols_needed = [off_num_col, off_den_col, def_num_col_other, def_den_col_other]
                if all(c in frame.columns for c in cols_needed):
                    # fourth_down_attempts is frequently 0 (many teams never face one in a
                    # single game, especially at the prev_week transform) - 0/0 would
                    # otherwise NaN out ~20-40% of rows (confirmed empirically) once
                    # dropped downstream. Treat "no attempts" as "no rate to adjust by"
                    # (0.0, i.e. neutral) rather than propagating NaN and losing the row -
                    # a reasonable default given there's genuinely no information to act on.
                    off_rate = (frame[off_num_col] / frame[off_den_col].replace(0, np.nan)).fillna(0.0)
                    def_rate_other = (frame[def_num_col_other] / frame[def_den_col_other].replace(0, np.nan)).fillna(0.0)
                    rate_name = off_num.replace("_conversion", "_rate")
                    register(prefix, transform, rate_name, off_rate, def_rate_other)

    frame = pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)
    return frame, feature_columns + new_col_names  # additive - raw columns are kept, see module docstring


def apply_engineered_features(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Orchestrates all three recommendations, in order. Called from
    load_and_validate_dataset.py via cleaning.build_clean_modeling_frame's
    feature_engineering_fn hook."""
    frame, feature_columns = consolidate_returning_production(frame, feature_columns)
    frame, feature_columns = consolidate_special_teams(frame, feature_columns)
    frame, feature_columns = add_opponent_adjusted_features(frame, feature_columns)
    return frame, feature_columns

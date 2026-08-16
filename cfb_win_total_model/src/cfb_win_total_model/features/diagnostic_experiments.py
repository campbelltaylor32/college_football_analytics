"""Diagnostic-only feature augmentation for the compression investigation
(scripts/diagnostics/feature_experiment.py). Every new column here is derived purely from
columns already present in data/processed/modeling_dataset.parquet -- no new DB queries.

This module is intentionally NOT wired into dataset.py's _FEATURE_BUILDERS/_FEATURE_DESCRIBERS
registries, so it can never be pulled into the production build_modeling_dataset() path. It
exists to test "would a handful of extra features/interactions reduce prediction compression"
under the identical walk-forward CV protocol, without touching the shipped feature set.
"""

from __future__ import annotations

import pandas as pd

# New standalone features/interactions added by augment_with_diagnostic_features(), in the
# order they're computed. Exposed so callers can extend get_feature_columns()'s output with
# exactly these names.
DIAGNOSTIC_FEATURE_COLUMNS = [
    "talent_margin_vs_schedule",
    "talent_advantage_game_count",
    "talent_advantage_ratio",
    "interaction_talent_sos",
    "interaction_returning_prior_wins",
    "interaction_qb_continuity_off_efficiency",
    "interaction_coaching_change_turnover",
]


def augment_with_diagnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a COPY of df with DIAGNOSTIC_FEATURE_COLUMNS appended. Each column is guarded
    so the function degrades gracefully (fills NaN) rather than raising if a source column is
    ever renamed, instead of hard-failing a diagnostic script over a missing ingredient."""
    out = df.copy()

    def _get(col: str) -> pd.Series:
        return out[col] if col in out.columns else pd.Series(float("nan"), index=out.index)

    # Talent relative to scheduled opponents: own preseason talent minus the average
    # preseason talent of teams actually on the schedule (both already on the raw composite
    # talent scale -- NOT mixing a z-scored own-talent with a raw-scale opponent average).
    out["talent_margin_vs_schedule"] = _get("talent") - _get("avg_opponent_prior_talent")

    # "Games with a preseason talent advantage": n_opponents_above_own_talent already counts
    # opponents whose prior talent EXCEEDS the team's own season-t talent, so its complement
    # (within the team's actual game count) is exactly this ingredient.
    out["talent_advantage_game_count"] = _get("n_games") - _get("n_opponents_above_own_talent")
    # Ratio form sidesteps comparing raw counts across teams with 12- vs 13-game schedules,
    # and avoids leaning on the endogenous n_games count directly (see
    # outputs/diagnostics_compression/tables/n_games_endogeneity_check.csv).
    n_games = _get("n_games")
    out["talent_advantage_ratio"] = out["talent_advantage_game_count"] / n_games.replace(0, pd.NA)

    # talent x strength-of-schedule
    out["interaction_talent_sos"] = _get("talent") * _get("avg_opponent_prior_talent")

    # returning production x prior-season performance
    out["interaction_returning_prior_wins"] = _get("returning_percent_ppa") * _get("prior_season_wins")

    # QB continuity (1 - departure indicator, i.e. did the starter return) x offensive
    # efficiency. qb_departure_indicator is boolean/0-1; treat missing as departed (0
    # continuity) rather than fabricating continuity for teams with unknown QB status.
    qb_departed = _get("qb_departure_indicator").astype("float")
    qb_continuity = 1 - qb_departed.fillna(1)
    out["interaction_qb_continuity_off_efficiency"] = qb_continuity * _get("off_epa_per_play")

    # coaching change x roster turnover
    coaching_change = _get("coaching_change_indicator").astype("float")
    out["interaction_coaching_change_turnover"] = coaching_change * _get("net_roster_turnover_pct")

    return out

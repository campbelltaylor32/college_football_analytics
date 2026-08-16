"""Build temporal-transform and home/away-representation variants of the candidate
feature set. Pure column-selection/arithmetic - no fitting happens here, so it's safe to
call on any slice of rows (train, val, or holdout) without leaking anything.
"""
from __future__ import annotations

import pandas as pd

PREFIXES = ("home_", "away_")


def categorize_features(
    feature_columns: list[str],
) -> tuple[dict[tuple[str, str], dict[str, str]], list[str], list[str]]:
    """Split candidate feature columns into:
      temporal:     {(prefix, stat): {"prev_week": col, "avg_all": col, "avg3": col}}
      non_temporal: home_/away_-prefixed columns with no temporal suffix (talent, coaching,
                    returning production)
      context:      columns with no home_/away_ prefix at all (week, spread, neutral_site,
                    conference_game)
    """
    temporal: dict[tuple[str, str], dict[str, str]] = {}
    non_temporal: list[str] = []
    context: list[str] = []

    for col in feature_columns:
        prefix = next((p for p in PREFIXES if col.startswith(p)), None)
        if prefix is None:
            context.append(col)
            continue
        rem = col[len(prefix):]
        if rem.startswith("prev_week_"):
            stat, transform = rem[len("prev_week_"):], "prev_week"
        elif rem.endswith("_avg_all"):
            stat, transform = rem[: -len("_avg_all")], "avg_all"
        elif rem.endswith("_avg3"):
            stat, transform = rem[: -len("_avg3")], "avg3"
        elif rem.endswith("_prev_week"):
            # engineered_features.py names its columns with a *trailing* _prev_week
            # suffix (matchup_adj_<stat>_prev_week, special_teams_net_score_prev_week),
            # unlike raw columns which use a *leading* prev_week_ prefix - without this
            # branch these would fall through to non_temporal and bypass transform-
            # ablation filtering entirely (a real bug: they'd leak into every transform
            # variant regardless of which one the ablation was testing).
            stat, transform = rem[: -len("_prev_week")], "prev_week"
        else:
            non_temporal.append(col)
            continue
        temporal.setdefault((prefix, stat), {})[transform] = col

    return temporal, non_temporal, context


def build_transform_variant(
    frame: pd.DataFrame, feature_columns: list[str], transforms: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the feature set keeping only the requested temporal transform(s) for every
    game-stat base column. "trend" is engineered fresh as avg3 - avg_all (only emitted for
    (prefix, stat) pairs that have both). Non-temporal and context columns always pass
    through unchanged - they only exist in one form to begin with."""
    temporal, non_temporal, context = categorize_features(feature_columns)

    out_cols = list(non_temporal) + list(context)
    out = frame[out_cols].copy()

    for (prefix, stat), available in temporal.items():
        for t in transforms:
            if t == "trend":
                if "avg3" in available and "avg_all" in available:
                    new_col = f"{prefix}trend_{stat}"
                    out[new_col] = frame[available["avg3"]] - frame[available["avg_all"]]
                    out_cols.append(new_col)
            elif t in available:
                col = available[t]
                out[col] = frame[col]
                out_cols.append(col)

    return out[out_cols], out_cols


def apply_home_away_representation(
    frame: pd.DataFrame, feature_columns: list[str], mode: str
) -> tuple[pd.DataFrame, list[str]]:
    """mode == "raw_dual": pass through unchanged.
    mode == "differential": replace every home_X/away_X pair with diff_X = home_X - away_X.
    Columns that only exist on one side (shouldn't normally happen given the R layer's
    symmetric construction, but guarded) pass through under their original name.
    """
    if mode == "raw_dual":
        return frame[feature_columns].copy(), list(feature_columns)

    if mode != "differential":
        raise ValueError(f"Unknown home_away_representation mode: {mode!r}")

    pairs: dict[str, dict[str, str]] = {}
    unpaired: list[str] = []
    for col in feature_columns:
        if col.startswith("home_"):
            pairs.setdefault(col[len("home_"):], {})["home"] = col
        elif col.startswith("away_"):
            pairs.setdefault(col[len("away_"):], {})["away"] = col
        else:
            unpaired.append(col)

    out = frame[unpaired].copy()
    new_cols = list(unpaired)
    for key, sides in pairs.items():
        if "home" in sides and "away" in sides:
            new_col = f"diff_{key}"
            out[new_col] = frame[sides["home"]] - frame[sides["away"]]
            new_cols.append(new_col)
        else:
            only_col = sides.get("home") or sides.get("away")
            out[only_col] = frame[only_col]
            new_cols.append(only_col)

    return out[new_cols], new_cols

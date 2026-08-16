"""Expand config-declared base-stat name patterns into full column names and build the
candidate feature list / cleaned modeling frame.

Base stat names in config/data.yaml (e.g. "point_differential") apply to every
home_/away_ prefix and every temporal-transform suffix present in the source CSV
(prev_week_<stat>, <stat>_avg_all, <stat>_avg3) - this module does that expansion once so
downstream code always works with real column names.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from cfb_cover_model.engineered_features import apply_engineered_features

# Bookkeeping columns produced by targets.py / data.py that must never be treated as
# candidate features (they are the label, the leakage-adjacent components used to build it,
# or splitting keys handled separately by modeling/splits.py).
NON_FEATURE_BOOKKEEPING_COLUMNS = [
    "home_points",
    "away_points",
    "home_minus_away",
    "signed_spread",
    "cover_margin",
    "is_push",
    "home_covered",
    "season",  # used for walk-forward splitting only, not fed to the model - see
    # docs/assumptions_and_limitations.md: including season as a raw numeric feature would
    # let a model extrapolate on "which year" rather than on-field signal.
]

PREFIXES = ("home_", "away_")
TEMPORAL_SUFFIXES = {
    "prev_week": "{prefix}prev_week_{stat}",
    "avg_all": "{prefix}{stat}_avg_all",
    "avg3": "{prefix}{stat}_avg3",
}


def expand_base_stats(base_stats: list[str], columns: pd.Index) -> list[str]:
    """Expand base stat names into every matching real column (home/away x 3 transforms).
    Silently skips a (prefix, transform) combination that isn't present in `columns` -
    the three temporal transforms are only defined for game-stat columns, not for
    talent/coaching/returning-production base names.
    """
    expanded: set[str] = set()
    colset = set(columns)
    for stat in base_stats:
        for prefix in PREFIXES:
            for template in TEMPORAL_SUFFIXES.values():
                col = template.format(prefix=prefix, stat=stat)
                if col in colset:
                    expanded.add(col)
    return sorted(expanded)


def build_excluded_columns(data_cfg: dict, columns: pd.Index) -> dict[str, list[str]]:
    """Return the full set of columns excluded from the candidate feature set, grouped by
    reason (useful for docs/data_dictionary.md and for sanity-checking column counts)."""
    known_bad = expand_base_stats(data_cfg["known_bad_base_stats"], columns)
    deterministic_redundant = expand_base_stats(
        data_cfg["deterministic_redundant_base_stats"], columns
    )
    return {
        "id_columns": list(data_cfg["id_columns"]),
        "leakage_adjacent_columns": list(data_cfg["leakage_adjacent_columns"]),
        "known_bad_columns": known_bad,
        "deterministic_redundant_columns": deterministic_redundant,
        "bookkeeping_columns": [
            c for c in NON_FEATURE_BOOKKEEPING_COLUMNS if c in columns
        ],
    }


def candidate_feature_columns(data_cfg: dict, columns: pd.Index) -> list[str]:
    """All columns eligible to enter feature selection - everything except the label and
    everything explicitly excluded above."""
    excluded = build_excluded_columns(data_cfg, columns)
    excluded_flat: set[str] = set()
    for cols in excluded.values():
        excluded_flat.update(cols)
    excluded_flat.discard("home_covered")  # label handled separately, never a feature anyway
    return [c for c in columns if c not in excluded_flat and c != "home_covered"]


def build_clean_modeling_frame(
    df: pd.DataFrame,
    data_cfg: dict,
    feature_engineering_fn: Callable[[pd.DataFrame, list[str]], tuple[pd.DataFrame, list[str]]]
    | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Push-filtered, NA-dropped modeling frame plus the candidate feature column list.

    Expects df to already have gone through targets.add_push_and_targets and
    targets.drop_pushes. Returns (frame, feature_columns) where frame retains season/week/
    game_id/home_team/away_team alongside the label and features, so callers can split and
    inspect without re-joining anything; feature_columns is the subset actually meant to be
    fed to a model.

    feature_engineering_fn, if given (see engineered_features.py::apply_engineered_features),
    runs after the base candidate columns are assembled but *before* the final NA-drop, so
    any new engineered columns it adds (e.g. a ratio that can be NaN on 0 attempts) go
    through the same NA-handling as every other feature rather than needing their own.
    """
    feature_columns = candidate_feature_columns(data_cfg, df.columns)
    # home_favored is excluded from feature_columns (leakage-adjacent, see data.yaml) but
    # kept in the saved frame anyway, purely to construct the "always_favorite" diagnostic
    # baseline in modeling/classifiers.py - it is never fed to a model as a predictor.
    keep_columns = sorted(
        set(feature_columns)
        | {
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_covered",
            "cover_margin",
            "home_favored",
        }
    )
    frame = df[keep_columns].copy()

    for col in ("neutral_site", "conference_game"):
        if col in frame.columns and frame[col].dtype == bool:
            frame[col] = frame[col].astype(int)

    if feature_engineering_fn is not None:
        frame, feature_columns = feature_engineering_fn(frame, feature_columns)

    if data_cfg.get("drop_rows_with_any_na", True):
        before = len(frame)
        frame = frame.dropna(subset=feature_columns + ["home_covered"]).reset_index(drop=True)
        dropped = before - len(frame)
        if dropped:
            frame.attrs["n_rows_dropped_for_na"] = dropped

    return frame, feature_columns


def prepare_week_frame(df: pd.DataFrame, data_cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """The build_clean_modeling_frame equivalent for a single week's CFB_Pred_Week_<N>.csv
    (future games, no label, no push filtering).

    Critically, this applies the *same* engineered_features.apply_engineered_features step
    as historical training data - without it, a week file's raw columns don't match what a
    model trained on the engineered feature set (matchup_adj_*, special_teams_net_score_*,
    the trimmed returning_production family) expects, and scoring fails with a KeyError (a
    real bug caught while building scripts/generate_weekly_predictions.py - the original
    single-model scripts/generate_week_predictions.py had this same latent bug for any
    engineered-feature-using model, since it never called this function).

    No NA-drop here, deliberately: dropping a row from a week's *predictions* would silently
    hide a game rather than being a training-data hygiene step - any resulting NaNs should
    surface as an explicit error at prediction time, not a quietly shorter output.
    """
    feature_columns = candidate_feature_columns(data_cfg, df.columns)
    keep_columns = sorted(
        (set(feature_columns) | {"game_id", "season", "week", "home_team", "away_team"})
        & set(df.columns)
    )
    frame = df[keep_columns].copy()

    for col in ("neutral_site", "conference_game"):
        if col in frame.columns and frame[col].dtype == bool:
            frame[col] = frame[col].astype(int)

    frame, feature_columns = apply_engineered_features(frame, feature_columns)
    return frame, feature_columns

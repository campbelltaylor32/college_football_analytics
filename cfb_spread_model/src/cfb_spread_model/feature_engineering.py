"""Home-vs-away differential and recent-form trend features -- pure arithmetic recombinations
of already-lagged, already-leakage-verified home_*/away_* columns (docs/data_leakage_rules.md),
no new data pull, no new leakage risk. Both deliberately exclude prev_week_* (the noisiest
temporal transform, already excluded from this project's default feature set for the same
reason -- see config/data.yaml's excluded_column_patterns). Selected via
config/data.yaml's feature_representation toggle ("raw_dual" default vs "differential"),
applied by data.build_feature_matrix.
"""

from __future__ import annotations

import pandas as pd

from cfb_spread_model.data import build_temporal_triplet_groups, non_temporal_side_columns

_DIFF_TRANSFORMS = ("avg_all", "avg3")
_PYTHAGOREAN_EXPONENT = 2.0
_PYTHAGOREAN_TRANSFORMS = ("avg_all", "avg3")


def _paired_non_temporal_bases(columns: list[str]) -> dict[str, tuple[str, str]]:
    """base_metric -> (home_column, away_column) for non-temporal columns present on both
    sides. `columns` must already exclude id columns (home_team/away_team would otherwise be
    misclassified as a non-temporal "team" metric and produce a nonsensical string subtraction)."""
    non_temporal = non_temporal_side_columns(columns)
    home_cols = {c[len("home_"):]: c for c in non_temporal if c.startswith("home_")}
    away_cols = {c[len("away_"):]: c for c in non_temporal if c.startswith("away_")}
    return {base: (home_cols[base], away_cols[base]) for base in home_cols.keys() & away_cols.keys()}


def build_pythagorean_features(df: pd.DataFrame, columns: list[str], exponent: float = _PYTHAGOREAN_EXPONENT) -> pd.DataFrame:
    """<side>_pythagorean_win_pct_<transform> = PF**exponent / (PF**exponent + PA**exponent + eps),
    built from each side's already-lagged rolling points-for (<side>_points_<transform>) and
    points-allowed (<side>_points_allowed_<transform>) columns -- pure arithmetic recombination,
    same no-new-leakage guarantee as the functions below. Classic exponent=2 (default) tracked
    the season about as well as a data-fit exponent (~2.18) in a 2025 retrospective check (see
    ../../cfb_pythagorean_model/); kept at 2 for simplicity. Once present, these columns are
    picked up automatically by build_differential_features/build_trend_features (base metric
    "pythagorean_win_pct") -- diff_/trend_ versions require no additional code.
    """
    new_cols = {}
    for side in ("home", "away"):
        for transform in _PYTHAGOREAN_TRANSFORMS:
            pf_col, pa_col = f"{side}_points_{transform}", f"{side}_points_allowed_{transform}"
            if pf_col in columns and pa_col in columns:
                pf_k, pa_k = df[pf_col] ** exponent, df[pa_col] ** exponent
                new_cols[f"{side}_pythagorean_win_pct_{transform}"] = pf_k / (pf_k + pa_k + 1e-6)
    return pd.DataFrame(new_cols, index=df.index)


def build_differential_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """diff_<transform>_<base> = home_* - away_*, for avg_all/avg3 base metrics and
    non-temporal metrics present on both sides. Returns ONLY the new columns (same index as
    df) -- callers concat as needed. `columns` must already exclude id columns."""
    groups = build_temporal_triplet_groups(columns)
    by_base: dict[str, dict[str, dict[str, str]]] = {}
    for (side, base), transform_map in groups.items():
        by_base.setdefault(base, {})[side] = transform_map

    new_cols = {}
    for base, sides in by_base.items():
        if "home" not in sides or "away" not in sides:
            continue
        for transform in _DIFF_TRANSFORMS:
            home_col = sides["home"].get(transform)
            away_col = sides["away"].get(transform)
            if home_col and away_col:
                new_cols[f"diff_{transform}_{base}"] = df[home_col] - df[away_col]

    for base, (home_col, away_col) in _paired_non_temporal_bases(columns).items():
        new_cols[f"diff_{base}"] = df[home_col] - df[away_col]

    return pd.DataFrame(new_cols, index=df.index)


def build_trend_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """trend_<side>_<base> = <side>_*_avg3 - <side>_*_avg_all, for every base metric with both
    present, per side. Positive = playing better recently than the team's own season baseline.
    `columns` must already exclude id columns."""
    groups = build_temporal_triplet_groups(columns)
    new_cols = {}
    for (side, base), transform_map in groups.items():
        avg3_col = transform_map.get("avg3")
        avg_all_col = transform_map.get("avg_all")
        if avg3_col and avg_all_col:
            new_cols[f"trend_{side}_{base}"] = df[avg3_col] - df[avg_all_col]
    return pd.DataFrame(new_cols, index=df.index)


def apply_differential_representation(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """Replaces the raw home_*/away_* avg_all/avg3/non-temporal columns with diff_*/trend_*
    columns. prev_week_* columns, context columns (spread, home_favored, etc.), ids, and the
    label are left untouched -- this only removes the specific columns it derived new features
    from, nothing else."""
    candidate_columns = [c for c in df.columns if c not in id_columns]
    diff_df = build_differential_features(df, candidate_columns)
    trend_df = build_trend_features(df, candidate_columns)

    groups = build_temporal_triplet_groups(candidate_columns)
    consumed = set()
    for transform_map in groups.values():
        for transform in _DIFF_TRANSFORMS:
            col = transform_map.get(transform)
            if col:
                consumed.add(col)
    for home_col, away_col in _paired_non_temporal_bases(candidate_columns).values():
        consumed.add(home_col)
        consumed.add(away_col)

    remaining = [c for c in df.columns if c not in consumed]
    return pd.concat([df[remaining], diff_df, trend_df], axis=1)

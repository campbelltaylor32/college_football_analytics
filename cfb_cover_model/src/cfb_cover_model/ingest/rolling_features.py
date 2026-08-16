"""Port of the rolling-window + lag logic split across
../R Scripts/Full_CFB_Game_Outcome_Historical.R (section 1.12, computing avg_all/avg3) and
Merge_Predictors_CFB_Historical.R (section 2.4-2.5, the prev_week_ rename + the actual
point-in-time lag).

Mechanics (verified against the R source, not guessed): for each base stat X, the rolling
step produces X (raw, this week's own value), X_avg_all (cumulative mean *including* the
current row), and X_avg3 (trailing-3-game mean *including* the current row). All three are
then shifted by exactly one row within (team, season), ordered by week - X becomes
"prev_week_X" (last week's raw value), while X_avg_all/X_avg3 keep their names but now mean
"as of last week's game" rather than "as of this week's game". The shift, not the "prev_week"
name, is what prevents lookahead.

fill_value/drop_incomplete control what happens to avg3 before week 3 (no full trailing
window exists yet) - see docs/api_ingestion.md for why the historical-backfill and
live-prediction paths deliberately use different policies here (this project's own choice,
not something to re-derive per call site).
"""
from __future__ import annotations

import pandas as pd


def add_rolling_averages(
    team_week: pd.DataFrame,
    stat_cols: list[str],
    fill_value: float | None,
) -> pd.DataFrame:
    """team_week: one row per (team, week, year), with stat_cols populated as raw values
    for that week's own game. Adds {stat}_avg_all (cumulative mean including the current
    row) and {stat}_avg3 (trailing-3 mean including the current row) - both still
    "as of and including this row's own week", not yet point-in-time-shifted. See
    add_rolling_and_lag (historical path) and pipeline.build_current_week_rows (live path)
    for the two different ways these get turned into point-in-time-safe features."""
    df = team_week.sort_values(["team", "year", "week"]).reset_index(drop=True)
    grouped = df.groupby(["team", "year"], sort=False)

    avg_all_cols, avg3_cols = {}, {}
    for col in stat_cols:
        avg_all_cols[f"{col}_avg_all"] = grouped[col].transform(lambda s: s.expanding().mean())
        avg3 = grouped[col].transform(lambda s: s.rolling(window=3, min_periods=3).mean())
        if fill_value is not None:
            avg3 = avg3.fillna(fill_value)
        avg3_cols[f"{col}_avg3"] = avg3

    return pd.concat([df, pd.DataFrame(avg_all_cols), pd.DataFrame(avg3_cols)], axis=1)


def add_rolling_and_lag(
    team_week: pd.DataFrame,
    stat_cols: list[str],
    fill_value: float | None,
    drop_incomplete: bool,
) -> pd.DataFrame:
    """Historical-backfill path only: each row is a *specific completed game*, so its
    prev_week_{stat}/{stat}_avg_all/{stat}_avg3 must reflect strictly-earlier games than
    that row's own week - computed by shifting the rolling averages (and the raw stat,
    renamed prev_week_{stat}) by exactly one row within (team, year), ordered by week.

    Do NOT use this for the live "predict the upcoming week" path - there is no raw row
    for a not-yet-played game to shift *into*, so shifting would make the broadcast row
    (via slice_max/tail(1)) one game staler than intended. See
    pipeline.build_current_week_rows, which instead takes the latest *un-lagged* row
    directly - that row's own current values already mean "as of the most recent
    completed game", which is exactly the right thing to call prev_week_{stat} for the
    next, not-yet-played game. This was a real bug caught during R-pipeline validation -
    see docs/api_ingestion.md."""
    df = add_rolling_averages(team_week, stat_cols, fill_value)

    lag_cols = [f"{col}_avg_all" for col in stat_cols] + [f"{col}_avg3" for col in stat_cols] + list(stat_cols)
    grouped2 = df.groupby(["team", "year"], sort=False)
    lagged = grouped2[lag_cols].shift(1)
    lagged = lagged.rename(columns={col: f"prev_week_{col}" for col in stat_cols})

    id_cols = df[["team", "week", "year"]]
    result = pd.concat([id_cols, lagged], axis=1)

    lagged_avg_all = [f"{col}_avg_all" for col in stat_cols]
    lagged_avg3 = [f"{col}_avg3" for col in stat_cols]
    lagged_prev_week = [f"prev_week_{col}" for col in stat_cols]

    if drop_incomplete:
        result = result.dropna(subset=lagged_avg_all + lagged_avg3 + lagged_prev_week)

    return result

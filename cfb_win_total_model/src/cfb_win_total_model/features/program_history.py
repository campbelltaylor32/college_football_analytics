"""Program-level historical/rolling features. Multi-year windows (config/features.yaml
`rolling_windows`, default [2,3,5]) that ALWAYS end at season t-1 -- no window ever reaches
into season t. Reuses targets.build_target_table, roster_turnover.build_roster_turnover_features
(for per-season returning_pct), and coaching's coach-of-record derivation so the same season-t
identity logic isn't duplicated.

A `rolling_window_actual_seasons_w` count column accompanies every rolling stat so a partial
window (e.g. only 2 of 5 seasons available for a team new to the DB's coverage era) is visible
to downstream code rather than silently treated as equivalent to a full window.

Simplification, documented: the COVID-shortened 2020 season is included in rolling windows as
a raw historical data point (unlike modeling.yaml's excluded_seasons, which only governs
train/validation folds) -- a team's 5-year rolling win total spanning 2020 will reflect that
season's shortened schedule as-is.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.config import FeaturesConfig
from cfb_win_total_model.database import run_query
from cfb_win_total_model.features import coaching as coaching_mod
from cfb_win_total_model.features.roster_turnover import build_roster_turnover_features
from cfb_win_total_model.targets import build_target_table
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)


def _source_seasons(target_season: int, max_window: int) -> list[int]:
    """All rolling windows end at t-1; the widest configured window determines how far back
    source data is pulled."""
    return list(range(target_season - max_window, target_season))


def _pull_point_diff_by_season(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["school", "season", "point_diff_per_game"])
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"SELECT school, season, points, points_allowed FROM game_team_stats WHERE season IN ({placeholders})"
    df = run_query(sql, params=params, engine=engine)
    df["diff"] = df["points"] - df["points_allowed"]
    return df.groupby(["school", "season"], as_index=False)["diff"].mean().rename(columns={"diff": "point_diff_per_game"})


def _pull_epa_by_season(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["school", "season", "off_epa_per_play"])
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"SELECT season, pos_team AS school, epa FROM plays WHERE season IN ({placeholders}) AND pos_team IS NOT NULL"
    df = run_query(sql, params=params, engine=engine)
    return df.groupby(["school", "season"], as_index=False)["epa"].mean().rename(columns={"epa": "off_epa_per_play"})


def _pull_talent_by_season(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["school", "season", "talent"])
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"SELECT school, season, talent FROM team_talent WHERE season IN ({placeholders})"
    return run_query(sql, params=params, engine=engine)


def _pull_returning_pct_by_season(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        rt = build_roster_turnover_features(engine, target_season=s)
        if not rt.empty:
            frames.append(rt[["school", "season", "returning_pct"]])
    if not frames:
        return pd.DataFrame(columns=["school", "season", "returning_pct"])
    return pd.concat(frames, ignore_index=True)


def _coach_identity_by_season(engine: Engine, target_season: int) -> pd.DataFrame:
    history = coaching_mod._pull_coaches_history(engine, target_season)
    if history.empty:
        return pd.DataFrame(columns=["school", "season", "coach_name"])
    cor = coaching_mod._coach_of_record(history)
    cor["coach_name"] = cor["first_name"].fillna("") + " " + cor["last_name"].fillna("")
    return cor[["school", "season", "coach_name"]]


def build_program_history_features(engine: Engine, target_season: int, features_cfg: FeaturesConfig) -> pd.DataFrame:
    max_window = max(features_cfg.rolling_windows)
    lookback_start = target_season - max_window
    seasons = list(range(lookback_start, target_season))  # up to t-1, inclusive
    logger.info(f"Building program_history features for target_season={target_season} (rolling windows {features_cfg.rolling_windows}, seasons {seasons})")

    wins = build_target_table(engine, seasons=seasons)
    point_diff = _pull_point_diff_by_season(engine, seasons)
    epa = _pull_epa_by_season(engine, seasons)
    talent = _pull_talent_by_season(engine, seasons)
    returning = _pull_returning_pct_by_season(engine, seasons)
    coach_by_season = _coach_identity_by_season(engine, target_season)

    schools = sorted(set(wins["school"]) if not wins.empty else set())
    if not schools:
        return pd.DataFrame(columns=["school", "season"])

    rows = []
    for school in schools:
        row: dict = {"school": school, "season": target_season}
        school_wins = wins[wins["school"] == school].set_index("season")["regular_season_wins"]
        school_pd = point_diff[point_diff["school"] == school].set_index("season")["point_diff_per_game"]
        school_epa = epa[epa["school"] == school].set_index("season")["off_epa_per_play"]
        school_talent = talent[talent["school"] == school].set_index("season")["talent"]
        school_returning = returning[returning["school"] == school].set_index("season")["returning_pct"]
        school_coach = coach_by_season[coach_by_season["school"] == school].set_index("season")["coach_name"]

        if len(school_wins) >= 2 and (target_season - 1) in school_wins.index and (target_season - 2) in school_wins.index:
            row["recent_trend_wins"] = school_wins[target_season - 1] - school_wins[target_season - 2]
        else:
            row["recent_trend_wins"] = None

        for w in features_cfg.rolling_windows:
            window_seasons = [s for s in range(target_season - w, target_season)]
            wv = school_wins.reindex(window_seasons).dropna()
            pdv = school_pd.reindex(window_seasons).dropna()
            epav = school_epa.reindex(window_seasons).dropna()
            talv = school_talent.reindex(window_seasons).dropna()
            retv = school_returning.reindex(window_seasons).dropna()
            coachv = school_coach.reindex(window_seasons).dropna()

            row[f"rolling_win_total_{w}"] = wv.sum() if len(wv) else None
            row[f"win_volatility_{w}"] = wv.std() if len(wv) >= 2 else None
            row[f"rolling_point_diff_{w}"] = pdv.mean() if len(pdv) else None
            row[f"rolling_epa_{w}"] = epav.mean() if len(epav) else None
            row[f"rolling_talent_{w}"] = talv.mean() if len(talv) else None
            row[f"roster_stability_{w}"] = retv.mean() if len(retv) else None
            row[f"coaching_stability_{w}"] = (1 - (coachv.nunique() - 1) / w) if len(coachv) else None
            row[f"rolling_window_actual_seasons_{w}"] = len(wv)

        rows.append(row)

    return pd.DataFrame(rows)


def describe_features() -> list[dict]:
    base = {
        "source_table": "targets/game_team_stats/plays/team_talent/team_rosters/coaches (multi-season)",
        "source_season": "t-w through t-1 for window size w",
        "category": "program_history",
        "known_before_kickoff": True,
        "missing_value_treatment": "NaN if fewer than the full window of seasons is available (see rolling_window_actual_seasons_w)",
    }
    rows = [
        {**base, "feature_name": "recent_trend_wins", "description": "wins[t-1] - wins[t-2]", "transformation": "diff", "expected_direction": "+"},
    ]
    for w in (2, 3, 5):
        rows.extend(
            [
                {**base, "feature_name": f"rolling_win_total_{w}", "description": f"Sum of wins over trailing {w} seasons ending t-1", "transformation": "sum", "expected_direction": "+"},
                {**base, "feature_name": f"win_volatility_{w}", "description": f"Std dev of wins over trailing {w} seasons ending t-1", "transformation": "std", "expected_direction": "-"},
                {**base, "feature_name": f"rolling_point_diff_{w}", "description": f"Avg point differential/game over trailing {w} seasons ending t-1", "transformation": "mean", "expected_direction": "+"},
                {**base, "feature_name": f"rolling_epa_{w}", "description": f"Avg offensive EPA/play over trailing {w} seasons ending t-1", "transformation": "mean", "expected_direction": "+"},
                {**base, "feature_name": f"rolling_talent_{w}", "description": f"Avg talent composite over trailing {w} seasons ending t-1", "transformation": "mean", "expected_direction": "+"},
                {**base, "feature_name": f"roster_stability_{w}", "description": f"Avg returning_pct over trailing {w} seasons ending t-1", "transformation": "mean", "expected_direction": "+"},
                {**base, "feature_name": f"coaching_stability_{w}", "description": f"Fraction of trailing {w} seasons under the same coach of record", "transformation": "1-(n_distinct_coaches-1)/w", "expected_direction": "+"},
                {**base, "feature_name": f"rolling_window_actual_seasons_{w}", "description": f"How many of the {w} trailing seasons actually had win data (partial-window indicator)", "transformation": "count", "expected_direction": "context"},
            ]
        )
    return rows

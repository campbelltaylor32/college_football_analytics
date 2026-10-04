"""Play-by-play efficiency (EPA/play and success rate, both sides of the ball) as a second,
lower-noise read on each game alongside its final score.

Points over a 3-4 game sample are noisy -- turnovers, return TDs, garbage-time scoring and
uncapped blowouts all move a scoring-only SRS a lot. Per-play efficiency stabilizes much
faster. Rather than build a second rating system, each team-game's net efficiency edge
(offensive EPA/play minus EPA/play allowed, same for success rate) is converted into a
"points-equivalent margin" via a regression fit on past seasons, then blended with the actual
scoring margin inside rating_engine.update_ratings. Opponent adjustment, home-field correction,
the non-FBS pool and preseason-prior fading all then come from the existing SRS machinery
unchanged.

`opponent_adjusted_off_def` separately produces reporting-only offense/defense columns -- they
describe both sides of the ball per team but never feed the rating itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Scrimmage plays only: special teams (kickoffs, punts, field goals), accepted penalties with
# no play, and admin rows (timeouts, period ends) all excluded. Taken from the observed 2025
# plays.play_type distribution.
SCRIMMAGE_PLAY_TYPES = (
    "Rush", "Pass Reception", "Pass Incompletion", "Pass Completion", "Sack",
    "Rushing Touchdown", "Passing Touchdown",
    "Interception Return", "Interception Return Touchdown",
    "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Recovery (Opponent) Touchdown",
    "Fumble Return Touchdown", "Fumble", "Safety",
)


@dataclass
class GarbageTimeConfig:
    """Plays in period >= min_period with the offense's pre-snap win probability outside
    [wp_low, wp_high] are dropped. Plays with a null wp_before are kept."""
    min_period: int = 3
    wp_low: float = 0.05
    wp_high: float = 0.95


@dataclass
class EfficiencyCalibration:
    """raw margin ~= coef_epa * epa_net + coef_sr * sr_net (no intercept -- a game's two team
    rows are exact negatives of each other)."""
    coef_epa: float
    coef_sr: float
    r2: float
    mae: float
    n_team_games: int

    def predict(self, epa_net, sr_net):
        return self.coef_epa * np.asarray(epa_net) + self.coef_sr * np.asarray(sr_net)


def load_team_game_efficiency(
    engine,
    seasons: list[int],
    max_week: int | None = None,
    garbage: GarbageTimeConfig | None = None,
) -> pd.DataFrame:
    """One row per (game, offense, defense): scrimmage plays, mean EPA, mean success.
    Aggregated SQL-side so a multi-season calibration never pulls millions of raw rows.
    `max_week` (exclusive) mirrors update_ratings.py's "week < N" convention."""
    from cfb_power_ratings.database import run_query

    garbage = garbage or GarbageTimeConfig()
    season_ph = ", ".join(f":s{i}" for i in range(len(seasons)))
    type_ph = ", ".join(f":pt{i}" for i in range(len(SCRIMMAGE_PLAY_TYPES)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    params.update({f"pt{i}": pt for i, pt in enumerate(SCRIMMAGE_PLAY_TYPES)})
    params.update({"min_period": garbage.min_period, "wp_low": garbage.wp_low, "wp_high": garbage.wp_high})
    week_clause = ""
    if max_week is not None:
        week_clause = "AND week < :max_week"
        params["max_week"] = max_week

    sql = f"""
        SELECT season, week, game_id, pos_team, def_pos_team,
               COUNT(*) AS plays, AVG(epa) AS epa, AVG(success) AS sr
        FROM plays
        WHERE season IN ({season_ph})
          AND play_type IN ({type_ph})
          AND pos_team IS NOT NULL AND def_pos_team IS NOT NULL AND pos_team <> def_pos_team
          AND epa IS NOT NULL AND success IS NOT NULL
          {week_clause}
          AND NOT (period >= :min_period AND wp_before IS NOT NULL
                   AND (wp_before < :wp_low OR wp_before > :wp_high))
        GROUP BY season, week, game_id, pos_team, def_pos_team
    """
    out = run_query(sql, params=params, engine=engine)
    for col in ["epa", "sr"]:
        out[col] = out[col].astype(float)
    return out


def team_game_frame(eff: pd.DataFrame) -> pd.DataFrame:
    """Offense-vs-defense rows -> one row per (game_id, team) carrying both sides of the ball:
    this team's offense (off_*) and what its defense allowed (def_*_allowed, i.e. the
    opponent's offensive row). Games where either side has no scrimmage plays are dropped."""
    keys = ["season", "week", "game_id"]
    off = eff.rename(columns={
        "pos_team": "team", "def_pos_team": "opponent", "plays": "off_plays", "epa": "off_epa", "sr": "off_sr",
    })
    dfn = eff.rename(columns={
        "def_pos_team": "team", "pos_team": "opponent", "plays": "def_plays", "epa": "def_epa_allowed", "sr": "def_sr_allowed",
    })
    tg = off.merge(dfn, on=keys + ["team", "opponent"], how="inner")
    tg["epa_net"] = tg["off_epa"] - tg["def_epa_allowed"]
    tg["sr_net"] = tg["off_sr"] - tg["def_sr_allowed"]
    return tg


def fit_efficiency_to_points(
    engine, seasons: list[int], fbs_by_season: dict[int, set[str]], garbage: GarbageTimeConfig | None = None,
) -> EfficiencyCalibration:
    """No-intercept OLS of each FBS-vs-FBS team-game's raw scoring margin on its net EPA/play
    and net success rate, over `seasons`. The caller is responsible for passing only seasons
    strictly before the one being rated (no lookahead)."""
    from cfb_power_ratings.database import run_query

    tg = team_game_frame(load_team_game_efficiency(engine, seasons, garbage=garbage))
    season_ph = ", ".join(f":s{i}" for i in range(len(seasons)))
    games = run_query(
        f"SELECT game_id, home_team, home_points, away_points FROM games "
        f"WHERE completed = 1 AND season IN ({season_ph})",
        params={f"s{i}": s for i, s in enumerate(seasons)}, engine=engine,
    )
    tg = tg.merge(games, on="game_id", how="inner")
    is_fbs_pair = [
        (t in fbs_by_season.get(s, set())) and (o in fbs_by_season.get(s, set()))
        for s, t, o in zip(tg["season"], tg["team"], tg["opponent"])
    ]
    tg = tg[is_fbs_pair]
    home_margin = tg["home_points"].astype(float) - tg["away_points"].astype(float)
    y = np.where(tg["team"] == tg["home_team"], home_margin, -home_margin)
    X = tg[["epa_net", "sr_net"]].to_numpy(dtype=float)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return EfficiencyCalibration(
        coef_epa=float(coef[0]), coef_sr=float(coef[1]),
        r2=1 - ss_res / ss_tot if ss_tot else 0.0,
        mae=float(np.mean(np.abs(y - pred))),
        n_team_games=len(y),
    )


def efficiency_margins(team_games: pd.DataFrame, calibration: EfficiencyCalibration) -> pd.DataFrame:
    """(game_id, team, eff_margin): each team-game's efficiency expressed as a raw
    points-equivalent margin -- the shape rating_engine.update_ratings expects."""
    out = team_games[["game_id", "team"]].copy()
    out["eff_margin"] = calibration.predict(team_games["epa_net"], team_games["sr_net"])
    return out


def opponent_adjusted_off_def(
    team_games: pd.DataFrame,
    completed_games: pd.DataFrame,
    fbs_teams: set[str],
    non_fbs_pool_name: str = "generic_low_major",
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Reporting-only, per FBS team: opponent- and site-adjusted offensive EPA/play and
    success rate (adj_off_*) and the same allowed on defense (adj_def_*, lower = better).

    Per metric, fits  off_metric(team t vs defense d) = mu + O_t + D_d + h * site  by ridge
    least squares (O/D penalized toward 0 with strength `ridge_alpha` game-rows, mu/h
    effectively unpenalized). site = +1 home / -1 away / 0 neutral. Non-FBS teams share one pooled O and D.
    Reported values are mu + O_t and mu + D_t, i.e. what the team would post against an
    average opponent on a neutral field."""
    from cfb_power_ratings.srs import games_to_team_game_frame

    site = games_to_team_game_frame(completed_games)[["game_id", "team", "is_home", "neutral_site"]]
    rows = team_games.merge(site, on=["game_id", "team"], how="inner")
    if rows.empty:
        return pd.DataFrame(columns=["team", "adj_off_epa", "adj_def_epa", "adj_off_sr", "adj_def_sr"])

    def entity(t):
        return t if t in fbs_teams else non_fbs_pool_name

    off_ent = rows["team"].map(entity)
    def_ent = rows["opponent"].map(entity)
    entities = sorted(set(off_ent) | set(def_ent))
    idx = {e: i for i, e in enumerate(entities)}
    n_ent, n = len(entities), len(rows)
    site_sign = np.where(rows["neutral_site"].astype(bool), 0.0, np.where(rows["is_home"], 1.0, -1.0))

    # columns: [mu, h, O_0..O_k, D_0..D_k]
    X = np.zeros((n, 2 + 2 * n_ent))
    X[:, 0] = 1.0
    X[:, 1] = site_sign
    X[np.arange(n), 2 + off_ent.map(idx).to_numpy()] = 1.0
    X[np.arange(n), 2 + n_ent + def_ent.map(idx).to_numpy()] = 1.0
    # mu/h get a negligible ridge only so the solve stays well-posed when h is unidentified
    # (e.g. an all-neutral-site slice); O/D get the real shrinkage.
    penalty = np.diag([1e-8, 1e-8] + [ridge_alpha] * (2 * n_ent))

    out = pd.DataFrame({"team": entities})
    for metric, off_col in [("epa", "off_epa"), ("sr", "off_sr")]:
        y = rows[off_col].to_numpy(dtype=float)
        beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        mu = beta[0]
        out[f"adj_off_{metric}"] = mu + beta[2:2 + n_ent]
        out[f"adj_def_{metric}"] = mu + beta[2 + n_ent:]

    return out[out["team"].isin(fbs_teams)].reset_index(drop=True)

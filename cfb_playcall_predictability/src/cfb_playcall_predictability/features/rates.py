"""Shared "prior season + season-to-date" rate machinery, used by both team tendencies and
opponent-defense strength.

Given per (team, game) counts `n_<m>` and sums `sum_<m>` for each metric m, returns per
(game_id, team):
  prior_<m> -- the team's full previous season, shrunk toward that season's league average
  std_<m>   -- this season's games strictly BEFORE this game, shrunk toward prior_<m>

Leakage rule: a row for game g only ever sees seasons < season(g) and, within the season,
games with game_number < game_number(g). The current game never contributes to its own row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["team", "season"]


def _shrink(total: pd.Series, n: pd.Series, prior: pd.Series, k: float) -> pd.Series:
    return (total + k * prior) / (n + k)


def prior_and_season_to_date(
    team_game: pd.DataFrame,
    metrics: list[str],
    k_prior: float,
    k_std: float,
) -> pd.DataFrame:
    """team_game: one row per (game_id, team) with season, game_number, n_<m>, sum_<m>."""
    tg = team_game.sort_values(["team", "season", "game_number"]).reset_index(drop=True)
    out = tg[["game_id", "team", "season", "game_number"]].copy()

    season_totals = tg.groupby(KEYS)[[f"n_{m}" for m in metrics] + [f"sum_{m}" for m in metrics]].sum()
    league = season_totals.groupby("season").sum()

    for m in metrics:
        n_col, s_col = f"n_{m}", f"sum_{m}"
        league_rate = (league[s_col] / league[n_col]).rename(f"league_{m}")

        # Last season's team totals, attached to this season's rows.
        last = season_totals[[n_col, s_col]].reset_index()
        last["season"] += 1
        last = last.merge(
            league_rate.rename("league_prev").reset_index().assign(season=lambda d: d["season"] + 1),
            on="season", how="left",
        )
        last[f"prior_{m}"] = _shrink(last[s_col], last[n_col], last["league_prev"], k_prior)

        rows = out[KEYS].merge(last[KEYS + [f"prior_{m}"]], on=KEYS, how="left")
        # No prior season for this team (new to the data): fall back to last season's league rate.
        league_prev = out["season"].map(lambda s: league_rate.get(s - 1, np.nan))
        prior = rows[f"prior_{m}"].fillna(league_prev).to_numpy()
        out[f"prior_{m}"] = prior

        # Season-to-date: cumulative through the PREVIOUS game.
        g = tg.groupby(KEYS)
        cum_n = g[n_col].cumsum() - tg[n_col]
        cum_s = g[s_col].cumsum() - tg[s_col]
        out[f"std_n_{m}"] = cum_n.to_numpy()
        out[f"std_{m}"] = _shrink(cum_s, cum_n, pd.Series(prior), k_std).to_numpy()

    return out

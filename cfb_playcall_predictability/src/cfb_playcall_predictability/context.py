"""Game-level context from MySQL: schedule / FBS membership, pregame betting lines, and head
coaches. Everything here is known before kickoff."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_playcall_predictability.config import DataConfig
from cfb_playcall_predictability.database import run_query


def load_games(seasons: list[int], engine: Engine | None = None) -> pd.DataFrame:
    sql = """
        SELECT game_id, season, week, start_date, neutral_site,
               home_team, away_team, home_division, away_division
        FROM games
        WHERE season BETWEEN :s0 AND :s1
    """
    games = run_query(sql, {"s0": min(seasons), "s1": max(seasons)}, engine=engine)
    games["start_date"] = pd.to_datetime(games["start_date"])
    games["neutral_site"] = games["neutral_site"].fillna(0).astype(int)
    return games


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team), with the team's game number within its season. Season-to-date
    features use every game strictly before game_number."""
    home = games.assign(team=games["home_team"], opponent=games["away_team"],
                        is_fbs=games["home_division"].eq("fbs"), opp_is_fbs=games["away_division"].eq("fbs"))
    away = games.assign(team=games["away_team"], opponent=games["home_team"],
                        is_fbs=games["away_division"].eq("fbs"), opp_is_fbs=games["home_division"].eq("fbs"))
    tg = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "week", "start_date", "team", "opponent", "is_fbs", "opp_is_fbs"]
    ]
    tg = tg.sort_values(["team", "season", "start_date", "week", "game_id"])
    tg["game_number"] = tg.groupby(["team", "season"]).cumcount() + 1
    return tg.reset_index(drop=True)


def load_lines(seasons: list[int], cfg: DataConfig, engine: Engine | None = None) -> pd.DataFrame:
    """Spread + total per game. Each is taken from the first provider in cfg.line_providers that
    has it -- consensus often lacks a total in early seasons. `spread` is the home line
    (negative = home favored), as CFBD reports it."""
    sql = """
        SELECT game_id, provider, spread, over_under
        FROM betting_lines
        WHERE season BETWEEN :s0 AND :s1
    """
    lines = run_query(sql, {"s0": min(seasons), "s1": max(seasons)}, engine=engine)
    rank = {p: i for i, p in enumerate(cfg.line_providers)}
    lines["rank"] = lines["provider"].map(rank).fillna(len(rank))
    lines = lines.sort_values(["game_id", "rank"])
    out = []
    for col in ("spread", "over_under"):
        best = lines.dropna(subset=[col]).drop_duplicates("game_id")[["game_id", col]]
        out.append(best.set_index("game_id")[col].astype(float))
    return pd.concat(out, axis=1).reset_index()


def load_head_coaches(seasons: list[int], engine: Engine | None = None) -> pd.DataFrame:
    """Head coach per (school, season) -- the one with the most games when a school changed
    coaches mid-season -- plus a flag for a different head coach than last season. Display and
    one model feature only; play-calling is attributed to the team-offense."""
    sql = """
        SELECT school, season, first_name, last_name, games
        FROM coaches
        WHERE season BETWEEN :s0 AND :s1
    """
    c = run_query(sql, {"s0": min(seasons) - 1, "s1": max(seasons)}, engine=engine)
    c["head_coach"] = c["first_name"].str.strip() + " " + c["last_name"].str.strip()
    c = c.sort_values(["school", "season", "games"], ascending=[True, True, False])
    c = c.drop_duplicates(["school", "season"])[["school", "season", "head_coach"]]
    prev = c.assign(season=c["season"] + 1).rename(columns={"head_coach": "prev_head_coach"})
    c = c.merge(prev, on=["school", "season"], how="left")
    c["new_head_coach"] = (c["head_coach"] != c["prev_head_coach"]).astype(int)
    return c.rename(columns={"school": "team"})[["team", "season", "head_coach", "new_head_coach"]]

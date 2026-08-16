"""Non-rolling, pre-game-known game-context features, keyed by (team, game_id). Source:
`spine` (schedule_spine's output, already carries neutral_site/conference_game/home_away/
rest_days) plus an optional betting_lines merge.

Unlike every other features/ module, nothing here is rolled or lagged -- these are all
identity/schedule-structure facts about the target game ITSELF, known before kickoff by
construction (an upcoming game's home/away, neutral site, and rest days are all public
information well before the game is played).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.config import DataConfig
from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

BASE_COLS = ["team", "game_id", "season", "week", "home_away", "neutral_site", "conference_game", "rest_days"]


def _pull_betting_context(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    """Mean spread/over_under across providers per game_id -- same consensus approach as the
    legacy R pipeline (Full_CFB_Game_Outcome_Historical.R). spread is stored as abs(spread);
    direction is not needed here since this is attached per-team via home_away, not per-side."""
    if not seasons:
        return pd.DataFrame(columns=["game_id", "spread", "over_under"])
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"SELECT game_id, spread, over_under FROM betting_lines WHERE season IN ({placeholders})"
    lines = run_query(sql, params=params, engine=engine)
    if lines.empty:
        return pd.DataFrame(columns=["game_id", "spread", "over_under"])
    lines["spread"] = lines["spread"].abs()
    return lines.groupby("game_id", as_index=False)[["spread", "over_under"]].mean()


def build_game_context_features(engine: Engine, spine: pd.DataFrame, seasons: list[int], data_cfg: DataConfig) -> pd.DataFrame:
    out = spine[BASE_COLS].drop_duplicates(subset=["team", "game_id"]).copy()
    out["is_home"] = out["home_away"] == "home"
    out = out.drop(columns=["home_away"])  # raw string superseded by the is_home boolean feature

    if data_cfg.include_betting_context:
        betting = _pull_betting_context(engine, seasons)
        out = out.merge(betting, on="game_id", how="left")

    return out


def describe_features(include_betting_context: bool = False) -> list[dict]:
    """include_betting_context must match data.yaml's toggle -- betting-context rows are only
    registered when the column is actually present in the modeling table (see
    dataset.build_feature_registry, which passes data_cfg.include_betting_context through),
    so the registry never lists a column that isn't really there."""
    base = {
        "source_table": "games (schedule_spine) + optionally betting_lines",
        "source_season": "t (identity/schedule structure only, known before kickoff by construction)",
        "category": "game_context",
        "known_before_kickoff": True,
        "missing_value_treatment": "median-imputed downstream if betting_lines has no row yet for a future game",
    }
    rows = [
        {**base, "feature_name": "is_home", "description": "Team is the home team for this game", "transformation": "home_away == 'home'", "expected_direction": "+"},
        {**base, "feature_name": "neutral_site", "description": "Game is at a neutral site", "transformation": "n/a", "expected_direction": "context"},
        {**base, "feature_name": "conference_game", "description": "Game is a conference game", "transformation": "n/a", "expected_direction": "context"},
        {**base, "feature_name": "rest_days", "description": "Days since this team's previous game", "transformation": "see schedule_spine.attach_rest_days", "expected_direction": "context"},
    ]
    if include_betting_context:
        rows += [
            {**base, "feature_name": "spread", "description": "Consensus abs(spread) across betting_lines providers", "transformation": "mean across providers", "expected_direction": "context"},
            {**base, "feature_name": "over_under", "description": "Consensus over/under across betting_lines providers", "transformation": "mean across providers", "expected_direction": "+"},
        ]
    return rows

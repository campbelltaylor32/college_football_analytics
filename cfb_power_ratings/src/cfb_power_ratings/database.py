"""Single point of DB access for the whole project. Every feature/target module imports
run_query from here rather than building its own connection -- keeps SQL centralized,
auditable, and (via the engine param) mockable in tests. Same pattern as
cfb_win_total_model/src/cfb_win_total_model/database.py."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cfb_power_ratings.config import DatabaseConfig, load_database_config
from cfb_power_ratings.utils.logging import get_logger

logger = get_logger(__name__)

_ENGINE: Engine | None = None


def get_engine(cfg: DatabaseConfig | None = None) -> Engine:
    """Return a memoized SQLAlchemy engine. Built once per process and shared across
    scripts/tests so connection pooling is reused."""
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine

        cfg = cfg or load_database_config()
        _ENGINE = create_engine(cfg.sqlalchemy_url, pool_pre_ping=True)
        logger.info(f"Created SQLAlchemy engine for {cfg.host}:{cfg.port}/{cfg.database}")
    return _ENGINE


def run_query(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    """Execute parameterized SQL (":name" placeholders) and return a DataFrame. All SQL in
    this project is written as parameterized text -- never f-string interpolation of season
    ints, for consistency and auditability, even where the values are otherwise safe."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def get_fbs_teams_by_season(engine: Engine, season: int) -> set[str]:
    """A team is FBS for a season if its OWN division='fbs' that season -- opponent division
    is irrelevant. Ported from cfb_win_total_model/src/cfb_win_total_model/targets.py, which
    this project's feature/target modules rely on identically."""
    sql = """
        SELECT DISTINCT school FROM (
            SELECT home_team AS school FROM games WHERE season = :season AND home_division = 'fbs'
            UNION
            SELECT away_team AS school FROM games WHERE season = :season AND away_division = 'fbs'
        ) t
    """
    df = run_query(sql, params={"season": season}, engine=engine)
    return set(df["school"])

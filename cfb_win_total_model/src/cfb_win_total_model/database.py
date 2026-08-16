"""Single point of DB access for the whole project. Every feature/target module imports
run_query from here rather than building its own connection -- keeps SQL centralized,
auditable, and (via the engine param) mockable in tests."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cfb_win_total_model.config import DatabaseConfig, load_database_config
from cfb_win_total_model.utils.logging import get_logger

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


def table_row_count(table: str, engine: Engine | None = None) -> int:
    df = run_query(f"SELECT COUNT(*) AS n FROM {table}", engine=engine)
    return int(df["n"].iloc[0])


def distinct_seasons(table: str, season_col: str = "season", engine: Engine | None = None) -> list[int]:
    df = run_query(
        f"SELECT DISTINCT {season_col} AS season FROM {table} ORDER BY {season_col}", engine=engine
    )
    return df["season"].dropna().astype(int).tolist()

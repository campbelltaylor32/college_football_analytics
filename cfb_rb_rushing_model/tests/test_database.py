from __future__ import annotations

from sqlalchemy import text

from cfb_rb_rushing_model.database import distinct_seasons, run_query, table_row_count


def test_engine_connects(engine):
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_teams_row_count(engine):
    # Stable dimension table -- verified baseline is exact, see docs/assumptions_and_limitations.md
    assert table_row_count("teams", engine=engine) == 774


def test_run_query_parameterized(engine):
    df = run_query("SELECT COUNT(*) AS n FROM games WHERE season = :season", params={"season": 2023}, engine=engine)
    assert df["n"].iloc[0] > 0


def test_plays_season_range(engine):
    seasons = distinct_seasons("plays", engine=engine)
    assert min(seasons) <= 2014
    assert max(seasons) >= 2024

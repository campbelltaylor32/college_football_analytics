from __future__ import annotations

import pandas as pd

from cfb_rb_rushing_model.player_resolution import match_rate_summary, normalize_name, resolve_players

SUFFIXES = ["jr", "sr", "ii", "iii", "iv", "v"]


def test_normalize_name_strips_suffix():
    assert normalize_name("Bob Smith Jr.", SUFFIXES) == "bob smith"
    assert normalize_name("Marcus Johnson III", SUFFIXES) == "marcus johnson"


def test_normalize_name_strips_accents_and_punctuation():
    assert normalize_name("José O'Brien", SUFFIXES) == "jose obrien"


def test_normalize_name_casefolds_and_collapses_whitespace():
    assert normalize_name("  BOB   SMITH  ", SUFFIXES) == "bob smith"


def test_normalize_name_handles_non_string_input():
    assert normalize_name(None, SUFFIXES) == ""


def test_resolve_players_classification_is_exhaustive_and_mutually_exclusive(engine, data_cfg):
    """Every input row gets exactly one of exact/normalized/ambiguous/unmatched -- no row is
    dropped and no row gets two methods."""
    distinct_names = pd.DataFrame(
        {
            "rusher_player_name": ["Ashton Jeanty", "Definitely Not A Real Player Name"],
            "pos_team": ["Boise State", "Boise State"],
            "season": [2023, 2023],
        }
    )
    resolved = resolve_players(engine, distinct_names, [2023], data_cfg.positions, data_cfg.name_suffixes_to_strip)
    assert len(resolved) == len(distinct_names)
    assert resolved["roster_match_method"].isin(["exact", "normalized", "ambiguous", "unmatched"]).all()
    jeanty_row = resolved[resolved["rusher_player_name"] == "Ashton Jeanty"].iloc[0]
    assert jeanty_row["roster_match_method"] == "exact"
    assert pd.notna(jeanty_row["athlete_id"])

    fake_row = resolved[resolved["rusher_player_name"] == "Definitely Not A Real Player Name"].iloc[0]
    assert fake_row["roster_match_method"] == "unmatched"
    assert pd.isna(fake_row["athlete_id"])


def test_resolve_players_any_position_match_rate_exceeds_rb_only(engine, data_cfg):
    """Matching against the full roster (positions=None) should resolve at least as many
    rows as restricting to RB only -- verifies the None-positions code path (used by
    data_validation's floor check) actually widens the match, not narrows it."""
    distinct_names = pd.DataFrame(
        {"rusher_player_name": ["Ashton Jeanty"], "pos_team": ["Boise State"], "season": [2023]}
    )
    rb_only = resolve_players(engine, distinct_names, [2023], ["RB"], data_cfg.name_suffixes_to_strip)
    any_position = resolve_players(engine, distinct_names, [2023], None, data_cfg.name_suffixes_to_strip)
    assert (any_position["roster_match_method"] == "exact").sum() >= (rb_only["roster_match_method"] == "exact").sum()


def test_match_rate_summary_shares_sum_to_one_per_season():
    resolved = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "roster_match_method": ["exact", "exact", "normalized", "unmatched"],
        }
    )
    summary = match_rate_summary(resolved)
    assert abs(summary.groupby("season")["share"].sum().iloc[0] - 1.0) < 1e-9

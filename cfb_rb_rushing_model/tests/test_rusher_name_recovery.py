from __future__ import annotations

import pandas as pd

from cfb_rb_rushing_model.player_game_rushing import recover_rusher_names

ROSTERS = pd.DataFrame(
    [
        (2026, "Georgia", 3, "Nate", "Frazier"),
        (2026, "Georgia", 3, "Quintavius", "Johnson"),
        (2026, "Wofford", 6, "Gavin", "Modest Jr."),
        (2026, "Purdue", 21, "Devin", "Mockobee"),
        (2026, "Purdue", 22, "Dan", "Mockobee"),
    ],
    columns=["season", "team", "jersey", "first_name", "last_name"],
)


def _recover(rows):
    raw = pd.DataFrame(rows, columns=["rusher_player_name", "pos_team", "season", "play_text"])
    return recover_rusher_names(raw, ROSTERS)["rusher_player_name"].tolist()


def test_existing_names_not_on_the_roster_are_left_alone():
    assert _recover([("Somebody Else", "Georgia", 2026, "#3 N.Frazier rush middle for 4 yards")]) == ["Somebody Else"]


def test_name_variants_collapse_to_the_roster_spelling():
    assert _recover([
        ("Gavin Modest", "Wofford", 2026, "Gavin Modest 3 Yd Run"),
        ("G. Modest Jr.", "Wofford", 2026, "G. Modest Jr. run for 0 yds"),
        (None, "Wofford", 2026, "#6 G.Modest, Jr. rush middle for 1 yard gain"),
    ]) == ["Gavin Modest Jr."] * 3


def test_jersey_format_picks_the_player_whose_initial_and_last_name_match():
    # Georgia #3 is both an RB and an LB; only the RB matches N. Frazier
    assert _recover([(None, "Georgia", 2026, "(14:14) Shotgun #3 N.Frazier rush middle for 7 yards gain to the UGA49")]) == ["Nate Frazier"]


def test_suffix_with_and_without_comma():
    assert _recover([
        (None, "Wofford", 2026, "No Huddle-Shotgun #6 G.Modest Jr. rush middle for 5 yards gain"),
        (None, "Wofford", 2026, "No Huddle-Shotgun #6 G.Modest, Jr. rush middle for 5 yards gain"),
    ]) == ["Gavin Modest Jr.", "Gavin Modest Jr."]


def test_wrong_jersey_falls_back_to_unique_initial_and_last_name():
    assert _recover([(None, "Georgia", 2026, "#30 N.Frazier rush left for 2 yards gain")]) == ["Nate Frazier"]


def test_ambiguous_or_unknown_rusher_stays_null():
    # two D. Mockobees and the jersey doesn't break the tie; an unknown player has no match
    assert _recover([
        (None, "Purdue", 2026, "#99 D.Mockobee rush left for 2 yards gain"),
        (None, "Purdue", 2026, "#8 Z.Nobody rush left for 2 yards gain"),
    ]) == [None, None]


def test_full_name_formats_are_taken_as_written():
    assert _recover([
        (None, "Central Michigan", 2026, "Angel Flores 50 Yd Run (Caden VanHuis PAT blocked)"),  # no roster row: as written
        (None, "Purdue", 2026, "Devin Mockobee run for 3 yds to the PUR 24"),
    ]) == ["Angel Flores", "Devin Mockobee"]


def test_kneel_downs_are_not_rushers():
    assert _recover([(None, "Georgia", 2026, "(00:39) Kneel down by #16 J.Mason at UGA32 for loss of 1 yard")]) == [None]

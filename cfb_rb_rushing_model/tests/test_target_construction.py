"""Synthetic fixture tests for targets.merge_realized_onto_eligible -- the core design
decision in targets.py: an eligible spine row with no matching realized-game row must end up
with rushing_yards=0/carries=0 in the target table, NOT be dropped."""

from __future__ import annotations

import pandas as pd

from cfb_rb_rushing_model.targets import merge_realized_onto_eligible


def _eligible_row(athlete_id, game_id):
    return {"athlete_id": athlete_id, "game_id": game_id, "team": "Ohio", "opponent": "Akron", "season": 2023, "week": 3, "eligible": True}


def test_eligible_row_with_no_realized_carries_is_zero_filled_not_dropped():
    eligible_rows = pd.DataFrame([_eligible_row(1, 100), _eligible_row(2, 100)])
    realized = pd.DataFrame(
        [{"athlete_id": 1, "game_id": 100, "carries": 18, "rushing_yards": 95, "explosive_runs": 2, "red_zone_carries": 3, "success_rate": 0.5, "yards_per_carry": 5.3}]
    )
    result = merge_realized_onto_eligible(eligible_rows, realized)

    assert len(result) == 2  # player 2's row survives despite no realized-carries match
    player2 = result[result["athlete_id"] == 2].iloc[0]
    assert player2["rushing_yards"] == 0
    assert player2["carries"] == 0
    assert player2["played"] == False  # noqa: E712

    player1 = result[result["athlete_id"] == 1].iloc[0]
    assert player1["rushing_yards"] == 95
    assert player1["played"] == True  # noqa: E712


def test_all_eligible_rows_preserved_regardless_of_realized_match():
    eligible_rows = pd.DataFrame([_eligible_row(i, 100) for i in range(5)])
    realized = pd.DataFrame(columns=["athlete_id", "game_id", "carries", "rushing_yards", "explosive_runs", "red_zone_carries", "success_rate", "yards_per_carry"])
    result = merge_realized_onto_eligible(eligible_rows, realized)
    assert len(result) == 5
    assert (result["rushing_yards"] == 0).all()
    assert (~result["played"]).all()


def test_no_negative_carries_after_fill():
    eligible_rows = pd.DataFrame([_eligible_row(1, 100)])
    realized = pd.DataFrame(columns=["athlete_id", "game_id", "carries", "rushing_yards", "explosive_runs", "red_zone_carries", "success_rate", "yards_per_carry"])
    result = merge_realized_onto_eligible(eligible_rows, realized)
    assert (result["carries"] >= 0).all()

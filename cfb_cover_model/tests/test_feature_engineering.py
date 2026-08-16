import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.feature_engineering import build_transform_variant, categorize_features


def test_engineered_prev_week_suffix_recognized_as_temporal():
    """Regression test for a real bug: engineered_features.py names columns with a
    *trailing* _prev_week suffix (matchup_adj_<stat>_prev_week), unlike raw columns which
    use a *leading* prev_week_ prefix. Before the fix, these fell through to
    non_temporal and bypassed transform-ablation filtering entirely - always present
    regardless of which transform was being tested, silently favoring whichever ablation
    candidate happened to be evaluated (since prev_week-suffixed engineered features leaked
    into every combination)."""
    columns = [
        "home_special_teams_net_score_prev_week",
        "home_special_teams_net_score_avg_all",
        "home_special_teams_net_score_avg3",
        "away_matchup_adj_third_down_rate_prev_week",
        "away_matchup_adj_third_down_rate_avg3",
        "home_rushing_usage",  # genuinely non-temporal - must stay non_temporal
    ]
    temporal, non_temporal, _context = categorize_features(columns)

    assert ("home_", "special_teams_net_score") in temporal
    assert temporal[("home_", "special_teams_net_score")]["prev_week"] == "home_special_teams_net_score_prev_week"
    assert ("away_", "matchup_adj_third_down_rate") in temporal
    assert temporal[("away_", "matchup_adj_third_down_rate")]["prev_week"] == "away_matchup_adj_third_down_rate_prev_week"
    assert "home_rushing_usage" in non_temporal


def test_build_transform_variant_excludes_engineered_prev_week_when_not_requested(rng):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "home_special_teams_net_score_prev_week": [1.0, 2.0],
            "home_special_teams_net_score_avg3": [3.0, 4.0],
            "home_rushing_usage": [0.5, 0.6],
        }
    )
    feature_columns = list(frame.columns)

    variant, cols = build_transform_variant(frame, feature_columns, ["avg3"])
    assert "home_special_teams_net_score_prev_week" not in cols
    assert "home_special_teams_net_score_avg3" in cols
    assert "home_rushing_usage" in cols  # non-temporal always passes through

#!/usr/bin/env python
"""Stage 1 (cheap, no training) prototype of "predict per-game win probability, sum to a
season total" as an alternative to directly regressing on season win counts. Uses two
pre-existing per-game fields already in the `games` table (home_pregame_elo/away_pregame_elo
and home_post_win_prob/away_post_win_prob -- CFBD's field name is "post_win_prob" but it is
in fact a pregame win-probability estimate, verified below) rather than building a new
per-game classifier from scratch. If this cheap check looks promising, a heavier Stage 2
(train an actual per-game logistic regression on pre-game-known features, walk-forward CV)
would be the next step -- not built here per the user's stated preference for a cheap check
first; see the module docstring note at the bottom of main().

Writes only to outputs/diagnostics_compression/experiments/ -- reads the live DB read-only
and outputs/model_comparison/oof_predictions.csv (for the gradient_boosting comparison), never
writes to outputs/model_comparison/ or outputs/models/.

Usage:
    python scripts/diagnostics/win_probability_sum_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.database import get_engine, run_query
from cfb_win_total_model.dataset import NON_FEATURE_COLS  # noqa: F401  (imported for parity/consistency with sibling scripts)
from cfb_win_total_model.modeling.evaluation import evaluate_predictions, regression_slope_intercept, std_range_summary
from cfb_win_total_model.modeling.train import TARGET_COL
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS, OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

GAMES_QUERY = """
SELECT game_id, season, home_team, away_team, home_points, away_points,
       home_pregame_elo, away_pregame_elo, home_post_win_prob, away_post_win_prob, completed
FROM games
WHERE season BETWEEN :start_season AND :end_season AND completed = 1
"""


def _elo_implied_prob(team_elo: pd.Series, opp_elo: pd.Series) -> pd.Series:
    """Standard logistic Elo win-probability transform (400-point scale, the same constant
    used by chess/CFBD Elo systems)."""
    return 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))


def _sanity_check_favorite_win_rate(games: pd.DataFrame) -> None:
    """Verifies home_post_win_prob/pregame_elo actually predict game outcomes above 50% --
    if this fails, the field's semantics need re-investigation before trusting anything
    downstream. Checked two ways: (a) the higher-rated side wins more than half the time,
    (b) games where home_post_win_prob is near a toss-up (0.4-0.6) really are close to 50/50
    for the home team -- calibration, not just discrimination."""
    completed = games.dropna(subset=["home_points", "away_points"])
    home_won = completed["home_points"] > completed["away_points"]

    elo_games = completed.dropna(subset=["home_pregame_elo", "away_pregame_elo"])
    elo_favorite_win_rate = (
        (home_won.loc[elo_games.index]) == (elo_games["home_pregame_elo"] > elo_games["away_pregame_elo"])
    ).mean()
    logger.info(f"Elo-favorite win rate across {len(elo_games)} completed games: {elo_favorite_win_rate:.3f} (expect ~0.65-0.75)")
    assert elo_favorite_win_rate > 0.55, (
        f"Elo favorite only wins {elo_favorite_win_rate:.3f} of games -- pregame_elo semantics look wrong, "
        "do not trust the Elo-implied-probability variant below"
    )

    wp_games = completed.dropna(subset=["home_post_win_prob"])
    wp_favorite_win_rate = ((home_won.loc[wp_games.index]) == (wp_games["home_post_win_prob"] > 0.5)).mean()
    toss_up = wp_games[(wp_games["home_post_win_prob"] >= 0.4) & (wp_games["home_post_win_prob"] < 0.6)]
    toss_up_home_win_rate = home_won.loc[toss_up.index].mean() if len(toss_up) else float("nan")
    logger.info(
        f"home_post_win_prob-favorite win rate across {len(wp_games)} completed games: {wp_favorite_win_rate:.3f} "
        f"(expect > 0.7); toss-up bucket (0.4-0.6, n={len(toss_up)}) home win rate: {toss_up_home_win_rate:.3f} (expect ~0.5)"
    )
    assert wp_favorite_win_rate > 0.6, (
        f"home_post_win_prob favorite only wins {wp_favorite_win_rate:.3f} of games -- despite the 'post' in its "
        "name this field is documented as CFBD's PREGAME win probability, but this doesn't look right; do not "
        "trust the win_prob_sum_baseline variant below"
    )
    logger.info("Sanity check PASSED: both pregame_elo and home_post_win_prob discriminate winners well above 50%")


def _stack_team_perspective(games: pd.DataFrame) -> pd.DataFrame:
    home = games.rename(
        columns={
            "home_team": "school",
            "away_team": "opponent",
            "home_pregame_elo": "team_elo",
            "away_pregame_elo": "opp_elo",
            "home_post_win_prob": "team_winprob",
        }
    )[["game_id", "season", "school", "opponent", "team_elo", "opp_elo", "team_winprob"]]

    away = games.rename(
        columns={
            "away_team": "school",
            "home_team": "opponent",
            "away_pregame_elo": "team_elo",
            "home_pregame_elo": "opp_elo",
            "away_post_win_prob": "team_winprob",
        }
    )[["game_id", "season", "school", "opponent", "team_elo", "opp_elo", "team_winprob"]]

    return pd.concat([home, away], ignore_index=True)


def build_predicted_wins_table(games: pd.DataFrame, fbs_school_seasons: pd.DataFrame) -> pd.DataFrame:
    stacked = _stack_team_perspective(games)
    stacked["elo_implied_prob"] = _elo_implied_prob(stacked["team_elo"], stacked["opp_elo"])

    # Restrict to the same (school, season) FBS universe the production model is scored on,
    # rather than re-deriving FBS-membership logic from division flags.
    stacked = stacked.merge(fbs_school_seasons[["school", "season"]], on=["school", "season"], how="inner")

    agg = stacked.groupby(["school", "season"]).agg(
        n_games_total=("game_id", "count"),
        n_games_with_winprob=("team_winprob", "count"),
        predicted_wins_via_winprob_sum=("team_winprob", "sum"),
        n_games_with_elo=("elo_implied_prob", "count"),
        predicted_wins_via_elo_sum=("elo_implied_prob", "sum"),
    ).reset_index()

    agg["winprob_coverage_pct"] = agg["n_games_with_winprob"] / agg["n_games_total"]
    agg["elo_coverage_pct"] = agg["n_games_with_elo"] / agg["n_games_total"]
    return agg


def build_oof_shaped_frame(predicted_wins: pd.DataFrame, actual_wins: pd.DataFrame, model_col: str, model_name: str) -> pd.DataFrame:
    merged = predicted_wins.merge(actual_wins, on=["school", "season"], how="inner")
    merged = merged.dropna(subset=[model_col])
    return pd.DataFrame(
        {
            "school": merged["school"],
            "season": merged["season"],
            "fold_validation_season": merged["season"],
            "model_name": model_name,
            "y_true": merged[TARGET_COL],
            "y_pred": merged[model_col],
        }
    )


def build_comparison_table(oof_shaped: pd.DataFrame, walk_forward_seasons: list[int], holdout_season: int) -> pd.DataFrame:
    baseline_oof = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv")
    baseline_gb = baseline_oof[baseline_oof["model_name"] == "gradient_boosting"]
    baseline_holdout = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "holdout_2025_predictions.csv")

    rows = []

    def _add_row(label: str, split: str, y_true, y_pred):
        metrics = evaluate_predictions(y_true, y_pred)
        summary = std_range_summary(y_true, y_pred)
        slope_a_on_p, _ = regression_slope_intercept(y_pred, y_true)
        slope_p_on_a, _ = regression_slope_intercept(y_true, y_pred)
        rows.append(
            {
                "model_name": label,
                "split": split,
                "n": summary["n"],
                "mae": metrics["mae"],
                "std_actual": summary["std_actual"],
                "std_pred": summary["std_pred"],
                "std_ratio_pred_over_actual": summary["std_ratio_pred_over_actual"],
                "slope_actual_on_pred": slope_a_on_p,
                "slope_pred_on_actual": slope_p_on_a,
            }
        )

    for model_name in oof_shaped["model_name"].unique():
        group = oof_shaped[oof_shaped["model_name"] == model_name]
        wf_group = group[group["season"].isin(walk_forward_seasons)]
        if not wf_group.empty:
            _add_row(model_name, "walk_forward_seasons_pooled", wf_group["y_true"], wf_group["y_pred"])
        holdout_group = group[group["season"] == holdout_season]
        if not holdout_group.empty:
            _add_row(model_name, "holdout_2025", holdout_group["y_true"], holdout_group["y_pred"])

    _add_row("gradient_boosting", "walk_forward_seasons_pooled", baseline_gb["y_true"], baseline_gb["y_pred"])
    _add_row("gradient_boosting_final_refit_2025", "holdout_2025", baseline_holdout["y_true"], baseline_holdout["y_pred"])

    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    modeling_cfg = load_modeling_config()
    engine = get_engine()

    walk_forward_seasons = modeling_cfg.walk_forward_validation_seasons
    holdout_season = modeling_cfg.final_holdout_season
    start_season, end_season = min(walk_forward_seasons), holdout_season

    logger.info(f"Querying games for seasons {start_season}-{end_season}...")
    games = run_query(GAMES_QUERY, params={"start_season": start_season, "end_season": end_season}, engine=engine)
    logger.info(f"{len(games)} completed games loaded")

    logger.info("Running sanity check on home_pregame_elo / home_post_win_prob semantics...")
    _sanity_check_favorite_win_rate(games)

    dataset_df = pd.read_parquet(DATASET_PATH)
    fbs_school_seasons = dataset_df[["school", "season"]].drop_duplicates()
    actual_wins = dataset_df[["school", "season", TARGET_COL]].drop_duplicates()

    predicted_wins = build_predicted_wins_table(games, fbs_school_seasons)
    predicted_wins_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "win_prob_sum_predicted_wins.csv"
    predicted_wins.to_csv(predicted_wins_path, index=False)
    logger.info(
        f"Coverage: mean winprob_coverage_pct={predicted_wins['winprob_coverage_pct'].mean():.2%}, "
        f"mean elo_coverage_pct={predicted_wins['elo_coverage_pct'].mean():.2%} "
        "(teams/seasons with partial coverage are summed over whatever games ARE available; "
        "n_games_with_* columns expose exactly how many that was per team-season, nothing is silently dropped)"
    )

    oof_winprob = build_oof_shaped_frame(predicted_wins, actual_wins, "predicted_wins_via_winprob_sum", "win_prob_sum_baseline")
    oof_elo = build_oof_shaped_frame(predicted_wins, actual_wins, "predicted_wins_via_elo_sum", "elo_implied_sum_baseline")
    oof_shaped = pd.concat([oof_winprob, oof_elo], ignore_index=True)
    oof_shaped_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "win_prob_sum_oof_predictions.csv"
    oof_shaped.to_csv(oof_shaped_path, index=False)
    logger.info(f"Wrote {len(oof_shaped)} win-prob-sum OOF-shaped predictions -> {oof_shaped_path}")

    comparison = build_comparison_table(oof_shaped, walk_forward_seasons, holdout_season)
    comparison_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "win_prob_sum_vs_direct_regression_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Win-probability-sum vs. direct regression comparison:\n{comparison.to_string(index=False)}")
    logger.info(f"Wrote comparison table -> {comparison_path}")

    logger.warning(
        "CAVEAT (read before treating the numbers above as a fair win): home_pregame_elo "
        "updates game-by-game WITHIN a season as results come in, and home_post_win_prob is "
        "computed per-game using whatever information is available at that game's kickoff -- "
        "including that season's already-played games. Summing these across a team's full "
        "season therefore blends in in-season information (week 10's win prob already reflects "
        "weeks 1-9 of THAT season) that the production season-level model deliberately does not "
        "have access to before the season starts. The dramatically better MAE/std-ratio numbers "
        "above are NOT an apples-to-apples 'this approach is better, switch to it' result -- they "
        "reflect an easier, partially in-season task. A fair comparison would need each game's win "
        "probability estimated using only information available before that season began, which "
        "neither home_pregame_elo nor home_post_win_prob provide out of the box. See "
        "docs/diagnostics_compression_report.md for how this is weighed."
    )
    logger.info(
        "Stage 2 (a from-scratch per-game logistic regression trained walk-forward on "
        "pre-game-known features, rather than relying on CFBD's own Elo/win-prob fields) is "
        "NOT built here -- per the agreed scope, Stage 1's numbers above should be reviewed "
        "first to decide whether that additional investment is warranted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

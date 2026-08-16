"""
Extends the base Pythagorean analysis with opponent-quality weighting: does
adjusting for how good an opponent was improve the fit to actual 2025 win%,
and is preseason recruiting talent a good proxy for "how good" an opponent
actually was?

Two opponent-quality proxies are compared:
  - talent: preseason Scaled_Talent (recruiting composite, z-scored)
  - srs: a season-derived Simple Rating System rating computed from actual
    2025 point margins (the standard Sports-Reference-style method), used
    as a ceiling benchmark since it reflects real in-season performance
    rather than a preseason guess.

For each proxy, opponent-adjusted points-for/against are built as:
  weight = 1 + alpha * opponent_quality_z
  PF_adj = sum(points_for * weight)
  PA_adj = sum(points_against / weight)
with alpha fit to minimize error against actual win% (k fixed at 2, so the
comparison to the classic baseline isolates the effect of the adjustment).
"""
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythagorean_analysis import (
    FBS_TEAMS_CSV,
    OUTPUT_DIR,
    RESULTS_CSV,
    build_team_game_frame,
    load_fbs_teams,
    pythagorean_win_pct,
    report_metrics,
)

SRS_ITERATIONS = 200
ALPHA_BOUNDS = (-0.3, 0.3)


def compute_srs(team_games: pd.DataFrame, fbs_teams: set) -> pd.Series:
    """Season-derived Simple Rating System rating, FBS-vs-FBS games only."""
    fbs_games = team_games[team_games["team"].isin(fbs_teams) & team_games["opponent_is_fbs"]].copy()
    fbs_games["margin"] = fbs_games["points_for"] - fbs_games["points_against"]

    teams = sorted(fbs_teams)
    srs = pd.Series(0.0, index=teams)
    for _ in range(SRS_ITERATIONS):
        opp_srs = fbs_games["opponent"].map(srs).values
        fbs_games["adj_margin"] = fbs_games["margin"] + opp_srs
        new_srs = fbs_games.groupby("team")["adj_margin"].mean().reindex(teams).fillna(0.0)
        new_srs -= new_srs.mean()
        srs = new_srs
    return srs


def build_adjusted_pf_pa(team_games: pd.DataFrame, quality_z: pd.Series, alpha: float, fbs_teams: set) -> pd.DataFrame:
    games = team_games[team_games["team"].isin(fbs_teams)].copy()
    opponent_quality = games["opponent"].map(quality_z).fillna(0.0)
    weight = np.where(games["opponent_is_fbs"], 1.0 + alpha * opponent_quality, 1.0)
    games["pf_weighted"] = games["points_for"] * weight
    games["pa_weighted"] = games["points_against"] / weight
    return games.groupby("team").agg(pf_adj=("pf_weighted", "sum"), pa_adj=("pa_weighted", "sum"))


def fit_alpha(team_games: pd.DataFrame, actual_win_pct: pd.Series, quality_z: pd.Series, fbs_teams: set) -> float:
    def mse_for_alpha(alpha: float) -> float:
        adj = build_adjusted_pf_pa(team_games, quality_z, alpha, fbs_teams)
        pred = pythagorean_win_pct(adj["pf_adj"], adj["pa_adj"], 2.0).reindex(actual_win_pct.index)
        return mean_squared_error(actual_win_pct, pred)

    result = minimize_scalar(mse_for_alpha, bounds=ALPHA_BOUNDS, method="bounded")
    return result.x


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    fbs_teams = load_fbs_teams(FBS_TEAMS_CSV)
    team_games = build_team_game_frame(RESULTS_CSV, fbs_teams)

    base = pd.read_csv(OUTPUT_DIR / "team_pythagorean_2025.csv").set_index("team")
    base = base[base.index.isin(fbs_teams)]
    base_metrics = pd.read_csv(OUTPUT_DIR / "metrics_summary_2025.csv")
    fitted_k_label = base_metrics.loc[1, "variant"]

    talent = pd.read_csv(FBS_TEAMS_CSV).set_index("team")["Scaled_Talent"]

    srs_raw = compute_srs(team_games, fbs_teams)
    srs_z = (srs_raw - srs_raw.mean()) / srs_raw.std()
    print(f"SRS check -- mean={srs_z.mean():.4f}, std={srs_z.std():.4f}")
    print("Top 5 SRS:\n", srs_z.sort_values(ascending=False).head(5))
    print("Bottom 5 SRS:\n", srs_z.sort_values().head(5))

    r, p = pearsonr(talent.reindex(srs_z.index), srs_z)
    print(f"\nTalent vs. SRS correlation: r={r:.3f} (p={p:.4g})\n")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(talent.reindex(srs_z.index), srs_z, alpha=0.6)
    ax.set_xlabel("Preseason Scaled_Talent")
    ax.set_ylabel("Season-derived SRS (z-scored)")
    ax.set_title(f"2025: Preseason talent vs. actual team quality (r={r:.2f})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "talent_vs_srs_correlation.png", dpi=150)

    actual = base["actual_win_pct"]

    talent_alpha = fit_alpha(team_games, actual, talent, fbs_teams)
    talent_adj = build_adjusted_pf_pa(team_games, talent, talent_alpha, fbs_teams)
    talent_pred = pythagorean_win_pct(talent_adj["pf_adj"], talent_adj["pa_adj"], 2.0)

    srs_alpha = fit_alpha(team_games, actual, srs_z, fbs_teams)
    srs_adj = build_adjusted_pf_pa(team_games, srs_z, srs_alpha, fbs_teams)
    srs_pred = pythagorean_win_pct(srs_adj["pf_adj"], srs_adj["pa_adj"], 2.0)

    print(f"Fitted alpha -- talent-weighted: {talent_alpha:.4f}, SRS-weighted: {srs_alpha:.4f}\n")

    metrics = [
        report_metrics("Classic (k=2, unadjusted)", actual, base["pyth_win_pct_exp2"]),
        report_metrics(f"{fitted_k_label}, unadjusted", actual, base["pyth_win_pct_fit"]),
        report_metrics(f"Talent-weighted (alpha={talent_alpha:.3f})", actual, talent_pred.reindex(actual.index)),
        report_metrics(f"SRS-weighted (alpha={srs_alpha:.3f})", actual, srs_pred.reindex(actual.index)),
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary_opponent_adjusted_2025.csv", index=False)

    results = base.copy()
    results["talent_scaled"] = talent
    results["srs_z"] = srs_z
    results["pf_adj_talent"] = talent_adj["pf_adj"]
    results["pa_adj_talent"] = talent_adj["pa_adj"]
    results["pyth_win_pct_talent"] = talent_pred
    results["pf_adj_srs"] = srs_adj["pf_adj"]
    results["pa_adj_srs"] = srs_adj["pa_adj"]
    results["pyth_win_pct_srs"] = srs_pred
    results = results.sort_values("actual_win_pct", ascending=False)
    results.to_csv(OUTPUT_DIR / "opponent_adjusted_2025.csv")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, base["pyth_win_pct_exp2"], alpha=0.5, label="Classic (unadjusted)")
    ax.scatter(actual, talent_pred.reindex(actual.index), alpha=0.5, label="Talent-weighted")
    ax.scatter(actual, srs_pred.reindex(actual.index), alpha=0.5, label="SRS-weighted")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect prediction")
    ax.set_xlabel("Actual win %")
    ax.set_ylabel("Pythagorean predicted win %")
    ax.set_title("2025 CFB: Opponent-adjusted Pythagorean win %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_opponent_adjusted.png", dpi=150)

    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

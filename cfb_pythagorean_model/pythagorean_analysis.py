"""
Tests how well the Pythagorean expectation (points-for/points-against, Bill
James style) predicts actual 2025 CFB season winning percentage per team.

Reads the cumulative 2025 season results CSV (weeks 1-14), aggregates to
team-season points-for/points-against/wins, then compares the Pythagorean
win% prediction (classic exponent=2, and an exponent fit to this season)
against actual win% using MAE, RMSE, and R^2.
"""
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "Data" / "CFB_Gambling_Results_2025_14.csv"
FBS_TEAMS_CSV = REPO_ROOT / "Data" / "CFB_Team_Talent_Data_2025.csv"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_fbs_teams(fbs_teams_csv: pathlib.Path) -> set:
    return set(pd.read_csv(fbs_teams_csv)["team"])


def build_team_game_frame(results_csv: pathlib.Path, fbs_teams: set) -> pd.DataFrame:
    """One row per team-game: team, opponent, points_for, points_against, win.

    Covers every division in the source file (not just FBS) so each FBS
    team's full slate -- including money games vs. non-FBS opponents --
    is represented; `opponent_is_fbs` flags which rows have an opponent
    with a quality signal (talent/SRS) available.
    """
    games = pd.read_csv(results_csv)
    games = games[games["season"] == 2025]
    games = games.dropna(subset=["home_points", "away_points"])

    home = games[["home_team", "away_team", "home_points", "away_points"]].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_points": "points_for",
            "away_points": "points_against",
        }
    )
    away = games[["away_team", "home_team", "away_points", "home_points"]].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_points": "points_for",
            "home_points": "points_against",
        }
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
    team_games["opponent_is_fbs"] = team_games["opponent"].isin(fbs_teams)
    return team_games


def load_team_season(results_csv: pathlib.Path, fbs_teams: set) -> pd.DataFrame:
    team_games = build_team_game_frame(results_csv, fbs_teams)

    # Every FBS team's full slate counts (including money games vs. FCS/etc.),
    # but we only report on FBS teams -- the source file spans all divisions.
    team_season = team_games.groupby("team").agg(
        games=("win", "count"),
        wins=("win", "sum"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
    )
    team_season["actual_win_pct"] = team_season["wins"] / team_season["games"]
    team_season = team_season.reset_index()
    return team_season[team_season["team"].isin(fbs_teams)].reset_index(drop=True)


def pythagorean_win_pct(points_for: pd.Series, points_against: pd.Series, exponent: float) -> pd.Series:
    pf_k = points_for**exponent
    pa_k = points_against**exponent
    return pf_k / (pf_k + pa_k)


def fit_best_exponent(team_season: pd.DataFrame) -> float:
    def mse_for_exponent(k: float) -> float:
        pred = pythagorean_win_pct(team_season["points_for"], team_season["points_against"], k)
        return mean_squared_error(team_season["actual_win_pct"], pred)

    result = minimize_scalar(mse_for_exponent, bounds=(0.5, 6.0), method="bounded")
    return result.x


def report_metrics(name: str, actual: pd.Series, predicted: pd.Series) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)
    print(f"{name:>28s}   MAE={mae:.4f}   RMSE={rmse:.4f}   R^2={r2:.4f}")
    return {"variant": name, "mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    fbs_teams = load_fbs_teams(FBS_TEAMS_CSV)
    team_season = load_team_season(RESULTS_CSV, fbs_teams)

    team_season["pyth_win_pct_exp2"] = pythagorean_win_pct(
        team_season["points_for"], team_season["points_against"], 2.0
    )
    best_k = fit_best_exponent(team_season)
    team_season["pyth_win_pct_fit"] = pythagorean_win_pct(
        team_season["points_for"], team_season["points_against"], best_k
    )

    print(f"2025 season, {len(team_season)} teams, fitted exponent k={best_k:.3f}\n")
    metrics = [
        report_metrics("Classic exponent (k=2)", team_season["actual_win_pct"], team_season["pyth_win_pct_exp2"]),
        report_metrics(f"Fitted exponent (k={best_k:.3f})", team_season["actual_win_pct"], team_season["pyth_win_pct_fit"]),
    ]
    metrics_df = pd.DataFrame(metrics)

    team_season = team_season.sort_values("actual_win_pct", ascending=False)
    team_season.to_csv(OUTPUT_DIR / "team_pythagorean_2025.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary_2025.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(team_season["actual_win_pct"], team_season["pyth_win_pct_exp2"], alpha=0.6, label="Classic (k=2)")
    ax.scatter(team_season["actual_win_pct"], team_season["pyth_win_pct_fit"], alpha=0.6, label=f"Fitted (k={best_k:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect prediction")
    ax.set_xlabel("Actual win %")
    ax.set_ylabel("Pythagorean predicted win %")
    ax.set_title("2025 CFB: Actual vs. Pythagorean-expected win %")
    ax.legend(loc="upper left")

    r2_exp2 = metrics_df.loc[metrics_df["variant"] == "Classic exponent (k=2)", "r2"].iloc[0]
    r2_fit = metrics_df.loc[metrics_df["variant"].str.startswith("Fitted exponent"), "r2"].iloc[0]
    ax.text(
        0.98, 0.04,
        f"$R^2$ (k=2) = {r2_exp2:.3f}\n$R^2$ (k={best_k:.2f}) = {r2_fit:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_actual_vs_pyth.png", dpi=150)

    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

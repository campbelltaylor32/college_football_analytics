"""
Which 2025 FBS teams deviated most from their classic (k=2) Pythagorean
expectation? A positive deviation means a team won more than its point
differential predicted (overperformed / "lucky" in close-game terms); a
negative deviation means it won less (underperformed).

Reads pythagorean_analysis.py's output, so run that first.
"""
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythagorean_analysis import OUTPUT_DIR

TOP_N = 10


def main() -> None:
    team_season = pd.read_csv(OUTPUT_DIR / "team_pythagorean_2025.csv")
    team_season["deviation"] = team_season["actual_win_pct"] - team_season["pyth_win_pct_exp2"]
    team_season = team_season.sort_values("deviation", ascending=False)

    overperformers = team_season.head(TOP_N)
    underperformers = team_season.tail(TOP_N).sort_values("deviation")

    cols = ["team", "games", "wins", "actual_win_pct", "pyth_win_pct_exp2", "deviation"]
    table = pd.concat([overperformers, underperformers])[cols].reset_index(drop=True)
    table.to_csv(OUTPUT_DIR / "deviation_top_bottom_10.csv", index=False)

    print(f"Top {TOP_N} overperformers (actual win% > Pythagorean expectation):")
    print(overperformers[cols].to_string(index=False))
    print(f"\nTop {TOP_N} underperformers (actual win% < Pythagorean expectation):")
    print(underperformers[cols].to_string(index=False))

    plot_data = pd.concat([overperformers, underperformers]).sort_values("deviation")
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in plot_data["deviation"]]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(plot_data["team"], plot_data["deviation"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Actual win% − Pythagorean expected win% (k=2)")
    ax.set_title(f"2025 CFB: Top {TOP_N} / Bottom {TOP_N} Pythagorean deviation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deviation_top_bottom_10.png", dpi=150)

    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

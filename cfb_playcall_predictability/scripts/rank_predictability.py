"""Rank FBS offenses by how predictable their run/pass calls are, plus a year-over-year
stability check on the score itself.

Writes:
  outputs/rankings/<target>_predictability.csv   one row per team, 1 = least predictable
  outputs/rankings/team_season_predictability.csv every out-of-sample team-season (2019-<target>)
  outputs/rankings/stability.csv                  correlation of team scores, season vs next season

Usage: python scripts/rank_predictability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_playcall_predictability.config import load_modeling_config
from cfb_playcall_predictability.modeling.artifacts import load_latest_production_artifact
from cfb_playcall_predictability.predictability import predictability_table, team_scores
from cfb_playcall_predictability.utils.logging import get_logger
from cfb_playcall_predictability.utils.paths import (
    DATA_PROCESSED_DIR, OUTPUTS_MODEL_COMPARISON, OUTPUTS_RANKINGS, ensure_output_dirs,
)

logger = get_logger("rank_predictability")
# A full season is ~700-900 calls per team; demand most of one for the stability comparison.
MIN_FULL_SEASON_CALLS = 400


def main() -> None:
    cfg = load_modeling_config()
    ensure_output_dirs()
    model, features, meta = load_latest_production_artifact()
    best = meta["model"]
    if cfg.target_season in meta["trained_on_seasons"]:
        raise RuntimeError(f"Production model was trained on {cfg.target_season} -- scores would be in-sample")

    df = pd.read_parquet(DATA_PROCESSED_DIR / "modeling_dataset.parquet")
    target = df[df["season"] == cfg.target_season].copy()
    target["p_pass"] = model.predict_proba(target[features])[:, 1]
    logger.info(f"Scored {len(target):,} {cfg.target_season} calls (weeks {target['week'].min()}-"
                f"{target['week'].max()}) with {best}")

    table = predictability_table(target, "p_pass", cfg.min_team_plays, cfg.bootstrap_reps,
                                 cfg.ci_level, cfg.random_seed)
    table["through_week"] = int(target["week"].max())
    out_path = OUTPUTS_RANKINGS / f"{cfg.target_season}_predictability.csv"
    table.to_csv(out_path, index=False)

    show = ["unpredictability_rank", "team", "head_coach", "games", "calls", "pass_rate",
            "predictability", "predictability_lo", "predictability_hi", "auc", "accuracy"]
    logger.info(f"Least predictable:\n{table[show].head(15).round(3).to_string(index=False)}")
    logger.info(f"Most predictable:\n{table[show].tail(15).iloc[::-1].round(3).to_string(index=False)}")

    # Year-over-year stability: every season's out-of-sample predictions from the same model
    # family (walk-forward validation + holdout), plus the target season from production.
    oos = pd.read_parquet(OUTPUTS_MODEL_COMPARISON / "oos_predictions.parquet")
    seasons = []
    for season, g in oos.groupby("season"):
        s = team_scores(g, f"p_{best}", MIN_FULL_SEASON_CALLS)
        seasons.append(s.assign(season=season))
    seasons.append(team_scores(target, "p_pass", cfg.min_team_plays).assign(season=cfg.target_season))
    long = pd.concat(seasons, ignore_index=True)
    long.to_csv(OUTPUTS_RANKINGS / "team_season_predictability.csv", index=False)

    wide = long.pivot(index="team", columns="season", values="predictability")
    order = sorted(wide.columns)
    rows = []
    for a, b in zip(order[:-1], order[1:]):
        if b != a + 1:   # no out-of-sample predictions for 2020 -> skip the 2019 -> 2021 gap
            continue
        pair = wide[[a, b]].dropna()
        rows.append({
            "season": a, "next_season": b, "teams": len(pair),
            "pearson": pair[a].corr(pair[b]), "spearman": pair[a].corr(pair[b], method="spearman"),
            # The target season is in progress (min_team_plays, not a full season) -> noisier.
            "next_season_partial": b == cfg.target_season,
        })
    stability = pd.DataFrame(rows)
    stability.to_csv(OUTPUTS_RANKINGS / "stability.csv", index=False)
    logger.info(f"Year-over-year stability of team predictability:\n{stability.round(3).to_string(index=False)}")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

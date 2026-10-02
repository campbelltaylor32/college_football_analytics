"""Per-team predictability scores from out-of-sample P(pass) predictions.

Primary score: predictability = 1 - LL_model / LL_null
  LL_model -- the model's mean log loss on the team's calls
  LL_null  -- the log loss of always guessing the team's own overall pass rate (its entropy)
So the score is the share of the team's run/pass uncertainty that situation + tendencies +
game flow explain. A balanced 50/50 offense is not automatically "unpredictable": what matters
is whether you can tell WHEN it will pass. Low (or negative) = hard to call; high = predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cfb_playcall_predictability.modeling.evaluation import EPS, binary_entropy

SITUATIONS = {
    "early_down_neutral": lambda d: d["down"].isin([1, 2]) & (d["pos_score_diff_start"].abs() <= 8) & (d["period"] <= 3),
    "third_down": lambda d: d["down"] == 3,
    "red_zone": lambda d: d["yards_to_goal"] <= 20,
}


def _with_call_log_loss(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    p = df[p_col].clip(EPS, 1 - EPS)
    y = df["is_pass"]
    return df.assign(ll=-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _score(n: float, passes: float, ll_sum: float) -> float:
    return 1 - (ll_sum / n) / binary_entropy(passes / n)


def team_scores(df: pd.DataFrame, p_col: str, min_plays: int = 0) -> pd.DataFrame:
    d = _with_call_log_loss(df, p_col)
    d["correct"] = ((d[p_col] >= 0.5) == d["is_pass"]).astype(int)
    rows = []
    for team, g in d.groupby("pos_team"):
        n, passes = len(g), g["is_pass"].sum()
        rate = passes / n
        rows.append({
            "team": team,
            "games": g["game_id"].nunique(),
            "calls": n,
            "pass_rate": rate,
            "mean_pred_pass": g[p_col].mean(),
            "log_loss": g["ll"].mean(),
            "null_log_loss": float(binary_entropy(rate)),
            "predictability": _score(n, passes, g["ll"].sum()),
            "auc": roc_auc_score(g["is_pass"], g[p_col]) if 0 < passes < n else np.nan,
            "accuracy": g["correct"].mean(),
            "majority_accuracy": max(rate, 1 - rate),
        })
    out = pd.DataFrame(rows)
    out["accuracy_lift"] = out["accuracy"] - out["majority_accuracy"]
    return out[out["calls"] >= min_plays].reset_index(drop=True)


def bootstrap_predictability(df: pd.DataFrame, p_col: str, reps: int, level: float, seed: int) -> pd.DataFrame:
    """Resample each team's GAMES with replacement (calls within a game aren't independent)."""
    d = _with_call_log_loss(df, p_col)
    per_game = d.groupby(["pos_team", "game_id"]).agg(n=("ll", "size"), passes=("is_pass", "sum"), ll=("ll", "sum"))
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for team, g in per_game.groupby(level="pos_team"):
        arr = g[["n", "passes", "ll"]].to_numpy(dtype=float)
        idx = rng.integers(0, len(arr), size=(reps, len(arr)))
        tot = arr[idx].sum(axis=1)                       # reps x 3
        rate = tot[:, 1] / tot[:, 0]
        scores = 1 - (tot[:, 2] / tot[:, 0]) / binary_entropy(rate)
        rows.append({"team": team, "predictability_lo": np.quantile(scores, lo_q),
                     "predictability_hi": np.quantile(scores, hi_q)})
    return pd.DataFrame(rows)


def situational_scores(df: pd.DataFrame, p_col: str, min_plays: int = 25) -> pd.DataFrame:
    """Predictability within each situation, using that situation's own pass rate as the null."""
    frames = []
    for name, mask_fn in SITUATIONS.items():
        s = team_scores(df[mask_fn(df)], p_col, min_plays=min_plays)
        frames.append(s[["team", "calls", "pass_rate", "predictability"]].rename(
            columns={c: f"{name}_{c}" for c in ["calls", "pass_rate", "predictability"]}
        ))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="team", how="outer")
    return out


def predictability_table(df: pd.DataFrame, p_col: str, min_plays: int, reps: int, level: float,
                         seed: int) -> pd.DataFrame:
    scores = team_scores(df, p_col, min_plays)
    ci = bootstrap_predictability(df[df["pos_team"].isin(scores["team"])], p_col, reps, level, seed)
    out = scores.merge(ci, on="team").merge(situational_scores(df, p_col), on="team", how="left")
    if "head_coach" in df.columns:
        coach = df.groupby("pos_team")["head_coach"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
        out = out.merge(coach.rename("head_coach"), left_on="team", right_index=True, how="left")
    out = out.sort_values("predictability").reset_index(drop=True)
    out.insert(0, "unpredictability_rank", np.arange(1, len(out) + 1))  # 1 = least predictable
    return out

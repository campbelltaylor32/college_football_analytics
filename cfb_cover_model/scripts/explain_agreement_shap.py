#!/usr/bin/env python
"""SHAP explanation of the production agreement signal - logistic_regression (Track A) +
xgboost_regressor (Track B) - loaded straight from the latest versioned production artifact
(outputs/models/production/latest.json), so it explains exactly the models that score live
weeks rather than a refit.

Each model is explained in its own native output space:
  - logistic_regression: linear SHAP in log-odds of home cover (exact for a linear model on
    the artifact's own StandardScaler'd inputs).
  - xgboost_regressor: TreeSHAP in points of predicted cover_margin (the base XGBRegressor
    inside ResidualProbabilityRegressor; cover probability is a monotone Phi() of this).

Because the two output scales differ, cross-model comparisons use each feature's share of
that model's total mean |SHAP|. "What matters for agreement" is answered three ways:
  1. shap_beeswarm_<model>.png        - each model's own global drivers
  2. shap_agreement_drivers.png       - importance share in both models side by side, plus
                                        the per-game correlation of the two models' SHAP
                                        values for that feature (do they push the same way?)
  3. shap_agreement_bet_profile.png   - mean SHAP on agreement-bet games vs. all games: which
                                        features actually lift games over both thresholds

Rows explained are the artifact's reference_training_features.parquet, i.e. the training
rows themselves - this is an in-sample description of the fitted models, not a measure of
out-of-sample predictive value (see explain_model.py for holdout permutation importance).

Requires `pip install shap`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PRODUCTION_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models" / "production"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison" / "shap"
TOP_N = 20

CLASSIFIER_COLOR = "#2a78d6"
REGRESSOR_COLOR = "#eb6834"
NEUTRAL_COLOR = "#8a8984"
TEXT_SECONDARY = "#52514e"


def load_latest_artifact() -> tuple[Path, dict]:
    version = json.loads((PRODUCTION_DIR / "latest.json").read_text())["version"]
    version_dir = PRODUCTION_DIR / version
    return version_dir, json.loads((version_dir / "metadata.json").read_text())


def classifier_shap(classifier, X: pd.DataFrame) -> shap.Explanation:
    scaler, clf = classifier.named_steps["scaler"], classifier.named_steps["clf"]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    explainer = shap.LinearExplainer(clf, shap.maskers.Independent(X_scaled, max_samples=len(X_scaled)))
    explanation = explainer(X_scaled)
    explanation.data = X.to_numpy()  # color the beeswarm by raw (unscaled) feature values
    return explanation


def regressor_shap(regressor, X: pd.DataFrame) -> shap.Explanation:
    return shap.TreeExplainer(regressor.base_regressor)(X)


def save_beeswarm(explanation: shap.Explanation, title: str, xlabel: str, path: Path) -> None:
    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False, plot_size=(10, 9))
    ax = plt.gca()
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def build_driver_table(lr_vals: np.ndarray, xgb_vals: np.ndarray, columns: list[str], agree: np.ndarray) -> pd.DataFrame:
    lr_abs, xgb_abs = np.abs(lr_vals).mean(0), np.abs(xgb_vals).mean(0)
    sign_corr = [
        np.corrcoef(lr_vals[:, j], xgb_vals[:, j])[0, 1] if lr_vals[:, j].std() > 0 and xgb_vals[:, j].std() > 0 else np.nan
        for j in range(len(columns))
    ]
    df = pd.DataFrame(
        {
            "feature": columns,
            "lr_mean_abs_shap_logodds": lr_abs,
            "xgb_mean_abs_shap_points": xgb_abs,
            "lr_share": lr_abs / lr_abs.sum(),
            "xgb_share": xgb_abs / xgb_abs.sum(),
            "shap_corr_lr_vs_xgb": sign_corr,
            "lr_mean_shap_agree_bets": lr_vals[agree].mean(0) if agree.any() else np.nan,
            "lr_mean_shap_all": lr_vals.mean(0),
            "xgb_mean_shap_agree_bets": xgb_vals[agree].mean(0) if agree.any() else np.nan,
            "xgb_mean_shap_all": xgb_vals.mean(0),
        }
    )
    df["combined_share"] = (df["lr_share"] + df["xgb_share"]) / 2
    # Agreement-bet lift in each model's own share units, so the two scales are comparable.
    df["lr_agree_lift_share"] = (df["lr_mean_shap_agree_bets"] - df["lr_mean_shap_all"]) / lr_abs.sum()
    df["xgb_agree_lift_share"] = (df["xgb_mean_shap_agree_bets"] - df["xgb_mean_shap_all"]) / xgb_abs.sum()
    return df.sort_values("combined_share", ascending=False).reset_index(drop=True)


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL_COLOR)
    ax.spines["bottom"].set_color(NEUTRAL_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_agreement_drivers(table: pd.DataFrame, path: Path) -> None:
    top = table.head(TOP_N).iloc[::-1]
    y = np.arange(len(top))
    fig, (ax_share, ax_corr) = plt.subplots(1, 2, figsize=(13, 9), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})

    h = 0.38
    ax_share.barh(y + h / 2, top["lr_share"] * 100, height=h, color=CLASSIFIER_COLOR, label="logistic_regression")
    ax_share.barh(y - h / 2, top["xgb_share"] * 100, height=h, color=REGRESSOR_COLOR, label="xgboost_regressor")
    ax_share.set_yticks(y, top["feature"], fontsize=9)
    ax_share.set_xlabel("Share of model's total mean |SHAP| (%)")
    ax_share.set_title("Importance in each model", loc="left", fontsize=11)
    ax_share.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax_share)

    corr = top["shap_corr_lr_vs_xgb"].fillna(0)
    colors = [CLASSIFIER_COLOR if c >= 0 else NEUTRAL_COLOR for c in corr]
    ax_corr.barh(y, corr, height=0.6, color=colors)
    ax_corr.axvline(0, color=NEUTRAL_COLOR, linewidth=1)
    ax_corr.set_xlim(-1, 1)
    ax_corr.set_xlabel("Per-game SHAP correlation, LR vs XGB")
    ax_corr.set_title("Do they push the same way?", loc="left", fontsize=11)
    style_axes(ax_corr)
    for yi, c in zip(y, corr):
        ax_corr.text(c + (0.03 if c >= 0 else -0.03), yi, f"{c:+.2f}", va="center",
                     ha="left" if c >= 0 else "right", fontsize=8, color=TEXT_SECONDARY)

    fig.suptitle(f"Top {TOP_N} features by combined importance across the agreement pair", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_bet_profile(table: pd.DataFrame, n_agree: int, path: Path) -> None:
    ranked = table.assign(joint=table["lr_agree_lift_share"] + table["xgb_agree_lift_share"])
    top = ranked.reindex(ranked["joint"].abs().sort_values(ascending=False).index).head(TOP_N).iloc[::-1]
    y = np.arange(len(top))
    h = 0.38
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(y + h / 2, top["lr_agree_lift_share"] * 100, height=h, color=CLASSIFIER_COLOR, label="logistic_regression")
    ax.barh(y - h / 2, top["xgb_agree_lift_share"] * 100, height=h, color=REGRESSOR_COLOR, label="xgboost_regressor")
    ax.axvline(0, color=NEUTRAL_COLOR, linewidth=1)
    ax.set_yticks(y, top["feature"], fontsize=9)
    ax.set_xlabel("Mean SHAP on agreement bets minus mean SHAP on all games\n(% of model's total mean |SHAP|; positive = pushes toward home cover)")
    ax.set_title(f"What lifts a game into an agreement bet (n={n_agree} in-sample agreement bets)", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    version_dir, metadata = load_latest_artifact()
    selected_columns = json.loads((version_dir / "selected_columns.json").read_text())
    classifier = joblib.load(version_dir / "classifier.joblib")
    regressor = joblib.load(version_dir / "regressor.joblib")
    X = pd.read_parquet(version_dir / "reference_training_features.parquet")[selected_columns]
    print(f"Explaining artifact {metadata['version']} on {len(X)} reference rows, {len(selected_columns)} features")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lr_exp = classifier_shap(classifier, X)
    xgb_exp = regressor_shap(regressor, X)

    # Sanity: SHAP must reconstruct each model's own output.
    lr_logit = classifier.decision_function(X)
    assert np.allclose(lr_exp.values.sum(1) + lr_exp.base_values, lr_logit, atol=1e-6)
    xgb_margin = regressor.predict_margin(X)
    assert np.allclose(xgb_exp.values.sum(1) + xgb_exp.base_values, xgb_margin, atol=1e-3)

    lr_flag = classifier.predict_proba(X)[:, 1] >= metadata["classifier_threshold"]
    xgb_flag = regressor.predict_proba(X)[:, 1] >= metadata["regressor_threshold"]
    agree = lr_flag & xgb_flag
    print(f"In-sample flags: LR={lr_flag.sum()}, XGB={xgb_flag.sum()}, agreement={agree.sum()}")

    save_beeswarm(lr_exp, "logistic_regression - SHAP (log-odds of home cover)", "SHAP value (log-odds)",
                  OUT_DIR / "shap_beeswarm_logistic_regression.png")
    save_beeswarm(xgb_exp, "xgboost_regressor - SHAP (points of predicted cover margin)", "SHAP value (points)",
                  OUT_DIR / "shap_beeswarm_xgboost_regressor.png")

    table = build_driver_table(lr_exp.values, xgb_exp.values, selected_columns, agree)
    table.to_csv(OUT_DIR / "shap_agreement_drivers.csv", index=False)
    plot_agreement_drivers(table, OUT_DIR / "shap_agreement_drivers.png")
    plot_agreement_bet_profile(table, int(agree.sum()), OUT_DIR / "shap_agreement_bet_profile.png")

    summary = {
        "artifact_version": metadata["version"],
        "rows_explained": len(X),
        "n_lr_flags": int(lr_flag.sum()),
        "n_xgb_flags": int(xgb_flag.sum()),
        "n_agreement_bets": int(agree.sum()),
        "rank_corr_lr_vs_xgb_importance": float(table["lr_share"].rank().corr(table["xgb_share"].rank())),
        "lr_vs_xgb_output_corr": float(np.corrcoef(lr_logit, xgb_margin)[0, 1]),
    }
    (OUT_DIR / "shap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(table.head(TOP_N)[["feature", "lr_share", "xgb_share", "shap_corr_lr_vs_xgb", "lr_agree_lift_share", "xgb_agree_lift_share"]].round(3).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

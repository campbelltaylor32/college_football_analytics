"""Track C: a meta-learner trained on out-of-fold base-model predictions.

Nesting rule: the meta-learner must never train on a base model's prediction for a row
that base model was itself trained on. Within a given *outer* walk-forward fold's training
rows, this module runs its own smaller *inner* expanding-window walk-forward to produce
out-of-fold base predictions for the meta-learner's training data, then separately fits
each base model on the *entire* outer-training set to produce the actual base predictions
for the outer validation rows the stack is ultimately scored on.

Simplification (documented in docs/modeling_methodology.md): all base models here share
the single feature set the calling outer fold already selected (via feature_selection/
selection.py) - the inner OOF loop does not re-run correlation/embedded selection at each
inner step, to keep nested compute tractable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def safe_inner_min_seasons(season_train: pd.Series, configured_min: int) -> int:
    """Clamp the configured inner-walk-forward minimum so there's always at least one
    season left to validate on, even for the outer fold with the least training history
    (min_train_seasons seasons, by construction of modeling/splits.py). Without this, the
    earliest outer fold produces zero meta-training rows and every downstream fit errors
    on an empty array rather than silently leaking or crashing confusingly."""
    n_seasons = season_train.nunique()
    return max(1, min(configured_min, n_seasons - 1))


def generate_oof_base_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cover_margin_train: pd.Series,
    season_train: pd.Series,
    base_model_specs: list[dict],
    min_inner_train_seasons: int,
) -> pd.DataFrame:
    seasons = sorted(season_train.unique())
    oof = pd.DataFrame(
        index=X_train.index, columns=[m["name"] for m in base_model_specs], dtype=float
    )

    for i in range(min_inner_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        tr_idx = season_train.index[season_train.isin(train_seasons)]
        va_idx = season_train.index[season_train == val_season]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        for spec in base_model_specs:
            model = spec["builder"]()
            if spec["kind"] == "classifier":
                model.fit(X_train.loc[tr_idx], y_train.loc[tr_idx])
            else:
                model.fit(X_train.loc[tr_idx], cover_margin_train.loc[tr_idx])
            proba = model.predict_proba(X_train.loc[va_idx])[:, 1]
            oof.loc[va_idx, spec["name"]] = proba

    return oof


def fit_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cover_margin_train: pd.Series,
    season_train: pd.Series,
    X_val: pd.DataFrame,
    base_model_specs: list[dict],
    meta_builder: Callable,
    min_inner_train_seasons: int,
) -> tuple[np.ndarray, dict]:
    oof = generate_oof_base_predictions(
        X_train, y_train, cover_margin_train, season_train, base_model_specs, min_inner_train_seasons
    )
    meta_train_mask = oof.notna().all(axis=1)
    meta_X_train = oof.loc[meta_train_mask]
    meta_y_train = y_train.loc[meta_train_mask]

    meta_model = meta_builder()
    meta_model.fit(meta_X_train, meta_y_train)

    val_base_preds = {}
    for spec in base_model_specs:
        model = spec["builder"]()
        if spec["kind"] == "classifier":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, cover_margin_train)
        val_base_preds[spec["name"]] = model.predict_proba(X_val)[:, 1]

    meta_X_val = pd.DataFrame(val_base_preds, index=X_val.index)[
        [m["name"] for m in base_model_specs]
    ]
    stacked_proba = meta_model.predict_proba(meta_X_val)[:, 1]

    report = {
        "n_meta_train_rows": int(meta_train_mask.sum()),
        "meta_train_seasons": sorted(int(s) for s in season_train.loc[meta_train_mask].unique()),
        "base_model_names": [m["name"] for m in base_model_specs],
    }
    return stacked_proba, report

"""Load the trained preseason model (scripts/train_preseason_model.py's output) and score a
target season's teams -- shared by scripts/generate_preseason_ratings.py,
scripts/update_ratings.py, and scripts/backtest_season.py so none of them duplicate this
load-and-predict logic."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from cfb_power_ratings.dataset import FEATURE_COLUMNS, build_modeling_dataset
from cfb_power_ratings.utils.paths import OUTPUTS_MODELS

MODEL_PATH = OUTPUTS_MODELS / "preseason_model.joblib"
METADATA_PATH = OUTPUTS_MODELS / "preseason_model_metadata.json"


def load_preseason_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained preseason model at {MODEL_PATH} -- run "
            "scripts/train_preseason_model.py first."
        )
    return joblib.load(MODEL_PATH)


def load_preseason_model_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def predict_preseason_ratings(engine, season: int, features_cfg, srs_history_start_season: int, model=None) -> pd.Series:
    """One rating per FBS team entering `season`, indexed by team name, using the persisted
    PRODUCTION model (trained on all eligible history through its own final_holdout_season --
    including `season` itself, if season <= that cutoff). Appropriate for a genuinely future
    season (e.g. 2026), where there's nothing to leak. NOT appropriate for reconstructing what
    a past season's preseason ratings honestly would have looked like -- use
    predict_out_of_sample_preseason_ratings for that instead (see its docstring for why).

    Requires season-t rows to already exist in team_talent/coaches/returning_production/
    team_rosters/recruiting_players (a documented prerequisite -- run
    SQL Scripts/ingest_to_mysql.R for the new season first; this function does not ingest
    anything itself)."""
    model = model or load_preseason_model()
    df = build_modeling_dataset(engine, [season], features_cfg, srs_history_start_season)
    if df.empty:
        raise ValueError(f"No FBS teams found for season={season} -- has the DB been ingested for this season?")
    preds = model.predict(df[FEATURE_COLUMNS])
    return pd.Series(preds, index=df["team"], name="preseason_rating")


def predict_out_of_sample_preseason_ratings(
    engine, season: int, features_cfg, modeling_cfg, model_name: str | None = None
) -> pd.Series:
    """One rating per FBS team entering `season`, from a model trained ONLY on seasons strictly
    before `season` -- a genuinely honest reconstruction of "what would this season's preseason
    ratings have looked like," never having seen that season's own outcome. Deliberately
    independent of whatever the persisted production model (`load_preseason_model`) saw during
    its own training, which includes recent seasons on purpose (it's built to make the best
    real prediction for a genuinely future season, not to stay backtest-clean for past ones).

    `model_name` defaults to the production model's own selected type (e.g. "ridge") so the
    comparison is apples-to-apples on model architecture, just not on training data.
    """
    from cfb_power_ratings.modeling.models import get_candidate_models

    model_name = model_name or load_preseason_model_metadata()["model_name"]
    seasons = list(range(modeling_cfg.full_feature_start_season, season + 1))
    df = build_modeling_dataset(engine, seasons, features_cfg, modeling_cfg.srs_history_start_season)
    train_df = df[
        (df["season"] < season) & (~df["season"].isin(modeling_cfg.excluded_seasons))
    ].dropna(subset=["target_srs"])

    model = get_candidate_models([model_name], modeling_cfg.random_seed)[model_name]
    model.fit(train_df[FEATURE_COLUMNS], train_df["target_srs"])

    target_rows = df[df["season"] == season]
    if target_rows.empty:
        raise ValueError(f"No FBS teams found for season={season} -- has the DB been ingested for this season?")
    preds = model.predict(target_rows[FEATURE_COLUMNS])
    return pd.Series(preds, index=target_rows["team"], name="preseason_rating")

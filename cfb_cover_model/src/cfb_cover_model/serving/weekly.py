"""Score an upcoming week against the current production artifact and write the picks CSV.

The gap this fills: the FastAPI `/predictions/week` route scores but only returns JSON, and
`scripts/generate_weekly_predictions.py` refits both models from scratch every run instead
of loading the persisted artifact. `scripts/run_weekly.py` needs neither - it has just
retrained and wants next week's picks from *that* fit, on disk, in the same
`live_<season>_week_<week>_dual_model_predictions.csv` shape
`scripts/run_weekly_monitoring.py` already knows how to grade.
"""
from __future__ import annotations

from pathlib import Path

from cfb_cover_model.config import load_data_config
from cfb_cover_model.serving.artifact import PROJECT_ROOT, load_latest_artifact
from cfb_cover_model.serving.scoring import score_week

PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"


def score_and_save_upcoming_week(
    season: int,
    week: int,
    *,
    out_dir: Path = PRED_DIR,
) -> Path:
    """Pull `week`'s schedule/lines/form live from the CFBD API, score every game with the
    latest production artifact (logistic_regression + xgboost_regressor agreement signal),
    and write outputs/predictions/live_<season>_week_<week>_dual_model_predictions.csv.

    Raises RuntimeError if no artifact has been trained yet, or ValueError (from the ingest
    pipeline) if the season hasn't produced enough completed weeks to build features."""
    artifact = load_latest_artifact()
    if artifact is None:
        raise RuntimeError(
            "No production artifact exists yet - run scripts/train_production_artifact.py "
            "(or a full scripts/run_weekly.py pass) before scoring a week."
        )

    from cfb_cover_model.ingest import cfbd_client, pipeline

    client = cfbd_client.get_client()
    week_df_raw = pipeline.build_current_week_rows(client, season, week)
    if week_df_raw.empty:
        raise ValueError(f"No games returned for season={season} week={week}")

    out = score_week(artifact, week_df_raw, load_data_config())
    out["season"] = season  # the live frame's own season column is unreliable (all-NaN in
    # real weekly CSVs); the caller's value is authoritative, same as the API route

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"live_{season}_week_{week}_dual_model_predictions.csv"
    out.to_csv(out_path, index=False)
    return out_path

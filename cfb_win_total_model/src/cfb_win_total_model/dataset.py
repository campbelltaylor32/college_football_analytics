"""Assembles the one-row-per-(school,season) modeling table by joining targets.py's win
counts against every enabled feature module's output. The FBS team-season universe
(targets.get_fbs_teams_by_seasons) is always the LEFT side of every join -- feature tables
never inflate or shrink the row universe, they only ever attach columns (left join) or leave
NaN + a `<module>_missing` flag.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.config import FeaturesConfig
from cfb_win_total_model.features import coaching, program_history, roster_turnover, schedule, talent_recruiting
from cfb_win_total_model.features import prior_performance, returning_production
from cfb_win_total_model.targets import build_target_table, get_fbs_teams_by_seasons
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

NON_FEATURE_COLS = {"school", "season", "regular_season_wins", "regular_season_losses", "scheduled_games"}

_FEATURE_BUILDERS = {
    "prior_performance": lambda engine, season, cfg: prior_performance.build_prior_performance_features(
        engine, season, cfg.explosiveness_epa_threshold
    ),
    "returning_production": lambda engine, season, cfg: returning_production.build_returning_production_features(
        engine, season, cfg.winsorize_percent_ppa_limits
    ),
    "talent_recruiting": lambda engine, season, cfg: talent_recruiting.build_talent_recruiting_features(engine, season, cfg),
    "roster_turnover": lambda engine, season, cfg: roster_turnover.build_roster_turnover_features(engine, season),
    "coaching": lambda engine, season, cfg: coaching.build_coaching_features(engine, season, cfg),
    "schedule": lambda engine, season, cfg: schedule.build_schedule_features(engine, season, cfg),
    "program_history": lambda engine, season, cfg: program_history.build_program_history_features(engine, season, cfg),
}

_FEATURE_DESCRIBERS = {
    "prior_performance": prior_performance.describe_features,
    "returning_production": returning_production.describe_features,
    "talent_recruiting": talent_recruiting.describe_features,
    "roster_turnover": roster_turnover.describe_features,
    "coaching": coaching.describe_features,
    "schedule": schedule.describe_features,
    "program_history": program_history.describe_features,
}


def build_modeling_dataset(engine: Engine, target_seasons: list[int], features_cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info(f"Building modeling dataset for target_seasons={target_seasons}")

    base = get_fbs_teams_by_seasons(engine, target_seasons)
    target = build_target_table(engine, seasons=target_seasons)
    df = base.merge(target, on=["school", "season"], how="inner")

    for name, builder in _FEATURE_BUILDERS.items():
        if not features_cfg.feature_groups.get(name, True):
            logger.info(f"Feature group '{name}' disabled via config; skipping")
            continue

        frames = [builder(engine, s, features_cfg) for s in target_seasons]
        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning(f"Feature group '{name}' produced no rows for any target season")
            df[f"{name}_missing"] = True
            continue

        combined = pd.concat(frames, ignore_index=True)
        present_keys = combined[["school", "season"]].drop_duplicates().assign(**{f"_has_{name}": True})

        df = df.merge(combined, on=["school", "season"], how="left")
        df = df.merge(present_keys, on=["school", "season"], how="left")
        df[f"{name}_missing"] = df[f"_has_{name}"].isna()
        df = df.drop(columns=[f"_has_{name}"])

    if df.duplicated(subset=["school", "season"]).any():
        raise AssertionError("Duplicate (school, season) rows in modeling dataset -- feature join fan-out bug")

    return df


def build_feature_registry(features_cfg: FeaturesConfig) -> pd.DataFrame:
    """Concatenates every enabled feature module's describe_features() output -- the
    auto-generated, code-co-located source of truth for outputs/feature_analysis/feature_registry.csv.
    """
    rows: list[dict] = []
    for name, describer in _FEATURE_DESCRIBERS.items():
        if not features_cfg.feature_groups.get(name, True):
            continue
        rows.extend(describer())
        # dataset.py adds one coarse "was this whole module's row present" flag per enabled
        # module, on top of any finer-grained *_missing flag the module defines itself.
        rows.append(
            {
                "feature_name": f"{name}_missing",
                "description": f"True if the {name} feature module produced no row at all for this (school, season)",
                "source_table": "dataset.py join presence check",
                "source_season": "n/a",
                "transformation": "merge indicator",
                "known_before_kickoff": True,
                "missing_value_treatment": "n/a (this IS the missingness signal)",
                "expected_direction": "context",
                "category": name,
            }
        )
    registry = pd.DataFrame(rows)
    column_order = ["feature_name", "description", "source_table", "source_season", "transformation", "known_before_kickoff", "missing_value_treatment", "expected_direction", "category"]
    return registry[column_order]

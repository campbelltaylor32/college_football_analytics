"""Expanding-window walk-forward validation splits. Never a random split across team-season
rows -- every fold's training set is strictly earlier seasons than its validation season."""

from __future__ import annotations

from dataclasses import dataclass

from cfb_win_total_model.config import ModelingConfig


@dataclass(frozen=True)
class Fold:
    train_seasons: list[int]
    validation_season: int


def generate_walk_forward_folds(cfg: ModelingConfig) -> list[Fold]:
    folds = []
    for val_season in cfg.walk_forward_validation_seasons:
        train_seasons = [
            s for s in range(cfg.full_feature_start_season, val_season) if s not in cfg.excluded_seasons
        ]
        assert max(train_seasons) < val_season, "walk-forward fold leaked a future season into training"
        if len(train_seasons) < cfg.min_train_seasons:
            continue
        folds.append(Fold(train_seasons=train_seasons, validation_season=val_season))
    return folds


def final_holdout_fold(cfg: ModelingConfig) -> Fold:
    train_seasons = [
        s for s in range(cfg.full_feature_start_season, cfg.final_holdout_season) if s not in cfg.excluded_seasons
    ]
    assert max(train_seasons) < cfg.final_holdout_season
    return Fold(train_seasons=train_seasons, validation_season=cfg.final_holdout_season)

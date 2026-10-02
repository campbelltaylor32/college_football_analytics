"""Build the play-level modeling dataset -> data/processed/modeling_dataset.parquet.

Usage: python scripts/build_modeling_dataset.py [first_season] [last_season]
Requires scripts/pull_pbp.R to have cached every season in range.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_playcall_predictability.config import load_data_config, load_features_config
from cfb_playcall_predictability.dataset import build_modeling_dataset, feature_columns
from cfb_playcall_predictability.utils.logging import get_logger
from cfb_playcall_predictability.utils.paths import DATA_PROCESSED_DIR, ensure_output_dirs

logger = get_logger("build_modeling_dataset")


def main() -> None:
    data_cfg = load_data_config()
    first = int(sys.argv[1]) if len(sys.argv) > 1 else data_cfg.first_season
    last = int(sys.argv[2]) if len(sys.argv) > 2 else data_cfg.current_season
    ensure_output_dirs()

    df = build_modeling_dataset(list(range(first, last + 1)), data_cfg, load_features_config())
    out_path = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
    df.to_parquet(out_path, index=False)

    summary = df.groupby("season").agg(
        calls=("is_pass", "size"), pass_rate=("is_pass", "mean"),
        teams=("pos_team", "nunique"), max_week=("week", "max"),
    )
    logger.info(f"Per-season summary:\n{summary.to_string()}")
    missing = df[feature_columns(df)].isna().mean().sort_values(ascending=False)
    logger.info(f"Feature missing rates (top 10):\n{missing.head(10).to_string()}")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

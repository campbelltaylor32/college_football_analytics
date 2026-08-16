#!/usr/bin/env python
"""Thin sequencer for the compression-diagnostics suite -- runs each stage script's main()
in order. Stages are independent (no stage's output feeds another stage's input): the cheap,
no-retraining diagnostics run first so the core findings surface fastest even on a partial run.

Usage:
    python scripts/diagnostics/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIAGNOSTICS_DIR.parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIAGNOSTICS_DIR))

import compute_compression_diagnostics
import feature_experiment
import win_probability_sum_prototype

from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = [
    ("compute_compression_diagnostics", compute_compression_diagnostics.main),
    ("feature_experiment", feature_experiment.main),
    ("win_probability_sum_prototype", win_probability_sum_prototype.main),
]


def main() -> int:
    for name, stage_main in STAGES:
        logger.info(f"=== Stage: {name} ===")
        rc = stage_main()
        if rc != 0:
            raise RuntimeError(f"Stage '{name}' exited with code {rc}")
    logger.info("Compression diagnostics suite complete. See outputs/diagnostics_compression/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

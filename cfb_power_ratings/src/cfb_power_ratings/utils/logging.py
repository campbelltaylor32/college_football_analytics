"""Shared logging setup. No module in this project uses print() for anything but final
human-readable script summaries -- everything else routes through get_logger() so a full
pipeline run produces one consistent, timestamped log stream."""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("cfb_power_ratings")
    if not _CONFIGURED:
        root.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    qualified = name if name.startswith("cfb_power_ratings") else f"cfb_power_ratings.{name}"
    return logging.getLogger(qualified)

"""Shared logging setup. No module in this project uses print() -- everything routes through
get_logger() so run_pipeline.py produces one consistent, timestamped log stream."""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("cfb_win_total_model")
    if not _CONFIGURED:
        root.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    qualified = name if name.startswith("cfb_win_total_model") else f"cfb_win_total_model.{name}"
    return logging.getLogger(qualified)

"""Thin wrapper over the official `cfbd` Python client, for the handful of things this project
needs live from the API rather than the MySQL cache: the transfer-portal endpoint (not in the
DB schema at all -- see features/roster_turnover.py) and an upcoming week's real schedule
(games/betting_lines in the DB are completed-games-only -- see scripts/update_ratings.py).
Same pattern as cfb_cover_model/src/cfb_cover_model/ingest/cfbd_client.py."""
from __future__ import annotations

import os

import cfbd

from cfb_power_ratings.config import _load_env_once


def get_client() -> cfbd.ApiClient:
    _load_env_once()
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CFBD_API_KEY not found -- expected it in the repo root's .env (see ../.env.example)."
        )
    config = cfbd.Configuration(access_token=api_key)
    return cfbd.ApiClient(config)

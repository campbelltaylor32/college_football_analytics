"""Entrypoint for `uvicorn cfb_cover_model.api.main:app` or `python -m cfb_cover_model.api.main`.
docker-compose.yml instead targets `cfb_cover_model.api.app:create_app --factory` directly,
which is equivalent but skips loading this module's `if __name__` block.
"""
from __future__ import annotations

import uvicorn

from cfb_cover_model.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

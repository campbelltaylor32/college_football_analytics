FROM python:3.12-slim

# libgomp1 is required at runtime by xgboost/lightgbm on slim base images (not just at
# build time) - omitting it produces an import-time OSError inside the container even
# though everything installs fine.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e ".[dev,boosting,serving]"

# No CMD here - docker-compose.yml sets the command per service (api vs. scheduler), since
# both services share this one image.

#!/usr/bin/env bash
# Pipeline state that is gitignored but needed between weekly runs, kept as assets on the
# `pipeline-state` GitHub Release of this repo:
#   cover_state.tar.gz - cover-model CFBD cache + extended history, production artifacts,
#                        the historical predictor CSVs in Data/, the preseason ratings model
#   cfb_db.sql.gz      - mysqldump of the cfb_football DB (power ratings + ingest_to_mysql.R)
#
#   ci/state.sh download          fetch both assets into ./_state
#   ci/state.sh unpack-cover      restore cover_state.tar.gz into the working tree
#   ci/state.sh restore-db        load cfb_db.sql.gz into $CFB_DB_* MySQL
#   ci/state.sh pack-cover        rebuild _state/cover_state.tar.gz from the working tree
#   ci/state.sh dump-db           rebuild _state/cfb_db.sql.gz from $CFB_DB_* MySQL
#   ci/state.sh upload [asset..]  replace the release assets (default: both)
#   ci/state.sh seed              one-time, from the Mac: pack + dump + create the release
#
# Run from the repo root. Needs `gh` (GH_TOKEN in CI) and the mysql client for the DB steps.
set -euo pipefail

TAG=pipeline-state
DIR=_state
COVER_PATHS=(
  cfb_cover_model/data/raw
  cfb_cover_model/data/processed
  cfb_cover_model/outputs/models/production
  Data/CFB_Gambling_Predictors_Final_PBP.csv
  Data/CFB_Gambling_Results.csv
  cfb_power_ratings/outputs/models/preseason_model.joblib
)

: "${CFB_DB_HOST:=127.0.0.1}" "${CFB_DB_PORT:=3306}" "${CFB_DB_USER:=root}" "${CFB_DB_NAME:=cfb_football}"
mysql_args() {
  echo -h"$CFB_DB_HOST" -P"$CFB_DB_PORT" -u"$CFB_DB_USER" ${CFB_DB_PASSWORD:+-p"$CFB_DB_PASSWORD"}
}

mkdir -p "$DIR"
cmd=${1:-}; shift || true
case "$cmd" in
  download)
    gh release download "$TAG" --dir "$DIR" --clobber --pattern '*.gz'
    ls -la "$DIR"
    ;;
  unpack-cover)
    tar -xzf "$DIR/cover_state.tar.gz"
    ;;
  restore-db)
    # shellcheck disable=SC2046
    mysql $(mysql_args) -e "CREATE DATABASE IF NOT EXISTS \`$CFB_DB_NAME\`"
    # shellcheck disable=SC2046
    gunzip -c "$DIR/cfb_db.sql.gz" | mysql $(mysql_args) "$CFB_DB_NAME"
    ;;
  pack-cover)
    tar -czf "$DIR/cover_state.tar.gz" "${COVER_PATHS[@]}"
    ls -la "$DIR/cover_state.tar.gz"
    ;;
  dump-db)
    # shellcheck disable=SC2046
    mysqldump $(mysql_args) --single-transaction --quick --no-tablespaces "$CFB_DB_NAME" | gzip -6 > "$DIR/cfb_db.sql.gz"
    ls -la "$DIR/cfb_db.sql.gz"
    ;;
  upload)
    assets=("$@")
    [ ${#assets[@]} -eq 0 ] && assets=(cover_state.tar.gz cfb_db.sql.gz)
    gh release upload "$TAG" "${assets[@]/#/$DIR/}" --clobber
    ;;
  seed)
    "$0" pack-cover
    "$0" dump-db
    gh release view "$TAG" >/dev/null 2>&1 || \
      gh release create "$TAG" --title "Pipeline state (automated)" --latest=false \
        --notes "Gitignored state for the weekly GitHub Actions jobs. Overwritten every run; see ci/state.sh."
    "$0" upload
    ;;
  *)
    sed -n '2,17p' "$0"; exit 1
    ;;
esac

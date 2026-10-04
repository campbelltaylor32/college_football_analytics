# Weekly GitHub Actions jobs

Two scheduled workflows publish the CFB tool pages on campbelltaylor32.github.io:

| Workflow | When (UTC) | Does |
|---|---|---|
| `weekly-results.yml` | Sun 15:00 | MySQL ingest of the finished week → cover-model ingest + monitoring (grades last week's picks) → power ratings + charts → Team Dashboard data → rebuild/render Picks, Power Rankings, Dashboard |
| `weekly-picks.yml` | Tue 15:00 | cover-model retrain through last week → score this week → game cards/PDF → rebuild/render Picks |

Both commit `outputs/` to `main` here, push `tools/` + `docs/` to the site repo's `main`, and share a `cfb-pipeline` concurrency group. Weeks come from the CFBD calendar via `ci/resolve_weeks.py`. Use **Run workflow** with `season` / `completed_week` to re-run a specific week, and untick `push` for a dry run (the built pages are uploaded as a run artifact instead).

## State

Gitignored inputs live as assets on the `pipeline-state` release and are restored/saved by `ci/state.sh`:

- `cover_state.tar.gz` holds the cover-model CFBD cache and extended history, the production artifacts, `Data/CFB_Gambling_*.csv`, and `preseason_model.joblib`.
- `cfb_db.sql.gz` is a dump of the `cfb_football` MySQL DB. Only the Sunday job restores and re-saves it.

## One-time setup

1. Repo secrets (Settings → Secrets and variables → Actions):
   - `CFBD_API_KEY`
   - `SITE_REPO_TOKEN`: a fine-grained PAT scoped to `campbelltaylor32.github.io` with **Contents: read and write**.
2. Seed the state from the Mac (local MySQL running, repo root, `gh` logged in): `ci/state.sh seed`.
3. Dry-run each workflow from the Actions tab with `push` unticked.

Python deps are pinned in `*/requirements-ci.txt` (frozen from the local venvs) so pickled models load with the versions that trained them. Re-freeze them if you upgrade the local venvs.

If a run falls over, re-run it or fall back to the manual steps in the site repo's `CLAUDE.md`. Nothing is pushed unless every step succeeds.

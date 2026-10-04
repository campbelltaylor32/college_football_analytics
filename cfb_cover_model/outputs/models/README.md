Trained models are no longer refit-on-the-fly only (see docs/project_story.md 'What actually
shipped' for the original walk-forward-vs-holdout context on why nothing was persisted at
first). `outputs/models/production/` now holds versioned production artifacts written by
`scripts/train_production_artifact.py` and served by the FastAPI app in `src/cfb_cover_model/api/`
- see `docs/serving_and_monitoring.md` for the full artifact layout, the `latest.json` pointer
mechanism, and retention/pruning. Contents of `production/` are gitignored (retrained weekly,
tied to the library versions installed at train time) - only the `.gitkeep` placeholder is
tracked.

# Play-Call Predictability: Run vs Pass (FBS, 2026)

## Question
Which offenses are the hardest to call, and which are the easiest? We train a play-level run/pass classifier on 11 seasons of FBS play-by-play. It knows everything a well-prepared defensive coordinator would know before the snap:
- down, distance, field position, clock and score
- the pregame spread and total
- the offense's tendencies from last season and this season so far
- the opponent defense's strength
- how this game has gone for the offense so far

We then score every 2026 call and ask, team by team, how much of the run/pass decision the model can see coming.

Play calling is attributed to the **team-offense**. CFBD has no offensive-coordinator or play-caller data, so the head coach is shown for context only.

## Data sources
- **Play-by-play:** `cfbfastR::load_cfb_pbp()` release files, 2014–2026 regular season (`scripts/pull_pbp.R`). For the current season, any completed week the release is still missing is back-filled from the CFBD API.
- **MySQL `cfb_football` (`SQL Scripts/`):**
  - `games`: FBS membership, neutral site, schedule order
  - `betting_lines`: spread and total
  - `coaches`: head coach and new-coach flag
- Why not the MySQL `plays` table: it has no score or timeout columns, and its play ids can't be joined exactly to the release. Details are in `docs/assumptions_and_limitations.md`.

## Method
1. **Label.**
   - Pass = dropbacks, including sacks and interceptions. Run = designed runs.
   - Scrambles are counted as runs, because the source can't separate them.
   - Kneels, spikes, overtime and garbage time are dropped.
   - Garbage time is a margin over 38 in Q2, over 28 in Q3, or over 22 in Q4.
   - That leaves about 1.05M calls for 2015–2025 and 31k for 2026 (weeks 1–4).
2. **Features.** 78 features, every one knowable before the snap. `docs/data_leakage_rules.md` lists the rules and `tests/test_leakage.py` enforces them.
   - **Situation:** down, distance, yards to goal, clock, score, score × time, win probability, timeouts, home.
   - **Market:** expected margin, game total, implied team total.
   - **Tendencies:** team pass rate in 10 situation buckets (1st & 10, 3rd & long, red zone, trailing, …). Each has a prior-season version and a season-to-date version shrunk toward it, plus a new-head-coach flag.
   - **Strength:** the offense's and the opponent defense's run/pass EPA and success rate, prior season and season-to-date.
   - **In-game flow:** pass rate so far (shrunk toward the team's norm), run/pass success and EPA so far, the previous call and its result, same-call streak, and the run/pass mix on the current drive.
3. **Model.**
   - Walk-forward validation: train on every earlier season, validate on 2019 and 2021–2024. 2025 is the holdout.
   - The final model is refit on 2015–2025, so 2026 is fully out of sample.

   | Model | Walk-forward log loss | 2025 holdout log loss | Holdout AUC | Holdout accuracy |
   |---|---|---|---|---|
   | **LightGBM (chosen)** | **0.5851** | **0.5876** | **0.749** | **67.6%** |
   | XGBoost | 0.5854 | 0.5878 | 0.749 | 67.6% |
   | Logistic, all features + splines | 0.6084 | 0.6112 | 0.725 | 65.8% |
   | Baseline: league situational logistic | 0.6319 | 0.6317 | 0.697 | 64.1% |
   | Baseline: team pass rate only | 0.6829 | 0.6853 | 0.563 | 54.1% |

   - The model is well calibrated (holdout ECE 0.7%; `outputs/model_comparison/calibration_holdout.csv`), so no post-hoc calibration is applied.
   - Top features by gain:
     - down and distance (31%)
     - in-game pass rate (10%)
     - clock (13%)
     - field position (6%)
     - win probability
     - the previous call's result
4. **Predictability score**, per team: **`1 − (model log loss) / (log loss of always guessing the team's own pass rate)`**.
   - It is the share of the team's run/pass uncertainty the model explains. 0 means no better than knowing the mix. Higher means more predictable.
   - It accounts for run/pass mix: the correlation with pass rate is 0.01. On raw accuracy, run-heavy option teams look "predictable" just from always guessing run.
   - Teams need at least 150 qualifying calls.
   - 90% confidence intervals come from bootstrapping games.

## Results (2026, through week 4)
Full table: `outputs/rankings/2026_predictability.csv`. Charts: `outputs/figures/2026_extremes.png` (all FBS) and `outputs/figures/2026_extremes_logos_power5.png` (Power 5 + Notre Dame, with logos). The FBS median score is 0.11.

**Least predictable** (the model barely beats knowing their run/pass mix):

| Rank | Offense | Head coach | Pass rate | Score (90% CI) |
|---|---|---|---|---|
| 1 | Utah State | Bronco Mendenhall | 55% | −0.02 (−0.04, 0.00) |
| 2 | Mississippi State | Jeff Lebby | 52% | 0.00 (−0.07, 0.04) |
| 3 | Ball State | Mike Uremovich | 55% | 0.00 (−0.10, 0.10) |
| 4 | Air Force | Troy Calhoun | 17% | 0.00 (−0.06, 0.07) |
| 5 | Utah | Morgan Scalley | 52% | 0.02 (−0.03, 0.05) |
| 6 | Northwestern | David Braun | 48% | 0.03 (−0.02, 0.03) |
| 7 | UMass | Joe Harasymiak | 40% | 0.03 (−0.01, 0.07) |
| 8 | Oklahoma State | Eric Morris | 53% | 0.03 (−0.03, 0.08) |
| 9 | New Mexico | Jason Eck | 39% | 0.03 (−0.01, 0.06) |
| 10 | Tennessee | Josh Heupel | 43% | 0.04 (−0.02, 0.08) |

**Most predictable:**

| Rank | Offense | Head coach | Pass rate | Score (90% CI) |
|---|---|---|---|---|
| 1 | Navy | Brian Newberry | 24% | 0.28 (0.18, 0.31) |
| 2 | Georgia State | Dell McGee | 47% | 0.21 (0.12, 0.26) |
| 3 | Marshall | Tony Gibson | 50% | 0.21 (0.17, 0.22) |
| 4 | Wake Forest | Jake Dickert | 54% | 0.20 (0.13, 0.23) |
| 5 | Central Michigan | Matt Drinkall | 33% | 0.20 (0.10, 0.26) |
| 6 | West Virginia | Rich Rodriguez | 26% | 0.20 (0.15, 0.22) |
| 7 | UL Monroe | Bryant Vincent | 51% | 0.19 (0.10, 0.22) |
| 8 | Oregon State | JaMarcus Shephard | 70% | 0.19 (0.12, 0.24) |
| 9 | Baylor | Dave Aranda | 52% | 0.18 (0.15, 0.21) |
| 10 | Colorado | Deion Sanders | 49% | 0.18 (0.13, 0.22) |

What stands out:
- **Run-heavy option offenses split.** Navy is the most predictable offense in FBS: when it throws, the model sees it coming (AUC 0.83). Air Force runs just as much (17% passes), but its few passes are close to impossible to anticipate. Army sits in the middle (#22 least predictable).
- **Unpredictable doesn't mean balanced.** The least-predictable group runs from 17% to 60% pass rate, and the most-predictable group from 24% to 70% (`2026_pass_rate_vs_predictability.png`).
- **Is it a trait?** Partly. Team scores correlate r = 0.30–0.40 from one full season to the next (2021→2022 through 2024→2025), and 0.25 from 2025 to the four-week 2026 sample (noisier, since 2026 is partial) (`outputs/rankings/stability.csv`, `figures/stability.png`). Some of a team's score carries over from year to year, but a lot of it is season-specific (QB, scheme, staff changes) or noise.

## Caveats
- **Four weeks of data.** About 160–280 calls per team, so the CIs are wide. Treat the ranks as tiers, and the middle of the table as indistinguishable. Re-run weekly; the CIs tighten as the season goes on.
- **Team, not play caller.** Teams where the OC calls plays are still ranked under the offense. The head coach column is for context.
- **Scrambles count as runs.** Offenses with running QBs carry some label noise.
- **"Predictable" is relative to a well-informed model.** It already knows each team's tendencies, so it measures what a prepared defense still can't anticipate, not deviation from league norms.
- More detail: `docs/assumptions_and_limitations.md`.

## Running it
Run from the repo root. The MySQL credentials go in `cfb_playcall_predictability/.env` (copy `.env.example`), and `CFBD_API_KEY` goes in the repo-root `.env`, used for the current-season back-fill.

```bash
cd cfb_playcall_predictability
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
Rscript scripts/pull_pbp.R                    # cache PBP (re-run weekly; past seasons are skipped)
.venv/bin/python scripts/build_modeling_dataset.py
.venv/bin/python scripts/train_models.py      # walk-forward comparison + production fit (~5 min)
.venv/bin/python scripts/rank_predictability.py
.venv/bin/python scripts/plot_predictability.py
cd .. && Rscript cfb_playcall_predictability/scripts/plot_extremes_logos_power5.R   # Power 5 + ND logo chart
cd cfb_playcall_predictability && .venv/bin/python -m pytest
```

Weekly in-season refresh: `pull_pbp.R` → `build_modeling_dataset.py` → `rank_predictability.py` → `plot_predictability.py`. Retraining isn't needed in-season, because the production model never sees 2026.

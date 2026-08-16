# CFB Pythagorean Win% Model — 2025 Season

Standalone, retrospective analysis: how well does the Pythagorean expectation
(points-for / points-against, Bill James style) predict a team's actual
winning percentage for the 2025 college football season?

This is descriptive, not predictive — it uses full-season point totals, so
it can't be used to forecast games in advance. It's a sanity check on how
much of a team's win-loss record is "explained" by scoring margin alone.

## Data

- `../Data/CFB_Gambling_Results_2025_14.csv` — cumulative 2025 season game
  results (weeks 1-14), one row per game with home/away teams and final
  scores. Spans all divisions (FBS, FCS, D2, D3, NAIA), so results are
  filtered to the 134 FBS teams found in `../Data/CFB_Team_Talent_Data_2025.csv`.
  Each FBS team's full slate (including money games vs. non-FBS opponents)
  counts toward its points-for/against and win total, matching its real
  season record.

## Method

For each FBS team: sum points scored (PF) and points allowed (PA) across all
2025 games, compute actual win% = wins / games played, and compare against
two Pythagorean predictions:

- **Classic** (`k=2`): `PF^2 / (PF^2 + PA^2)`
- **Fitted**: exponent numerically fit (via `scipy.optimize.minimize_scalar`)
  to minimize squared error against actual 2025 win% across all 134 teams.

## Run it

```
.venv/bin/python pythagorean_analysis.py
```

## Results (2025 season, 134 FBS teams)

| Variant | MAE | RMSE | R² |
|---|---|---|---|
| Classic (k=2) | 0.0781 | 0.1012 | 0.7974 |
| Fitted (k=2.181) | 0.0770 | 0.1004 | 0.8006 |

Point differential explains roughly 80% of the variance in season win%
(R² ≈ 0.80), with a typical miss of ~0.08 win% (under 1 win over a
12-game season). The best-fit exponent (~2.18) is close to the classic
value of 2, and barely improves on it — the textbook Pythagorean formula
already does most of the work for college football.

## Outputs

- `outputs/team_pythagorean_2025.csv` — per-team PF/PA/games/wins, actual
  win%, and both Pythagorean predictions.
- `outputs/metrics_summary_2025.csv` — MAE/RMSE/R² for both variants.
- `outputs/scatter_actual_vs_pyth.png` — actual vs. predicted win%, both
  exponents, with a 45° reference line.

## Opponent-adjusted extension: does strength of schedule matter?

`opponent_adjusted_analysis.py` tests whether weighting each game's points
by opponent quality — so running up the score on a weak team counts for
less, and points allowed to a strong team are discounted — actually
improves the fit. Two opponent-quality proxies are compared:

- **Talent**: preseason `Scaled_Talent` (recruiting composite, fixed before
  the season).
- **SRS**: a Simple Rating System rating computed from actual 2025 point
  margins (Sports-Reference-style iterative method, FBS-vs-FBS games only)
  — a ceiling benchmark since it reflects real in-season performance rather
  than a preseason guess.

For each proxy, `weight = 1 + alpha * opponent_quality_z`, with
`PF_adj = sum(PF * weight)` and `PA_adj = sum(PA / weight)`; `alpha` is fit
per proxy to minimize error against actual win% (k held at 2, so the
comparison isolates the effect of the adjustment itself).

### Is talent a good proxy for actual team quality?

**Moderately.** Preseason `Scaled_Talent` correlates with the season-end SRS
rating at **r = 0.687** (p < 1e-19) — talent explains under half the
variance in how a team actually performed. That's a real signal, but a long
way from a stand-in for what actually happened on the field (injuries,
coaching, player development, and plain variance all move teams a lot from
their recruiting baseline over a season).

### Does opponent-adjustment improve the Pythagorean fit?

Barely, and in a counterintuitive direction:

| Variant | MAE | RMSE | R² |
|---|---|---|---|
| Classic (k=2, unadjusted) | 0.0781 | 0.1012 | 0.7974 |
| Fitted k=2.181, unadjusted | 0.0770 | 0.1004 | 0.8006 |
| Talent-weighted (alpha=-0.034) | 0.0781 | 0.1000 | 0.8023 |
| SRS-weighted (alpha=-0.041) | 0.0769 | 0.0995 | 0.8043 |

The best-fit `alpha` for *both* proxies is small and **negative** —
opposite of the "reward playing good teams" hypothesis — and the R² gain
over the unadjusted baseline is marginal (0.797 → 0.804, using the "true"
SRS benchmark). Two takeaways:

1. **The opponent-adjustment effect, if real, is small.** Raw points-for/
   against already capture most of what matters; schedule strength adds
   only a sliver on top, at least with this simple multiplicative weighting
   scheme on a single ~11-game season.
2. **SRS-weighted still narrowly beats talent-weighted** (R² 0.8043 vs.
   0.8023), consistent with the r=0.687 correlation — actual in-season
   performance is a better opponent-quality signal than a preseason
   recruiting number, but here neither moves the needle much.

### Run it

```
.venv/bin/python opponent_adjusted_analysis.py
```

(Run `pythagorean_analysis.py` first — this script reads its
`team_pythagorean_2025.csv` / `metrics_summary_2025.csv` outputs.)

### Additional outputs

- `outputs/opponent_adjusted_2025.csv` — per-team PF/PA, talent- and
  SRS-adjusted PF/PA, talent score, SRS rating, and all four predicted
  win%s.
- `outputs/metrics_summary_opponent_adjusted_2025.csv` — MAE/RMSE/R² for
  all four variants.
- `outputs/talent_vs_srs_correlation.png` — preseason talent vs.
  season-derived SRS, annotated with r.
- `outputs/scatter_opponent_adjusted.png` — actual vs. predicted win% for
  the classic, talent-weighted, and SRS-weighted variants.

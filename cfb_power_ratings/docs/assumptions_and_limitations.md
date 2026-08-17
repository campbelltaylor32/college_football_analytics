# Assumptions and limitations

## Home-field advantage

One league-average constant (~4.33 points, verified live against the full historical window),
applied identically to every team, every season. Real HFA plausibly varies by program (altitude,
crowd size, travel distance for the visitor) and has likely drifted over 2013-2025 (COVID-era
empty/reduced-capacity stadiums in particular). Not modeled. If a future iteration wants
per-team or per-season HFA, `srs.estimate_home_field_advantage` and
`site_adjusted_margin`/`compute_srs`'s `hfa` parameter would need to become team- or
season-indexed rather than a single float threaded through everywhere.

## Non-FBS opponents

Pooled into one fixed-rating pseudo-team (`generic_low_major`) rather than individually rated.
A good money-game win over a real FCS quality opponent and a blowout over the worst FCS program
in the country count identically toward this pooled rating's calibration. Reasonable for v1
(this project's focus is FBS-vs-FBS ratings), but means margin-of-victory information against
non-FBS opponents is only partially used.

## `phantom_games` — the in-season blending weight

Defaults to 5 (`config/modeling.yaml`), chosen as a plausible "faded out by week 8-10" starting
point and then checked (not rigorously optimized) via `scripts/backtest_season.py`'s 3/5/8
sweep on the 2024 season, where 5 had the lowest market-spread MAE and Brier score of the three
tested. Not validated across multiple seasons, not searched over a finer grid, and not
validated as season-invariant (a season with a lot of early-season upsets might want a
different fade-out rate than a chalk-heavy one). Revisit if `backtest_season.py` run against
additional seasons shows a different value is consistently better.

## Preseason model accuracy ceiling

~6.4 points pooled walk-forward MAE against actual end-of-season SRS (verified, see README
"Results"). This is a real, substantial improvement over "assume nothing changes" (7.9) or
"assume league average" (10.2), but it is not small in absolute terms — a lot of a season's
outcome (injuries, in-season coaching changes, breakout/bust individual performances,
transfer-portal churn that happens *during* a season) is genuinely not knowable in August. The
in-season blending engine exists specifically because the preseason model alone isn't expected
to stay accurate all season — see `docs/methodology.md` section 4.

## Market-spread validation isn't a target

The consensus/market-average-spread correlation (0.74-0.84) and backtest MAE (~3.8-4.0 points)
are **external sanity checks**, not something the model is trained to match. A well-calibrated
rating system should track the market reasonably closely (the market is a strong signal), but
this project doesn't optimize against it directly, unlike `cfb_cover_model`/`cfb_spread_model`,
whose whole point is finding an edge *against* the market. Don't read "correlates with the
market" as "beats the market" — no claim of that kind is made or tested here.

## `betting_lines.provider = 'consensus'` coverage gap

CFBD's own literal "consensus" provider field is populated for 2013-2022 (verified: 522-1201
games/season) but drops to 29 games in 2023 and 0 in 2024-2025. Every place in this project
that needs a market-average spread (`evaluate_against_consensus_spread`,
`backtest_season.py`'s `_market_spreads`) works around this by averaging `spread` across
**every** provider reporting a given game, not filtering to the literal `'consensus'` string —
verified this restores full per-season coverage (~730-1500 games/season) through 2025.

## Transfer portal data before 2021

`cfbd_recruiting_transfer_portal` has zero rows before 2021 (verified live and confirmed in
`cfb_transfer_portal_flow/README.md`). `roster_turnover.py` falls back to inferring turnover
from roster-set differences for those seasons, which cannot distinguish a transfer from a true
freshman arriving on campus (both just show up as "new athlete_id on the roster") — a real,
documented gap in what the pre-2021 version of this feature actually measures, unlike the
portal-endpoint version.

## `team_rosters.year` (eligibility class) is corrupted for older seasons

Verified live: in 2015, 81.5% of `team_rosters` rows have `year` equal to the season itself
(e.g. `2015`) rather than a plausible class value (1-6) — a raw ingest artifact. Corruption
decreases over time (2018: ~30%, 2023: ~6%, 2025: ~4%, 2026: 0%, clean). This is CFBD's *only*
class/eligibility field for roster players — there's no separate age field to use instead.
`features/roster_experience.py` filters to plausible rows and gates team-seasons with too few
valid rows to NaN, but a class-based feature built on top of this data was tested and reverted
for hurting model accuracy (see `docs/methodology.md`) — the corruption is real enough that even
after filtering, it appears to still be net-harmful for training seasons prior to ~2023, even
though the feature would be reasonably clean for 2026 itself.

## No true future-season schedule strength adjustment

Preseason ratings for a not-yet-started season (e.g. 2026) don't account for that season's
actual schedule difficulty (who a team plays) at all -- deliberately, since a power rating
should be schedule-independent by design (that's what opponent-adjustment during the season is
for). This means the preseason number reflects team quality only, not "how hard is their
schedule" -- a genuinely different question this project doesn't answer.

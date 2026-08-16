# Transfer Portal Flow (Conference-Level, 2021–2025)

## Question

Which conferences are benefiting from transfer-portal movement, and which are losing talent to it?

## Data source

`cfbfastR::cfbd_recruiting_transfer_portal(year)` for 2021–2025 — the only seasons CFBD has portal records for (verified live: 0 rows 2015–2020). This is a dedicated origin/destination/rating endpoint, not used anywhere else in this repo (the existing `cfb_win_total_model` roster-turnover feature only *infers* transfers by diffing rosters year-over-year, with no team-pair or rating data on the move itself). Of ~14,400 total portal entries across the 5 seasons, **10,309** have both an origin and a destination on record (the rest either haven't landed yet or aren't tracked).

Conference assignment comes from `cfbd_team_info(year, only_fbs = TRUE)`, pulled fresh the same way `cfb_talent_distribution` does. Transfers to/from a school that doesn't resolve to an FBS conference that season (FCS, juco, D2, etc.) are bucketed into a single `Non-FBS/Other` node rather than dropped, so the flow chart stays a true "all transfers" picture. Every individual transfer is attributed to whichever conference the school actually belonged to that season, so realignment (e.g. Oklahoma/Texas joining the SEC in 2024) is handled correctly in the flow accounting.

## Method

`transfer_portal_flow.R` — pulls, joins, aggregates, and renders three charts.

The flow chart is deliberately **not a geographic map**. An earlier version placed conferences at their real-world venue centroids, but conference realignment during this exact window (Oklahoma/Texas to the SEC, the Pac-12's near-total collapse, multiple G5 programs changing leagues) means "where a conference is" isn't a stable, meaningful position — and geography was never actually informative for conference-to-conference flow to begin with. It's now a **circular diagram**: nodes run clockwise around a circle ordered by net talent (most negative at top, increasing clockwise), so a conference's position alone tells you whether it's a net importer or exporter, before even looking at the arcs. (One implementation note: `geom_curve()`, used for the arcs, is incompatible with `coord_map()`, a true geographic projection, which is part of why the map approach was worth abandoning — arcs rendered in raw unprojected space and shot off the edge of the plot.)

## Charts

1. **`transfer_portal_conference_flow.png`** — the circular conference-to-conference flow diagram. Node size = total portal activity (in + out), node color = net importer (navy) vs. net exporter (red) by aggregate recruiting rating, arc width = total talent moved between the pair, arrow points to the net importer of that pair.
2. **`transfer_portal_net_talent_by_conference.png`** — total recruiting-rating talent imported vs. exported per conference (the volume story — dominated by high-turnover depth-player churn, not just star power).
3. **`transfer_portal_quality_delta_by_conference.png`** — average rating of incoming transfers minus average rating of outgoing transfers, per conference (the "trading up vs. trading down" story, independent of volume).

## Results

- **Non-FBS/Other is, by far, the biggest net exporter** (-641 aggregate rating points, 1,552 out vs. only 193 in over 2,150 total moves) — FBS programs heavily poach FCS/juco/D2 talent upward through the portal, far more than FBS players drop down. This is also the sanity check that the origin/destination join is oriented correctly.
- **The raw volume numbers are counter-intuitive for the two biggest brands.** SEC (-108) and Big Ten (-11) both show *negative* net aggregate rating — they export more total transfers than they import (SEC: 1,352 out vs. 908 in; Big Ten: 1,208 out vs. 895 in). This reflects roster-crunch churn: blue bloods over-stock via high-school recruiting, and a lot of backups get squeezed out via the portal for playing time elsewhere, and each of those outgoing players still counts in the sum.
- **But on a per-transfer basis, SEC and Big Ten clearly trade up.** Average incoming rating vs. average outgoing rating: SEC 0.799 in vs. 0.616 out (+0.183), Big Ten 0.754 vs. 0.568 (+0.186) — both in the top half of all conferences. They lose a lot of players, but the ones they gain are meaningfully better-rated than the ones they lose. This is the real "who's benefiting" story for those two conferences: quality over quantity.
- **Big 12 wins outright on both measures** — the single largest net raw-talent gainer (+223, 1,032 in vs. 1,118 out) *and* the highest quality delta (+0.259, the largest gap between what it takes in and sends out). Pac-12 (+0.239) is a close second on quality despite modest volume (only 978 total moves, the smallest of any major conference in this dataset — consistent with the conference's 2023–24 realignment collapse shrinking its pool of transfer partners).
- **Conference USA is the only conference that trades down on average** (-0.042 quality delta) — it's a modest net importer by raw count/talent (+23), but the players it brings in are, on average, rated slightly lower than the ones it loses.

**Caveat:** `rating` is each player's *original high-school recruiting rating*, frozen at signing — it doesn't reflect how they actually performed in college before transferring. A team gaining a "highly-rated" transfer who busted, or losing a lightly-recruited player who developed into a star, isn't distinguishable from this data alone.

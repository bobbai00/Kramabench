# Failure dive — archeology-hard-9 (all-arm common-core failure)

## Task
Q: What is the correlation (to 6 dp) between the rank of ancient Roman cities
and the population of their corresponding modern cities with a population
over one million? For rank, if there is an 'or', average the two numbers.
An ancient city == a current city if the distance between them is < 0.1
degrees. If there are multiple ancient cities, take the last sample. Round to
6 dp.

D: two files (the roman_cities/worldcities distance-match family — cf.
archeology-hard-7).
- `roman_cities.csv` — `Barrington Atlas Rank` (values like `"3"`,
  `"4 or 5"`), `Longitude (X)`, `Latitude (Y)`.
- `worldcities.csv` — `lng`, `lat`, `population`; filtered to `>1,000,000`.

## Solution
```
worldcities → filter(population > 1e6)
roman → clean_rank: mean of the digits in the rank string ("4 or 5"→4.5)
match: cKDTree(roman lon/lat) query_ball_point(modern, r=0.1)  [L2 disc]
   per modern match: take idxmax(rank) among the roman hits
pearson corr(matched rank, matched population) → round(6) = 0.015648
```
Four dense specs stacked: `>1M` filter, rank-averaging, **L2 disc r=0.1**
match, and the idxmax/"last sample" tie-break — a correlation over a *small*
matched set is hypersensitive to each.

## What DeltaStats3kD2 does (best arm, FAIL −0.098529)
- STEP 0–2 load; `roman_rank_parsed` (rank averaging), `modern_over_1m`.
- STEP 3–4 `roman_last_sample` → `roman_modern_matched` --> **the problematic
  step.** The match/tie-break differs from gold's L2-disc + idxmax-rank
  (likely an L∞ box or a different "last sample" resolution), changing which
  (rank, population) pairs enter the correlation.
- STEP 5 `rank_population_corr` → −0.098529 — **wrong sign** vs gold +0.0156.

## What the gold dataflow does at the missed step
Gold matches with a KD-tree L2 disc of radius 0.1 and, per modern city,
keeps the roman hit with the **maximum** rank (`idxmax`). On the tiny matched
set, swapping the metric or the per-match selection moves a few pairs and
flips the near-zero correlation's sign.

## Why it fell short
**Stacked method choices on a hypersensitive statistic, render-invariant.**
Identical to the archeology-hard-7 finding: the L2-vs-other-metric and the
tie-break live in the *English spec*, not in any renderable table, so more
rows/stats/history cannot carry them. Because the true correlation is ~0.016
(essentially noise), any small matching difference produces a large relative
error and often a sign flip.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 8 | −0.098529 |
| Delta3kSchemaOnly | 8 | +0.020731 (right sign, still off) |
| Delta5kSchemaOnly | 8 | −0.100105 |
| Latest3kSchemaOnly | 7 | −0.098529 |

Gold +0.015648. Answers **scatter across zero** (−0.10 … +0.02) — the
signature of a near-zero statistic amplifying tiny matching/tie-break
differences. Same distance-metric root cause as the archeology-hard-7 flip
case, here as a common-core failure. Render-invariant; only a pinned
matching spec would converge the arms.

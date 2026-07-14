# Failure dive — astronomy-easy-3 (all-arm common-core failure)

## Task
Q: Using Swarm Alpha satellite data (warmup dataset), find the average
atmospheric density measured between 450km and 500km altitude during 2015 for
available data points at 00:00 of each day in the initial state file. Clean up
the density measurement data (n/a values or 9.99E32).

D: two files.
- initial-state file — 715 rows, one per `File ID` (wu###), with a midnight
  `Timestamp` and `Altitude (km)`. Sample: `wu001 2014-01-31 00:00 … alt
  486.25`.
- Swarm-Alpha density series — 53,259 rows of `Timestamp, Orbit Mean Density
  (kg/m^3)` at 10-min cadence; sentinel `9.99E32` marks missing.

## Solution
```
initial states → filter(450 ≤ alt ≤ 500) ∧ year==2015 ∧ time==00:00
density → mask(≥ 9.99e31) → dropna
join initial-day midnights to density → mean(density) = 7.95e-13
```

## What DeltaStats3kD2 does (best arm, FAIL 8.02e-13)
- `initial_states` + `swarma_2015_raw` loaded.
- `initial_states_2015_midnight`: filter `450 ≤ Altitude ≤ 500`, 2015 → 120 rows.
- `swarma_2015_clean`: `mask(dens >= 9.99e31)` + dropna (sentinel handled
  correctly).
- `density_on_initial_days` → `avg_density_result`: `mean` = 8.02e-13.
  --> the only divergence is **which density points map to each initial-day
  midnight** (exact-timestamp vs nearest-in-window matching) — a sub-1%
  effect.

## What the gold dataflow does at the missed step
Gold's join selects a slightly different density set for the per-day 00:00
points (exact matching of the initial-file timestamps), yielding 7.95e-13.
The pipelines are otherwise identical.

## Why it fell short
**Convergent near-miss killed by exact-match scoring.** 8.02e-13 vs
7.95e-13 is **~0.9%** — semantically the right answer, but `numeric_exact`
requires an exact match, so it scores 0. All four arms produce the identical
8.02e-13. The residual comes from a timestamp-join nuance, not from any
render parameter. This is the clearest "scoring-strictness, not a real
capability gap" case in the common core: no context lever changes it, and
arguably the task/metric is at fault.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 4 | 8.02e-13 |
| Delta3kSchemaOnly | 4 | 8.02e-13 |
| Delta5kSchemaOnly | 7 | 8.02e-13 |
| Latest3kSchemaOnly | 7 | 8.02e-13 |

Gold 7.95e-13. **All four identical, ~1% off** — a convergent near-miss.
Flag for the paper: some common-core "failures" are exact-match-metric
artifacts on essentially-correct answers, distinct from genuine
capability gaps.

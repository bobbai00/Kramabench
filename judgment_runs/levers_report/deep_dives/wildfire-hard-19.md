# Failure dive — wildfire-hard-19 (all-arm common-core failure)

## Task
Q: In 2016, what percentage (2 dp) of fires were brought under control with
it raining moderately or heavily (>0.05 in) in the fire area on the same or
the day before the control day? Assume the narrowest fire diameter is 1km; if
a weather station falls in the fire area, use that station's detailed
observation. (answer type: numeric_approximate)

D: a 2016 fire-incident dataset (location, control date, extent) and a
weather-station observation dataset (station location + daily precipitation).
The task requires a **spatial join**: a station within the 1km-diameter fire
area supplies that fire's rain reading.

## Solution
```
fires 2016 → build 1km-radius area per fire
weather → for each fire, find stations inside the area
join by (fire, control-day OR day-before) → rain > 0.05 in?
percentage of fires with qualifying rain → round(2) = 32.76
```
The hard part is the geospatial containment + the same-or-prior-day rain
window.

## What the arms do
- **DeltaStats3kD2 (best arm) step-capped (26 steps), NO answer, score 0.**
- Delta3kSchemaOnly (22) → 27.35, score **0.858**; Delta5kSchemaOnly (16) →
  27.35, score 0.858 — closest, just under pass.
- Latest3kSchemaOnly (20) → 0.63, score 0.505 (spatial join collapsed).
--> the divergence is spatial-join fidelity + the rain-day window; the
27.35 arms approximate the containment well, the Latest arm mis-joins.

## What the gold dataflow does
Gold performs the 1km containment join and the same/prior-day rain test
precisely → 32.76. The 27.35 arms use a looser containment (fewer qualifying
fires).

## Why it fell short — and the counter-intuitive note
**Task-intrinsic geospatial complexity; the BEST arm did WORST.** This is the
one common-core task where the profile arm (DeltaStats3kD2) is strictly worse
than the leaner arms: it **churned to the step cap and returned nothing**,
while Delta 3k/5k schema-only finished at 0.858 (near pass). The stats/D2
render did not help a spatial-join task and its extra per-step weight
plausibly contributed to the step-budget exhaustion — an anti-example to
"more context is safer," consistent with the CASE_METRICS finding that stats
is pure tax when it doesn't address the task's actual difficulty.

## Cross-arm failure shape
| arm | steps | answer | score |
|---|---|---|---|
| DeltaStats3kD2 | 26 | (no response) | 0 |
| Delta3kSchemaOnly | 22 | 27.35 | 0.858 |
| Delta5kSchemaOnly | 16 | 27.35 | 0.858 |
| Latest3kSchemaOnly | 20 | 0.63 | 0.505 |

Gold 32.76. Not render-fixable on accuracy (geospatial join is the blocker),
but a pointed cost/robustness data point: the richest-context arm was the
only one to time out.

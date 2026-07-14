# Failure dive — wildfire-hard-14 (all-arm common-core failure)

## Task
Q: What is the correlation between the proportion of generally unsafe
air-quality days (per EPA) and the amount of land affected by fires in 2024?
Round to 2 dp. (answer type: numeric_approximate)

D: an EPA air-quality daily series (AQI category per day, to derive the
"generally unsafe" proportion per area) and a fire-land-affected dataset for
2024 (acreage per area).

## Solution
```
air quality → per area: proportion of days classified "generally unsafe"
fires 2024  → per area: total land affected
join on area → pearson correlation → round(2) = 0.65
```

## What DeltaStats3kD2 does (best arm, FAIL 0.42, score 0.74)
- Builds both per-area series and correlates. --> **the problematic step** is
  the "generally unsafe" day classification and/or the area join grain:
  0.42 vs 0.65 is the same *sign and direction* but a materially weaker
  correlation, consistent with a slightly different AQI threshold for
  "generally unsafe" or a different area-matching that dilutes the pairing.

## What the gold dataflow does at the missed step
Gold's "generally unsafe" cutoff and area alignment produce a tighter 0.65
correlation. The difference is the categorical threshold definition, not the
mechanics.

## Why it fell short
**Convergent near-miss on a threshold definition, render-invariant — and a
metric near-pass.** All four arms produce the identical 0.42, scoring
**0.7386** on the numeric_approximate RAE metric — below the 0.9 pass line
but clearly "directionally correct." The residual is the AQI "generally
unsafe" cutoff (a convention), invisible to any render parameter.

## Cross-arm failure shape
| arm | steps | answer | score |
|---|---|---|---|
| DeltaStats3kD2 | 6 | 0.42 | 0.739 |
| Delta3kSchemaOnly | 6 | 0.42 | 0.739 |
| Delta5kSchemaOnly | 6 | 0.42 | 0.739 |
| Latest3kSchemaOnly | 5 | 0.42 | 0.739 |

Gold 0.65. **All four identical (0.42), all at 0.739** — a convergent
near-miss that just misses the pass threshold. Like astronomy-easy-3, part of
the common core is "close-but-not-close-enough," here on a categorical
threshold convention rather than a hard capability gap. Render-invariant.

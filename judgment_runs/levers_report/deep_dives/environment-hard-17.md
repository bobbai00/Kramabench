# Failure dive — environment-hard-17 (all-arm common-core failure)

## Task
Q: What is the seasonal exceedance rate (%, 2 dp) of Chatham's Bucks Creek
Beach in the summer (Jun/Jul/Aug) with the most rainfall in its area? Impute
missing values with the median of the month in non-missing years.

D: the Bucks Creek Beach water-quality datasheet (per-day `Violation`) and
`monthly_precipitations_<city>.csv` (per-year monthly rainfall for Chatham).

## Solution
```
rainfall: sum Jun+Jul+Aug per Year (2002–2020) → sort desc → max_year        (A)
beach: filter Bucks Creek, summer months, impute missing = month-median over
       non-missing years → restrict to max_year → exceedance% = 47.37         (B)
```
Load-bearing step (A): **first select the single summer with the highest
total rainfall**, then compute the exceedance rate *for that year only*.

## What DeltaStats3kD2 does (best arm, FAIL 21.43)
- Loads beach + rainfall; imputes; computes an exceedance rate. --> **the
  problematic step**: it does not pick the max-rainfall *year* and restrict
  to it (or picks the wrong one) — 21.43 is the exceedance over a different
  slice (likely all summers pooled, or the wrong year), vs gold's
  single-year 47.37.

## What the gold dataflow does at the missed step
Gold explicitly ranks years by summer-total rainfall, takes `max_year`, and
computes the exceedance rate within that year alone. The "with the most
rainfall" clause is a *year-selector*, which the arms treat as descriptive.

## Why it fell short
**Convention misread of a selector clause, render-invariant.** All four arms
produce the **identical 21.43** — they read "the summer with the most
rainfall" as scenery rather than a filter selecting one year. The rainfall
file and beach data were both available; nothing about rows/stats/history
tells the agent to rank-and-select the year.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 10 | 21.43 |
| Delta3kSchemaOnly | 9 | 21.43 |
| Delta5kSchemaOnly | 10 | 21.43 |
| Latest3kSchemaOnly | 15 | 21.43 |

Gold 47.37. **All four identical** — maximal render-invariance; a convergent
misread of the max-rainfall-year selector. Convention family
(cf. archeology-hard-12, environment-hard-7).

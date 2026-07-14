# Failure dive — archeology-hard-2 (all-arm common-core failure)

## Task
Q: Across the years, what was the percent of years the wet-dry index was
increasing? Give your answer in percentage, round your answer to 2 decimal
places.

D: same two Excel files as archeology-hard-1.
- `climateMeasurements.xlsx` — **dirty header** (`header=0, skiprows=5`);
  carries a `Wet_dry_index` column and an `Age_ky.1` time axis. Chronological
  year is `1950 - Age_ky.1·1000`, which **reverses** the raw row order
  (raw file is deepest/oldest last).
- `radiocarbon_database_regional.xlsx` — loaded by gold but not load-bearing
  for the increasing-fraction.

## Solution
```
climate.xlsx (skiprows=5) → dropna → year = 1950 - Age_ky.1*1000
  → sort by chronological year
  → for each year: wet_dry(year) > wet_dry(previous chronological year)?
  → fraction of years that are increasing · 100 → round(2) = 38.42
```
Two load-bearing subtleties: the skiprows-5 header, and comparing each
sample to its **chronological** predecessor (after the BP→year flip), not to
its raw-row predecessor.

## What DeltaStats3kD2 does (best arm, FAIL 47.29)
- STEP 0–2 load; `climate_reloaded` (defeats the skiprows-5 header — profile
  lever again flags the unnamed columns).
- STEP 3–5 `wet_dry_series` (×2) → `wet_dry_increasing_pct` --> **the
  problematic step.** Computes the sign of consecutive differences on the
  series **in raw/loaded order** (or an ambiguous ordering), giving the
  fraction of positive steps ≈ 47.29%. It does not re-sort by the
  chronological year derived from `Age_ky.1·1000`, so "previous sample"
  is the wrong neighbor.

## What the gold dataflow does at the missed step
Gold converts `Age_ky.1` to chronological year, sorts ascending, and only
then evaluates "increasing vs the previous year." Because the age axis
inverts the raw order, the raw-order diff sign is systematically different →
the ~9-point gap (38.42 vs ~47–50).

## Why it fell short
**Method-choice on ordering, downstream of the header trap.** The header
sub-problem is render-assisted (D2 profile → one reload); the ordering
convention (sort by derived chronological year before differencing) is a
reasoning choice with no render surface. Every arm cleared the header but
none re-sorted correctly.

## Cross-arm failure shape
| arm | steps | answer | note |
|---|---|---|---|
| DeltaStats3kD2 | 7 | 47.29 | fewest steps, header via profile |
| Delta3kSchemaOnly | 11 | 49.82 | more header struggle |
| Delta5kSchemaOnly | 15 | 50.25 | " |
| Latest3kSchemaOnly | 19 | 49.78 | most churn on the header |

**Clustered 47–50% vs gold 38.42** — not identical (unlike hard-12) because
the exact diff/ordering improvisation varies, but all in the same wrong
neighborhood: none applied the chronological re-sort. Note the step-count
gradient (7→19) tracks header-struggle by mode exactly as CASE_METRICS
predicts (profile ≪ schema-only ≪ Latest on dirty xlsx), yet accuracy is
render-invariant because the ordering bug is downstream.

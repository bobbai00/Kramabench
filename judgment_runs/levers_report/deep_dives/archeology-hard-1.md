# Failure dive — archeology-hard-1 (all-arm common-core failure)

## Task
Q: What is the average Potassium in ppm from the first and last time the
study recorded people in the Maltese area? Assume that Potassium is linearly
interpolated between samples. Round your answer to 4 decimal places.

D: two Excel files.
- `radiocarbon_database_regional.xlsx` — dated radiocarbon records incl. a
  regional column; the Maltese-area rows define the *first* and *last* study
  years (converted BP→CE via `1950 - date`).
- `climateMeasurements.xlsx` — **dirty header**: the real column names are on
  row 6 (`header=0, skiprows=5`); the top 5 rows are title/metadata, so a
  naive load yields 29 `Unnamed:` columns. Carries `K_ppm` (Potassium) and an
  `Age_ky.1` time axis (`year = 1950 - Age_ky.1·1000`).

## Solution
```
radiocarbon.xlsx ─ dropna ─ year=1950-date ─ filter(Maltese) ─ first & last year ┐
climate.xlsx (skiprows=5) ─ dropna ─ year=1950-Age_ky.1*1000 ─ K_ppm series      │
   → linear-interpolate K_ppm at the first-year and last-year points ────────────┘
   → mean(K@first, K@last) → round(4) = 8577.5298
```
Three hard sub-steps: (1) find the climate header row; (2) build the K–year
series in the right units; (3) linearly interpolate K at two specific years
and average.

## What DeltaStats3kD2 does (best arm, FAIL 380.8208, 14 steps)
- STEP 0–2 load both files; `climate_raw_preview`; `radiocarbon_malta_window`
  (finds the Maltese first/last window — the D2 profile flags the unnamed
  headers, which is what pushes it to reload).
- STEP 3–6 `climate_reload_header` → `climate_reparse` → `climate_sheets` →
  `climate_header_row29` → `climate_allstring_preview` — **four header
  reloads** to defeat the skiprows-5 trap (the profile lever earns its keep
  here: structure discovery succeeds).
- STEP 7–8 `climate_numeric_series` → `climate_k_timeseries` build a K series.
- STEP 9 `k_avg_over_malta_window` --> **the problematic step.** It averages
  K over the window rather than **interpolating K at exactly the first and
  last years and averaging those two points**; STEP 10–11 retry in BP units
  (`_bp`) but keep the wrong aggregation → 380.8208 (≈ mean of a raw slice,
  ~22× below gold).

## What the gold dataflow does at the missed step
Gold does not average a window. It finds the two boundary years (first, last
Maltese record), linearly interpolates the K_ppm series to those two
year values, and averages **only those two interpolated values**. The
"linearly interpolated between samples" clause is the operative spec; the
arm treated it as "mean over the interval."

## Why it fell short
**Method-choice on the arithmetic core, downstream of a header trap the
render lever DID help with.** Two-layer: the dirty-header sub-problem is
exactly what D2's `headers: 29 of 30 unnamed` profile is for — and the stats
arm used it to reach a correct K series (14 productive steps, furthest of any
arm). But the *interpolate-at-two-points-then-average* semantics is a
reasoning specification with no render surface; more rows/stats/history can't
carry it. So the profile lever moved the failure downstream but did not
close it.

## Cross-arm failure shape
| arm | steps | answer | where it broke |
|---|---|---|---|
| DeltaStats3kD2 | 14 | 380.8208 | header solved; window-mean not interpolation |
| Delta3kSchemaOnly | 9 | 71,980.9 | header/units mis-scaled (×1000 error) |
| Delta5kSchemaOnly | 4 | 0.2422 | gave up early on the header, degenerate value |
| Latest3kSchemaOnly | 26 | (no response) | churned on the header to the step cap |

**Same task, four different failure points** — a hard 3-hop task with a
dirty-xlsx header trap. Note the ordering: the arm with the richest structure
signal (D2 profile) got *furthest* (past the header) and the history-mode
arm churned to death on the header — consistent with "profile is an
anti-iteration lever for dirty headers" (CASE_METRICS F2/F5), but the
task's interpolation core defeats all four regardless. Render-relevant at
the header, render-invariant at the answer.

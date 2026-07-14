# Failure dive — archeology-hard-5 (all-arm common-core failure)

## Task
Q: In the Maltese dataset, find the year of the most northern Neolithic
sample, breaking ties by considering the later year. What is the maximum
aluminum value recorded in the climate dataset in the closest year to that
year? If there are multiple closest years, take the max aluminum value
measured across all of them. Round to 4 decimal places.

D: the two Excel files (as in hard-1/hard-2).
- `radiocarbon_database_regional.xlsx` — `Species`, `Culture`, latitude,
  `date` (BP). Maltese Neolithic *Homo sapiens* samples; "most northern" =
  max latitude, tie-break by later `year = 1950 - date`.
- `climateMeasurements.xlsx` — **dirty header** (`skiprows=5`); carries an
  aluminum column (`Al_ppm`) and `Age_ky.1` → year.

## Solution
```
radiocarbon → filter(Homo sapiens & Neolithic & Maltese)
   → argmax(latitude), tie-break later year → target_year
climate (skiprows=5) → year = 1950 - Age_ky.1*1000
   → rows at min |year - target_year|  (all ties)
   → max(Al_ppm) over those rows → round(4) = 66158.3691
```

## What DeltaStats3kD2 does (best arm, FAIL 0.0260, 14 steps)
- STEP 0–5 load both; `rcb_maltese`, `rcb_sheets`, `rcb_sheet1` isolate the
  Maltese Neolithic northern sample; header reloads on climate.
- STEP 6–13 `climate_raw`/`climate_head` repeated --> **churn on the
  skiprows-5 header**; `climate_year_aluminum` → `closest_year_max_aluminum`
  end at 0.0260 — a degenerate value (it picked a wrong/near-empty aluminum
  column or a fraction, having never stabilized the header).

## What the gold dataflow does at the missed step
Gold reads climate with `skiprows=5`, so `Al_ppm` is a real numeric column;
it selects the closest-year rows and takes their max Al → ~66,158. The arm
never got a clean `Al_ppm` series.

## Why it fell short
**Dirty-header trap + downstream selection, render-invariant on the answer.**
The best (profile) arm spent 14 steps fighting the header and still
produced a degenerate value; the other three loaded *something* and
converged on 36,828.7165 (a plausible-but-wrong closest-year/max pick).
Neither more rows nor the profile carried the two reasoning specs
("closest year, all ties" + "max aluminum") reliably enough.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 14 | 0.0260 (header churn → degenerate) |
| Delta3kSchemaOnly | 8 | 36,828.7165 |
| Delta5kSchemaOnly | 8 | 36,828.7165 |
| Latest3kSchemaOnly | 9 | 36,828.7165 |

Gold 66,158.3691. Three arms **identical (36,828.72)** via a common
wrong-column/closest-year path; the profile arm diverges by over-iterating
the header into a degenerate result — the counter-intuitive note that on
this task the extra structure signal *lengthened* the trajectory without
rescuing it. Render-invariant.

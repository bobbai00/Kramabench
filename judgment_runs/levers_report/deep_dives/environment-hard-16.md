# Failure dive — environment-hard-16 (all-arm common-core failure)

## Task
Q: How many marine beaches (2002–2023 inclusive) remained safe for swimming
the entire time (no violation at all throughout the seasons; if no data for a
beach in a year, assume safe)?

D: the per-year beach water-quality datasheets (marine beaches), each with a
`Beach Name` and a `Violation` column (values `yes`/`no`, mixed case) across
seasons and years.

## Solution
```
concat all years' marine rows
  → violated := set of Beach Name where Violation.lower() == 'yes' (any row)
  → safe := all beaches NOT in `violated`   (missing-year ⇒ still safe)
  → count(safe) = 60
```
Load-bearing: a beach is unsafe if it has **≥1** `yes` violation anywhere in
the whole 2002–2023 span; "assume safe" for missing data means absence of a
`yes` is safe.

## What DeltaStats3kD2 does (best arm, FAIL 75)
- Loads and aggregates the marine datasheets, counts beaches with no
  violation. --> **the problematic step** is the violation predicate /
  coverage: the arm counts 75 safe beaches vs gold's 60, i.e. it marks 15
  beaches safe that gold marks unsafe — most likely by (a) a narrower
  violation match (case/'yes' variants), (b) not unioning violations across
  **all** years, or (c) restricting to beaches present in a subset of years.

## What the gold dataflow does at the missed step
Gold lowercases `Violation`, takes the union of any-`yes` beaches across the
entire concatenated span, and subtracts from the full beach set. Its unsafe
set is larger (more beaches caught by ≥1 violation somewhere), so fewer
remain safe.

## Why it fell short
**Convention misread of "no violation at all … throughout" + missing-data
rule, render-invariant.** The `Violation` values were visible to every arm;
the divergence is how strictly "any violation, any year" is applied. All
arms land in a tight 73–75 band above gold's 60 — a convergent
over-counting of safe beaches, not an information gap.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 6 | 75 |
| Delta3kSchemaOnly | 4 | 75 |
| Delta5kSchemaOnly | 4 | 73 |
| Latest3kSchemaOnly | 5 | 74 |

Gold 60. **Clustered 73–75** — convergent violation-scope misread.
Render-invariant; convention family.

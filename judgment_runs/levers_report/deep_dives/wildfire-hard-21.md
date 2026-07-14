# Failure dive — wildfire-hard-21 (all-arm common-core failure)

## Task
Q: Based on NOAA data, what are the top 3 states that lost the most
residential property in value between 2005 and 2010 (inclusive)? Answer in
state full names and **do not discard rows with missing values
unnecessarily**. (answer type: list_exact)

D: `ZHVI.csv` (Zillow Home Value Index — residential property value series by
region/state over time; **has missing cells**), a NOAA wildfires dataset, and
a state-abbrev→name map.

## Solution
```
ZHVI → state-level residential value at 2005 and at 2010
  → value change 2005→2010 per state (KEEP rows with missing values —
     do not blanket-dropna, or you lose states like Washington)
  → nlargest(3, loss) → full state names = ['California','Washington','Idaho']
```
The explicit instruction "do not discard rows with missing values
unnecessarily" is the load-bearing spec: a naive `dropna()` removes
Washington.

## What the arms do
- Delta3kSchemaOnly → **`[]`**: --> at STEP 5 it concluded `ZHVI.csv` is a
  "zero-byte (empty) file" (`zhvi_debug.size_bytes = 0`, loaded as 0×0) and
  gave up — a **data-access misdiagnosis** (the other arms read it fine), so
  it returned an empty list.
- DeltaStats3kD2 / Delta5kSchemaOnly → `['California','Idaho','Montana']`
  (2/3 = 0.667): --> they `dropna()` and lose Washington, promoting
  Montana into 3rd.
- Latest3kSchemaOnly → `['Washington','California','Utah']` (2/3): keeps
  Washington but mis-ranks 3rd (Utah).

## What the gold dataflow does
Gold computes the 2005→2010 change while **retaining missing-value rows**
where appropriate, yielding California/Washington/Idaho. The two failure
modes are: dropping missing (loses Washington) and the file-empty give-up.

## Why it fell short
**A split of two render-invariant causes.** (1) The "don't discard missing"
instruction is a verbal spec no render parameter conveys — three arms either
ignored it (dropna → lost Washington) or mis-ordered the tail. (2) Delta3k's
empty-file give-up is an execution/robustness bug, not a context issue (the
file is readable; it mis-read a probe as size 0). Neither is fixable by
sampling/stats/history.

## Cross-arm failure shape
| arm | steps | answer | score |
|---|---|---|---|
| DeltaStats3kD2 | 6 | ['California','Idaho','Montana'] | 0.667 |
| Delta3kSchemaOnly | 7 | [] | 0 |
| Delta5kSchemaOnly | 9 | ['California','Idaho','Montana'] | 0.667 |
| Latest3kSchemaOnly | 9 | ['Washington','California','Utah'] | 0.667 |

Gold ['California','Washington','Idaho']. Three arms get 2/3 by different
tail states; one arm gives up. The shared root — ignoring the "keep missing"
instruction — is a convention/spec issue; Delta3k's `[]` is a separate
one-arm robustness failure. Render-invariant.

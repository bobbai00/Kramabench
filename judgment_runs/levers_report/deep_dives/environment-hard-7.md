# environment-hard-7 — deep dive

Counter-intuitive case: the anchor **Delta3kSchemaOnly** lost to ALL THREE rays
(Delta5kSchemaOnly, DeltaStats3kD2, and the history-less **Latest3kSchemaOnly**).
On its face a triple-lever convergence — but the semantic walk shows every arm
computed the bit-identical number and the split is a unit-convention coin flip.

## Task

Q: "What was the difference in bacterial exceedance rates (to 2 decimal places)
for marine beach samples collected in 2023 between communities with more than
50% environmental justice (EJ) populations and those with less than 25% EJ
populations?"

D: two CSVs under `data/environment/input/`.

- **water-body-testing-2023.csv** — 16,258 rows × 11 cols. One row per beach
  water sample in 2023. Real rows:
  `242, Provincetown, 1, Barnstable, 2023, 2023-07-10 00:00:00, 333 Commercial Street, Marine, Enterococci, 5, No`
  `126, Harwich, 1, Barnstable, 2023, 2023-07-11 00:00:00, Bank Street, Marine, Enterococci, 504.0, Yes`
  Relevant columns / quirks:
  - `Beach Type Description` ∈ {`Marine` (8,491), `Fresh` (7,767)} — **capital-M**;
    gold filters `== 'Marine'` without lowercasing.
  - `Violation` ∈ {`No` (15,145), `Yes` (1,113)} — **title-case**; an "exceedance"
    is `Violation == 'yes'` (lowercased).
  - `Community` is title-case (e.g. `Provincetown`) — the join key to EJ.
  - `Year` is uniformly 2023 (the file is already the 2023 slice; a `Year==2023`
    filter is a no-op).
- **environmental-justice-populations.csv** — 187 rows × 9 cols. One row per MA
  municipality. Real rows:
  `1, Acton, M, 3, 16, 18.8, 5621, 24021, 23.4`
  `3, Adams, I, 6, 8, 75.0, 6761, 8166, 82.8`
  Relevant columns / quirks:
  - `Municipality` title-case — join key to `Community`.
  - **`Percent of population in EJ BGs`** — the EJ share, on a **0–100 scale**
    (min 0.7, mean 42.9, max 100). `>50` selects 64 municipalities, `<25`
    selects 75. This column being a *percent already* is the semantic trap: the
    task's own "rate" answer is expected on the same ×100 scale, but the
    exceedance rate the agents compute (`mean` of a boolean) is a **fraction**.

## Solution

From `solutions/environment/environment-hard-7.py`, as an operator graph:

```
load water-body-testing-2023.csv                     load environmental-justice-populations.csv
  lower(Beach Name, Community, Violation)              lower(Municipality)
        │                                                    │
  filter Beach Type Description=='Marine'         ┌─ ej_ge_50 = pct > 50   (64 municipalities)
  (capital-M, NOT lowered)  → marine_df           └─ ej_le_25 = pct < 25   (75 municipalities)
        │                                                    │
        ├──── merge(inner, Community == Municipality) ───────┤
        │                                                    │
   marine_ej_ge_50                                      marine_ej_le_25
        │                                                    │
  rate_ge_50 = len(Violation=='yes')/len(grp) * 100   rate_le_25 = len(Violation=='yes')/len(grp) * 100
        └──────────────── answer = rate_ge_50 − rate_le_25 ──┘  →  10.87
```

Node annotations:
- **load** water file — plain `read_csv`, then lowercase `Beach Name`,
  `Community`, `Violation`.
- **marine filter** — predicate `Beach Type Description == 'Marine'` (capital-M,
  the one column *not* lowercased). No Year filter (file is 2023 already).
- **load** EJ file — lowercase `Municipality`.
- **EJ groups** — two masks on `Percent of population in EJ BGs`: `> 50` and
  `< 25` (the 25–50 band is dropped).
- **join** — inner-merge marine samples to each EJ group on
  `Community == Municipality`; grain = one row per (sample × matched community).
- **aggregation grain** — per group, `exceedance_rate = count(Violation=='yes') /
  count(group) * 100` — **a PERCENTAGE (×100)**.
- **final compute + format** — `rate_ge_50 − rate_le_25`, 2 dp → **10.87
  percentage points**.

Load-bearing arithmetic: the underlying quantity is
`0.1512360639844886 − 0.042488619119878605 = 0.10874744486461`.
`round(0.10874744486461, 2) = 0.11`; `round(0.10874744486461 * 100, 2) = 10.87`.
Gold's ×100 is the whole ballgame.

## What Delta3kSchemaOnly does (mode X — FAIL, answer 0.11)

- **step 0** — load `ej_populations` and `water_tests_2023`. Correct two sources.
- **step 1a** `marine_2023` — filter `Year==2023 & Beach Type Description=='Marine'`
  → 8,491 rows. Correct marine slice (matches gold's marine grain).
- **step 1b** `ej_groups` — keep `Percent of population in EJ BGs`, tag `gt50`
  (`>50`) / `lt25` (`<25`), drop the middle → 139 rows (64+75). Correct EJ split.
- **step 1c** `exceedance_diff` — inner-merge on `Community`==`Municipality`
  (keys left title-case on both sides — harmless, both sources are title-case so
  the match set is identical to gold's lowercased join); `is_exceed =
  Violation.lower()=='yes'`; `rates = groupby(ej_group)['is_exceed'].mean()`;
  `diff = rates['gt50'] − rates['lt25']`; `round(diff, 2)`.
  --> **DIVERGENCE.** Computes the bit-identical `0.10874744486461` but returns
  it as a **bare fraction** — `round(diff,2)` with **no ×100**. Falls short of
  gold plan items 6/7 (rate expressed as a percentage). Rendered result:
  `[exceedance_diff] Output 1x1: ... | diff | 0  0.11`.
- **step 2** — Final Answer: **0.11**. Wrong only by the ×100 convention.

Every upstream semantic is correct; the sole shortfall is the final-op unit
convention (proportion `0.11` instead of percentage `10.87`).

## What Latest3kSchemaOnly does (mode Y — PASS, answer 10.87)

Representative of the **shared winning pipeline** — Delta5kSchemaOnly and
DeltaStats3kD2 win by the identical mechanism (their final ops carry `*100` /
name the output `diff_pct_points`).

- **step 0** — load `water_2023`, `ej_pops`. Correct.
- **step 1** `marine_2023` — filter `Year==2023 & Beach Type lower=='marine'`;
  add `Community_norm` (lowered) and `Violation_bool` (yes→True) → 8,491 rows.
  Matches gold marine grain (near-miss recovered: it *does* lowercase Beach Type,
  but 'Marine' is uniform so the row set is identical to gold's capital-M filter).
- **step 2** `ej_clean` — `Community_norm = lower(Municipality)`; `ej_pop_pct`
  numeric → 187 rows (EJ prep).
- **step 3** `marine_with_ej` — **left**-join marine to EJ on `Community_norm`
  → 8,491 rows with nulls where no EJ match. Grain-equivalent to gold's inner
  join: the downstream `>50` / `<25` masks drop the null-EJ rows anyway.
- **step 4** `ej_exceedance_diff` — `high = ej>50`, `low = ej<25`;
  `high_rate = mean(Violation_bool)`, `low_rate = mean`;
  `diff_pct_points = (high_rate − low_rate) * 100`.
  **×100** → 10.874744486461; also reports `high_n=2063`, `low_n=3295`.
  Rendered: `high_rate 0.1512360639844886  low_rate 0.042488619119878605
  diff_pct_points 10.874744486461  high_n 2063  low_n 3295`.
- **step 5** — Final Answer: **10.87**.

No divergence from the gold plan.

## Why Y succeeded but X failed

**It didn't — not for any reason a context lever touched.** Every arm ran the
same data pipeline and produced the **bit-identical underlying quantity**
`0.10874744486461` (re-derived from the raw CSVs against the anchor's own
title-case join: `round(v,2)=0.11`, `round(v*100,2)=10.87`). The only difference
is the final-op unit convention: the three winners multiplied by 100 (`*100` /
`diff_pct_points`), the anchor returned the raw fraction (`round(diff,2)`).

The evidence rendered at the anchor's divergence step was **identical in kind**
to the winners': the same `Percent of population in EJ BGs` column and the same
`Violation` field were visible to all four arms. **Nothing rendered told the
winners to ×100 and the anchor not to.** The task text ("difference in … rates …
to 2 decimal places") does not disambiguate percent vs fraction; gold chose ×100.
The ×100 is a free formatting choice made inside each arm's final-op plan,
*upstream* of any differential rendered observation between the arms — so
Latest's fuller history, the 5k budget, and the stats block all played no role.

Per SKILL — "reject method-choice divergence that predates the arms' first
rendered difference" — the ×100 choice predates any rendered divergence.

**Label: CHRONIC/VARIANCE.** environment-hard-7 is on `chronic_flippers.json`
(all three pairs tagged chronic). The apparent triple-lever convergence dissolves
under the walk: percentage is the majority "rate" convention, so three
independent arms landing on ×100 while the anchor lands on the bare fraction is
one coin landing the same way three times, not corroborating evidence of a
context gap. None of the three flips is attributable to a lever.

**Per-arm divergence table**

| arm | first divergence | gold-plan item fallen short | answer |
|---|---|---|---|
| Delta3kSchemaOnly (X) | step 1c `exceedance_diff`: `round(diff,2)`, no ×100 | 6/7 rate-as-percentage | 0.11 ✗ |
| Latest3kSchemaOnly (Y) | none | — | 10.87 ✓ |
| Delta5kSchemaOnly | none (final op `*100`) | — | 10.87 ✓ |
| DeltaStats3kD2 | none (final op `*100`) | — | 10.87 ✓ |

# environment-hard-7 — flip attribution walk

Anchor `DataflowSystemGPT52Delta3kSchemaOnly` loses to all three rays
(`Delta5kSchemaOnly` C1, `DeltaStats3kD2` C2, `Latest3kSchemaOnly` C3).
Task is on `chronic_flippers.json` — all three pairs annotated chronic*.

## Task + gold answer

QUESTION: "What was the difference in bacterial exceedance rates (to 2 decimal
places) for marine beach samples collected in 2023 between communities with more
than 50% environmental justice (EJ) populations and those with less than 25% EJ
populations?"

GOLD ANSWER: **10.87**

## Gold semantic plan

From `solutions/environment/environment-hard-7.py`:

1. **Source**: load `water-body-testing-2023.csv` (already the 2023 file).
   Lowercase `Beach Name`, `Community`, `Violation`.
2. **Marine filter**: `df['Beach Type Description'] == 'Marine'` (Beach Type NOT
   lowercased; compared to capital-M `'Marine'`). No explicit Year filter (file
   is already 2023).
3. **EJ source**: load `environmental-justice-populations.csv`; lowercase
   `Municipality`.
4. **EJ groups**: `ej_ge_50 = Percent of population in EJ BGs > 50`;
   `ej_le_25 = Percent of population in EJ BGs < 25`.
5. **Join key**: inner-merge marine on `Community` == EJ `Municipality`,
   separately per group.
6. **Aggregation grain**: per group, `exceedance_rate = len(Violation=='yes') /
   len(group) * 100` — **a PERCENTAGE (×100)**.
7. **Final compute + format**: `rate_ge_50 - rate_le_25`, printed to 2 dp →
   **10.87 percentage points**.

The load-time key detail: the underlying quantity is
`high_rate - low_rate = 0.1512360639844886 - 0.042488619119878605 =
0.10874744486461`. Gold multiplies by 100 → **10.87**. Reporting the same
quantity as a bare fraction to 2 dp → **0.11**.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (ANCHOR — FAIL, answer 0.11)

| step | action | semantics | matches gold? |
|---|---|---|---|
| 0 | load `ej_populations`, `water_tests_2023` | correct two sources | ✓ (plan 1,3) |
| 1a | `marine_2023` = filter Year==2023 & Beach Type=='Marine' | 8491 rows — correct marine/2023 filter | ✓ (plan 2) |
| 1b | `ej_groups` = keep `Percent of population in EJ BGs`, tag gt50 (>50) / lt25 (<25), drop middle → 139 rows | correct EJ grouping | ✓ (plan 4) |
| 1c | `exceedance_diff`: inner merge on `Community`==`Municipality`; `is_exceed = Violation.lower()=='yes'`; `rates = groupby(ej_group).mean()`; `diff = rates['gt50'] - rates['lt25']`; **`round(diff, 2)`** | computes 0.10874744 **as a fraction, NO ×100** → **0.11** | ✗ (plan 6/7: missing ×100) |
| 2 | Final Answer: 0.11 | — | ✗ |

**First divergence: step 1c (final compute op `exceedance_diff`).** Every
upstream semantic is correct and the underlying ratio is bit-identical to the
winners. The sole shortfall vs gold plan item 6/7: the anchor expresses the
exceedance-rate difference as a **proportion (0.1087 → 0.11)** rather than a
**percentage (×100 → 10.87)**.

Evidence quote (rendered before step 2):
`[exceedance_diff] Output 1x1: ... | diff | 0  0.11` — i.e. the code returned
`round(0.10874744, 2)`.

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (C1 — PASS, answer 10.87)

| step | action | semantics | matches gold? |
|---|---|---|---|
| 0 | load `water_2023`, `ej_pops` | correct | ✓ |
| 1 | `marine_2023`: Year==2023 & Beach Type lower=='marine'; add `Violation_flag` (yes→1) | 8491 rows | ✓ (plan 2) |
| 2 | `marine_with_ej`: strip `Community`, inner-merge on `Community`; rename EJ pct → `ej_pop_pct` | 7118 rows | ✓ (plan 5) |
| 3 | `exceedance_rate_diff`: `high=ej>50`, `low=ej<25`; `high_rate = mean*100`, `low_rate = mean*100`; **diff in pct points** | **×100** → 10.874744486461 | ✓ (plan 6/7) |
| 4 | Final Answer: 10.87 | — | ✓ |

**No divergence from gold plan.** Evidence quote (rendered before step 4):
`[exceedance_rate_diff] ... high_ej_exceed_rate_pct 15.12360639844886  low ...
4.24886191198786  diff_pct_points 10.874744486461001`.

## Walk: DataflowSystemGPT52DeltaStats3kD2 (C2 — PASS, answer 10.87)

| step | action | semantics | matches gold? |
|---|---|---|---|
| 0 | load `water_body_testing_2023`, `ej_populations` | correct | ✓ |
| 1 | `marine_2023`: Beach Type lower=='marine' & Year==2023; keep Community/Code/Violation; `violation_flag` (yes→1) | 8491 rows | ✓ (plan 2) |
| 2 | `marine_with_ej`: lowercase both keys (`Community_norm`,`Municipality_norm`), inner-merge; rename → `ej_pop_pct` | 7118 rows | ✓ (plan 5) |
| 3 | `ej_exceedance_rate_diff`: `high=ej>50`,`low=ej<25`; `(high_rate-low_rate)*100`; `round(...,2)` | **×100** → 10.87 | ✓ (plan 6/7) |
| 4 | Final Answer: 10.87 | — | ✓ |

**No divergence from gold plan.** Evidence quote (rendered before step 4):
`[ej_exceedance_rate_diff] ... difference_exceedance_rate_pct_points | 0  10.87`.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (C3 — PASS, answer 10.87)

| step | action | semantics | matches gold? |
|---|---|---|---|
| 0 | load `water_2023`, `ej_pops` | correct | ✓ |
| 1 | `marine_2023`: Year==2023 & Beach Type lower=='marine'; add `Community_norm`, `Violation_bool` | 8491 rows | ✓ (plan 2) |
| 2 | `ej_clean`: `Community_norm` = lower(Municipality); `ej_pop_pct` numeric | 187 rows | ✓ (plan 3/4 prep) |
| 3 | `marine_with_ej`: **left**-join on `Community_norm` | 8491 rows (nulls where no EJ match) | ✓ (grain-equiv: nulls dropped by >50/<25 filters) |
| 4 | `ej_exceedance_diff`: `high=ej>50`,`low=ej<25`; `(high_rate-low_rate)*100`; also reports high_n/low_n | **×100** → 10.874744486461; high_n=2063, low_n=3295 | ✓ (plan 6/7) |
| 5 | Final Answer: 10.87 | — | ✓ |

**No divergence from gold plan.** Evidence quote (rendered before step 5):
`[ej_exceedance_diff] ... high_rate 0.1512360639844886  low_rate
0.042488619119878605  diff_pct_points 10.874744486461  high_n 2063  low_n 3295`.

## Anchor divergence vs the three winners at the SAME decision

The decision the anchor gets "wrong" is the **final-compute unit convention**
(fraction vs percentage), NOT any data/evidence-driven step:

- Upstream, all four arms are semantically identical: same two sources, same
  `Year==2023 & Beach Type=='Marine'` filter (all four → **8491** marine rows),
  same `>50` / `<25` split on `Percent of population in EJ BGs`, same
  Community↔Municipality inner/left join, same `Violation=='yes'` rate.
- All four compute the **bit-identical** underlying quantity
  `0.10874744486461` (verified: `round(0.10874744486461, 2)=0.11`;
  `round(0.10874744486461*100, 2)=10.87`).
- The three winners' final op multiplies by 100 (`*100` / `diff_pct_points`);
  the anchor's final op returns the raw fraction (`round(diff,2)`, no `*100`).

Rendered evidence at the anchor's divergence step was **identical in kind** to
the winners': the same `Percent of population in EJ BGs` column and the same
`Violation` field were visible to every arm. **Nothing rendered told the
winners to ×100 and the anchor not to.** The ×100 is a free formatting choice
made inside the agent's final-op plan, upstream of any differential rendered
observation between the arms.

## Pair verdicts

**C1 Delta5k > Delta3k — CHRONIC/VARIANCE.** The anchor computed the identical
0.10874744; it lost only by omitting the ×100 percentage convention (0.11 vs
10.87). No differential rendered evidence drove the split. Not attributable to
the 5k context lever.

**C2 Stats3kD2 > Delta3k — CHRONIC/VARIANCE.** Same mechanism. The stats block
played no role — the divergence is the percentage-vs-fraction convention at the
final op, not an evidence gap the stats lever could close.

**C3 Latest3k > Delta3k — CHRONIC/VARIANCE.** Same mechanism. Latest's fuller
history is irrelevant; the anchor's data pipeline was correct and its miss is
the ×100 convention.

**On the triple convergence:** this is NOT strong dual/triple-lever evidence of
a genuine evidence gap. It is one coin landing the same way three times — all
three rays happened to express "rate" as a percentage (the more common
convention, so agreement by chance is likely), while the anchor happened to
report the bare proportion. The question ("difference in ... rates ... to 2
decimal places") does not disambiguate percent vs fraction; gold chose ×100.
environment-hard-7 is on `chronic_flippers.json`, and this is precisely a
chronic roll on final-output units. Per the SKILL "reject method-choice
divergence that predates the arms' first rendered difference," the ×100 choice
is upstream of any rendered divergence — so all three flips are variance, none
attributable to a context lever.

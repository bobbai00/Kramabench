# biomedical-easy-2 — semantic trace walk (flip attribution)

## Task + gold answer

**Question:** What is the average age of patients with serous tumor samples analyzed in the study?
**Gold answer:** `68.5`

Arms compared (all `code` mode, 3k char limit, `max_operator_edits:0`):

| Arm | role | context_mode | column_stats | data_level | answer | verdict |
|---|---|---|---|---|---|---|
| DataflowSystemGPT52DeltaStats3kD2 | WINNER (C2) | delta | **true** | **2** | 68.5 | PASS |
| DataflowSystemGPT52Latest3kSchemaOnly | WINNER (C3) | **latest** | false | 1 | 68.5 | PASS |
| DataflowSystemGPT52Delta3kSchemaOnly | loser | delta | false | 1 | 68.1 | FAIL |

Lever map (single-knob vs the loser):
- **C2 = loser + `column_stats:true` (+ `data_level:2`)** → the STATS lever.
- **C3 = loser + `context_mode:latest`** → the MODE lever.

`biomedical-easy-2` **is present in `judgment_runs/levers_report/chronic_flippers.json`** — a known within-noise task (flips pass/fail under identical config).

## Gold semantic plan

From `solutions/biomedical/biomedical-easy-2.py`:

1. **Load** `1-s2.0-S0092867420301070-mmc1.xlsx`, default (only) sheet, default header.
2. **Filter** `Case_excluded == 'No'` (drop excluded cases).
3. **Filter** `Histologic_type in ['Endometrioid','Serous']` (tumor cases).
4. **Subset** to `Histologic_type == 'Serous'`.
5. **Compute** `Age.mean()` over that subset → `68.5`.

Answer-relevant path = {load} → {exclude Case_excluded} → {serous subset} → {mean Age}. Plan item 3 is a no-op for the serous mean (subsumed by the serous subset). The load-bearing filter that separates `68.5` from `68.1` is **plan item 2 (Case_excluded)**: dropping the serous case(s) whose `Case_excluded == 'Yes'` shifts the mean from `68.1` to `68.5`.

## Walk: DataflowSystemGPT52DeltaStats3kD2 (C2, WINNER — 68.5, PASS)

| step | op | semantics | plan-item |
|---|---|---|---|
| 0 | xlsx_raw | `read_excel(sheet_name=None)` → dict; render errors (KeyError on tuple) | probe, recovered |
| 1 | xlsx_raw | list sheet names → `UCEC_CPTAC3_meta_table_V2.1` | pre-load probe |
| 2 | meta | `read_excel(sheet_name='UCEC_CPTAC3_meta_table_V2.1')` → 153×179 | item 1 (load) |
| 3 | serous_age | `included = ~(Case_excluded.lower=='yes')`; `serous = Histologic_type.lower=='serous'`; `mean(Age[serous & included])` → **68.5** | items 2 ✓, 4 ✓, 5 ✓ |
| 4 | (text) | Final Answer: **68.5** | — |

**First divergence:** none on the answer-relevant path. The step-0 `sheet_name=None` render error is a recovered method probe, not a semantic miss. Matches gold.

**Evidence quote (rendered stats block, step-3 context):**
`"Case_excluded" (str): null=0, distinct=2, top_5={"No"=144, "Yes"=9}, duplicate_values=151`
`"Histologic_type" (str): null=49, distinct=4, top_5={"Endometrioid"=86, "Serous"=14, ...}`
C2's aggregation prose echoes it: "...(Case_excluded != 'Yes') with Histologic_type == 'Serous', then compute mean of numeric Age." The distribution makes the 9 excluded rows explicit.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (C3, WINNER — 68.5, PASS)

| step | op | semantics | plan-item |
|---|---|---|---|
| 0 | xlsx | `read_excel(sheet_name=None)` → dict; render errors | probe, recovered |
| 1 | xlsx | list sheet names → one sheet | pre-load probe |
| 2 | meta | `read_excel(sheet_name='UCEC_CPTAC3_meta_table_V2.1')` → 153×179 | item 1 (load) |
| 3 | serous_age | `df = df[Case_excluded.upper() not in {'YES','Y'}]`; `ser = Histologic_type.contains('serous')`; `mean(Age[ser])` → **68.5** | items 2 ✓, 4 ✓, 5 ✓ |
| 4 | (text) | Final Answer: **68.5** | — |

**First divergence:** none on the answer-relevant path. Matches gold.

**Evidence quote (step-3 context, schema-only — NO stats block):**
schema list `... Case_excluded (str), ...` and TSV header `... Case_excluded ...`; op summary "exclude cases with Case_excluded=YES, filter Histologic_type containing 'serous', and compute mean of Age as avg_age." Critically, C3's rendered `Case_excluded` evidence is **byte-identical to the loser's** (column name + type only; `grep -c top_5` = 0 for both). C3 had nothing about `Case_excluded` that the loser lacked.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (loser — 68.1, FAIL)

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | xlsx_preview | list sheet names → `UCEC_CPTAC3_meta_table_V2.1` | pre-load probe |
| 1 | meta | `read_excel(sheet_name='UCEC_CPTAC3_meta_table_V2.1')` → 153×179 | item 1 (load) ✓ |
| 2 | serous_avg_age | resolve `Histologic_type`/`Age` cols; `serous = Histologic_type.contains('serous')`; `round(mean(Age[serous]),1)` → **68.1** | items 4 ✓, 5 ✓; **DIVERGES: plan item 2 (Case_excluded) never applied** |
| 3 | (text) | Final Answer: **68.1** | — |

**First divergence:** step 2 (`serous_avg_age`). It averages `Age` over all serous rows without the `Case_excluded == 'No'` filter → `68.1` instead of `68.5`. Its op summary confirms the omission: "...`Histologic_type` contains 'serous' (case-insensitive) and compute the mean of `Age`, returning it rounded to 1 decimal place" — no mention of exclusion. The loser also took a shorter route (skipped the `sheet_name=None` probe both winners tripped on), reaching the aggregation in fewer steps.

**Rendered evidence at the divergence step (step 2 context):** schema-only, same class as C3 — `Case_excluded (str)` appears as a column name in the schema list and TSV header, but there is **no value-distribution stats block** (`grep -c top_5` = 0). The loser saw the *name* `Case_excluded` (a strong hint) but had no rendered distribution telling it that 9 rows are excluded.

## Pair verdicts

**What each winner had that the loser lacked, at the loser's divergence step (item 2):**
- vs **C2**: the STATS block `Case_excluded ... top_5={"No"=144,"Yes"=9}` — genuinely differential rendered evidence (loser is schema-only).
- vs **C3**: *nothing.* C3's `Case_excluded` render is byte-identical to the loser's (column name only, no stats). Only the delta→latest rendering mode differs, and at the aggregation step both had the meta observation equally fresh.

### C3 Latest3k > Delta3k → **CHRONIC-VARIANCE**
C3 and the loser rendered identical schema-only evidence for `Case_excluded` (column name, zero stats lines). The sole config delta is `context_mode` (latest vs delta), which surfaced no additional corrective signal about the exclusion. C3's correct filter vs the loser's omission is a reasoning coin-flip on shared evidence — and `biomedical-easy-2` is a confirmed chronic flipper. No rendered-evidence difference to attribute. Rejected as method/reasoning variance.

### C2 Stats3kD2 > Delta3k → **CHRONIC-VARIANCE** (with the one real differential signal noted)
C2 uniquely rendered the `Case_excluded` distribution `{"No"=144,"Yes"=9}`, which plausibly cued its exclusion filter — this is the only genuinely differential evidence in the whole comparison. But it fails the accept bar: (1) `biomedical-easy-2` is in `chronic_flippers.json`, so the outcome sits inside the noise floor; (2) C3 reaches the identical correct answer from schema-only evidence *identical to the loser's*, proving the stats block is **sufficient-but-not-necessary** — the correct filter is reachable without it, so the loser's omission is demonstrably stochastic (its schema-only sibling nailed it). On a single chronic trace we cannot separate "stats caused the filter" from "C2 would have filtered anyway, like C3 did." Attributing would replicate the aggregate-blindness / single-flip-on-a-chronic-task trap.

### Dual-lever convergence? **NO.**
The interesting angle does not hold. Dual-lever convergence requires both winners to independently surface the **same** corrective evidence. Here they do not: C2's corrective signal is the stats distribution; C3 has no `Case_excluded` signal beyond the column name the loser also had. Because the schema-only winner (C3) reproduces `68.5` on the loser's exact evidence, the loser's miss is stochastic — the textbook chronic pattern, not convergence. Consistent with the levers-report macro result (27 flips → 1 attributed = legal-hard-15).

Cost footnote (both winners passed; not the question here): C2 `$0.0261`, C3 `$0.0188`, loser `$0.0179` — the winners' extra spend is the recovered `sheet_name=None` probe, not the filter.

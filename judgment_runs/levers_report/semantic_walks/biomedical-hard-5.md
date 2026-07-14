# biomedical-hard-5 — semantic trace walk

Pair judged: **C3 Delta3k > Latest3k** (chronic\* — default CHRONIC-VARIANCE unless accept rules met).
- WINNER: `DataflowSystemGPT52Delta3kSchemaOnly` (C3) — PASS, answer 2.6563
- LOSER: `DataflowSystemGPT52Latest3kSchemaOnly` (C3) — FAIL, answer 2.4241

One-knob check: the meaningful diff is **`context_mode` delta→latest**. Both arms have `enable_inspect_tool` and `enable_render_prefs` at the default (Delta omits them = None, Latest writes them explicitly False — inert). Same 3k budget, schema-only. biomedical-hard-5 IS in the 23-task chronic-flipper set (confirmed against `chronic_flippers.json`). Stats: Delta 7 steps / $0.068, Latest 6 steps / $0.031 (Latest cheaper via a one-shot concat recovery — see below).

## Task + gold answer

- Question: "What is the median number of variants per Mbp for the serous tumor samples in the study? Round to 4 decimal places."
- Files: `1-s2.0-S0092867420301070-mmc1.xlsx` (clinical meta), `1-s2.0-S0092867420301070-mmc7.xlsx` (TMB / APM, multi-sheet)
- Gold answer: **2.6563** (`numeric`, 4 dp)

## Gold semantic plan

From `solutions/biomedical/biomedical-hard-5.py`:

1. **Load mmc1** default sheet (`UCEC_CPTAC3_meta_table_V2.1`, 153×179) → `clinical_df`.
2. **Filter cohort** on mmc1: `Case_excluded == 'No'` **AND** `Histologic_type ∈ {Endometrioid, Serous}`.
3. **Load mmc7** sheet **`B-APM subtypes`** (95×34) → `tmb_df`. This sheet carries `idx` + `Log2_variant_per_Mbp`.
   - (The `APP_Z_score idxmin` / `Age` lines in the .py are dead distractor code — not used in the returned value.)
4. **Restrict to serous**: `serous_cases = case_df[Histologic_type=='Serous']['idx'].tolist()` → **12 idx** (post Case_excluded).
5. **Key / join**: `tmb_df = tmb_df[tmb_df['idx'].isin(serous_cases)]` — inner-restrict mmc7 B-APM to those 12 idx.
6. **Compute**: `vpm = 2 ** tmb_df['Log2_variant_per_Mbp']`; **`np.median(vpm)`** → **2.6563**.

Two facts I verified in the data that decide this task:
- **mmc1 meta ALSO contains a `Log2_variant_per_Mbp` column**, and its values are **byte-identical** to mmc7 B-APM for every shared idx. So "which table's column" is a **red herring** — the value column is the same. What matters is the **row set**.
- The serous cohort has 14 rows if you filter only `Histologic_type≈serous`; the 2 extras are the **excluded** cases **S043 (Case_excluded='Yes', Log2=0.58)** and **S104 (Case_excluded='Yes', Log2=NaN)**. Neither excluded case appears in the B-APM subtypes sheet. So an inner join to B-APM drops both → 12; an explicit `Case_excluded=='No'` also drops both → 12. Miss both guards and S043's low 0.58 drags the median down (S104's NaN is skipped by pandas `.median()`).

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (WINNER, PASS, answer 2.6563)

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | mmc1, mmc7 load | `read_excel(sheet_name=None)` (dict) on both | plan 1/3; both error (dict return unsupported) |
| 1 | mmc1, mmc7 modify | `pd.ExcelFile(...).sheet_names` → sheet-name tables | **enumerate-sheets recovery**: reveals mmc7 = {README, A-predicted neoantigen, **B-APM subtypes**}, mmc1 = {UCEC meta} |
| 2 | mmc1_meta / mmc7_apm load; serous_samples; serous_variants_per_mbp; median | loads meta sheet + **`sheet_name='B-APM subtypes'`**; heuristic serous filter; attempts auto-key merge | plan 1/3 correct-sheet; merge **fails** (no common column) |
| 3 | serous_samples process | `Histologic_type contains 'serous'` → 14 rows, keeps `idx`→sample_id, participant_id, histology | plan 2 (partial — no `Case_excluded` filter, 14 not 12) |
| 4 | serous_variants_per_mbp process | `left.merge(right, left_on='sample_id', right_on='idx', how='inner')` → **12 rows**; `2**Log2_variant_per_Mbp` from mmc7 | **matches plan 4+5**: inner join to B-APM drops excluded S043/S104 → correct 12-cohort |
| — | median_serous_variants_per_mbp | `.median()` = 2.656346… | **matches plan 6** |
| 5 | final | "Final Answer: 2.6563" | correct |

**First divergence from gold: none material.** Delta never writes the `Case_excluded=='No'` filter (its serous_samples is 14), but the **inner join to `B-APM subtypes` on idx** implicitly enforces the analysis cohort — B-APM contains only non-excluded cases — landing exactly on gold's 12. It reads `Log2_variant_per_Mbp` from mmc7 (identical values to gold).

Evidence at the join step (step 4 context — the `mmc7_apm` schema rendered before Delta wrote the merge):
```
[mmc7_apm] Output 95x34: | idx  Log2_variant_per_Mbp  mutation_classification  APP_Z_score  ...
[serous_samples] Output 14x3: | sample_id  participant_id  histology | 0  S006  C3L-00098  Serous
```
B-APM subtypes was on the table as a **clean, joinable 95×34 frame with `idx` + `Log2_variant_per_Mbp` in the first two columns**. Delta joined the serous filter to it — the gold structure.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (LOSER, FAIL, answer 2.4241)

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | mmc7, mmc1 load | `read_excel(sheet_name=None)` (dict) on both | plan 1/3; both error (identical KeyError to Delta) |
| 1 | mmc7, mmc1 modify | **concat ALL sheets** with a `__sheet__` col | **concat recovery**: mmc7 → 705×60 blob (README+neoantigen+B-APM jammed together); mmc1 → 153×180. **B-APM never isolated.** DIVERGES from plan 3 |
| 2 | serous_mmc1 process | filter mmc1 where `Histologic_type contains 'serous'` → **14 rows**, 180 cols | plan 2 partial (no `Case_excluded` filter) |
| 3 | serous_median_variants_per_mbp process | `2 ** serous_mmc1['Log2_variant_per_Mbp']`; `.median()`; round 4 | **DIVERGES, plan 4/5**: computes over mmc1's own Log2 for all **14** serous, **no join to B-APM, no `Case_excluded` guard** → includes excluded **S043 (0.58)** |
| — | (result) | 2.4241 | wrong |
| 4 | final | "Final Answer: 2.4241" | wrong |

**First divergence from gold: step 1** — the **concat-all-sheets recovery**. By collapsing mmc7 into a 705-row blob it never isolated `B-APM subtypes` as a joinable table. That set up **step 3** as the payload error: it found `Log2_variant_per_Mbp` sitting directly in the mmc1 meta (mmc1 has its own copy) and computed the median in place over 14 serous rows, so it never applied the cohort restriction that a B-APM join (or a `Case_excluded=='No'` filter) would have given. Pandas `.median()` skips S104's NaN but keeps **S043 (Log2=0.58, excluded)**; that one low value pulls 2.6563 → 2.4241.

Evidence at the divergence step (step 3 context — everything rendered before Latest wrote the median):
```
[serous_mmc1] Output 14x180: From `mmc1`, filter rows where `Histologic_type` contains 'serous' ...
  | idx  Proteomics_Participant_ID  Case_excluded  ...  Histologic_type  ...
```
`Case_excluded` was **visible in the very schema Latest filtered on**, and the concatenated `mmc7` blob (`705x60`, columns `Sheet Description __sheet__ sample Variant_ID ...`) was too tangled to invite a clean B-APM join — so Latest took the in-place mmc1 column and skipped both cohort guards.

## Pair verdicts

**C3 Delta3k > Latest3k → CHRONIC-VARIANCE.**

Per-arm divergence table:

| arm | first divergence step | action | gold-plan item fallen short |
|---|---|---|---|
| Delta3kSchemaOnly (win) | — | — | none material (inner join to B-APM enforces the 12-cohort) |
| Latest3kSchemaOnly (loss) | step 1 | concat-all-sheets recovery → step-3 in-place median over 14 (incl. excluded S043) | plan 3 (isolate B-APM) → plan 4/5 (serous∩B-APM = 12 cohort) |

Why not ATTRIBUTED — the divergence **predates any lever-induced rendered difference** (the skill's explicit REJECT condition):
1. **The hinge is Step 1, the recovery from an identical Step-0 error.** Both arms hit the *same* `sheet_name=None` dict KeyError and, going into Step 1, had **byte-identical context** (one identical failed probe). The `context_mode` lever governs how *prior* steps render; with a single identical step in history there is **no differential rendering** at the decision point. The concat-vs-enumerate recovery choice is therefore a **model method choice**, not a lever effect.
2. **The value column is identical across tables**, so this is not an evidence-starvation story about missing a column — Latest had `Log2_variant_per_Mbp` right in front of it. The error is a *cohort/row-set* miss that flows from the Step-1 method choice (concat → no clean B-APM → no join → no cohort restriction), plus the absent `Case_excluded` guard that both arms omitted but only Delta's join covered for.
3. **Chronic gate.** biomedical-hard-5 is in the 23-task chronic-flipper set; the enumerate-vs-concat recovery on a dict-load error is exactly the kind of choice that coin-flips across reruns. No dual-lever convergence, no rerun evidence that Latest is evidence-starved on this decision.

Equivalent framing: **REJECTED-method-choice** (the divergence is a recovery-strategy method choice at Step 1, before compaction produced any differential context). Riding the chronic gate, the reported label is **CHRONIC-VARIANCE**. Net: Delta got the right answer because its enumerate-sheets recovery happened to reconstruct the gold's mmc7-B-APM join structure — a plan-shaped path that the lever did not cause.

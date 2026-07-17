# biomedical-hard-5 — deep dive (PROBE-STAR vintage; TWO pairs, opposite directions)

One task, **two single-knob pairs that flip in opposite directions** — the
first ATTRIBUTED, the second CHRONIC. Winners → **2.6563 ✓**, losers →
**2.4241 ✗**. Chronic flipper (`chronic_flippers.json`). GPT-5.2 raw-probe
prompt. Traces via `python3 scripts/extract_walk.py --sut <ARM> --task
biomedical-hard-5`.

| pair | knob | arm | role | answer | cost / steps | verdict |
|---|---|---|---|---|---|---|
| **1 (C2p)** | `column_stats` F→T (`data_level` 1→2) | `DeltaStats1kD2ProbePrompt` | **WIN** | **2.6563** | $0.0292 / 6 | **ATTRIBUTED** |
| 1 (C2p) | " | `Delta1kSchemaOnlyProbePrompt` | lose | 2.4241 | $0.0216 / 5 | (to stats knob) |
| **2 (C3p)** | `context_mode` latest vs delta | `Latest5kSchemaOnlyProbePrompt` | **WIN** | **2.6563** | $0.0309 / 7 | **CHRONIC** |
| 2 (C3p) | " | `Delta5kSchemaOnlyProbePrompt` | lose | 2.4241 | $0.0259 / 6 | (mirror-coin) |

One-knob validity gate passes for both (config diffs confirmed: Pair 1 =
`column_stats`+`data_level`; Pair 2 = `context_mode` only, both schema-only).

## Task

Q: "What is the median number of variants per Mbp for the serous tumor samples
in the study? Round the result to 4 decimal places."

Gold answer: **2.6563** (`numeric_exact` — 2.4241 fails hard).

D: two Excel files from the CPTAC3 UCEC (endometrial cancer) proteogenomics
study.

- `data/biomedical/input/1-s2.0-S0092867420301070-mmc1.xlsx` — the meta table,
  one sheet `UCEC_CPTAC3_meta_table_V2.1`, **153 rows × 179 cols** (default
  header). The 179-col width forces a "showing 50 of 179 cols" truncation in the
  rendered schema. It carries **its own `Log2_variant_per_Mbp` column** — the
  route every arm here actually used. Real serous rows (relevant cols;
  `vpm = 2**Log2_variant_per_Mbp`):

  ```
  idx   Participant_ID          Case_excluded  Histologic_type  Age    Log2_variant_per_Mbp  vpm
  S006  C3L-00098               No             Serous           63.0   1.6347                3.1053
  S016  C3L-00358               No             Serous           90.0   2.0666                4.1889
  S041  C3L-00963               No             Serous           59.0   1.2775                2.4241
  S043  C3L-01247               Yes            Serous           63.0   0.5805                1.4954   <- EXCLUDED, lowest vpm
  S097  C3N-01349               No             Serous           77.0   2.0666                4.1889
  S103  C3N-01825               No             Serous           70.0   1.0591                2.0836
  S104  C3N-01825_replication   Yes            Serous           NaN    NaN                   NaN      <- EXCLUDED, inert (NaN)
  ```

  Column semantics / quirks:
  - `Case_excluded` (str): study eligibility/QC flag, `"No"` = kept / `"Yes"` =
    dropped. Whole-table distribution **`No=144, Yes=9`**.
  - `Histologic_type` (str): tumor histology — `Endometrioid=86, Serous=14,
    Carcinosarcoma=3, Clear cell=1`, plus **49 NaN**.
  - `Log2_variant_per_Mbp` (numeric): log2 tumor mutational burden per Mbp;
    answer = median of `2**` this over the serous samples.
  - `idx` (str) sample id S001..S153; one `..._replication` row (S104).

- `data/biomedical/input/1-s2.0-S0092867420301070-mmc7.xlsx` — the gold's TMB
  source; sheet **`B-APM subtypes`**, **95 rows × 34 cols**, also carrying `idx`
  + `Log2_variant_per_Mbp`. **Quirk that makes the filter the sole
  discriminator:** this 95-row sheet **does not contain S043 or S104 at all** —
  so gold's mmc7 route drops the excluded serous cases *for free*, while the
  arms' mmc1 route (S043 present, vpm=1.4954) must apply `Case_excluded=='No'`
  explicitly.

**The quirk that decides the answer:** of the 14 serous rows, **12 are
`Case_excluded=='No'`, 2 are `'Yes'`** (S043 vpm=1.4954, S104 vpm=NaN). S104 is
NaN so inert; **the entire 2.6563-vs-2.4241 gap is the single row S043.**
Median over the 12 kept = **2.6563** (gold); keep S043 → 13 valid rows →
**2.4241** (median commutes with `2**x`, so the mmc1 route is numerically equal
to gold's mmc7 route). This is the task's stated "keeping excluded S043"
failure, reached via a missing filter.

## Solution

From `solutions/biomedical/biomedical-hard-5.py`, as an operator graph:

```
read_excel(mmc1.xlsx, default sheet+header)                    # load: 153x179
        │
   filter(Case_excluded == 'No')                               # drop 9 excluded  <-- LOAD-BEARING
        │
   filter(Histologic_type in {Endometrioid, Serous})           # tumor cases
        │
   serous_idx = subset(Histologic_type == 'Serous')['idx']     # 12 kept serous ids
        │
read_excel(mmc7.xlsx, sheet='B-APM subtypes') ─ filter(idx ∈ serous_idx)   # 12 rows (S043/S104 absent anyway)
        │
   vpm = 2 ** Log2_variant_per_Mbp  →  median  ──────────────► 2.6563
```

Node annotations: load = default read on both files; the decisive predicate is
**`Case_excluded=='No'`** (plan item 2); grain = serous kept cases; final
compute = `median(2**Log2_variant_per_Mbp)`, 4 dp. The arms substitute mmc1's
own `Log2_variant_per_Mbp` for the mmc7 join (numerically identical), so across
**every arm here the sole discriminator is the `Case_excluded` filter**.

## What the losers (mode X) do — both omit the filter → 2.4241, FAIL

Both losers load mmc1, subset `Histologic_type=='serous'`, and take
`median(2**Log2_variant_per_Mbp)` **without any `Case_excluded` guard** — S043
retained → 13 valid rows → 2.4241.

- **Pair 1 loser `Delta1kSchemaOnly`** — `serous_variants_median`:
  `serous = df[df['Histologic_type']...eq('serous')]`;
  `vals = to_numeric(serous['Log2_variant_per_Mbp']).dropna();
  2**vals.median()` → **2.4241**. --> **Divergence: plan item 2
  (`Case_excluded=='No'`) never applied.**
- **Pair 2 loser `Delta5kSchemaOnly`** — `serous_variant_per_mbp`:
  hunts a variant column, `ser = df[Histologic_type=='serous']`, emits
  `variant_per_mbp = 2**Log2_variant_per_Mbp`; then `median_serous_variant_per_mbp`
  → **2.4241**. --> **Same divergence: no `Case_excluded` filter.**

## What the winners (mode Y) do — both apply the filter → 2.6563, PASS

Both winners drop the excluded case before the median.

- **Pair 1 winner `DeltaStats1kD2`** — `serous_variant_mbp`:
  `df = df[(df['Case_excluded'].str.lower()=='no') &
  (df['Histologic_type'].str.lower()=='serous')]`;
  `variant_per_mbp = 2**df['Log2_variant_per_Mbp']`; then
  `median_serous_variant_mbp` → **2.6563**. Applies plan item 2 ✓ (12 rows).
- **Pair 2 winner `Latest5kSchemaOnly`** (7 steps — an extra preview probe):
  `serous_variants_median`:
  `df = df[df["Case_excluded"]...!= "yes"]`; then
  `serous = df[Histologic_type=="serous"]`;
  `median(2**Log2_variant_per_Mbp)` → **2.6563**. Applies the exclusion ✓.

## Why the winners passed but the losers failed

### Pair 1 (C2p, stats > schema @1k): ATTRIBUTED to the stats knob

**The one knob IS exactly the rendered difference.** At the filter step, the
stats winner's context rendered the value distribution of `Case_excluded`:

```
- "Case_excluded" (str): null=0, distinct=2, top_5={"No"=144, "Yes"=9}, duplicate_values=151
```

The **`"Yes"=9`** count directly surfaces that excluded cases exist → the arm
wrote `Case_excluded.str.lower()=='no'`. The schema-only loser, at the identical
1k budget, rendered **only the column name and type** — `Schema (showing 50 of
179 cols): Case_excluded (str)`, with `grep -c top_5 = 0` over its whole final
context. **No signal that any case is excluded** → the filter is omitted → S043
retained → 2.4241.

Both accept clauses are met, and uniquely tied to the one knob:
- **winner-evidence-explains-action** — the rendered `top_5={No=144,Yes=9}`
  names the 9 excluded cases and is the visible cue for the `=='No'` filter;
- **loser-absence-explains-error** — the schema-only render contains no
  excluded-case signal at all, and the loser's code omits precisely that filter;
- **the one knob (`column_stats`) is the rendered difference** — the value
  distribution appears only because stats is on.

This is the **answer-relevant profile line** pattern of the **legal-hard-15
family** — a rendered value-distribution stat that names the exact rows the
answer must drop. It is the **second attributed stats flip ever** recorded
(the first being legal-hard-15's dual-lever convergence): stats helps **precisely
when the rendered stat is answer-relevant** — here it names the 9 excluded cases
— and is the mirror of wildfire-hard-12 (same `DeltaStats1kD2` arm), where the
1k stats block was univariate/answer-irrelevant and stats was neutral-to-harmful.

**Verdict Pair 1: ATTRIBUTED to `column_stats`.**

### Pair 2 (C3p, latest > delta @5k): CHRONIC — mirror-coin

**Both 5k arms are schema-only, and neither renders the `Case_excluded`
distribution** — `grep -c top_5 = 0` for *both* the Latest winner and the Delta
loser; neither context contains `"Yes"=9`. The Latest winner *self-generated*
`Case_excluded != "yes"` from **column-name reasoning** (plus a more elaborate
7-step exploration with an extra preview), while the Delta loser reasoned only
as far as `Histologic_type`. **No rendered difference explains the flip** — the
`context_mode` knob surfaces no corrective signal here (at a single-load task,
DELTA's "new/changed only" and LATEST's "full latest state" both reduce to the
same one meta table).

**Decisive corroboration — the mirror pair.** On the **same C3p knob**
(`5kSchemaOnly`, `context_mode` latest vs delta), **biomedical-easy-2 flips the
other way**: there **DELTA** self-supplied `Case_excluded!='yes'` from the column
name and **LATEST** omitted it (Delta 68.5 ✓ vs Latest 68.1 ✗). Same knob, same
self-supplied-filter mechanism, **opposite arm wins** → the `Case_excluded`
filter is a **coin-flip across `context_mode`** on these chronic tasks, not a
lever effect.

**Verdict Pair 2: CHRONIC / rejected method-choice (variance).**

## Per-arm divergence table

| pair | arm | filter written | rows | answer | vs gold |
|---|---|---|---|---|---|
| 1 | **DeltaStats1kD2 (win)** | `Case_excluded=='no' & Histologic_type=='serous'` | 12 | **2.6563** | ✓ excludes S043 |
| 1 | Delta1kSchemaOnly (lose) | `Histologic_type=='serous'` only | 13 | 2.4241 | ✗ keeps S043 |
| 2 | **Latest5kSchemaOnly (win)** | `Case_excluded!='yes'` then serous | 12 | **2.6563** | ✓ self-supplied filter |
| 2 | Delta5kSchemaOnly (lose) | `Histologic_type=='serous'` only | 13 | 2.4241 | ✗ keeps S043 |

**Labels: Pair 1 ATTRIBUTED (`column_stats` — the answer-relevant profile line,
legal-hard-15 family, second attributed stats flip ever) · Pair 2 CHRONIC
(mirror-coin vs biomedical-easy-2).** Cost footnote (not the accuracy claim):
winners spend more — Pair 1 $0.0292 vs $0.0216 (stats bytes + 1 step), Pair 2
$0.0309 vs $0.0259 (Latest's extra preview step) — a cost the ATTRIBUTED pair
earns and the CHRONIC pair does not.

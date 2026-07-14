# biomedical-easy-2 — deep dive (counter-intuitive: history-less Latest3k beat Delta3k)

Three `code`-mode arms, 3k char limit, `max_operator_edits:0`. Single-knob map vs the loser:

| Arm | role | context_mode | column_stats / data_level | answer | verdict |
|---|---|---|---|---|---|
| **Latest3kSchemaOnly** (mode Y) | WINNER | **latest** | false / 1 | **68.5** | PASS |
| DeltaStats3kD2 (also won) | WINNER | delta | **true / 2** | **68.5** | PASS |
| **Delta3kSchemaOnly** (mode X) | loser | delta | false / 1 | 68.1 | FAIL |

Y = loser + `context_mode:latest` (the MODE knob). Stats3kD2 = loser + `column_stats:true (+data_level:2)` (the STATS knob). `biomedical-easy-2` is in `chronic_flippers.json`.

## Task
Q: What is the average age of patients with serous tumor samples analyzed in the study?

D: `data/biomedical/input/1-s2.0-S0092867420301070-mmc1.xlsx` — supplementary table mmc1 of the CPTAC3 UCEC (endometrial cancer) proteogenomics study. One sheet `UCEC_CPTAC3_meta_table_V2.1`, **153 rows × 179 cols**, default header (row 0), no header-offset quirk. The width (179 cols) forces a "showing first 25 / last 25" truncation in the rendered table preview. Real sample rows (relevant cols):

```
 idx Proteomics_Participant_ID Case_excluded Histologic_type  Age
S001                 C3L-00006            No    Endometrioid 64.0
S004                 C3L-00084           Yes  Carcinosarcoma  NaN
S006                 C3L-00098            No          Serous 63.0
S043                 C3L-01247           Yes          Serous 63.0   <- excluded serous, Age present
S104     C3N-01825_replication           Yes          Serous  NaN   <- excluded serous, Age NaN (inert)
```

Column semantics:
- `Case_excluded` (str): study eligibility/QC flag, `"No"` = kept / `"Yes"` = dropped from the analysis. Distribution `No=144, Yes=9`.
- `Histologic_type` (str): tumor histology — `Endometrioid=86, Serous=14, Carcinosarcoma=3, Clear cell=1`, plus **49 NaN** (non-tumor / unlabeled rows).
- `Age` (numeric): patient age, years; contains NaNs.
- `idx` (str) sample id S001..S153; `Proteomics_Participant_ID` e.g. `C3L-00006`, incl. one `..._replication` row (S104).

Quirk that decides the answer: of the 14 serous rows, **12 are `Case_excluded=='No'` and 2 are `'Yes'`** (S043 Age 63.0, S104 Age NaN). Serous mean over the 12 kept = **68.5** (gold); over all 13 non-null serous ages (adds S043) = 68.077 → **68.1**. S104 is NaN so inert — the entire 68.5-vs-68.1 gap is the **single row S043**.

## Solution
From `solutions/biomedical/biomedical-easy-2.py`, as an operator graph:

```
read_excel(mmc1.xlsx, default sheet+header)          # load: 153x179
        │
        ▼
filter(Case_excluded == 'No')                        # drop 9 excluded cases  <-- LOAD-BEARING
        │
        ▼
filter(Histologic_type in {Endometrioid, Serous})    # tumor cases (no-op for the serous mean)
        │
        ▼
subset(Histologic_type == 'Serous')                  # grain = serous kept cases (n=12 non-null Age)
        │
        ▼
mean(Age) ─────────────────────────────────────────► 68.5
```

Answer-relevant path = {load} → {`Case_excluded=='No'`} → {serous subset} → {mean `Age`}. Plan item 3 (tumor filter) is subsumed by the serous subset. The only filter that separates 68.5 from 68.1 is **`Case_excluded=='No'`**.

## What Delta3kSchemaOnly does (mode X, loser — 68.1, FAIL)
- **step 0** `xlsx_preview`: `ExcelFile(...).sheet_names` → `UCEC_CPTAC3_meta_table_V2.1` (pre-load probe; goes straight to sheet-name listing, skipping the `sheet_name=None` misstep both winners hit).
- **step 1** `meta`: `read_excel(sheet_name='UCEC_CPTAC3_meta_table_V2.1')` → 153×179. Load, plan item 1 ✓.
- **--> step 2** `serous_avg_age`: resolves `Histologic_type`/`Age` case-insensitively, `serous_mask = Histologic_type.contains('serous')`, `round(mean(Age[serous_mask]),1)` → **68.1**. **Divergence: `Case_excluded=='No'` (plan item 2) is never applied.** It averages all 13 non-null serous ages including the excluded S043 (Age 63) → 68.1. Its own op summary confirms the omission: "…`Histologic_type` contains 'serous' … and compute the mean of `Age`, returning it rounded to 1 decimal place" — no exclusion mentioned.
- **step 3** Final Answer: **68.1**.

## What Latest3kSchemaOnly does (mode Y, winner — 68.5, PASS)
- **step 0** `xlsx`: `read_excel(sheet_name=None)` → dict; renders a `KeyError` (method probe, recovered).
- **step 1** `xlsx`: `ExcelFile(...).sheet_names` → one sheet (pre-load probe).
- **step 2** `meta`: `read_excel(sheet_name='UCEC_CPTAC3_meta_table_V2.1')` → 153×179. Load, plan item 1 ✓.
- **step 3** `serous_age`: `df = df[Case_excluded.strip().upper() not in {'YES','Y'}]`; `ser = Histologic_type.contains('serous')`; `mean(Age[ser])` → **68.5**. Applies the exclusion (item 2 ✓), serous subset (item 4 ✓), mean (item 5 ✓). No near-miss on the answer path; the only detour was the recovered step-0 `sheet_name=None` probe.
- **step 4** Final Answer: **68.5**.

Sibling winner **DeltaStats3kD2** (also 68.5, PASS) runs the same shape: `sheet_name=None` probe (recovered) → `meta` load → `serous_age` with `included = ~(Case_excluded=='yes')` & `serous`, `mean(Age[serous & included])` → 68.5. Its final prose explicitly cites the exclusion: "excluding cases where `Case_excluded == 'Yes'`".

## Why Y succeeded but X failed
**The decisive fact: at the divergence step, Latest3k's rendered evidence about `Case_excluded` was BYTE-IDENTICAL to the loser's.** Both arms are schema-only, and both rendered exactly:

```
Schema (showing 50 of 179 cols): Case_excluded (str)
```

Column name + type, nothing else — `grep -c top_5` = **0** for both. Latest3k saw *nothing* about `Case_excluded` that Delta3k lacked; both had the single `meta` observation equally fresh at the aggregation step, so the `delta`→`latest` mode knob surfaced **no corrective signal** here. Latest3k applying the exclusion while Delta3k omitted it is therefore a reasoning coin-flip on identical evidence — a **stochastic omission**, not a lever effect. Combined with the task's presence in `chronic_flippers.json`: **Latest3k > Delta3k = CHRONIC-VARIANCE**, unattributable to the mode knob.

**The one genuinely differential signal lives in Stats3kD2, not in the winner.** Stats3kD2 alone rendered the value distribution:

```
"Case_excluded" (str): null=0, distinct=2, top_5={"No"=144, "Yes"=9}, duplicate_values=151
("Histologic_type" (str): … top_5={"Endometrioid"=86, "Serous"=14, "Carcinosarcoma"=3, "Clear cell"=1})
```

Those 9 explicit `"Yes"` rows plausibly cued its exclusion filter — the only differential rendered evidence in the entire three-arm comparison. But it is **sufficient-but-not-necessary**: Latest3k reached the identical 68.5 from evidence byte-identical to the loser's, proving the exclusion is reachable *without* the stats block. On a single chronic trace we cannot separate "the distribution caused the filter" from "Stats3kD2 would have filtered anyway, as Latest3k did on nothing extra." Attributing it would replicate the single-flip-on-a-chronic-task trap. So **Stats3kD2 > Delta3k = CHRONIC-VARIANCE too — with the `Case_excluded` distribution FLAGGED FOR RE-TEST**: does rendering the value distribution reliably lift the exclusion-filter rate across reruns and other exclusion-gated tasks?

**Dual-lever convergence? No.** Convergence would require both winners to independently surface the *same* corrective evidence. They do not: Latest3k has nothing beyond the column name the loser also had; Stats3kD2 has the distribution. Because the schema-only winner reproduces 68.5 on the loser's exact evidence, the loser's miss is demonstrably stochastic — the textbook chronic pattern, consistent with the levers-report macro result (27 flips → 1 attributed).

**Label: CHRONIC (both pairs).** Cost footnote (both winners passed; not the question): Stats3kD2 `$0.0261`, Latest3k `$0.0188`, loser `$0.0179` — the winners' extra spend is the recovered `sheet_name=None` probe, not the filter.

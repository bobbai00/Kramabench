# Semantic walk: legal-hard-15

## Task + gold answer

**Question:** How many total Identity Theft reports were there in 2024 from cross-state Metropolitan Statistical Areas?

**Gold answer:** 243377 (single integer).

**Judge marks:** Delta5kSchemaOnly `success=1` (answer 243377, PASS), DeltaStats3kD2 `success=1` (answer 243377, PASS), Delta3kSchemaOnly `success=0` (answer 593524, FAIL). Task is **NOT** in `chronic_flippers.json`. Pairs per `pairs.json`: C1 `Delta5kSchemaOnly > Delta3kSchemaOnly`, C2 `DeltaStats3kD2 > Delta3kSchemaOnly` — the levers report's one ATTRIBUTED flip (REPORT.md §5).

**Config diffs (validity gate):** C1 — exactly one behavioral knob: `max_operator_result_char_limit: 5000` vs `3000` (both arms `column_stats: false, data_level: 1, context_mode: delta`, cell limit 3000). C2 — the stats lever bundle `column_stats: true` + `data_level: 2` vs `false`/`1` (same 3k char limits, same delta mode). Vintage: the loser's config carries 5 later-vintage fields (`enable_inspect_tool`, `enable_render_prefs`, `fold_resolved_revisions_config`, `frontier_decay_config`, `probe_retirement_config`), all `null`/`false` = disabled; Delta5k predates `static_compaction` (absent), Stats3kD2 has it `false`. No behavioral delta from any of these. Trace dates: Delta5k 07-06, Stats3kD2 07-08, loser 07-12 — the loser's live trace is its post-star-recovery rerun (`logs/star-recovery-20260712_101634`), and its rerun history is itself evidence (see loser walk).

## Gold semantic plan

Source: `solutions/legal/legal-hard-15.py`

| # | Plan item |
|---|---|
| G1 | Load every CSV in `data/legal/input/csn-data-book-2024-csv/CSVs/State MSA Identity Theft data/` (52 files, one per state + DC + PR) with `skiprows=2`, `.dropna()` per file — strips the title row and header/footer junk |
| G2 | Concat all state frames, ignore_index |
| G3 | Derive state token = text after the first comma of `Metropolitan Area` (e.g. `GA-AL`); `is_cross_state` = token contains `-`. No filter on "Metropolitan" vs "Micropolitan" — cross-state Micropolitan areas count |
| G4 | `# of Reports` → int (strip thousands commas) |
| G5 | **`overall_df.drop_duplicates()` — full-row dedup, effectively key `(Metropolitan Area, # of Reports)`.** The load-bearing step: the FTC data book lists every cross-state MSA once under EACH member state's file, with identical name and count |
| G6 | Sum `# of Reports` over `is_cross_state` rows → **243377** |

Data facts (verified against the CSVs with `.venv/bin/python`, exact repro): 52 files; naive concat = 764 rows of which **`raw.duplicated().sum()` = 359 (47%)**; after junk-row cleaning 452 named rows → 401 unique on `(name, reports)`; the 51 removed rows are ALL cross-state repeats (cross-state set: 94 rows → 43 unique MSAs). **Every one of the 43 cross-state MSAs appears exactly once per member state** — e.g. `Columbus, GA-AL` in `Alabama.csv` + `Georgia.csv`; `Washington-Arlington-Alexandria, DC-VA-MD-WV` 4 copies in `DistrictofColumbia/Maryland/Virginia/WestVirginia.csv`; zero exceptions. Answer arithmetic: gold pipeline with its `drop_duplicates` line deleted returns **593524 — exactly the loser's answer**; 593524 − 243377 = 350147 = the duplicated mass. The anchor's other recorded rerun answer 242682 = dedup done but scope over-trimmed to `Metropolitan Statistical Area` only, dropping the two cross-state Micropolitan rows (LaGrange GA-AL 453 + Lebanon-Claremont NH-VT 242 = 695 = 243377 − 242682) — also reproduced exactly. One decision decides the task: are the repeated MSA rows duplicates to drop, or distinct data?

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (WINNER C1 — PASS)

**Final answer:** 243377 — correct. 5 agent steps (6 counted), 39,716 input tokens, 22.5s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `msa_state_it_2024_load` | `glob` all 52 CSVs, `pd.read_csv` (default header = title row), concat → 764x2 | G1/G2 ✓ (junk rows still in-band; deferred to clean) |
| 1 | `msa_state_it_2024_clean` | rename cols; drop header/footer/annotation rows by exact-string blacklist; `to_numeric` on de-comma'd reports; keep notna → 452x2 | G4 ✓, G1-residue ✓ — no dedup yet (G5 not yet due) |
| 2 | `msa_state_it_2024_cross_state` | `is_cross_state` = regex `,\s*([A-Z]{2}(?:-[A-Z]{2})+)\s` on name notna; filter → **94x2 materialized intermediate** | G3 ✓ (includes Micropolitan) |
| 3 | `msa_state_it_2024_cross_state_total` | **`df.drop_duplicates(subset=['msa'])`** then `sum` → 1x1 = 243377 | **G5 ✓** (key `msa` ≡ gold's full-row key here — copies are identical rows) G6 ✓ |
| 4 | (text) | Final Answer: 243377 | ✓ |

**First semantic divergence:** none load-bearing (dedup placed post-filter instead of pre-sum on full rows — same result; exact repro: 94 → 43 → 243377).

**Rendered evidence at the dedup decision (step 3 input, DELTA block for `msa_state_it_2024_cross_state`, 3,471 chars, 50 of 94 rows visible):** the repetition is literally on screen — ~a dozen identical pairs inside the window, starting 9 rows apart:

```
0	Columbus, GA-AL Metropolitan Statistical Area	1302
1	LaGrange, GA-AL Micropolitan Statistical Area	453
...
9	Columbus, GA-AL Metropolitan Statistical Area	1302
10	LaGrange, GA-AL Micropolitan Statistical Area	453
...
12	Chicago-Naperville-Elgin, IL-IN Metropolitan Statistical Area	37486
...
16	Chicago-Naperville-Elgin, IL-IN Metropolitan Statistical Area	37486
```

plus `Washington-Arlington-Alexandria, DC-VA-MD-WV ... 19689` at rows 6, 82 AND 87, `Memphis, TN-MS-AR ... 5502` at 3/76, `Chattanooga, TN-GA ... 1135` at 8/73, `Sioux City, IA-NE-SD ... 132` at 22/71, `Kingsport-Bristol ... 342` at 75/80, `Winchester, VA-WV ... 191` at 83/90. The dedup edit lands in exactly the step after this render. Before it, no render showed a duplicate PAIR in-window: the step-1 raw render (2,243 chars, rows 0–18 + 746–763) shows `5 Columbus, GA-AL ... 1,302` once inside the AL block plus the repeated `758 Metropolitan Area | # of Reports` header; the step-2 clean render (3,439 chars, rows 0–26 + 426–451) shows each cross-state MSA once.

## Walk: DataflowSystemGPT52DeltaStats3kD2 (WINNER C2 — PASS)

**Final answer:** 243377 — correct. 5 agent steps (6 counted), 40,523 input tokens, 28.3s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `it_msa_2024_raw` | `glob` + concat (adds all-null `__source_file` col) → 764x3 | G1/G2 ✓ |
| 1 | `it_msa_2024_clean` | junk-row blacklist; de-comma + `to_numeric`; **`df.drop_duplicates(subset=['metropolitan_area','reports'])`** → 401x2 | G4 ✓, **G5 ✓ at the EARLIEST correct placement, with gold's effective key** |
| 2 | `it_msa_2024_cross_state` | extract `,\s*([^,]+)\s` token, split on `-`, `n_states = len(set(...))`, keep `n_states > 1` → 43x3 | G3 ✓ (includes Micropolitan) |
| 3 | `it_msa_2024_cross_state_total` | `sum` → 1x1 = 243377 | G6 ✓ |
| 4 | (text) | Final Answer: 243377 | ✓ |

**First semantic divergence:** none (exact repro: 401 → 43 → 243377).

**Rendered evidence at the dedup decision (step 1 input, DELTA block for `it_msa_2024_raw`, 1,722 chars):** the visible sample is 9 head + 9 tail rows — **row-for-row the same window the loser saw** (`0 NaN`, `1 Metropolitan Area | # of Reports`, `2 Anniston-Oxford` … `5 Columbus, GA-AL ... 1,302` … tail `755–763` footer junk), containing no duplicate pair. The ONLY duplication signal is the D2 profile appended to the block:

```
Column Schema and stats:
- "__source_file" (str): null=764, distinct=0
- "Unnamed: 1" (str): null=260, distinct=343, format=thousands, duplicate_values=161
- "Metropolitan Areas: Identity Theft Reports" (str): null=156, distinct=404, duplicate_values=204
Output Table profile:
- empty rows: 156 of 764 rows are entirely null
- empty columns: ["__source_file"] are entirely null
- duplicate rows: 359 of 764 (47%)
- headers: 1 of 3 columns are unnamed ("Unnamed: 1")
```

The `drop_duplicates` line is written in that very step, before any filtering — evidence and action in the same step, nothing else differing.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (loser in BOTH pairs — FAIL)

**Final answer:** 593524 — the no-dedup sum (= gold's own pipeline minus its `drop_duplicates` line). 4 agent steps (5 counted), 30,583 input tokens, 18.8s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `msa_it_2024_load` | `glob` + concat, fills `source_file` per row (`Alabama.csv` …) → 764x3 | G1/G2 ✓ |
| 1 | `msa_it_2024_clean` | junk-row blacklist; strict `^(\d+)$` numeric extract → 452x3 | G4 ✓ — **no dedup** (not yet fatal) |
| 2 | `cross_state_msa_total_2024` | regex `,\s*([A-Z]{2}(?:-[A-Z]{2})+)\b` filter **AND sum fused in one op** → 1x1 = 593524 | G3 ✓ — **DIVERGES at G5: sums all 94 cross-state rows, one per member state per MSA (double/triple/quadruple-counting)**; G6 executed on un-deduped rows |
| 3 | (text) | Final Answer: 593524 | wrong by exactly the 350,147 duplicated mass |

**First semantic divergence:** step 2, op `cross_state_msa_total_2024`, plan item G5 — the aggregation omits dedup. Its filter is semantically equivalent to the winners' (includes Micropolitan; exact repro: same 94-row set, sum 593524).

**Rendered evidence — the absence:** no render this arm ever saw contained a duplicate pair or a duplicate count. Step-1 raw render (1,366 chars, rows 0–8 + 755–763): `5 Columbus, GA-AL Metropolitan Statistical Area | 1,302 | Alabama.csv` appears ONCE (the `source_file` column even hands it per-row provenance the stats arm's all-null `__source_file` lacked — a GA-AL area filed under Alabama — and it still doesn't dedup); head window ends at row 8, before `LaGrange, GA-AL` at row 12. Step-2 clean render (1,922 chars, rows 0–11 + 441–451): each visible cross-state MSA once per window. Step-3 render: the bare fused total `1x1: 593524` — zero duplication signal at the terminal decision. SchemaOnly config renders no profile anywhere.

**Run-to-run instability on exactly this decision (corroboration from the report's rerun records):** REPORT.md §5: "Anchor (3k, schema-only) … **unstable across its own reruns**: over-dedup (242682), then no-dedup (593524) — two different wrong answers bracketing gold | fails twice"; FINDINGS.md: "anchor coin-flips 242682/593524 around gold". Both wrong answers reproduce exactly from the data (242682 = dedup + Metropolitan-only over-trim dropping LaGrange 453 + Lebanon-Claremont 242; 593524 = no dedup). Evidence-starved, the arm guesses a different dedup policy each run.

## Pair verdicts

**C1 Delta5k > Delta3k: ATTRIBUTED** (confirming the levers report, with one refinement to its evidence wording). One knob (`max_operator_result_char_limit` 5000 vs 3000). The winner's dedup edit is explained by its decision-time render — the materialized 94-row cross-state table whose 50-row window shows ~12 literal duplicate pairs (`Columbus, GA-AL … 1302` at rows 0 AND 9; quoted above) — and the loser's error is explained by absence: across its whole trajectory nothing rendered ever showed a repeated row or a dup count, and it fused filter+sum so no cross-state intermediate ever rendered at all. The report's line "raw repetition visible in the wider sample (Columbus, GA-AL inside the AL section; repeated headers at row 758)" needs refinement: both those hints were visible to the LOSER too (its 3k raw render shows row 5 `Columbus, GA-AL … Alabama.csv` and tail rows 755–763 with the repeated header) and moved neither arm — the differential, load-bearing render is the step-3 filtered-intermediate sample, unique to the winner. The fuse-vs-materialize structural divergence does not trigger the method-choice rejection (it postdates the arms' first rendered difference — step-1 raw render 19+18 rows vs 9+9); but because sweep-era thoughts are empty, whether the 5k budget caused the winner to materialize the inspectable intermediate is not directly evidenced, so this verdict rests on the skill's two strongest acceptance forms, both present: (a) the loser coin-flips across its own reruns on exactly this decision (242682/593524, both reproduced), and (b) dual-lever convergence with C2 — two different levers independently delivering the same missing fact and flipping to the same correct answer.

**C2 Stats3kD2 > Delta3k: ATTRIBUTED — the clean single-lever case.** One lever (`column_stats`+`data_level 2`). At the decision step the two arms' visible samples are row-for-row identical (9 head + 9 tail of the same 764-row concat; the loser even had strictly more per-row signal via its filled `source_file` column); the ONLY informational delta is the profile block — decisively **`- duplicate rows: 359 of 764 (47%)`** plus `duplicate_values=204` on the MSA column — and the winner writes `drop_duplicates(subset=['metropolitan_area','reports'])` in that same step, at the earliest correct placement, with gold's effective key. Winner's evidence explains its action; loser's absence explains its error; the loser's rerun coin-flip seals it.

Artifacts: `system_scratch/{DataflowSystemGPT52Delta5kSchemaOnly,DataflowSystemGPT52DeltaStats3kD2,DataflowSystemGPT52Delta3kSchemaOnly}/legal-hard-15/`; repro numbers generated with `.venv/bin/python` against `data/legal/input/State MSA Identity Theft data/`.

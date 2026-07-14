# biomedical-hard-7 — semantic trace walk

Pair judged: **C2 Delta3k > Stats3kD2** (chronic\* — default CHRONIC-VARIANCE unless accept rules met).
- WINNER: `DataflowSystemGPT52Delta3kSchemaOnly` (C2) — PASS, answer 16
- LOSER: `DataflowSystemGPT52DeltaStats3kD2` (C2) — FAIL, answer 15

One-knob check: configs differ only on the **stats block** — `column_stats` false→true and `data_level` 1→2 (these are the same conceptual lever; loser also carries newer null/off schema fields `frontier_decay_config`/`fold_resolved_revisions_config`/`probe_retirement_config`/`enable_inspect_tool`/`enable_render_prefs`, all inert). Same code vintage (Jul 8 traces), chronic-flipper task.

## Task + gold answer

- Question: "How many are the significant genes by acetylproteomics?"
- File: `data/biomedical/input/1-s2.0-S0092867420301070-mmc3.xlsx`
- Gold answer: **16** (`numeric_exact`)

## Gold semantic plan

From `solutions/biomedical/biomedical-hard-7.py`:

1. **Load** the xlsx with `sheet_name=None` (all sheets → dict of DataFrames).
2. **README lookup**: take the `README` sheet; find the row whose `Description` contains "acetylproteomics" (case-insensitive) and read its `Sheet` value → resolves to `D-SE-acetyl`.
3. **Count with header correction**: `num_genes = len(dfs[acetyl_sheet]) + 1`. The sheet is a bare list of genes with NO header row, so pandas consumes the first gene (`BRD8`) as the column name. `len()` counts only the 15 data rows; the **+1 re-adds the header gene** → 16.
4. **Output**: single integer 16.

The whole difficulty of this task is item 3 — the header-is-a-gene +1. Sheet discovery (item 2) is the easy part. Note: the E1-report framing for this task was "README sheet explains where data lives; check who misses the right sheet" — that failure mode does **not** bind this pair (see below); both arms find `D-SE-acetyl`.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (WINNER, PASS, answer 16)

| step | action (op) | semantics | vs gold plan |
|---|---|---|---|
| 1 | `acetylproteomics_xlsx` load | `read_excel(sheet_name=None)` | plan item 1; errors (dict return unsupported) but the KeyError **renders the README mapping**: `D-SE-acetyl → "Significant genes by acetylproteomics in Figure 1"` |
| 2 | `acetylproteomics_xlsx` modify | `pd.ExcelFile(...).sheet_names` → 7-row table | confirms sheet list (README, A-MSI…F-SS-phospho); redundant w/ README but harmless |
| 3 | `acetyl_sig_genes` load | `read_excel(sheet_name='D-SE-acetyl')` → 15 rows × 1 col, header `BRD8` | **matches plan item 2** — correct sheet |
| 4 | `acetyl_sig_gene_count` process | `shape[0] + (0 if columns.size==0 else 1)` = 15 + 1 = **16** | **matches plan item 3** — re-adds the header gene |
| 5 | final | "Final Answer: 16" | correct |

**First divergence from gold: none** — the winner executes the full plan, including the header +1.

Evidence at the decision step (step 4 context, Event 4 observation — the only info the model had when it wrote the +1):
```
Output Table: 15 rows, 1 cols
	BRD8
0	DHX15
1	SSB
...
14	PMS2
Schema (1 cols): BRD8 (str)
```
The operator summary shows the reasoning was explicit: *"count … (including the single column header gene plus all rows)."* It read `BRD8` (a gene) sitting in the header slot and corrected. No stats block was present (schema-only arm).

## Walk: DataflowSystemGPT52DeltaStats3kD2 (LOSER, FAIL, answer 15)

| step | action (op) | semantics | vs gold plan |
|---|---|---|---|
| 1 | `acetylproteomics_xlsx` load | `read_excel(sheet_name=None)` | plan item 1; same KeyError, same README mapping rendered |
| 2 | `acetylproteomics_xlsx` modify | `read_excel(sheet_name='D-SE-acetyl')` → 15 rows × 1 col, header `BRD8` | **matches plan item 2** — correct sheet (reached directly from README, one step fewer) |
| 3 | `significant_acetyl_gene_count` process | `len(acetylproteomics_xlsx)` = **15** | **FALLS SHORT of plan item 3** — no header +1 |
| 4 | final | "Final Answer: 15" | wrong (off by the header gene) |

**First divergence from gold: step 3** — action `significant_acetyl_gene_count`, `len(...)=15`. It fell short of gold plan item 3 (header-gene +1). Summary reasoning: *"count rows (each row is a significant gene)"* — treated the `BRD8` header as a non-data label rather than a 16th gene.

Evidence at the divergence step (step 3 context, Event 2 observation — everything the model saw before writing `len()`):
```
Output Table: 15 rows, 1 cols
	BRD8
0	DHX15
...
14	PMS2
Schema (1 cols): BRD8 (str)
Column Schema and stats:
- "BRD8" (str): null=0, distinct=15
```
The last two lines (`Column Schema and stats: - "BRD8" (str): null=0, distinct=15`) are the **stats block** the lever adds.

### Stats block: ADD or DISPLACE on this task?

- **Displace? No.** The loser still rendered the *identical* 15-row preview with `BRD8` in the header slot and gene rows `DHX15…PMS2`. The header-is-a-gene signal was present, byte-for-byte, in both arms. The task is tiny (~5.6k input tokens/step, single file, no compaction pressure), so nothing was crowded out.
- **Add? Yes, and mildly counter-productive.** The block adds `distinct=15`, foregrounding "15" as the count of the column's values and framing the column as a clean 15-item set. If anything it *reinforced* `len()=15` and drew attention to the row count rather than surfacing that `BRD8` is an off-by-one header gene. It did not add the piece of information that would have flipped the answer.

## Pair verdicts

**C2 Delta3k > Stats3kD2 → CHRONIC-VARIANCE.**

Per-arm divergence table:

| arm | first divergence step | action | gold-plan item fallen short |
|---|---|---|---|
| Delta3kSchemaOnly (win) | — | — | none (full plan incl. +1) |
| DeltaStats3kD2 (loss) | step 3 | `len(acetylproteomics_xlsx)`=15 | item 3 (header-gene +1) |

Why not ATTRIBUTED: the accept rule ("winner's evidence explains its action AND loser's absence explains its error") fails in both halves.
1. **Both arms had identical evidence for the hinge.** The header-is-a-gene signal (`BRD8` in the header slot above gene rows) rendered the same in both. The winner is the *leaner* schema-only arm — it succeeded with strictly *less* rendered info, so its +1 is not explained by any evidence the loser lacked.
2. **The loser was not evidence-starved.** It saw the same table plus a stats block; the block's only addition (`distinct=15`) pulled toward the wrong answer, it did not remove the winning signal.
3. The flip is a reasoning coin-flip on the subtle off-by-one header inference over identical evidence. biomedical-hard-7 is in the 23-task chronic-flipper set, consistent with exactly this kind of self-reruns coin-flip. No dual-lever convergence.

Not REJECTED-method-choice either (both arms use the same method: load `D-SE-acetyl`, count; the divergence is the +1, which post-dates identical rendered evidence). But the flip is unmeasurable against twin noise and rides the chronic gate → **CHRONIC-VARIANCE**. If anything, this trace is weak counter-evidence that the stats block *helps* accuracy here: its `distinct=15` line nudged toward the wrong count.

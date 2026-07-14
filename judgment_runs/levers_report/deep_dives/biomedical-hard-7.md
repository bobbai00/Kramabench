# biomedical-hard-7 — deep dive (counter-intuitive: schema-only Delta3k beat the stats arm DeltaStats3kD2)

Two `delta`-mode arms, 3k char limit. Single-knob map:

| Arm | role | context_mode | column_stats / data_level | answer | verdict |
|---|---|---|---|---|---|
| **Delta3kSchemaOnly** (mode Y) | WINNER | delta | **false / 1** | **16** | PASS |
| **DeltaStats3kD2** (mode X) | loser | delta | **true / 2** | **15** | FAIL |

The only *active* knob is the **stats block**: `column_stats` false→true and `data_level` 1→2 (same conceptual lever). The loser also carries newer schema fields (`enable_inspect_tool:false`, `enable_render_prefs:false`, `frontier_decay_config:null`, `fold_resolved_revisions_config:null`, `probe_retirement_config:null`) — all inert. Same code vintage (Jul-8 traces). `biomedical-hard-7` is in `chronic_flippers.json`.

## Task
Q: How many are the significant genes by acetylproteomics?

D: `data/biomedical/input/1-s2.0-S0092867420301070-mmc3.xlsx` — supplementary table mmc3 of the CPTAC3 UCEC (endometrial cancer) proteogenomics study. **7 sheets**, one small README map plus per-assay result tables:

```
'README'              -> (6, 2)     Sheet | Description  (the discovery map)
'A-MSI'               -> (103, 10)   MSI-H determination
'B-SE-proteomics'     -> (207, 1)    bare gene list  (headerless, +1 quirk)
'C-SE-phospho'        -> (61, 1)     bare gene list  (headerless, +1 quirk)
'D-SE-acetyl'         -> (15, 1)     bare gene list  (headerless, +1 quirk)  <- this task
'E-Mutation-adjacent' -> (17, 12)    mutation calls, adjacent tissue
'F-SS-phospho'        -> (630, 17)   significant phospho sites
```

**`README` sheet** — well-formed, its own header row is consumed correctly; 6 data rows map each `Sheet` to a `Description`:

```
                 Sheet                                         Description
0                A-MSI                                 MSI-H determination
1      B-SE-proteomics  Significant genes by global proteomics in Figure 1
2         C-SE-phospho  Significant genes by phosphoproteomics in Figure 1
3          D-SE-acetyl   Significant genes by acetylproteomics in Figure 1   <- resolves the question
4  E-Mutation-adjacent        Mutation calling results for adjacent tissue
5         F-SS-phospho   Significant phospho sites for pairwise comparison
```

**`D-SE-acetyl` sheet — the load-bearing quirk.** It is a *bare list of gene symbols with NO header row*. The three `SE-` sheets (B/C/D) all share this shape; D is the acetyl one. Read two ways:

```
# pandas DEFAULT (header=0)  -> 15 rows, column name is the FIRST GENE
columns: ['BRD8']            shape (15, 1)    len() = 15
       BRD8                  # <- "BRD8" is a GENE sitting in the header slot, not a label
0     DHX15                  1  SSB      2  FUS       3  PARP1    4  TRIM33
5     JADE3                  6  CEBPZ    7  TBL1XR1   8  MYC      9  FBXO22
10  ZSCAN18                 11  CDK1    12  TOP2A    13  H2AFJ   14  PMS2

# header=None  -> the RAW cells: 16 rows, BRD8 is data cell (0,0)
shape (16, 1)   iloc[0,0] = 'BRD8'      len() = 16   <- the TRUE gene count
```

Semantic meaning: every cell is one significant gene. Because the sheet has no header row, `pd.read_excel` with default settings eats the first gene (`BRD8`) as the column name, so `len()` returns 15 — one short. The correct count is **16**.

## Solution
From `solutions/biomedical/biomedical-hard-7.py`, as an operator graph:

```
read_excel(mmc3.xlsx, sheet_name=None)              # dict of 7 sheets, each default header=row 0
        │
        ├── dfs['README']                            # 6x2 map: Sheet | Description
        │      └─ filter  Description.str.contains("acetylproteomics", case=False)   # grain = 1 row
        │             └─ take .Sheet[0]  ─────────►  "D-SE-acetyl"    # sheet resolution (the EASY part)
        │
        └── dfs['D-SE-acetyl']                        # bare 1-col gene list; header ate gene #1 (BRD8)
               └─ len(...) = 15                       # counts the 15 DATA rows only
                     └─ + 1  ────────────────────►  16    # RE-ADD the header-consumed gene  <-- the whole task
                           └─►  answer = 16  (numeric_exact)
```

Answer-relevant path = {load `sheet_name=None`} → {README predicate `contains("acetylproteomics")` → `Sheet` = `D-SE-acetyl`} → {`len(D-SE-acetyl)` = 15} → **{+1 header-gene correction}** → 16. Item 2 (sheet discovery) is easy and *both arms get it*. The entire 16-vs-15 gap is **plan item 3 — the header-is-a-gene +1**.

## What DeltaStats3kD2 does (mode X, loser — 15, FAIL)
- **step 1** `acetylproteomics_xlsx` load: `read_excel(sheet_name=None)`. The engine rejects the dict return with a `KeyError`, but the error text **renders the README map**, including the line `D-SE-acetyl  Significant genes by acetylproteomics in Figure 1`. This is how it learns the sheet — no clean README read needed.
- **step 2** `acetylproteomics_xlsx` modify: `read_excel(sheet_name='D-SE-acetyl')` → 15 rows × 1 col, header `BRD8`. Correct sheet, reached directly from the README error (one step fewer than the winner). Plan item 2 ✓.
- **--> step 3** `significant_acetyl_gene_count` process: `pd.DataFrame({'significant_gene_count':[len(acetylproteomics_xlsx)]})` = **15**. Op summary: *"count rows (each row is a significant gene) and output the count as `significant_gene_count`."* **Divergence from plan item 3:** it treats the `BRD8` header slot as a non-data label rather than a 16th gene, so it never applies the +1.
- **step 4** Final Answer: **15** (off by exactly the header gene).

## What Delta3kSchemaOnly does (mode Y, winner — 16, PASS)
- **step 1** `acetylproteomics_xlsx` load: same `read_excel(sheet_name=None)` → same `KeyError`, same README map rendered.
- **step 2** `acetylproteomics_xlsx` modify: `pd.ExcelFile(...).sheet_names` → 7-row table (`README, A-MSI … F-SS-phospho`). Confirms the sheet list; redundant with the README the KeyError already surfaced, but harmless — this is the winner's one extra step.
- **step 3** `acetyl_sig_genes` load: `read_excel(sheet_name='D-SE-acetyl')` → 15 rows × 1 col, header `BRD8`. Correct sheet. Plan item 2 ✓.
- **step 4** `acetyl_sig_gene_count` process: `shape[0] + (0 if columns.size==0 else 1)` = 15 + 1 = **16**. Op summary: *"count the number of significant genes listed in the 'D-SE-acetyl' sheet (including the single column header gene plus all rows)."* Plan item 3 ✓ — it explicitly re-adds the header gene.
- **step 5** Final Answer: **16**. First divergence from gold: **none** — it executes the full plan including the +1.

## Why Delta3kSchemaOnly succeeded but DeltaStats3kD2 failed
**The decisive fact: at the count decision, the two arms' rendered evidence on the `D-SE-acetyl` frame is BYTE-IDENTICAL except for two stats lines.** Both are `delta` arms and both rendered the same 15-row preview with `BRD8` sitting in the header slot above the gene rows. Verified from each arm's `inputMessages` at its count step —

Loser (before step 3, the `len()` decision):
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

Winner (before step 4, the `+1` decision):
```
Output Table: 15 rows, 1 cols
	BRD8
0	DHX15
...
14	PMS2
Schema (1 cols): BRD8 (str)
```

The only difference is the loser's trailing two lines — the stats block the knob adds:
```
Column Schema and stats:
- "BRD8" (str): null=0, distinct=15
```

- **Displace? No.** The header-is-a-gene signal (`BRD8` in the header slot above `DHX15…PMS2`) rendered identically in both. The task is tiny — ~5.6–5.7k input tokens/step, one small file, zero compaction pressure — so nothing was crowded out. The stats block is a pure **ADD**.
- **Add? Yes, and it points the WRONG way.** `distinct=15` foregrounds "15" as *the* count of the column's values and frames `BRD8` as a clean 15-item set. Far from surfacing that `BRD8` is an off-by-one header gene, it **reinforces `len()=15`** and pulls attention to the row count. It did not add the one fact that flips the answer.
- **The winner is the LEANER arm and still won.** Delta3kSchemaOnly succeeded on *strictly less* rendered information (no stats block). So its +1 cannot be explained by any evidence the loser lacked, and the loser was not evidence-starved — it saw everything the winner did, plus a misleading extra line.

The operative divergence lives entirely in the *reasoning label* placed on identical bytes: winner *"header gene + rows"* vs loser *"each row is a gene."* The +1 was a **reasoning step, not rendered evidence**.

**Label: method-choice / CHRONIC-VARIANCE.** The accept rule ("winner's evidence explains its action AND loser's absence explains its error") fails on both halves — the winner had no extra evidence, and the loser lost no winning signal. Not REJECTED-method-choice in the "predates the first rendered difference" sense (both arms use the same load-and-count method), but the deciding +1 post-dates identical rendered evidence, so it is a reasoning coin-flip, not a lever effect. `biomedical-hard-7` sitting in `chronic_flippers.json` is consistent with exactly this subtle off-by-one flipping across self-reruns; no dual-lever convergence. **If anything this trace is weak counter-evidence that the stats block helps accuracy here** — its `distinct=15` line nudged toward the wrong count.

Cost footnote (not the question; loser passed neither so no both-pass cohort): winner `$0.0169` (6 steps), loser `$0.0141` (5 steps). The winner spent *more*, entirely from its redundant `ExcelFile.sheet_names` probe — not from anything the stats block saved. So on this task more rendered info (loser) was both wrong and cheaper, and the extra dollars (winner) bought a redundant probe, not the correct answer; the +1 was free.

# Semantic walk: environment-easy-3

## Task + gold answer

**Question:** How many beaches had a higher bacterial exceedance rate for water samples collected in 2013 compared to 2012, excluding those with no samples in 2012?

**Gold answer:** 268 (single integer).

**Judge marks:** Latest3kSchemaOnly `success=1` (answer 268, PASS), Delta3kSchemaOnly `success=0` (answer 267, FAIL). Task is **NOT** in `chronic_flippers.json` — this is one of the two non-chronic flips in the levers set. Config diff between arms: `context_mode: latest` vs `delta` (the delta config additionally carries 5 later-vintage fields, all `null`/`false` = disabled — no behavioral delta).

## Gold semantic plan

Source: `solutions/environment/environment-easy-3.py`

| # | Plan item |
|---|---|
| G1 | Load `data/environment/input/water-body-testing-2012.csv` and `...-2013.csv` (plain `read_csv`, header row present) |
| G2 | Normalize `Beach Name` to lowercase |
| G3 | **Composite beach key**: `str(Community Code) + str(County Code) + Beach Name` — same-named beaches in different towns are DIFFERENT beaches |
| G4 | Per year, group by beach key: `total_samples = count`, `num_violations = (Violation.lower()=='yes').sum()`, `exceedance_rate = violations/samples` |
| G5 | **Inner join** 2012 and 2013 aggregates on the beach key (implements "excluding those with no samples in 2012") |
| G6 | Count rows where `rate_2013 > rate_2012` → **268** |

Data facts (verified against the CSVs, `.venv/bin/python`): 15,377 / 15,388 rows; zero NaN in `Beach Name` and `Violation`; zero rows with off-year `Year`; casing is uniform (lowercasing changes nothing); `Community Code → County Code` is a functional dependency (0 codes map to >1 county). Therefore the minimal correct key is `(Community Code, Beach Name)` — exactly equivalent to gold's triple key on this data. Key arithmetic: composite → 1171 (2012) / 1176 (2013) groups, inner join 1138, count **268**; `Beach Name` alone → 1155 / 1162 groups (16 / 14 cross-community name collisions collapse), inner join 1124, count **267**. The key choice alone flips the answer; every other code difference between the arms is a no-op (verified by exact repro of both traces' pipelines).

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (WINNER — PASS)

**Final answer:** 268 — correct. 5 agent steps, 41,650 input tokens, 68.6s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `wb2012`, `wb2013` | `pd.read_csv(...)` both files, default params | G1 ✓ |
| 1 | `exceed_rate_2012` | drop `Violation` NaN (no-op, 0 NaNs); `groupby(['Community Code','Beach Name'], dropna=False)`; samples=size, exceed=(lower=='yes').sum(), rate=exceed/samples → 1171x5 | G3 ✓ (equivalent variant: omits County Code — redundant under the CC→County FD; omits lowercasing — casing uniform) G4 ✓ |
| 2 | `exceed_rate_2013` | same key, same aggregation → 1176x5 | G3 ✓ G4 ✓ |
| 3 | `higher_exceed_2013_count` | inner merge on `['Community Code','Beach Name']`; `samples_2012 > 0` guard (no-op after inner join); count `rate_2013 > rate_2012` → 1x1 = 268 | G5 ✓ G6 ✓ |
| 4 | (text) | Final Answer: 268 | ✓ |

**First semantic divergence:** none load-bearing. The key variant at step 1 is data-equivalent to gold (identical 1171/1176/1138 cardinalities, identical count 268 on re-execution).

**Rendered evidence before step 1 (the key-decision step)** — full `inputMessages` of raw step 2, LATEST grammar:

> `### Operator \`wb2012\` (DataLoading)` … `Output Table: 15377 rows, 11 cols` — 4 head rows (all `242 Provincetown … 333 Commercial Street`) + 4 tail rows (all `287 Sturbridge … Yogi Bear Campground`) + `Schema (11 cols): Community (str), Year (numeric), Violation (str), Community Code (numeric), … Beach Name (str)` — and the same block for `wb2013`.

Nothing rendered shows a beach-name collision (the 8 visible rows per file cover exactly two beaches, one community each) and SchemaOnly mode renders no uniqueness/nunique stats. The composite-key choice is NOT explained by any rendered evidence.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (loser — FAIL)

**Final answer:** 267 — off by one. 4 agent steps, 33,636 input tokens, 37.4s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `wb2012`, `wb2013` | `pd.read_csv(...)` both files — **code identical to winner's step 0** | G1 ✓ |
| 1 | `exceedance_rates` | per year: filter `Year==year` (no-op), drop `Beach Name` NaN (no-op); **`groupby('Beach Name')`** — single-column key; samples/exceedances/rate per name; inner merge on `Beach Name`; `samples_2012 > 0` guard; `higher_in_2013 = rate_2013 > rate_2012` → 1124x8 | G4 ✓ G5 ✓-shape — **DIVERGES at G3**: key collapses 16 (2012) / 14 (2013) cross-community same-named beaches into single pooled pseudo-beaches |
| 2 | `count_higher` | sum of `higher_in_2013` → 1x1 = 267 | G6 correct-given-step-1 |
| 3 | (text) | Final Answer: 267 | off by one |

**First semantic divergence:** step 1, op `exceedance_rates`, plan item G3 — grouping/join key = `Beach Name` alone. Exact repro on the CSVs reproduces the trace end-to-end (1155/1162 groups, 1124 joined rows as rendered, count 267) and confirms this single choice, not the Year/notna filters, flips 268 → 267.

**Rendered evidence before step 1 (its key-decision step)** — full `inputMessages` of raw step 2, DELTA grammar:

> `## Agent Event 2` — Action block echoing the two loader codes, then `- operator wb2012 added` / `result:` … `Output Table: 15377 rows, 11 cols` — **the same 4 head rows + 4 tail rows + the identical `Schema (11 cols): …` line** as the winner's render, and the same for `wb2013`.

## Were the arms' contexts informationally identical before the divergence?

Both arms diverge-or-commit at the SAME point: the first processing op (raw step 2), with exactly one prior action (the two loads). Byte-level diff of the two decision-time contexts finds only:

1. **Grammar scaffolding** — LATEST `### Operator \`x\`` + one-line Summary vs DELTA `## Agent Event 2` + Action code echo + `- operator x added`. The tabular payload is identical: same 8 sample rows per file, same schema line, same row/col counts. No stats in either (SchemaOnly).
2. **Task file-list order** — Latest3k's prompt lists `...-2013.csv` first, Delta3k's lists `...-2012.csv` first. This difference exists in the initial task message (raw step 1, before ANY action), i.e., it predates all mode rendering — it is harness manifest ordering, not a context-mode effect.

Neither difference carries information about beach-name uniqueness. **Informationally, w.r.t. the key decision, the contexts were identical.** The winner's evidence does not explain its action (nothing rendered motivates compositing the key), and the loser's absence does not explain its error (the same `Community Code` column sat equally visible in its schema line and sample rows).

**Confounder checks (49 GPT-5.2 arms on this task):**

- **Key choice does not track mode.** Within the winner's own mode family: Latest5k and Latest7k chose `groupby('Beach Name')` → 267; every LatestStats/LatestColumnStats arm (which additionally rendered column stats) → 267. Within the Delta family: DeltaWin3kCompressPromptAware chose `groupby(['Community Code','Beach Name'])` → 268. Overall the composite key/268 appears in 4/49 GPT-5.2 arms (Latest3kSchemaOnly, LatestSchemaConverge, FullSchemaNoStats, DeltaWin3kCompressPromptAware) scattered across latest, delta-windowed, and full modes. (FullStatsOn=265 is a third, unrelated modeling variant.)
- **File order is not the driver either.** Crosstab first-listed-year × answer: 2013-first → 3×268 / 12×267; 2012-first → 1×268 / 30×267. Both orders overwhelmingly yield 267.
- The ~8% composite-key rate also explains the task's absence from `chronic_flippers.json`: three identical-config twin pairs simply never sampled the rare branch — "non-chronic" here is twin-sample sensitivity, not determinism.

## Pair verdicts

**C3 Latest3k > Delta3k: REJECTED-method-choice** (confirming the levers report, with one refinement to its wording). The hinge is a single a-priori modeling decision — composite `(Community Code, Beach Name)` key vs `Beach Name` alone — committed at each arm's first processing step. Strictly, the divergence does not *predate the arms' first rendered difference* (DELTA/LATEST grammar already differs in the one prior observation render); it predates any **informative** rendered difference: the decision-time contexts carried identical tables, identical schema, no collision or uniqueness evidence, and no stats. The acceptance test fails on both sides (winner's render doesn't explain its action; loser's doesn't explain its error), and the mode-independence survey is decisive — the winner's own mode at 5k/7k budgets picks the losing key, a Delta-family arm picks the winning key, and stats-enriched arms still pick the losing key. The flip is a low-probability (≈4/49) sampling event on an unforced join-key convention, not a context_mode effect.

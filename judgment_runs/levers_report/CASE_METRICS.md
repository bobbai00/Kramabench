# Case Metrics: what the operators & files look like in each comparison case

Programmatic per-case metrics for the three lever comparisons, over the four
Venn cases of each pair: **A-only wins, B-only wins, both-pass-A-cheaper,
both-pass-B-cheaper** (cheaper = material, ≥10% of the dearer arm — the twin
noise band). Companion to `REPORT.md` (flip attribution) and
`COMPARISON_PIPELINE.md` (method).

Reproduce (also dumps full per-op JSON next to this file):

```bash
./kb.py case-metrics --sut DataflowSystemGPT52Delta3kSchemaOnly DataflowSystemGPT52Delta5kSchemaOnly   # C1
./kb.py case-metrics --sut DataflowSystemGPT52Delta3kSchemaOnly DataflowSystemGPT52DeltaStats3kD2      # C2
./kb.py case-metrics --sut DataflowSystemGPT52Delta3kSchemaOnly DataflowSystemGPT52Latest3kSchemaOnly  # C3
```

Full console outputs: `case_metrics/C{1,2,3}_console.txt`.

## Metric definitions

- **depth** — longest path from a source to the operator in the final DAG
  (`workflow.json`); role = source (fan-in 0) / interior / sink (fan-out 0).
- **out rows/cols** — the operator's true output cardinality, parsed from the
  always-rendered `Output Table: N rows, M cols` line of its LAST rendered
  observation (both DELTA and LATEST grammars).
- **in rows** — input cardinality: sum of parents' out-rows for interiors; for
  sources, the **input-table cardinality** = the engine-loaded row count of the
  file, taken from the stats arm's full-data profiles (fallback: raw line
  count for line-oriented formats).
- **file facts** — extension, size on disk, loaded rows.
- **data issues** (engine-measured, from stats-arm `Output Table profile` on
  full data): duplicate-row %, empty-row %, empty columns, unnamed-header
  columns, max per-column null %, str-typed column share.
- **render pressure** — capped% (observation hit the char cap: elision/
  truncation marks or near-cap block), rendered/actual rows (table body lines
  actually shown ÷ true out-rows).
- **multi-edit%** — ops with ≥2 landed code versions (where history renders).

Caveats: exclusive-win cells are 3–6 tasks and mostly chronic flippers
(accuracy there is noise-dominated — profile them as *where the lever binds*,
not as attributed wins; the one attributed flip remains legal-hard-15). Ops
are the FINAL DAG only — deleted probes show up in steps/cost, not op counts.
Both-pass cost cells are 19–37 tasks and are the robust populations.

---

## C1 — sampling cap (Delta3kSchemaOnly vs Delta5kSchemaOnly)

Venn: 3k-only **4** | both **76** | 5k-only **5** | neither 19.
Both-pass cost: 3k materially cheaper on **26**, 5k on **19**.

| metric | 3k-only wins (4t) | 5k-only wins (5t) | both, 3k cheaper (26t) | both, 5k cheaper (19t) |
|---|---|---|---|---|
| chronic share | 4/4 | 4/5 | 5/26 | 4/19 |
| ops/task (3k\|5k) | 8.0\|7.8 | 6.2\|4.2 | 5.8\|6.1 | 6.1\|4.5 |
| depth med; rows@d0/d1/d2+ (3k) | 1; 2,965/779/2 | 1; 325/860/1 | 0; 153/153/1 | 1; 56/24/1 |
| src input-table rows med | 1,388 | 324 | 153 | 153 |
| file KB med / rows med | 220 / 1,388 | 30 / 862 | 31 / 433 | 7 / 95 |
| dirty-file % (of distinct files) | 64% | 33% | 50% | 65% |
| rendered/actual rows med (3k\|5k) | 2.8%\|21.5% | 14.8%\|5.1% | 11.6%\|56.2% | 43.1%\|**86.2%** |
| capped% (3k\|5k) | 59\|52 | 65\|71 | 66\|58 | 47\|44 |
| multi-edit% (3k\|5k) | 31\|16 | 10\|5 | 9\|16 | 24\|19 |

**Where 5k wins exclusively** (environment-hard-7/-8/-11, astronomy-easy-4,
legal-hard-15): NOT big files — small, *wide-dirty* ones. The beach
datasheets: carson 39KB/1,134 rows with **7 unnamed cols, 65% max-null**;
pleasure_bay 30KB/862 rows with **9 unnamed, 92% max-null**. The 3k arm
compensates with probe DAGs (6.2 vs 4.2 ops/task; environment-hard-8: 12 steps
/12 ops vs 6/5). More sample rows resolve column semantics that the schema
line can't carry when headers are unnamed.

**Where 3k wins exclusively** (all 4 chronic): big-ish files (med 1,388 rows,
p90 44,691; worldcities 4.6MB, omni2 2.8MB .dat) — tables so large that BOTH
caps elide heavily (3k shows 2.8%, 5k 21.5% of rows). The extra rows carry no
decision-relevant info; outcomes there coin-flip (hence all-chronic).

**Where 3k is cheaper (26 tasks, the modal case)**: small-to-mid tables
(files med 31KB/433 rows) where 5k renders 56% of all rows vs 3k's 12% at the
SAME step counts — pure render-byte waste. Includes the whole
water-body-testing family (15–16k rows, clean: schema alone suffices).

**Where 5k is cheaper (19 tasks)**: tiny but dirty tables (files med 7KB /
**95 rows**, 65% dirty — 6 dup, 7 empty-rows, 7 unnamed of 23 files). At 5k
the render crosses the **full-visibility threshold** (rendered/actual med
86.2% vs 43.1%): the table is simply *all there*, so the agent skips probe
iteration (4.5 vs 6.1 ops/task, multi-edit 19% vs 24%; biomedical-easy-9:
4 steps/2 ops vs 11/10; archeology-easy-4: 9/5 vs 13/11).

**C1 reading**: the sampling cap is not a "big data" lever. Above ~1k rows
every cap elides (info lives in schema/stats); the cap pays exactly in the
~40–150-row band where 5k shows the whole table and 3k doesn't, and in
wide-dirty small tables where rows disambiguate unnamed columns. Small+clean
→ 3k cheaper; small+dirty → 5k wins or is cheaper.

---

## C2 — profiling (Delta3kSchemaOnly vs DeltaStats3kD2)

Venn: schema-only **3** | both **77** | stats-only **6** | neither 18.
Both-pass cost: schema-only materially cheaper on **37**, stats on **19**.

| metric | schema-only wins (3t) | stats-only wins (6t) | both, schema cheaper (37t) | both, stats cheaper (19t) |
|---|---|---|---|---|
| chronic share | 3/3 | 5/6 | 7/37 | 4/19 |
| ops/task (sch\|st) | 3.3\|3.3 | 5.7\|6.3 | 5.9\|6.3 | 6.6\|5.3 |
| src input-table rows med | 1,388 | 387 | 153 | 387 |
| file KB med / rows med | 1,271 / 6,658 | 39 / 1,119 | 30 / 187 | 72 / 153 |
| unnamed-header files | 0/4 | **8/13** | 19/55 | 5/23 |
| null≥20% files | 3/4 | 10/13 | 20/55 | 10/23 |
| str-share med | 67% | **100%** | 100% | 100% |
| non-.csv files | 1/4 | 3/13 | 13/55 | **11/23** (.gpkg/.cdf/.lst/.xlsx/.html) |
| multi-edit% (sch\|st) | 10\|10 | 15\|3 | 11\|14 | **32\|20** |

**Where stats wins exclusively**: the beach-datasheet family + legal-hard-1 —
small (39KB med), 100% string-typed, wide-dirty files: 8/13 have unnamed
headers, 10/13 have ≥20% nulls (wollaston 11 unnamed/79% null, tenean 5/83%,
malibu 5/80%...). The profile lines (`headers: N of M columns are unnamed`,
per-column nulls, distinct counts) ARE the missing information; the schema
line alone renders these files as opaque `Unnamed: k (str)` walls. Same
failure surface 5k fixes in C1 — profiling and sampling are substitutes here
(environment-hard-8 is fixed by either lever, like legal-hard-15 — both
chronic-tagged, but the convergence is consistent).

**Where schema-only wins exclusively** (all 3 chronic): big clean-structured
numeric-ish files (worldcities 4.6MB, Fire_Weather 1.3MB/6,658 rows; 0
unnamed, 0 dup) — the profile adds bytes, not decisions.

**Where schema-only is cheaper (37 tasks — the always-on tax, case-resolved)**:
the modal task: small (30KB/187 rows med), half-clean files, interior
cardinality med 126 rows. Stats pays its +13.9% here for facts the agent never
uses. This cohort is why always-on profiling loses on cost overall.

**Where stats is cheaper (19 tasks)**: format weirdness — 11/23 files are
non-CSV (.gpkg ×2, .cdf, .lst ×2, .xlsx ×5, .html). Schema-only gropes at
these loaders (multi-edit 32%, its highest anywhere in this grid; steps 12→6
wildfire-hard-6, 14→12 astronomy-hard-8, 13→7 archeology-easy-4): the
profile substitutes for probe iteration on opaque loads (usa.gpkg: 100%
max-null profile immediately flags the geometry-only load; climateMeasurements
.xlsx: 29 unnamed + 5% empty rows says "header row is wrong" in one line).

**C2 reading**: profiling is an *anti-iteration* lever for dirty/wide/weird
inputs, not a general-accuracy lever. Its value concentrates precisely where
str-share=100% + unnamed headers (semantics invisible in schema) or non-CSV
formats (loader correctness invisible in samples). On clean tabular data it
is pure tax — hence targeted, per-operator delivery (render-prefs direction).

---

## C3 — history (Delta3kSchemaOnly vs Latest3kSchemaOnly)

Venn: Delta-only **5** | both **75** | Latest-only **4** | neither 20.
Both-pass cost: Delta materially cheaper on **20**, Latest on **34**.

| metric | Delta-only wins (5t) | Latest-only wins (4t) | both, Delta cheaper (20t) | both, Latest cheaper (34t) |
|---|---|---|---|---|
| chronic share | 5/5 | 3/4 | 2/20 | 6/34 |
| ops/task (Δ\|L) | 7.8\|**11.4** | 6.0\|6.0 | 6.2\|8.1 | 5.6\|4.6 |
| sinks share in Latest arm | **32/57 = 56%** | 8/24 | 47/161 = 29% | 39/156 = 25% |
| interior out-rows med (Δ\|L) | 126\|44 | **3,648\|8,491** | 39\|50 | 364\|240 |
| src input-table rows med | 153 | 1,134 | 153 | 630 |
| file KB med / rows med | 191 / 328 | 191 / 1,134 | 7 / 153 | 33 / 387 |
| dirty-file % | 58% | 29% | 41% | 62% |
| multi-edit% (Δ\|L) | **36\|21** | 4\|8 | 21\|19 | 15\|10 |
| formats | .tle/.text/.dat/.xlsx heavy | clean big .csv | mixed | .csv-dominant (36/47) |

**Where Delta wins exclusively** (all 5 chronic, but the structural signature
is stark): weird small formats (.tle, .text, .dat, omni2, TLE files — the
levers-report "difficulty lever" fingerprint) with the grid's highest
multi-edit share (36%). The Latest arm doesn't just fail — it **churns into
probe graveyards**: 11.4 ops/task of which 56% are dead-end sinks
(wildfire-hard-17: 25 ops, **19 unconsumed probe sinks**, 26 steps, $0.19 vs
Delta's 9 ops/13 steps/$0.09; astronomy-hard-9: 18 ops/9 sinks/26 steps/$0.45
vs 7/10/$0.11 — the identical-colscan-³× exhibit from REPORT.md §6). Without
rendered failure history, the same probe idea gets rebuilt as a NEW operator.

**Where Latest wins exclusively**: the inverse world — big clean CSV
pipelines (water-body-testing 15–16k rows, 0 unnamed, 0 dup), multi-edit
4–8% (nothing worth remembering), and **high mid-pipeline cardinality**
(interior out-rows med 8,491 in the Latest arm vs 3,648 — concat/union-shaped
flows; Latest's reduce-share 43% vs Delta 71% = more expanding ops). One
current-state render of a wide concat beats replaying its event history.

**Where Delta is cheaper (20 tasks)**: mid-size DAGs where Latest re-probes
what history would have told it (biomedical-hard-1: 25 steps/21 ops vs
Delta's 7/9 on the 73k-row mmc2.xlsx; wildfire-hard-6: 20/25 vs 12/8) —
same churn mechanism as the exclusive wins, below the failure threshold.

**Where Latest is cheaper (34 tasks — the biggest cost cohort in the grid)**:
short clean runs — 4.6 ops/task, multi-edit 10%, legal-easy family dominant
(36/47 files .csv, but 62% carry the legal docket dup/empty/unnamed trio the
agent handles in one shot). At equal-ish steps (5/4, 5/5...) Latest's
current-DAG render is simply fewer bytes than Delta's event log
(legal-hard-2: $0.12 vs $0.34). History is dead weight when nothing is
revised.

**C3 reading**: history is an *iteration-memory* lever. Its value tracks
multi-edit share (36% where Delta wins, 10% where Latest is cheaper) and its
cost signature is structural: sink-share of the Latest DAG (56% on Delta-win
tasks vs 25% baseline) is a programmatic churn detector — orphan probes
accumulate exactly where failure history is missing. Expanding, clean,
single-pass pipelines invert the sign.

---

## Cross-cutting insights (the numbers that generalize)

1. **The winning lever is predicted by the input table's failure mode, not its
   size.** Unnamed-header/wide-null/100%-str small tables → sampling OR
   profiling (substitutes: environment-hard-8, legal-hard-15 are fixed by
   either); format-quirk iteration (.tle/.gpkg/.cdf/.xlsx sheets) → profiling
   for loader sanity, history for probe memory; big clean tables → every
   extra byte is tax, leanest arm wins.
2. **Cardinality collapses at depth 1–2 everywhere** (rows med at
   depth0/1/2+ ≈ hundreds/tens/1–5 in all 12 cells; answer sinks are 1×1;
   reduce-share 57–100%). All render pressure lives at the data edge —
   per-operator knobs (render prefs) only need to exist for sources and
   first-hop transforms.
3. **A cap raise pays only across the full-visibility threshold.** 5k-cheaper
   cohort: file rows med 95, rendered/actual 86% (vs 43% at 3k) — the table
   becomes fully visible and probes disappear. Above ~1k rows both caps elide
   (2.8%/21.5% shown) and the raise is waste: 3k-cheaper cohort med 433-row
   files at identical steps. Rule of thumb: render caps should be set per
   input-table cardinality (≲100 rows: show all; ≳1k: schema/stats only),
   which is exactly what per-op render prefs enable.
4. **Dirtiness is measurable and predictive.** Engine-profile dirtiness
   (dup/empty/unnamed/max-null) sorts the cases: in C1 the cells that pay
   for more rows are the dirty-small ones (5k-cheap: 65% dirty at med 95
   rows; 5k's exclusive wins: 7–9-unnamed-col datasheets) while clean small
   tables land in 3k-cheaper; in C2 stats-wins files are 8/13
   unnamed-headed; the C3 Latest-cheap legal cohort is 62% dirty but
   *stereotyped* (the same docket dup/empty/unnamed trio each time →
   one-shot handled, no lever needed). A cheap static router could read
   these four numbers per file and pick sampling depth / stats on-off per
   source.
5. **Churn has a DAG signature.** Sink-share (unconsumed leaves) of the
   final DAG: 56% where Latest fails by churning vs 25–29% elsewhere;
   ops/task 11.4 vs 4.6–8.1. This is a trace-free, programmatic thrash
   detector — usable as a run-time trigger (e.g. inject history / suggest
   deletes when sink-share crosses ~40%) and as the quantified motivation for
   the render-prefs delete-nudge.
6. **Multi-edit share remains the universal sorter** (from the pipeline doc,
   now with per-case values): 31–36% where the richer arm wins (C1-3k-only,
   C3-Delta-only), 3–10% where the leaner/stateless arm wins (C1/C2/C3
   B-only and Latest-cheap cells). Iteration begets information value.

## Where this feeds the paper

- §levers: insight 1 is the case-level version of "levers are
  evidence-delivery channels" — with file-level predictors.
- §render-prefs v2: insights 2–4 give the *targeting policy* (edge ops only;
  cap by input cardinality; stats by dirtiness) that v1's agent-chosen prefs
  lacked.
- §method: insight 5 (sink-share) and the case-metrics command extend the
  comparison pipeline with structural, trace-free diagnostics.

# Findings from the case-metrics iteration — detailed evidence

Six findings from the per-Venn-case metric drill over C1 (sampling cap:
Delta3kSchemaOnly vs Delta5kSchemaOnly), C2 (profiling: vs DeltaStats3kD2),
and C3 (history: vs Latest3kSchemaOnly). Every number is programmatic and
reproducible:

```bash
./kb.py case-metrics --sut <A> <B>                 # per-case tables (C1/C2/C3 in case_metrics/)
.venv/bin/python scripts/analyze_case_findings.py  # full-population tables (F1/F3/F4)
```

Companions: `CASE_METRICS.md` (per-case tables), `REPORT.md` (flip
attribution), `../COMPARISON_PIPELINE.md` (method). Pass = answer-type metric
≥ 0.9; "materially cheaper" = cost gap ≥ 10% of the dearer arm (the twin-noise
band); chronic flippers per `chronic_flippers.json`.

---

## F1. A render-cap raise pays only across the full-visibility threshold — the "small file, small sample" intuition inverts

**Claim.** Raising the per-observation cap 3k→5k changes what the agent can
*decide* only for output tables of ≲100 rows — where the bigger cap tips the
table from partially to fully visible. Above ~1k rows both caps elide almost
everything (schema is the only signal either way), so a raise is pure byte
tax. Consequently the cap's payoff concentrates on **small dirty** tables,
not big ones.

**Full-population evidence** (every rendered operator output in both C1 arms,
all ~103 tasks; coverage = table rows actually rendered ÷ true `Output Table`
rows; fully-visible = coverage ≥ 90%):

| out-rows band | n (3k) | coverage med 3k | fully-visible 3k | n (5k) | coverage med 5k | fully-visible 5k |
|---|---|---|---|---|---|---|
| 2–40 | 181 | 86.7% | 49.2% | 145 | **100.0%** | **66.9%** |
| 41–100 | 39 | 50.0% | 20.5% | 49 | **80.8%** | **44.9%** |
| 101–1k | 101 | 4.9% | 0.0% | 96 | 8.3% | 0.0% |
| >1k | 129 | 0.1% | 0.0% | 124 | 0.1% | 0.0% |

The 41–100-row band is the contested zone: the raise moves median coverage
50%→81% and doubles the fully-visible share (20.5%→44.9%). At 101+ rows
neither cap ever reaches 90% coverage (0.0% in both arms) — the raise buys
3.4 percentage points of coverage on 101–1k tables and nothing at >1k. (The
1-row band is excluded: narrow one-row tables render without enough tab
characters for the counter, so its coverage stat is an artifact.)

**Where that translates to money** (C1 both-pass cells):

- *5k materially cheaper, 19 tasks*: the smallest files in the whole grid —
  median **7KB / 95 rows**, 65% dirty (6 dup≥5%, 7 empty-rows, 7
  unnamed-header files of 23). Rendered/actual rows med **86.2% vs 43.1%**:
  at 5k the table is simply all there, and probe iteration disappears —
  4.5 vs 6.1 ops/task, multi-edit 19% vs 24%. Exhibits:
  - biomedical-easy-9: 3k needs 11 steps / 10 ops ($0.05); 5k answers in
    4 steps / 2 ops ($0.02).
  - archeology-easy-4: 13 steps / 11 ops ($0.07) → 9 / 5 ($0.05).
  - wildfire-easy-2: 8 steps / 6 ops ($0.06) → 6 / 4 ($0.03).
- *3k materially cheaper, 26 tasks*: small-to-mid CLEAN tables — files med
  31KB / 433 rows, the water-body-testing family (15–16k rows, 0 unnamed,
  0 dup) prominent. 5k renders 56.2% of all rows vs 3k's 11.6% at the same
  median step counts — pure render waste. Exhibits: environment-easy-1
  ($0.01 vs $0.04), astronomy-easy-1 (6 steps/$0.04 vs 10/$0.09).

**Exclusive wins** (small cells, mostly chronic — read as *where the lever
binds*, not attributed accuracy):

- 5k-only (5 tasks, 4 chronic): wide-dirty small tables. carson_beach
  39KB/1,134 rows with **7 unnamed cols, 65.1% max-null**; pleasure_bay
  30KB/862 rows with **9 unnamed, 91.8% max-null**. The 3k arm compensates
  with probe DAGs (6.2 vs 4.2 ops/task; environment-hard-8: 12 steps/12 ops
  vs 6/5). Extra rows are how the agent learns what unnamed columns MEAN.
- 3k-only (4 tasks, 4/4 chronic): big files (med 1,388 rows, p90 44,691 —
  worldcities 4.6MB, omni2_2024.dat 2.8MB) where both caps show <22% of
  rows; outcome differences there are coin-flips, consistent with chronic
  tagging.

**Implication.** Set render caps per input-table cardinality, not globally:
≤100 rows → show everything (kills probes); ≥1k rows → schema/stats only
(rows are decoration). This is precisely the policy a per-operator
`outputSummary` knob can express — but driven by measured cardinality, not
agent goodwill (render-prefs v1's failure).

---

## F2. The winning lever is predicted by the input table's *failure mode*, not its size

**Claim.** Across all three comparisons, which arm wins/is-cheaper on a task
is predicted by WHAT IS WRONG with its source files — not by how big they
are. Size sorts nothing (5k's wins are on 30–39KB files; schema-only's wins
on 1.3–4.6MB files); the engine-measurable issue vector
(dup%, empty-rows%, unnamed-headers, max-null%, str-share) sorts everything.

**The decisive-file gallery** (engine-measured on full data, from the stats
arm's `Output Table profile`):

| file | size / rows | measured issues | which lever answers it — where |
|---|---|---|---|
| carson_beach_datasheet.csv | 39KB / 1,134 | 7 unnamed, 65.1% max-null, 100% str | rows (5k: C1 B-only) OR profile (stats: C2 B-only) |
| pleasure_bay…datasheet.csv | 30KB / 862 | 9 unnamed, 91.8% max-null | same pair (environment-hard-11) |
| wollaston_beach_datasheet.csv | 81KB / 1,906 | 11 unnamed, 78.8% max-null | profile (C2 stats-wins cell) |
| climateMeasurements.xlsx | 2.4MB / 8,366 | **29 unnamed, 5.0% empty rows, 2.5% dup** | profile says "header row is wrong" in one line (C2 stats-cheaper: archeology-easy-4 13→7 steps) |
| usa.gpkg | 17MB / 3,142 | 100% max-null (geometry-only load), 85% str | profile flags the bad load instantly (C2 stats-cheaper cell) |
| TLE/43180.tle | 18KB / 134 | 6.0% duplicate rows (epoch dups) | history — the C3 Delta-only churn task astronomy-hard-9 |
| omni2.text / omni2_2024.dat | 17KB / 2.8MB | 0 visible issues; header-less fixed-width | history (parse-iteration memory): C3 Delta-only cell |
| worldcities.csv | 4.6MB / 44,691 | 0 unnamed, 0 dup, 75.1% max-null (sparse optional cols) | nobody — leanest arm cheapest; its exclusive-win appearances are all chronic |
| water-body-testing-20xx.csv | 1.4–1.5MB / 15–16k | fully clean (0/0/0) | nobody — sits in every "leaner arm cheaper" cell |

**The substitutability evidence.** Two tasks are fixed by *either* of two
different levers, independently:

- environment-hard-8 (beach datasheets): fixed by 5k rows (C1 B-only, 12→6
  steps) AND by stats (C2 B-only, 12→5 steps). Chronic-tagged, but the same
  convergence both times.
- legal-hard-15 (FTC duplicate MSA): fixed by 5k AND by stats — the one
  ATTRIBUTED flip in REPORT.md §5 (anchor coin-flips 242682/593524 around
  gold; either `duplicate rows: 359/764 (47%)` or the visible raw repetition
  settles it).

Different levers deliver the SAME missing fact at different prices —
evidence-delivery channels, not independent capabilities (the levers-report
synthesis, now with file-level predictors).

**Implication.** A static per-source router is plausible: read five numbers
per file (dup%, empty%, unnamed, max-null, str-share — all engine-computed
at load time) and set sampling depth + stats on/off per operator. Wide-dirty
string tables → rows or profile; weird formats (.gpkg/.cdf/.xlsx multi-sheet)
→ profile; header-less numeric formats with parse iteration → history;
clean tables → minimum everything.

---

## F3. Churn has a DAG signature: orphan-sink share — a tail flag, not a median separator

**Claim.** When the Latest arm fails by churning (rebuilding probes it has no
memory of), the failure is visible STRUCTURALLY in the final workflow: the
DAG fills with unconsumed leaf operators (sinks nothing reads). This is a
high-precision tail detector — it does NOT separate pass/fail at the median,
and claiming it as a general predictor would be wrong:

| population | sink-share med (pass) | sink-share med (fail) |
|---|---|---|
| Latest3kSchemaOnly, all tasks | 33.3% | 33.3% |
| Delta3kSchemaOnly (control) | 25.0% | 30.0% |

**The tail flag** — `sink-share ≥ 50% AND ops ≥ 8` (full population, both
arms):

| arm | flagged | pass-rate flagged | pass-rate rest | steps med | cost med |
|---|---|---|---|---|---|
| Latest3k | **7/103** | **29%** | 80% | **25** vs 6 | **$0.16** vs $0.03 |
| Delta3k (control) | 2/103 | 50% | 78% | 13 vs 6 | $0.07 vs $0.03 |

The seven flagged Latest tasks:

| task | sink-share | ops | steps | cost | outcome |
|---|---|---|---|---|---|
| wildfire-hard-17 | 76% (19/25) | 25 | 26 | $0.19 | FAIL (Delta: 9 ops, 13 steps, $0.09, PASS) |
| archeology-hard-1 | 65% | 20 | 26 | $0.24 | FAIL |
| environment-hard-17 | 62% | 13 | 15 | $0.10 | FAIL |
| archeology-hard-2 | 60% | 15 | 19 | $0.15 | FAIL |
| biomedical-hard-1 | 52% | 21 | 25 | $0.16 | PASS — churn below the failure line: Delta passes it in 7 steps / 9 ops / $0.05 |
| astronomy-hard-9 | 50% | 18 | 26 | $0.45 | FAIL (Delta: 7 ops, 10 steps, $0.11, PASS — the REPORT §6 exhibit: the identical column-scan probe rebuilt three times) |
| environment-hard-8 | 50% | 10 | 12 | $0.09 | PASS |

Reading: the flag fires 3.5× more often under Latest than Delta (7 vs 2 per
103) — rendered history *suppresses the churn mode*. Flagged tasks run 4×
the steps and 5× the cost of the rest, and even the flagged PASSES are the
expensive ones (biomedical-hard-1 is exactly the top of C3's
both-pass-Delta-cheaper cell). In the C3 Delta-only-wins cell the aggregate
Latest DAG is 56% sinks (32/57 ops) vs 23% on the Delta side.

**Implication.** Sink-share is computable at run time from the live DAG with
zero trace analysis. As a trigger (≥50% with ≥8 ops), it identifies
churn-in-progress: inject failure history, or surface a delete-unused-probes
nudge — the quantified motivation for the render-prefs delete principle and
a history-on-demand mechanism.

---

## F4. Cardinality collapses within 1–2 hops of the sources — all render pressure lives at the data edge

**Claim.** Dataflow DAGs here are steeply contractive: median operator output
falls two orders of magnitude within two hops, so the entire render-budget
question is about sources and first-hop transforms.

**Full-population evidence** (anchor arm, all rendered ops, n=583):

| depth | n | out-rows med | out-rows p90 | cells med |
|---|---|---|---|---|
| 0 (sources) | 265 | **153** | 16,086 | 1,212 |
| 1 | 169 | **39** | 5,873 | 153 |
| 2 | 94 | **1** | 275 | 3 |
| 3+ | 55 | **1** | 31 | 3 |

Per-case confirmation: across all 12 case cells of C1/C2/C3, rows@depth-2+
median is 1–5; answer sinks are 1×1 everywhere; reduce-share (ops emitting
fewer rows than they consume) runs 57–100%. The single systematic exception:
C3's Latest-only-wins cell, where interior cardinality is HIGH
(med 3,648 Delta-side / 8,491 Latest-side — concat/union pipelines over the
water-body files; Latest's reduce-share 43% vs Delta's 71%). Expanding
pipelines are exactly where re-rendering event history is most expensive and
a single current-state render wins — the mechanism behind Latest's exclusive
wins.

**Implication.** Per-operator render knobs only need to exist for depth 0–1
operators (sources + first transform); everything deeper is 1×1-to-tens and
costs nothing to render fully. A depth-aware default (full detail at the
edge, schema-only past depth 2) approximates the optimal policy with no
agent involvement.

---

## F5. Profiling is an anti-iteration lever: it pays on dirty/wide/weird inputs and is pure tax on clean tabular data

**Claim.** Column stats + table profiles don't make the agent smarter in
general — they *replace probe iteration* where the schema line is blind:
unnamed/string-typed tables and non-CSV formats. Everywhere else they are
rendered bytes the agent never uses.

**Evidence, tax side** (C2 both-pass): schema-only is materially cheaper on
**37 tasks** vs stats' 19 — the modal KramaBench task (files med 30KB / 187
rows, half fully clean; interior outputs med 126–144 rows) never consults a
profile. This is the case-level anatomy of the levers-report aggregate
(+13.9% total cost, +29% uncached input/step for always-on stats).

**Evidence, payoff side** (C2 both-pass, stats materially cheaper, 19
tasks): 11 of 23 distinct files are non-CSV (.gpkg ×2, .cdf, .lst ×2,
.xlsx ×5, .html) — and the schema-only arm posts its highest multi-edit
share anywhere in the grid, **32%** (vs stats' 20%), i.e. it iterates on
loaders where the profile answers in one render. Step counts:

- wildfire-hard-6: 12 → 6 steps ($0.10 → $0.05)
- archeology-easy-4: 13 → 7 ($0.07 → $0.05) — climateMeasurements.xlsx, where
  `headers: 29 of 30 columns are unnamed` + `empty rows: 2 of 8365` says
  "re-load with header=4" immediately
- astronomy-hard-8: 14 → 12 ($0.17 → $0.12, the 86,400-row .cdf)
- biomedical-easy-9: 11 → 6 ($0.05 → $0.04)

**Evidence, exclusive wins** (C2 stats-only, 6 tasks, 5 chronic): the
wide-dirty family again — 8/13 files unnamed-headed, 10/13 with ≥20%
max-null, str-share median **100%**, sizes med 39KB. Schema-only renders
these as `Unnamed: 1 (str), Unnamed: 2 (str), …` walls; the profile carries
the only usable semantics.

**Implication.** Deliver stats per-operator, triggered by measurable
conditions (str-share = 100%, unnamed > 0, non-CSV extension), not
always-on. That recovers the 37-task tax while keeping the 19-task payoff —
the render-prefs v2 targeting policy, now with its trigger conditions
measured.

---

## F6. Multi-edit share sorts every cell of the grid — iteration begets information value

The winner-side multi-edit share (ops with ≥2 landed code versions) across
all 12 cells:

| cell | C1 (3k vs 5k) | C2 (schema vs stats) | C3 (Delta vs Latest) |
|---|---|---|---|
| A-only wins (anchor-side share) | **31%** | 10% | **36%** |
| B-only wins (B-side share) | 5% | 3% | 8% |
| both-pass, A cheaper (A share) | 9% | 11% | 21% |
| both-pass, B cheaper (B share) | 19% | 20% | 10% |

Exclusive wins for the evidence-rich side coincide with heavy iteration
(31–36%: the .tle/.text/.dat/.xlsx parse-iteration tasks); exclusive wins for
the lean/stateless side with near-zero iteration (3–8%: clean single-shot
pipelines). Where nobody iterates, the cheaper context wins; where someone
iterates, the context that *carries* the iteration wins. Consistent with the
COMPARISON_PIPELINE sorter (24–36% vs 3–9%) measured there on different
cells — this is now confirmed on all twelve.

---

## Caveats (apply to every finding)

- Exclusive-win cells are 3–6 tasks, 75–100% chronic flippers — they locate
  *where a lever binds*; the only attributed accuracy flip remains
  legal-hard-15 (REPORT.md §5). The cost cells (19–37 tasks) are the robust
  populations, and all step/cost exhibits above come from same-vintage,
  recovery-equalized arms.
- Operator metrics are computed on the FINAL DAG: deleted probes appear in
  steps/cost but not in op counts (sink-share therefore *under*-states churn
  where the agent cleans up — biasing F3 against us, not for us).
- Source input-table cardinality comes from the stats arm's engine profiles
  (full-data facts), joined by file path; multi-input in-rows are summed.
- All cost figures are cache-aware `cost_usd` (raw token deltas mislead —
  see COMPARISON_PIPELINE "Known traps").

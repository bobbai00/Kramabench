# Where the headroom actually is — a replicate-grounded study of dynamic knob tuning

Date 2026-07-27. Substrate: gpt-5-mini dataflow, oracle mode, 7 knob configs
× 5 clean single-shot replicates = **35 full runs, 94 matched tasks,
3,290 task-runs**. Score = KramaBench-native answer-type metric (continuous).
Process metrics: M5/M6/M7 (materialization) and M10 v3 (per-step verdicts),
2,154 cells judged. Five mechanism dives on flipped cases.

The question this file answers: *is there room to raise accuracy and lower cost
by tuning knobs dynamically during a trace, and where exactly is that room?*

Short version: **yes, but not where the static A/B campaign pointed.** The
per-task knob choice is worth ~nothing once you cross-validate it. The room is
(a) **per-operator** knob decisions with data-observable triggers, (b) a
verification protocol for the coin class, (c) upstream benchmark fixes. Cost is
the easier win than accuracy.

---

## 1. The variance budget — what the score is actually made of

Decomposing every score cell (task × knob × rep), 7 knobs × 5 reps × 94 tasks,
**after excluding the quota-outage scoring cells described in §2a**:

| source | share of total variance |
|---|---|
| **task identity** | 63.2% |
| **replicate noise (same task, same knob)** | **27.6%** |
| knob choice within a task | 9.2% |

Replicate noise is **3× larger than the entire knob effect**. Every
single-run A/B in the campaign was measuring a channel that carries a tenth of
the variance with an instrument whose noise carries a quarter.

Per-knob stability, **5 reps each, null-judge cells excluded** (n=94):

| knob | mean | std | flipped |
|---|---|---|---|
| anchor Delta1k | 0.617 | 0.025 | 38.3% |
| **C1 Delta5k** | 0.660 | 0.033 | **28.7%** |
| C2 DeltaStats1k | 0.633 | 0.018 | 36.2% |
| C5 DeltaStats2k | 0.658 | 0.029 | 36.5% |
| C4 DeltaStats5k | 0.649 | 0.036 | 35.1% |
| **C3 Latest1k+code** | **0.673** | 0.038 | 33.0% |
| C6 LatestStats1k | 0.626 | 0.013 | 44.2% |

flipped = any score difference across reps. Cost, 5% two-sided trimmed,
task-first: $0.0132–0.0147/task, C1 cheapest. Note the knob ranking is *not*
stable against the correction — at 3 contaminated reps C5 led at 0.668; at 5
clean reps C3 leads at 0.673 with C1 and C5 within 0.015. The knobs are a
0.05-wide cluster, which is the whole point of §1.

---

## 2. Outcome classes — the population you are actually optimizing

Every task, 35 runs, pass = score ≥ 0.5 (94 matched tasks):

| class | n | share of the score deficit |
|---|---|---|
| stable_pass (35/35 pass) | 28 | 3% |
| mostly_pass (≥80%) | 22 | 8% |
| **coin (20–80%)** | **26** | **28%** |
| mostly_fail (≤20%) | 5 | 16% |
| **never_pass (0/35)** | **13** | **45%** |

**Three-quarters of the remaining loss sits in tasks that are either
never-solved (45%) or non-deterministic (28%).** The classic knob question —
"which static rendering config" — operates on the 9.2% variance slice and
touches neither.

### 2a. Measurement contamination — a scoring OUTAGE, not judge sampling

An exhaustive audit of all 41 arms (4,222 scored cells): group every cell by
(task, normalized answer) and look for identical answers scored differently.
**10 groups across 8 tasks, every one `string_approximate`.** All the usual
suspects — `'wollaston beach'`, `'district of columbia'`, `'true'`,
`'great basin area'`, `'u.s. space force'`.

The discordance is **not** per-call sampling noise. It clusters perfectly by
arm: `anchor/rep1` scores 6/6 of them low, `anchor/rep0` scores 0/8 low. Root
cause, from `evaluation.json`:

```json
"llm_paraphrase": null,        // Replicate1/2, scored 2026-07-24 11:26
"token_usage_metrics": 0       // the judge call never completed
```

vs the same task/answer in Replicate0 (scored 07-25 18:52): `"llm_paraphrase": 1`,
`token_usage_metrics: 484`. Timestamp 07-24 11:26 is exactly the documented
OpenAI quota exhaustion window (`VARIANCE_REPORT.md`: "quota exhausted mid-run,
~11:00 2026-07-24"). `GPTInterface.evaluate_paraphrase` swallows the exception
and returns `None`; the harness stores `null`; every downstream reader treats
it as **0**.

Scope: **100 cells, 10 tasks × 10 arms** — Replicate1 and Replicate2 of anchor,
C1, C2, C4, C3 (the five arms scored during the outage). C5/C6's reps were
scored 4 hours later and are clean; all code-agent arms are clean.

Impact of treating `null` as missing rather than 0:

| knob | mean before → after | flipped% before → after | std before → after |
|---|---|---|---|
| anchor Delta1k | 0.602 → 0.617 | 43.3% → 38.3% | 0.051 → **0.025** |
| C1 Delta5k | 0.642 → 0.660 | 34.6% → 28.7% | 0.048 → 0.033 |
| C2 DeltaStats1k | 0.617 → 0.633 | 41.3% → 36.2% | 0.035 → **0.018** |
| C4 DeltaStats5k | 0.633 → 0.649 | 40.4% → 35.1% | 0.066 → 0.036 |
| C3 Latest1k+code | 0.655 → 0.673 | 38.5% → 33.0% | 0.056 → 0.038 |
| C5 DeltaStats2k | 0.666 (clean) | 36.5% | 0.029 |
| C6 LatestStats1k | 0.631 (clean) | 44.2% | 0.013 |

The contaminated arms' apparent replicate std **halves**. A large part of what
the variance study called "run-to-run randomness" in those five knobs was a
scoring outage. Two fixes, both cheap:

1. **Never let a failed judge call score as 0.** `metrics.py`/harness should
   propagate `null` as *unscored* and the aggregator should exclude the task,
   the way `kb` now does. A silent 0 on API failure is the single most
   dangerous defect in the measurement stack.
2. **Make the paraphrase verdict deterministic and cached.** The call at
   `benchmark/llm_tools/gpt_interface.py:110` passes no `temperature` and no
   `seed`, so it runs at OpenAI's default temperature 1.0. Set `temperature=0`,
   and memoize per normalized `(predicted, target)` pair so a pair is judged
   once. (Related latent bug at `gpt_interface.py:128-131`: the verdict is
   extracted by substring — `"no" in answer` fires on "not", "none", "cannot".)

Re-score those 100 cells before publishing any variance number.

---

## 3. The central negative result: history-based per-task routing is dead

An oracle that picks the best knob per task using observed rep-means reads
**+0.105 accuracy at −12% cost**. That number is an artifact. Cross-validate
it — pick the knob on half the reps, score on the held-out half — and:

| router (7 knobs × 5 reps, corrected matrix) | held-out score |
|---|---|
| best static knob (C3 Latest1k+code) | 0.673 |
| CV router, always route | 0.665 |
| CV router, only when select-half margin ≥ 0.2 | 0.679 |
| **NULL (knob labels destroyed, same procedure)** | **0.645** |
| real knob-identity signal (router − null) | +0.020 … +0.035 (router sd 0.015) |
| **router − best static** | **+0.006, i.e. nothing** |

Read this carefully, because the two comparisons say different things. Knob
identity *does* carry a little real information (+0.03 over a label-destroyed
null, ~2σ). But the act of selecting a knob from noisy history **costs** about
as much as that information is worth: the null sits 0.028 *below* the static
default, because "pick whichever knob looks best on the observed reps" reliably
picks noise. Net, the router lands on top of the static default, +0.006.

**You cannot beat picking one good static config by remembering per-task
performance.** Inspecting which tasks a large-margin router fires on:
`legal-hard-18` (helps 54% of firings, hurts 0%), `astronomy-hard-9` (helps
77%), `archeology-hard-7` (helps 84%) — a genuine handful — against a tail of
chronic coins that happened to land high on the select half and get hurt.

Only **6 of 98 tasks** show a clean knob separation (one knob's worst rep beats
another's best rep by ≥0.5): environment-easy-3, legal-easy-4, legal-hard-18,
legal-hard-29, environment-hard-12, biomedical-hard-8.

**Corollary:** a router must be driven by *observable signals in the data*, not
by remembered per-task performance. That is the DYNAMIC_KNOBS thesis, and this
study is its quantitative justification — the alternative has now been measured
at zero.

### The target is per-STEP context assembly, not per-run config

The unit that matters is one ReAct step: at the moment the engine renders an
operator's result into the next prompt, it decides that operator's sampling
size, whether to attach a column profile, which prior version of the data and
of the code to carry forward. Corpus shape for that controller (3 arms × 102
tasks, 1,418 operators): **4.63 operators per task, 54% of them at DAG depth 0**
— the data edge, where every dive's decisive moment lives. Depth ≥2 is 8% of
operators and, per the earlier F4 finding, carries almost no render pressure.

So the controller has ~5 decisions per task, ~2–3 of which are load-time, and
the evidence in §4 says each one wants a *different* setting. Section 5's
step-0 divergence result sharpens it further: the first operator's render is
the highest-leverage single decision in the trace.

### Brute resampling is not the answer

Realizable strategies simulated by resampling actual runs (no ground truth in
the loop, answers compared by normalized string):

| strategy | acc | $/task | $/pass |
|---|---|---|---|
| static C5 DeltaStats2k | 0.687 | $0.0150 | **$0.0218** |
| static C1 Delta5k | 0.658 | $0.0144 | $0.0219 |
| vote-3 same knob (C3) | 0.706 | $0.0477 | $0.0676 |
| cascade C5→C3→C1 (agree-or-third) | **0.723** | $0.0370 | $0.0511 |

Cascades buy +0.036 accuracy for 2.5× the money. **Cost per pass gets 2.3×
worse.** Self-consistency is a real accuracy lever here (best-of-5 reps reaches
0.785–0.826) but it is the expensive way to buy it.

---

## 4. Where the room is: five mechanism dives

Each dive pairs a passing and failing trace and identifies the decisive
context difference. All five yielded a trigger computable at operator
write/render time.

### D1 · environment-easy-3 — stats **poison** (15/15 pass schema-only, 1/12 pass with stats)
Gold 268, stats arms answer 267. The profile line
`"Beach Name" (str): null=0, distinct=1155, duplicate_values=14222`
reads as *"1155 beaches, so name is the entity key"*. It is not — 14 beach
names recur across towns. The agent grouped by name alone; the schema-only arm,
lacking the number, hedged with a composite key `[Beach Name, Community,
County, Year]` and hit gold. The error then **self-confirmed**: the next render
showed `1155 rows` + `distinct=1155`, a perfect 1:1, so it was never revisited.
→ **Rule:** never render bare marginal cardinality on a column about to be
grouped/joined when sibling qualifier columns exist (`*Code`, `Community`,
`County*`). Render joint-key candidacy instead: `Beach Name → 1155 groups;
+Community → 1169 (not unique alone)`.

### D2 · legal-easy-4 — rows **rescue** (0/5 at 1k, 5/5 at 5k)
The 1k render elided immediately after the first data row, so the only web
label the agent ever saw was `FTC - Web Reports (IDT)`. The 5k render showed
row 4: `FTC - Web Reports (Fraud & Other)` — proving the label is a shared
prefix. The 1k arm used `str.contains('FTC - Web Reports')`, summed 6 rows
instead of 3, answered 4,391,927 vs gold 2,111,635. It used **3 of 25 allowed
steps** and never probed. Column stats did *not* substitute (they give
`distinct=45`, never the values).
→ **Rule:** substring predicate whose literal is a strict prefix of a rendered
value, on a column whose distinct set was never fully shown → force one
selectivity check (matched-row count) before accepting a terminal aggregate.

### D3 · legal-hard-18 — budget primary, **delta** essential (stats2k 3/3, anchor 0/5, latest 0/3)
Needs three scalars from two files. The decisive row `Identity Theft |
1,135,291 | 17.54%` is row 3 of a 35-row table — elided at 1k. Latest-mode
fails for a different reason: it re-renders each operator's *current* result at
the budget every turn, so the one-time step-2 observation is overwritten.
Counted directly: `"17.54"` appears **0 times** in the latest arm's final input
message, **4 times** in the passing delta arm's.
→ **Rule:** multi-scalar-across-files/ratio queries must stay delta.
Raise budget when a *small* table (< ~50 rows) shows elision and its stats
carry `format=percent`/`format=thousands` — a label-keyed lookup row is hidden.

### D4 · biomedical-hard-8 — rows **harm**, monotone dose–response (stats1k 5/5, stats5k 0/4)
The predicate lives in the *sheet name* (`G-FDA approved drugs`, all 375 rows
qualify). A 5k render exposes the value vocabulary of
`interaction_claim_source`, which contains a literal `FDA` — so the agent
re-applies the predicate as a row filter, collapsing 39 genes to 4, answering
`[]`. Frequency of that wrong filter rises monotonically with the char budget:
1/6 at 1k, 2/4 at 2k, 4/5 at 5k, 6/6 at schema-only-5k. Across 59 runs: filter
present → 20 fail / 5 pass; absent → 32 pass / 2 fail.
→ **Rule:** cap rows when (i) the container name already encodes the
question's predicate *and* that token also appears as a value in a
low-cardinality column, or (ii) the next operator is `isin`/set-membership
against a reference list — distinct-count suffices, rows are noise.

### D5 · biomedical-easy-9 — stats **stabilize a coin** (schema-only 2/5, stats 6/6)
Sibling columns `FDR.phos` and `FDR.prot`; the query names "FDR" unqualified.
Schema-only arms pick at random. The stats fingerprint
`"FDR.phos": min=3.5e-05, max=0.05` vs `"FDR.prot": min=0.05, max=1` marks the
pre-filtered significance column unambiguously. Controlled: stats arms that
made the *same* load mistake as the failing arms still chose correctly.
→ **Rule:** ≥2 same-prefix/synonym candidate columns for one query term →
stats ON for that operator.

### D6 (control) · astronomy-hard-8 — a coin **no knob touches**
Passing and failing reps of the *same* config produce byte-identical merged
data (21 rows). The only divergence: `train_n = int(n*2/3)` → test 7 (pass) vs
`test_n = int(n*0.3)` → test 6 (fail). A rounding-direction coin flip on an
underspecified 70/30 split. Nothing in the render space reaches it; only a
spec change does. (A *sibling* failure in rep 4 — picking an all-zero CDF
variable — **is** catchable by a plausibility check.)

**The pattern across D1–D5: the same knob is right and wrong in the same run.**
Stats rescue D5's column pick and poison D1's key choice. Rows rescue D2's
predicate and poison D4's filter. That is the actual argument for dynamic
tuning — not "pick the best config per task", which we just measured at zero,
but **"pick per operator, from what that operator's data looks like."**

---

## 5. Where randomness comes from, and whether context reaches it

**Divergence is at the beginning, not the end.** For the 31 coin tasks identified in the
first (uncorrected) pass, comparing a passing and a failing rep of the same
config, the first differing operator was:

| position | count |
|---|---|
| **step 0 (the very first authored operator)** | **29 / 31** |
| mid-trace | 2 / 31 |

The coin is decided when the agent writes its first loader/plan, before any
observation exists. This kills the "richer downstream rendering fixes the coin"
theory and explains why every render knob's flip-set overlaps only ~43%
(pairwise Jaccard median) — each config reshuffles the same lottery rather than
removing it.

**More steps is not the cure.** Pass rate by trace length: 0.70 (1–2 steps),
0.65 (3–4), 0.61 (5–7), 0.49 (8–12), **0.30 (13+)**. Traces exceeding 1.5× the
gold subtask count pass at 0.47 vs 0.65 for fused ones. Length marks trouble;
it does not buy correctness.

**The trace does carry a usable distress signal — but the current judges cheat.**
Within coin cells only (n=649), the process metrics separate pass from fail
sharply:

| metric (coin cells, low vs high tercile) | pass rate low | pass rate high |
|---|---|---|
| M5 value-materialization | 0.28 | 0.67 |
| M7 (visible or fused) | 0.29 | 0.74 |
| M10 useful-step fraction | 0.27 | 0.72 |
| M10 wrong_param fraction | 0.74 | **0.28** |
| M10 thwarted | 0.50 | 0.50 |
| M10 redundant | 0.51 | 0.50 |

A ±0.45 spread inside the noise class means **the information needed to tell a
doomed run from a good one is present in the trace itself.** Caveat that
matters: M5 and M10 are gold-anchored, so this is an upper bound on
detectability, not a deployable detector. It licenses building an
outcome-blind proxy (M9's grounding/waste shape) and gating a re-run on it —
the cheap targeted version of the cascade in §3, spending the 2.5× only on the
~30% of runs the detector flags.

Note also which verdicts *don't* separate: `thwarted` (engine friction) and
`redundant` (looping) are flat at 0.50. Dataflow's characteristic failure
modes are not what decides its coins — **wrong_param, a decision error, is.**

---

## 6. What kind of task is winnable

Pass/coin/fail rates by task property (PASS = stable+mostly_pass):

| property | PASS | COIN | FAIL |
|---|---|---|---|
| wildfire | 76% | 24% | **0%** |
| legal | 57% | 30% | 13% |
| environment | 45% | 40% | 15% |
| archeology | 33% | 17% | **50%** |
| astronomy | 22% | 44% | 33% |
| biomedical | 17% | 50% | 33% |
| easy | 62% | 26% | 12% |
| hard | 41% | 36% | 23% |
| `string_exact` | **80%** | 15% | 5% |
| `numeric_approximate` | **100%** | 0% | 0% |
| `numeric_exact` | 43% | 25% | **33%** |
| `string_approximate` | 0% | **90%** | 10% |
| `list_exact` | 55% | 45% | 0% |

Reading:
- **The answer type predicts outcome better than any knob.** `numeric_exact`
  (49 tasks, no tolerance) carries 33% hard failures; `numeric_approximate`
  (1% tolerance) passes 100%. Several "failures" are near-misses under an
  exact-match gate — astronomy-easy-3's 0.9% gap is the known case.
- **Source count doesn't matter** (1 source 55% PASS, 4+ sources 60%), and
  neither does subtask count. Dirtiness and gold-convention specificity do —
  archeology (metadata-header xlsx, stacked sub-tables) is the failure pole;
  wildfire (clean CSVs) has *zero* hard failures.
- Domain rank is stable against the earlier gpt-5.2 campaign, and 11 of the 31
  coins are the same tasks that flipped at gpt-5.2 — **chronic across models**,
  which is what you'd expect if the coin is decided at plan time by an
  underspecified question rather than by rendering.

---

## 7. Answers to the three questions, and what to build

**Q1 — which statistics indicate room?**
1. `flipped%` per config (26–35%) is the honest measure of unclaimed accuracy;
   `floor` (per-task min over reps) is the pessimistic operating point and
   sits 0.12–0.22 below the mean. Report both, never single-run deltas.
2. Variance shares (task / knob / rep) — if rep share > knob share, an A/B is
   uninformative at any single-run sample size. It currently is, 2.8×.
3. CV-router-minus-null — the only trustworthy estimate of routing headroom.
4. Within-coin process-metric spread (M7, wrong_param) — measures whether a
   controller *could* see trouble. ±0.45 says yes.
5. Same-answer-different-score rate — measures judge contamination. Currently
   8/31 coins.

**Q2 — what is the randomness, and is it knob-reachable?**
It is **first-operator plan variance**: 29/31 coin divergences occur at step 0,
on choices the data does not disambiguate (which of two sibling columns, which
grouping key, which split convention). Rendering knobs act *after* that choice,
which is why they reshuffle rather than remove the coin, and why per-task
routing measures zero. Two things do reach it: (a) making the disambiguating
fact present *before* the first operator is authored — that is exactly what
D1/D5's column-level signals do; (b) a verification step — D2's missing
selectivity check, D6's all-zero-target plausibility check. Neither is a knob
setting; both are policies conditioned on operator content.

**Q3 — task characteristics and knob affinity?**
Affinity exists at the **column/operator** level, not the task level:
stats pay when the operator faces ambiguous sibling columns or dirty headers
(D5, S2) and cost accuracy when they publish bare cardinality on a key column
(D1); rows pay when a small table's discriminating row is being elided (D2,
D3) and cost accuracy when the render exposes a value vocabulary the agent can
mis-filter on (D4). Task-level correlates that do hold: exact-match numeric and
dirty-spreadsheet domains carry the failures; clean-CSV domains carry none.

**Build order (each independently measurable):**
1. **Pin the paraphrase judge** (cache per normalized answer). Removes 8 fake
   coins and un-contaminates every stability number. Hours of work.
2. **Ship the four rules D1/D2/D4/D5 as engine-side per-operator render
   policy** — joint-key candidacy instead of bare distinct; row-budget raise on
   zero-margin elision of a small table; row-budget cap on
   container-name-encodes-predicate and on set-membership next-ops; stats on
   sibling-candidate columns. All append-only, all cache-safe, all fire on a
   trigger so they cost nothing on the ~90 tasks that don't need them.
   Expected: cost recovery of the blanket stats tax, plus the D1/D4 poison
   removals, which are *deterministic* flips, not lottery tickets.
3. **Outcome-blind distress detector → single targeted re-run.** The gated
   version of §3's cascade: spend the 2.5× only where the detector fires.
   M9-shaped features, validated against the ±0.45 within-coin spread.
4. **Do not** build a per-task knob router from run history. It is measured at
   zero and it actively hurts chronic coins.

**Ceiling check.** With 11 never-pass tasks (3 of them known benchmark defects —
missing Kaggle file, `omni2.txt`/`.text` manifest typo, unseeded gold script)
and ~4 gold-private conventions, the realistic dataflow ceiling remains ~93/104,
consistent with `NEVER_SOLVED_CEILING.md`. Against a 0.688 static operating
point, the addressable band is roughly **0.69 → 0.80**, and most of it is
bought by removing wrong context, not by adding more.

---

### Provenance
Matrix + all statistics: `judgment_runs/mini_star/{rebuild,judgeaudit,matrix,features,quant2,diverge,predictors,cascade,honest2}.py`;
score matrix and outcome classes cached as `score_matrix.json` / `classes.json`
in the same directory. Stability metric: `kb.py stability` (rev 2026-07-27 —
flipped = any score difference across reps; std over per-rep means; mid-run
arms auto-skipped).

`rebuild.py` is the corrected loader: a task whose answer-type metric is JSON
`null` (the judge call failed) is **excluded**, not scored 0. Everything in
§1, §2, §3 and §6 is computed from that matrix — 7 knobs × 5 reps × 94 tasks.
§4's dives, §5's divergence localization and the process-metric terciles were
computed before the correction and are unaffected by it (they read traces and
judge caches, not the paraphrase metric). Section 5's coin list is the
pre-correction one (31 tasks); the corrected class count is 26.

Dives: 5 trace-pair investigations plus one control, mechanisms confirmed by
re-execution or direct token counts in the traces.

# The Levers of Dataflow Context: Sampling, Profiling, History

2026-07-12. All arms GPT-5.2, oracle mode, no compression, July-6/7 vintage,
and — per the recovery-first protocol — freshly equalized with symmetric
2× `--all-failed` recovery rounds before any comparison (star-recovery run
2026-07-12).

## 1. The levers

A dataflow ReAct agent's context is a rendered view over canonical,
versioned, re-executable state. That gives three independently controllable
levers a flat script agent does not have:

1. **Code versions** — every operator's edit history is recorded; the render
   can expose it (DELTA) or fold it into the current DAG (LATEST).
2. **Data versions** — every operator's result history across re-executions,
   likewise exposable or foldable.
3. **Result rendering** — per-operator **sampling** volume (char budget) and
   **profiling** (per-column statistics + Output Table profile), independent
   of what the backend computes (downstream operators always receive full
   data).

## 2. Design: a star of matched pairs

ANCHOR: `Delta3kSchemaOnly` (delta, 3k result chars, schema line only).
Each ray moves one thing:

| Ray | Arm | Lever isolated |
| --- | --- | --- |
| C1 sampling | `Delta5kSchemaOnly` | result char budget 3k→5k (single knob) |
| C2 profiling | `DeltaStats3kD2` | + column stats + Output Table profile (two rungs, one facet) |
| C3 history | `Latest3kSchemaOnly` | versions exposed → folded |

## 3. Method

- Pass = answer-type metric ≥ 0.9.
- **Twin-noise floor** (three same-config rerun pairs): 9–12 flips per
  IDENTICAL pair, ±3 net passes, ±10% cost; **23 chronic-flipper tasks**.
  A flip is credited to a lever only if the task is non-chronic AND the
  traces show the lever's information doing the work.
- **Recovery-first**: transient failures (watchdog kills, one-off spirals)
  converted to stable outcomes before comparison.
- Cost analysis uses the both-pass **same-answer cohort** with per-step
  invariants (totals are day-noise-prone; per-step rates are stable).
- These sweep arms carry empty thought contents (tool-calls only), so
  attribution reads operator code + rendered observations.

## 4. Recovered aggregates (103 shared tasks)

| Arm | Pass | Cost | Steps |
| --- | ---: | ---: | ---: |
| Anchor (Delta 3k schema-only) | 80 (77.7%) | $5.29 | 716 |
| C1 · 5k sampling | 81 (78.6%) | $5.47 | 699 |
| C2 · stats+profile | 83 (80.6%) | $6.28 | 726 |
| C3 · Latest | 79 (76.7%) | $5.31 | 829 |

Every accuracy delta (±1–3) sits at or inside the twin-noise band — the
aggregate view cannot see these levers. The information is in the cases.

## 5. Accuracy flips: 27 mined → 1 attributed

Post-recovery the star yields 27 direction-flips; 25 involve chronic-flipper
tasks and default to variance. The two survivors:

### ATTRIBUTED — `legal-hard-15` (wins in BOTH C1 and C2; the star's one real accuracy case)

Task: total identity-theft reports across cross-state MSAs (gold 243377).
The FTC source file lists every multi-state MSA once under EACH member
state's section — 47% of rows are duplicates by construction. One decision
decides the task: are repeated MSA rows legitimate duplicates to drop, or
distinct data?

Three evidence regimes, one decision:

| Arm | Duplicate evidence visible | Behavior | Answer |
| --- | --- | --- | --- |
| Anchor (3k, schema-only) | none (sample too short to show the per-state repetition; no stats) | **unstable across its own reruns**: over-dedup (242682), then no-dedup (593524) — two different wrong answers bracketing gold | fails twice |
| C1 (5k) | raw repetition visible in the wider sample (cross-state `Columbus, GA-AL` inside the AL section; repeated headers at row 758) | dedup at the total step | **243377** |
| C2 (stats+profile) | one profile line: `duplicate rows: 359 of 764 (47%)` + `duplicate_values=204` on the MSA column | dedup at the CLEAN step (earliest correct placement) | **243377** |

Converging causation: the same missing fact, delivered by either lever
(≈2,000 chars of extra sample, or 3 lines of profile), flips the same
failure to the same correct answer. The anchor's own run-to-run
instability on exactly this decision is the strongest form of the
attribution: without evidence, the choice is a coin flip.

### REJECTED — `environment-easy-3` (C3, Latest won 268 vs 267, gold 268)

Off-by-one from a join-key choice: Delta joined beaches on name only,
Latest on `(Community Code, Beach Name)` — one beach-name collision across
communities. The key choice was made at operator-creation time, before any
history existed in either context; no mechanism connects it to the history
lever. Modeling variance.

**C3 verdict on accuracy**: 0 attributable flips post-recovery among the
DIRECTLY compared star arms — consistent with the pre-recovery audit's 0/26.

### C3 mechanism exhibit — the thrash blind spot, directly observed (`astronomy-hard-9`)

The blind-spot prediction: LATEST folds superseded attempts away, so the
agent loses the OUTPUTS of its own past probes and repeats them. Task:
parse the fixed-width OMNI2 space-weather file's Ap column and find the
lag maximizing correlation with TLE-derived drag (gold: 24).

- **Delta (schema-only)**: 5 parsing submissions across 4 steps
  (spec ×2, raw ×1, ap ×2) — each attempt's OUTPUT stays inline in the
  trajectory; the agent never repeats a probe. 9 steps, answer 24, PASS.
- **Latest (schema-only)**: ground `omni2_spec` for SIX consecutive steps,
  then later submitted `omni2_ap_colscan` with numerically IDENTICAL
  column-slice code at steps 15, 16, 17, and a one-constant variant at
  18, 19, 20 — re-running a scan whose results it had already produced,
  because the superseded scans' outputs were no longer in view (the
  folded snapshot shows only the current version + latest result;
  attempt-reflection supplies edit counts, not past outputs). It exhausted
  all 25 steps and returned NO answer.

Caveat and override: astronomy-hard-9 is in the chronic set, so its
pass/fail alone proves nothing — but the twin gate permits trace override,
and identical-probe repetition is the mechanism itself, directly visible.
The arm-wide signature points the same way: Latest takes +13–18% more
steps than Delta in both C3 pairs, and the step-asymmetry tail is
one-directional (archeology-hard-5 7→19 with the parse-op ensemble
recreated at steps 9/10/11/12; astronomy-hard-8 7→15;
astronomy-hard-10 9→16; wildfire-hard-19 8→25).

**The counter-mechanism exists too** (from the pre-recovery audit):
`environment-hard-13` — Delta rewrote its tidy operator four times and then
answered from a STALE value carried in its own visible history ("returns
stale 12"). History prevents re-derivation but enables answering from
superseded state; the folded view prevents staleness but forces
re-derivation. Neither dominates on accuracy (0 net attributable); the
trade shows up as C3's cost structure (steps vs bytes, §6).

## 6. Cost: the both-pass same-answer cohorts (recovered)

| Ray | Cohort | Cost anchor→ray | Per-step input | Per-step uncached |
| --- | ---: | --- | --- | --- |
| C1 · 5k | 75 | $3.017 → $3.049 (**+1.0%**) | 7,999 → 8,176 | 1,316 → 1,477 |
| C2 · profile | 75 | $3.210 → $3.655 (**+13.9%**) | 8,175 → 9,360 | 1,360 → 1,757 |
| C3 · Latest | 73 | $2.612 → $2.686 (+2.8%) | 7,732 → **7,387** | 1,126 → **943** |

- **Sampling is nearly free at the margin** (+1.0%; anchor cheaper on only
  39/75). The 3k→5k budget only materializes on tasks whose tables actually
  exceed 3k — most render identically.
- **Profiling is the expensive lever**: +13.9% on identical-outcome tasks,
  +29% uncached input per step. It decorates every operator at every event —
  *paying everywhere to help somewhere* (its one decisive line in
  `legal-hard-15` is bought with ~$0.45 of profile text across the cohort).
  This is the measured motivation for targeted/per-operator profiling
  rather than an always-on flag.
- **History restructures cost rather than adding it** (at 3k schema-only):
  Latest runs MORE steps (490 vs 440, +11%) that are each LEANER
  (−4% input/step, −16% uncached/step). Delta = fewer, fatter steps
  (carrying history per step); Latest = leaner steps, more of them. At this
  operating point they net out (+2.8%); at stats-bearing operating points
  the audit found Delta's per-step weight dominating (+30% at identical
  steps). The history lever's cost sign depends on the data levers — the
  levers interact.

## 7. Interactions (secondary, mirrored pairs — pre-recovery, indicative only)

The mirrored (Latest-side) pairs were not re-recovered; directionally they
showed the levers are mode-dependent (profile bundle: +2.8 passes on Latest
vs 0 on Delta at 3k pre-recovery; sampling: +4 on Delta-stats vs −1 on
Latest-stats). Treat as hypotheses; the star is the controlled result.

## 8. Synthesis

1. **On oracle-mode KramaBench, the dataflow context levers are
   evidence-delivery channels, not accuracy dials.** After recovery and
   twin-noise filtering, 27 flips reduce to ONE attributable case — and in
   it, two different levers deliver the same missing fact. Aggregate
   accuracy is flat across the star; accuracy sensitivity to context lives
   in exploration settings (see explore_mode/analysis.md), not here.
2. **The levers are partial substitutes with very different price points.**
   A data-quality fact (duplication) can arrive as raw sample bytes (5k:
   ~free arm-wide but bulky per fact) or as profile lines (3 lines per
   fact, but +14% arm-wide because always-on). The efficient frontier is
   targeted delivery — per-operator, per-moment — which is exactly the
   write-time/pull direction of the render-prefs and inspect experiments.
3. **History (DELTA) buys neither accuracy (0 attributed) nor, at lean
   operating points, does it simply cost more — it trades step count for
   step weight.** Its liability is regime-dependent: neutral here,
   accuracy-harmful under exploration (dead-end replay), cost-harmful when
   per-step decorations are rich.
4. **Method matters more than any single number**: 85%+ of naive A/B flips
   on this benchmark are reproducible noise (23 chronic tasks; ±3 passes,
   ±10% cost between IDENTICAL configs). Recovery-first + twin-noise
   calibration + trace attribution is the minimum standard for claiming a
   context effect — and it reduced this entire star to one clean,
   mechanically-explained accuracy case and three clean cost mechanisms.

Artifacts: flips_recovered.json, chronic_flippers.json (this dir);
star-recovery logs logs/star-recovery-20260712_101634; case traces in
system_scratch/<arm>/legal-hard-15 and .../environment-easy-3.

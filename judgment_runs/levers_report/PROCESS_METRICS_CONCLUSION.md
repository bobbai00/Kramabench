# Extending the eval space for dataflow knobs — design, trials, conclusion (2026-07-20)

Goal: can we extend KramaBench's eval beyond binary accuracy so the dataflow
context knobs (rows / stats / history) and their combinations show measurable
effectiveness? Built four metrics, validated each against hand-labeled
semantic-walk verdicts, and stress-tested. Scripts: `scripts/{m1_plan_coverage,
m2_intermediate_coverage,m4_evidence_delivery,validate_metrics}.py`.

## The four metrics tried

| id | measures | source | cost | LLM? |
|---|---|---|---|---|
| (a) continuous mean | final answer, graded [0,1] | eval scores on disk | free | no |
| M1 | plan-step coverage in the agent's CODE | trace code | free | no |
| M2 | gold intermediate answers materialized (re-executed DAG) | re-exec | slow, fragile | no |
| M3 | per-step correctness, judged against gold answer | trace code | ~$0.05 | yes (gpt-4o-mini) |
| M4 | gold-step facts DELIVERED in the rendered context | trace observations | free | no |

## Validation — attribution (do flip gaps separate ATTRIBUTED wins from coins?)

Ground truth = the 3 probe-star ATTRIBUTED flips (legal-hard-2 rows,
biomedical-hard-5 stats, environment-hard-12 history) vs ~24 chronic coins,
winner/loser known (`validate_metrics.py`).

| metric | mean ATTRIBUTED gap | mean chronic \|gap\| | signal/noise | verdict |
|---|---|---|---|---|
| M1 (code) | +0.037 | 0.104 | 0.36 | **fails** (attributed within noise) |
| M2 (values) | +0.211 | 0.132 | 1.60 | weak/partial |
| M4 (delivery) | +0.149 | 0.095 | 1.58 | weak/partial |
| M3 (LLM semantic judge, gpt-4o-mini) | −0.037 | 0.071 | **−0.52** | **fails worst** (wrong direction) |

**No artifact metric — not even a semantic LLM judge — cleanly attributes
individual flips.** M3 was the decisive test: if semantic reasoning over the
code could catch the interpretation coins the token/value metrics miss, the
conclusion would change. It does not. M3's three ATTRIBUTED gaps are −0.11 / 0 /
0 (all ≤ 0), while chronic coins reach +0.50 — a NEGATIVE signal/noise ratio.
Why: environment-hard-12 (history) is 0/0 because the two arms' CODE is
identical (only the retained label differed — invisible to a code reader);
biomedical-hard-5 is 0/0 because the judge got route-confused (gold routes via
mmc7) and missed the code-visible `Case_excluded` filter that even M1 caught.
The judge adds noise, not signal. Chronic coins
produce artifact gaps as big or bigger than real knob wins (chronic
wildfire-hard-17 M4 +0.41 > attributed environment-hard-12 +0.04; chronic
legal-easy-9 M1 −0.62). M1 fails outright (the two evidence-driven attributions
— legal-hard-2 row-window, environment-hard-12 history — leave near-identical
CODE, so a code metric is blind: legal-hard-2 M1 gap = −0.06). M2/M4 do better
(~1.6× signal/noise) because they see value/render differences, but the
distributions overlap — not a per-flip classifier.

## Trial-and-error findings (what broke, what we learned)

1. **Re-executing agent code (M2) is fragile** — one agent op can spin at 100%
   CPU with no natural timeout; needed a per-task subprocess timeout guard.
2. **M2 is route/fusion-confounded.** Value-matching intermediates against ONE
   gold decomposition penalizes correct-but-different routes and fused
   pipelines: across a 30-task sample, 8% of *passing* arms scored near-zero M2
   (e.g. astronomy-easy-6: all arms PASS, M2 ≈ 0). Pass-vs-fail still separates
   (0.47 vs 0.29), but the confound + slowness rule it out as a primary metric.
3. **M1 is coin-blind by construction.** Knob wins act on the agent's EVIDENCE,
   not its code; two arms write near-identical code and diverge on
   interpretation (L2 vs L∞) or a retained label — invisible to a code metric.
4. **(a) is nearly inert.** 84/104 tasks are exact 0/1, so the continuous mean
   sits +1–3% above pass-rate and does nothing for the coin problem.

## The result that answers the goal — separate the LAYERS

The knob's causal chain is: **knob → evidence delivery (deterministic) → agent's
use of it (stochastic) → answer (coin).** Measuring at the wrong layer explains
every negative above.

- **Answer layer (accuracy):** coin-dominated. Aggregate knob gaps ≈ noise; the
  real accuracy effect is *conditional* — stats/budget recover ~5–11 specific
  tasks (the 82→93 oracle-union gap), concentrated, washed out by averaging.
- **Artifact layer (M1 code / M2 values):** no clean signal. The knob effect is
  either invisible (same code/values) or indistinguishable from chronic variance.
- **Delivery layer (M4):** a REAL aggregate signal — richer configs surface
  more gold source-facts — but weaker and narrower than first thought (see the
  correction below).

### CORRECTION (2026-07-20, prompted by "what facts does M4 probe?")

M4 probes, per subtask: the literal values in the `step` text + the subtask's
gold `answer` value. Two flaws this exposed:
- **Circularity.** M4 counted facts appearing ANYWHERE the agent looked,
  including its own computed outputs. On biomedical-hard-5 the apparent knob
  win (schema 0.24 vs stats 0.52) was the stats arm's *correct answer + inter-
  mediates rendering* — restricting to knob-controlled SOURCE renders collapses
  the gap to 0.24 vs 0.24 (0.28 → 0.01). The per-task "win" was re-measuring the
  answer, not delivery.
- **Wrong facts.** Subtasks encode SOLUTION facts (`'No'`, the kept IDs, the
  median), not the DIAGNOSTIC evidence the knob delivers (the `Case_excluded
  top_5={No=144, Yes=9}` line). `Yes=9` is nowhere in the subtasks, so M4 can't
  credit the knob for delivering it even in principle.

What SURVIVES (source-only M4, all 104, delta arms): the aggregate lean→rich
ordering holds — 1k **0.509** < stats **0.537** < 5k **0.571**. So richer render
configs genuinely surface more gold source-facts; the delivery-layer claim
holds at the AGGREGATE, but M4 is a coarse gauge, NOT a per-flip attributor.
(Latest arm reads 0.000 under source-only = a parser bug: latest render headers
differ; C3 needs the parser fixed.) A true per-step delivery metric needs
diagnostic-evidence annotations the subtasks don't provide — currently only the
manual semantic walks capture them.

M4 evidence delivery, all 104 tasks:

| arm | M4 delivery |
|---|---|
| Delta **1k** schema-only (leanest) | **0.648** |
| Delta stats-D2 1k | 0.683 |
| Latest 5k | 0.694 |
| Delta **5k** schema-only (richest) | **0.713** |

- **C1 char-cap (1k→5k): +0.065** — the biggest knob separation of any metric
  (accuracy range 0.04, M1 0.02). Char cap directly controls render volume.
- **C2 profile (schema→stats): +0.035** — stats surfaces the value distributions.
- **C3 history (delta→latest): −0.019** — near-tie, because history is retention
  not volume. Exactly how a delivery metric should rank the three knobs.

And per-task it is targeted, not just "more text": biomedical-hard-5 C2 pair
separates schema **0.24** vs stats **0.52** — stats delivered the specific
`Case_excluded` exclusion evidence the gold step needs.

## Conclusion (the reasonable stopping point)

**Yes, the eval space can be extended so the knobs show measurable
effectiveness — but only at the evidence-delivery layer, not accuracy or the
code/value artifact.** Knobs are evidence-delivery channels: M4 shows they
deterministically and monotonically control how much of the gold-relevant
evidence reaches the agent (lean 0.648 → rich 0.713), ranked exactly by
mechanism (char-cap > stats > history). This effect is **decoupled from the
answer** by the agent's stochastic use of delivered evidence (the no-rows smoke
test: stats line rendered, arm still failed), which is why accuracy is a coin
and artifact metrics don't attribute.

The defensible contributions:
1. **A layer separation**: knob effectiveness is real and measurable at
   delivery (M4), coin-decoupled at the answer. Evaluate knobs by *what they
   surface*, not by whether a stochastic agent used it on one roll.
2. **A method, not a magic metric**: twin-noise gating + semantic-walk
   attribution remain the way to read accuracy; no artifact metric replaces them
   (M1 fails, M2/M4 are ~1.6× signal/noise, not classifiers).
3. **Diagnostic utility** (bonus): M1/M4 cleanly separate "right pipeline, wrong
   benchmark" (wildfire-hard-18, high coverage + fail) from "couldn't build it"
   (wildfire-hard-19, low coverage) — useful for triage even where they don't
   attribute.

## KramaBench's OWN two builtin process metrics — actually run (2026-07-20)

Ran both builtin settings on 20 random tasks (seed=20), arm DeltaStats1kD2ProbePrompt.

**Pipeline-design "identify" judge** (`evaluate_data_pipeline`, gpt-4o-mini, fed
the serialized DAG), 20 tasks × 2 arms (stats + schema): **1.00 on every task,
both arms** — including tasks that FAIL the answer (biomedical-hard-5,
environment-hard-8/9, legal-hard-22, wildfire-hard-18). Zero spectrum, zero knob
contrast. It's a presence check, and our strong agent includes every step type,
so the judge says Yes to all. **Saturates too HIGH — useless for our agent.**

**Subtask-execution "implementation" metric** (`--run_subtasks`, re-ran the agent
on 122 subtasks, F1/success/RAE per gold answer_type; 0 no-response, engine
healthy): mean **0.281 (28%)** vs main-task pass ~73%. This **reproduces the
paper's central gap** (solve end-to-end ≫ implement the pieces; their systems:
55% e2e vs 20% subtask). But two problems for using it as "a richer spectrum":
- **Not rich — 94% bimodal** (69% hard-0, 25% hard-1, only 6% partial): list/
  exact metrics resolve to 0 or 1.
- **Format/isolation-confounded, understates capability.** biomedical-hard-5's
  MAIN task passes (2.6563) yet scores **0/5 subtasks** — the agent answers in
  prose, returns a variable name (`variant_linearized_values`), or the wrong ID
  column (`Participant_ID` C3L-… instead of `idx` S-…), and the F1 scorer (parses
  answers as Python literals) zeros them. So 28% measures answer-format-compliance
  + isolated-answerability as much as capability. It tracks capability only weakly
  (subtask mean 0.315 on main-pass vs 0.209 on main-fail, +0.11).

**Format-hint fix (2026-07-20, dataflow_system.py `_SUBTASK_FORMAT_HINTS` +
`_load_format_hints` synthesis):** subtasks were getting an EMPTY `Answer format:`
line (format_hint/<domain>.json covers only main tasks), so the agent free-formed.
Synthesizing a format sentence per subtask from its answer_type (list→"return a
comma-separated list", numeric→"single numeric value") fixed the formatting.
Before→after on the 20-task subtask spectrum: mean 0.281→**0.331**, hard-zeros
69%→**61%**, partial(0<x<1) 6%→**12%**, distinct score values 8→**17** (a genuinely
richer spectrum). This PARTITIONS the implementation score: ~5 pts were pure
format artifact (recovered); the residual **61% hard-zeros are NOT format** —
they're the isolation/dependency confound (biomedical-hard-5 stays ~0.05: subtask
3 given only mmc7 can't reconstruct the serous cases without subtask 2's handoff →
clean-but-wrong values), wrong-column choice (Participant_ID vs idx), and genuine
errors. Fully de-confounding would require CHAINED subtask execution (feed each
step the prior gold result), a harness change KramaBench doesn't do.

**Net:** the two builtin metrics fail in OPPOSITE directions on our strong agent —
design-judge saturates HIGH (1.0), subtask-exec craters LOW (0.28) — and neither
gives a clean or rich capability signal. The subtask number is the more useful
(reproduces the e2e≫implementation gap, and surfaces a real weakness: the agent's
sub-answers are loosely formatted), but to be a trustworthy metric it needs
format normalization (clean-list output prompt, or a fuzzy/semantic scorer that
accepts `Participant_ID`↔`idx` and prose). For KNOB effects specifically: the
design-judge shows none (both arms 1.0), and the subtask confound is
arm-independent, so knob signal would be swamped.

## Caveats / what's left
- M4 partly reflects render volume; the gold-step-signal weighting makes it
  "delivery of gold-relevant facts," and the conditional per-task view
  (biomedical-hard-5) is the non-tautological demonstration.
- M4's attribution is weak (1.6×) because delivery ≠ win; that's the point, not
  a bug — do not use it as a per-flip attributor.
- M3 (LLM correctness-anchored judge) was tested and FAILED (above) — semantic
  judgment does not recover the knob effect; the effect is upstream of the
  artifact the judge reads.
- Not yet done: M4 noise floor via a twin pair; the conditional-effect test
  formalized (knob accuracy gain restricted to its trigger subset, with a
  matched null); a stronger judge model for M3 (gpt-4o-mini may be too weak) —
  but the mechanism (identical code on the history/rows attributions) means a
  stronger judge cannot help on those cases regardless.

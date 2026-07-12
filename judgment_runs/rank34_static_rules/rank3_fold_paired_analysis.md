# Rank-3 paired analysis: DataflowSystemGPT52DeltaStats3kD2FoldControl vs DataflowSystemGPT52DeltaStats3kD2FoldResolved

Shared task directories: 104 (control 104, treatment 104)

## Configuration check

Tasks with configs in both arms: 104
Settings keys differing: {'fold_resolved_revisions_config': 104}
Unexpected diffs (should be empty): (none)

## Rule activation (treatment arm)

Tasks with >=1 rendered activation: 22 / 104
Sum of final-step signature counts: 62

| Task | Agent steps | Steps w/ signature | Final-step signatures |
| --- | ---: | ---: | ---: |
| `archeology-hard-5` | 11 | 7 | 2 |
| `astronomy-easy-1` | 11 | 2 | 0 |
| `astronomy-easy-4` | 10 | 2 | 4 |
| `astronomy-hard-12` | 25 | 13 | 3 |
| `astronomy-hard-9` | 7 | 1 | 1 |
| `biomedical-hard-1` | 7 | 2 | 2 |
| `biomedical-hard-3` | 5 | 1 | 2 |
| `biomedical-hard-5` | 6 | 1 | 4 |
| `biomedical-hard-8` | 7 | 2 | 2 |
| `legal-easy-12` | 5 | 1 | 2 |
| `legal-easy-13` | 5 | 1 | 2 |
| `legal-hard-17` | 5 | 1 | 2 |
| `legal-hard-22` | 6 | 1 | 2 |
| `legal-hard-23` | 5 | 1 | 4 |
| `legal-hard-24` | 6 | 2 | 4 |
| `legal-hard-28` | 6 | 2 | 2 |
| `legal-hard-8` | 5 | 1 | 2 |
| `wildfire-easy-2` | 12 | 3 | 8 |
| `wildfire-easy-3` | 7 | 1 | 2 |
| `wildfire-hard-4` | 9 | 3 | 4 |
| `wildfire-hard-5` | 8 | 3 | 2 |
| `wildfire-hard-6` | 10 | 3 | 6 |

Control-arm signature leak (must be []): []

## Accuracy (pass = answer-type metric >= 0.9)

| Outcome | Tasks |
| --- | ---: |
| Both pass | 73 |
| Control only | 6 |
| Treatment only | 4 |
| Both fail | 21 |

Passes: control 79/104 (76.0%), treatment 77/104 (74.0%)

Accuracy divergences (independent trajectories — attribution needs the trace):

- `astronomy-easy-4`: control-only
- `astronomy-hard-8`: control-only
- `biomedical-hard-5`: treatment-only
- `environment-hard-10`: control-only
- `environment-hard-11`: treatment-only
- `environment-hard-8`: control-only
- `environment-hard-9`: treatment-only
- `legal-hard-1`: treatment-only
- `legal-hard-15`: control-only
- `wildfire-hard-17`: control-only

## Fair paired usage (tasks with stats.json in both arms)

Paired usage tasks: 102

| Measure | Control | Treatment | Treatment − Control |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $4.731615 | $5.000183 | $0.268568 (+5.68%) |
| input_tokens | 6,152,971 | 6,262,406 | +109,435 (+1.78%) |
| cached_tokens | 5,134,464 | 5,131,008 | -3,456 (-0.07%) |
| output_tokens | 146,478 | 151,594 | +5,116 (+3.49%) |
| total_tokens | 6,299,449 | 6,414,000 | +114,551 (+1.82%) |
| num_steps | 668 | 677 | +9 (+1.35%) |
| Uncached input | 1,018,507 | 1,131,398 | +112,891 (+11.08%) |
| Cache hit (control) | | | 83.4% |
| Cache hit (treatment) | | | 81.9% |

## Activation-split cost (the mechanism isolation)

| Cohort | n | Cost C→T | Uncached input | Steps C→T |
| --- | ---: | --- | --- | --- |
| Activated (rule fired) | 22 | $1.6903 → $2.0523 (**+21.4%**) | 369,072 → 527,879 (**+43.0%**) | 191 → 200 |
| Non-activated | 80 | $3.0413 → $2.9478 (−3.1%) | 649,435 → 603,519 (−7.1%) | 477 → 477 |

Non-activated tasks are statistical noise around zero — the arms render
byte-identically when the rule is silent (flag-off parity held end-to-end).
Every dollar of the +5.68% aggregate regression comes from the 22 tasks where
the rule fired. Same-step activated cases isolate it further:
`legal-hard-23` (6 steps both arms) +31% cost, `legal-hard-17` (6/6) +29%,
`biomedical-hard-3` (6/6) +65%. Identical trajectory length, fold on → cost up:
pure prefix-cache invalidation.

## Un-latching observed (`astronomy-easy-1`)

Per-agent-step rendered fold counts: `[0,0,0,0,0,0,0,0,6,6,0]` — six folds
fired at step 9, persisted at step 10, then vanished at the final step. A late
DAG change broke the sole-consumer predicate (fan-out), un-folding the entire
history and re-paying it uncached in one turn. The audit's "one consumer
initially" eligibility is evaluated against the CURRENT leaf and is therefore
not monotone; latching it (once folded, folded forever) would remove this
churn mode but departs from the audit's stated predicate.

## Accuracy-flip attribution

Control-only flips (6): `astronomy-easy-4`, `astronomy-hard-8`,
`environment-hard-10`, `environment-hard-8`, `legal-hard-15`,
`wildfire-hard-17`.

- Five of six are NON-activated tasks — the rule rendered nothing in their
  treatment runs; the contexts were rule-untouched. Three (`environment-hard-8`,
  `legal-hard-15`, `wildfire-hard-17`) are chronic direction-flippers in the
  delta_vs_latest_3k audit.
- `astronomy-easy-4` (activated): the answer-determining choice — the
  peak-detection operator with prominence ≥ 20 / distance ≥ 5 — was created at
  agent step 7; the first fold rendered at step 8. The maxima years the
  treatment reported were fixed by a method choice made in a fold-free context.
  Not rule-attributable.

Retry scan (difflib similarity > 0.92 between a post-first-fold submission
and an earlier superseded version, across all 22 activated tasks): the only
flagged task is `astronomy-hard-12` — a both-fail, max-steps spiral where the
CONTROL arm also loops near-identical code on its own, disjoint, operator set
(21 high-sim resubmission pairs control vs 42 treatment). Task-intrinsic
thrash on independent trajectories; no clean case of an agent retrying a
folded attempt it could no longer see. The designed falsification signal
(resolution fact insufficient → resolved failure retried) did not occur in
the other 21 activated tasks.

## Verdict against the acceptance gate

| Gate | Result |
| --- | --- |
| 1. No attributable accuracy regression | **PASS** — 0/6 control-only flips attributable (5 non-activated, 1 predates first render) |
| 2. Same-answer cases keep the same dataflow | PASS on inspected cases (same-step cohort) |
| 3. Cache-aware `cost_usd` decreases | **FAIL** — +5.68% aggregate, +21.4% on activated tasks |
| 4. Steps/tool churn don't offset savings | FAIL on activated (+9 steps, +43% uncached) |

**Rank 3 as implemented does not graduate.** It is information-safe (accuracy
neutral within noise, no thrash-retry events) but cache-hostile: each fold is a
mid-prompt rewrite of an early trajectory region, and every subsequent step of
the task re-reads the invalidated suffix uncached. The per-turn carrying saving
(~0.5–2k tokens) never amortizes. This is the third independent replication of
the pattern (static compaction v3: +22% cost; frontier decay: cost-flat with
−4.4pt cache hit; rank-3 fold: +5.7%): **in-place context mutation loses to
the prompt cache regardless of how surgically the mutated bytes are chosen.**

The audit's own falsification clause anticipated the remedy: "retain the
semantic rule but move it to a cache-stable boundary." The fold semantics
belong at a boundary that is ALREADY a rewrite — e.g. inside the
edit-convergence compaction deck (fold superseded revisions into resolution
facts *when the deck folds anyway*), or applied once at a task-phase boundary —
not as a free-running per-event rule.

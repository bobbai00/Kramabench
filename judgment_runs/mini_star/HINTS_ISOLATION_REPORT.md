# A6 — isolating the `structuralHints` leg

**Why this run exists.** The user noticed that `__table_hints__` (`Output Table profile:` —
`duplicate rows: 359 of 764 (47%)`, `empty rows: 156 of 764`, unnamed headers) renders in **62-64%
of runs in EVERY A arm** (A1/A2/A3/A4/A5) and **0% of A0**, measured over 1,436 runs. I had been
describing A1 as a rows+stats policy and A4 as a provenance-prompt result, but both also flip
`sourceStructuralHints` — a third leg I never isolated. A2/A3 varied only stats. So A1's +5.3 and
A4's `legal-hard-29` 8/8 could have belonged to the profile facts instead. That was a real hole in
the experimental design, found by reading the arm configs rather than the results.

**Run:** 2026-07-29 00:49-03:12. `A6HintsOnlyReplicate1-4` (A0 base + `structuralHints` on sources
via the new `rolePolicy.hintsOnly`, *nothing else* — no row cap, no stats, no interior trim) vs
`A0ControlReplicate9-12` (fresh control, required because commit `81dc518be` changed the default
render). 160 runs at P8, `:3002` @ `3d2bbe187`, 20 hard tasks. Isolation verified pre-launch on
`legal-hard-29`: source rendered `stats: False` (A0-identical) + `profile: True`, rows uncapped at
36, interior untouched.

## Result: the hints leg is NOT the hidden cause

| arm | acc (avg ± rep std) | $/task | steps | reasoning | cache% |
|---|---|---|---|---|---|
| A6 hints-only | **50.1 ± 13.3** | **0.0204** | **6.5** | 6,346 | 82.7 |
| A0 fresh control | 58.2 ± 4.1 | 0.0219 | 7.0 | 7,249 | 83.6 |

A6 − A0fresh = **−7.8 pt**, SE-of-diff 6.50 → **1.20× SE, inside noise**, and pointing the *wrong
way* for the confound hypothesis. Per-rep: A6 `54.0 / 69.0 / 34.2 / 47.4`, A0fresh
`65.0 / 57.0 / 56.8 / 57.0`.

**So the prior attributions stand.** `RULE_A_FINAL.md` and `COMBINE_ROUND_REPORT.md` do **not**
need revising: A1's edge is not smuggled in by the profile facts, and A4's `legal-hard-29` result
(8/8 vs 2/8) is not either — A6 gets 0.75 there, the same as its own control, versus A4's 1.00.
The provenance mechanism remains the best-supported explanation for that task.

What the hints leg *is*: **cheap** (−7% $/task, −0.5 steps, −903 reasoning vs control) and
accuracy-neutral-to-negative on its own. Consistent with the standing pattern — single legs in
isolation (A5 49.5, A6 50.1) underperform the bundles they came from.

Per-task on the provenance-gold set (A6 4 reps, A0fresh 4 reps, A1/A4 8 reps):

| task | A6 | A0fresh | A1 | A4 |
|---|---|---|---|---|
| legal-hard-29 | 0.75 | 0.75 | 0.25 | **1.00** |
| legal-hard-16 | 0.75 | 0.25 | 0.62 | 0.62 |
| environment-hard-8 | 0.50 | 1.00 | 0.88 | 0.62 |
| environment-hard-9 | 0.90 | 0.65 | 0.64 | 0.65 |
| environment-hard-10 | 0.50 | 0.50 | 0.88 | 0.75 |
| environment-hard-13 | 1.00 | 1.00 | 0.88 | 0.75 |

## Free result: the stats-bound commit is accuracy-neutral

The fresh control also measures my own render changes (`81dc518be`: proof-based stats suppression,
head+tail column slicing, 200-char line clamp), since A0 reps 1-8 predate them:

| | acc | $/task | steps |
|---|---|---|---|
| A0 old sha (8 reps) | 59.1 ± 12.3 | 0.0241 | 7.4 |
| A0 new sha (4 reps) | 58.2 ± 4.1 | 0.0219 | 7.0 |

−0.9 pt (well inside noise) at −9% cost. The fixes are behaviourally inert and modestly cheaper,
which is what "lossless" was supposed to mean. Worth noting the fresh control's rep std is 4.1 vs
the old pool's 12.3 — 4 reps is a small sample, so don't over-read the stability.

## Incident: 58 runs destroyed and repaired

Mid-pool I restarted `:3002` to deploy `b9fd6d4f1` and smoke-test C9/C10 **while A6 was still
draining at 133/160**. In-flight runs died with `Connection to remote host was lost` /
`Connection refused`. Damage: A6 reps lost 1/2/14/1 answers, A0fresh lost 13/20/7/0 — A0 rep10 lost
all 20. The pre-repair numbers were A6 41.4 ± 20.8 vs A0fresh **25.2** ± 20.5, which I nearly
reported as a hints effect; the tell was that A0fresh is config-identical to a control that had
scored 59.1 across 8 reps, so a 34-point collapse had to be infrastructure.

Two lessons now in the ops rules:

1. **Never restart an agent service while any pool is live.** Gating a new pool on an old pool's PID
   does not protect the old pool from a service restart — the dependency is the shared service, not
   the process tree.
2. **A connection-killed run still writes `Total score is: 0.0`**, so every orchestrator's
   resume-skip (`grep 'Total score'`) treats it as complete. Repair must be keyed on a missing
   `system_scratch/<SUT>/<task>/response.txt`. The durable health check is
   **scored == answers == dirs** per arm, which is what caught this; it is now checked every cycle.

Repaired with `repairA6.sh` (58 runs, P6, keyed on missing `response.txt`, stale logs deleted so
resume-skip could not swallow them). Post-repair completeness: A6 20/20/19/20, A0fresh
20/20/19/20 — the remaining 19s are single genuine task failures, not damage.

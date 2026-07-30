# C9 / C10 / C11 — per-operator char budget at the raw-data boundary

**Run:** 2026-07-29 01:37–04:08. 9 arms (3 configs × 3 reps) × **full 104 tasks = 936 runs**,
`:3002` @ `b9fd6d4f1`, `dirty=False`, no `sourceProvenanceHint` (clean char-budget tests, not
confounded by A4). Integrity: `scored == answers` on all nine arms, 0 connection errors, 0
instant-fails.

Configs:
- **C9** LATEST+code — sources 5k + stats, every downstream op 1k + no stats
- **C10** DELTA — same split (the char-budget leg binds on DELTA event renders)
- **C11** LATEST+code — 5k + stats for **all** ops (uniform-rich reference)

Accuracy is KramaBench's own score (native measures CSVs reduced with `compute_scores.py`'s
formula), never self-computed.

## Result

| arm | acc (avg ± std) | easy | hard | $/task | in tok | out tok | cache% | reasoning | steps |
|---|---|---|---|---|---|---|---|---|---|
| C9 source-rich LATEST | **67.5 ± 0.1** | 80.7% | 58.6% | 0.0175 | 58,256 | 6,829 | 82.0% | 5,271 | 6.1 |
| C10 source-rich DELTA | 67.1 ± 0.8 | 80.0% | 58.3% | **0.0140** | **50,815** | **5,283** | 80.8% | **4,109** | **5.5** |
| C11 uniform-rich LATEST | 68.1 ± 2.0 | 80.7% | 59.5% | 0.0168 | 56,680 | 6,508 | 81.8% | 5,205 | 6.1 |

Per-rep OVERALL — C9 `67.4 / 67.4 / 67.7`, C10 `67.2 / 68.0 / 66.0`, C11 `66.5 / 70.9 / 66.8`.

## Q1 — does rich-source / lean-downstream beat uniform-rich on cost? **No.**

C9 − C11: accuracy **−0.6 pt** (0.48× SE, inside noise) and cost **+4.3%** — the split is
*slightly more expensive*, with **more** input tokens (+1,576), not fewer. The core dataflow-rule
claim fails on its own terms.

**Why, and it's the same reason as every previous byte-trim:** capping downstream operators at 1k
saves nothing because **derived tables were already under 1k**. Downstream ops here are aggregates
and filters — a `groupby` result is 1-6 rows. The earlier footprint work said exactly this
(interior/sink ops are 3-5% of the render footprint); this run confirms it at full scale with a real
per-operator char budget rather than a row proxy. C9 left the *source* budget at 5k — identical to
C11 — so it never touched the only place bytes actually live, and paid a small overhead for the
per-operator override machinery.

That is now the **fourth** independent confirmation that render levers live at the raw-data
boundary: sampling/stats footprint analysis, A2/A3 stats trims, A1's row-cap legs, and now char
budgets.

One genuine positive: C9 is astonishingly **reproducible** — std **0.1** across 3 reps
(67.4/67.4/67.7) vs C11's 2.0. Constraining downstream renders appears to remove a source of
run-to-run variance even though it does not move the mean. With n=3 that is suggestive, not
established.

## Q2 — the same split under DELTA: **20% cheaper at equal accuracy**

C10 − C9: accuracy **−0.5 pt** (inside noise, and the three reps straddle C9: 67.2/68.0/66.0) at
**−19.8% cost**, −7,441 input tokens, −1,162 reasoning, −0.5 steps.

This is the actual finding of the run, and it was not the hypothesis. Under DELTA the char budget
caps **every event render** of an operator, so a source re-rendered across 6 events pays the cap
6 times instead of paying full price 6 times — the saving compounds with trajectory length in a way
it cannot in a single LATEST snapshot. Cache is marginally worse (−1.2pp) but the raw byte reduction
dominates.

Caveat worth stating: this contradicts the earlier explore-mode result where LATEST beat DELTA
(66.3 vs 58.7). Those runs were no-oracle exploration; these are `--use_truth_subset` full-104. The
honest reading is that DELTA's disadvantage is regime-dependent, and under this regime the
char-budget split makes DELTA the cost-efficient choice at parity accuracy.

Per-workload (avg of 3 reps) shows the parity is an average of real differences, not uniformity:

| arm | archeology | astronomy | biomedical | environment | legal | wildfire |
|---|---|---|---|---|---|---|
| C9 | 38.9% | 33.3% | 55.6% | 73.8% | 76.2% | **90.2%** |
| C10 | 38.9% | 30.6% | 51.9% | 72.1% | **82.2%** | 84.1% |
| C11 | 33.3% | **41.7%** | **63.0%** | 71.3% | 76.7% | 89.9% |

C10 wins legal by +6 pts and loses wildfire by −6; C11 wins astronomy and biomedical. Nothing here
survives the ±4-5pt floor at 3 reps, but it means "equal overall" is masking workload-level
structure worth a targeted look.

## Manipulation evidence

Verified before launch on `legal-hard-29`, both modes:

| | source | downstream |
|---|---|---|
| C9 LATEST | 4,086 ch block, **20 rows**, stats + profile | 3,322 / 916 ch, **6 / 1 rows**, no stats |
| C10 DELTA | first event result 3,496 ch | later events 1,162 / 1,079 ch |

The knob itself was new work: `maxOperatorResultCharLimit` is a *single* number sent to the JVM
inside the execute request, so "5k source / 1k downstream" was previously inexpressible. The
render-time per-operator `tuple.maxChars` (commit `0b132212a`) fixes that and binds in both context
modes.

## Throughput vs concurrency (documented so the next run doesn't guess)

| concurrency | throughput |
|---|---|
| P5 (engine shared with a foreign pool) | 2.0/min |
| P8 (exclusive) | 1.9/min |
| P12 | 3.5/min |
| **P18** | **8.7/min** (6.6/min while a P6 repair shared the engine) |

The inherited note that "P6 causes instant-fails" is **stale** — 0 instant-fails at P8, P12, P18,
and briefly ~P30. 936 runs completed in 2h31m.

## Data caveat

`astronomy-hard-11` (nearly every arm) and `biomedical-hard-1` (C10 arms) hit the orchestrator's
`timeout 900` and score 0. These are **pre-existing heavy-task timeouts** — both are also missing in
P4-P5-era arms — not a product of the P18 push: run wall-times are p50 **70s**, p95 372s, max 842s
against the 900s cap. They fail uniformly across arms, so the three-way comparison stays fair; they
depress all three absolute numbers equally.

## Verdict

1. **Do not ship the lean-downstream char split.** No accuracy change, no cost saving, mild
   overhead. The bytes are not downstream.
2. **The DELTA + char-budget combination is the cost-efficient config** on this benchmark:
   −20% cost at parity accuracy vs the LATEST equivalent. Worth a 6-8 rep confirmation before
   adopting, and worth re-testing in the no-oracle explore regime where LATEST previously won.
3. `tuple.maxChars` stays in the codebase (default 0 = inert). It is the right primitive; this
   particular policy was just aimed at the wrong operators.

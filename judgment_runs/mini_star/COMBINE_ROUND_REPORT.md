# Combine round: A4 (provenance principle) and A5 (A1+B2 history)

**Run:** 2026-07-28 20:02–21:24 (ORCHA4), 120 runs, P5, `:3002` @ `2259ebf2d`, `dirty=False`,
no retries. Controls = the 8-rep A0/A1 pools. Coexisted with a foreign `:3001` pool all evening;
zero instant-fails.

## Headline

| arm | acc (avg ± rep std) | $/task | steps | reasoning | reps |
|---|---|---|---|---|---|
| **A4 = A1 render + provenance principle** | **71.7 ± 2.4** | **0.0234** | **6.7** | 7,404 | 3 |
| A3 no source stats | 66.5 ± 7.4 | 0.0277 | 8.2 | 8,324 | 3 |
| A1 rich-source/lean-interior | 64.4 ± 8.4 | 0.0239 | 7.3 | 7,277 | 8 |
| A2 anomaly-density stats | 61.2 ± 5.8 | 0.0236 | 7.4 | 7,199 | 3 |
| A0 control | 59.1 ± 12.3 | 0.0241 | 7.4 | 7,596 | 8 |
| A5 = A1 render + B2 data history | 49.5 ± 4.0 | 0.0239 | 7.5 | 7,479 | 3 |

## A4: the first rule that clears the bar

**A4 − A0 = +12.6 pt at −2.9% cost.** SE of the difference ≈ 4.6 (A4 SEM 1.4 at 3 reps, A0 SEM 4.3
at 8) → the delta is **~2.7× SE**, passing the pre-registered ≥2×SEM rule that every previous arm
failed. And it is not carried by dice — the gain lands exactly where the mechanism predicted:

| provenance-gold task | A4 | A1 (8) | A0 (8) |
|---|---|---|---|
| legal-hard-29 | **1.00** | 0.25 | 0.25 |
| legal-hard-16 | **1.00** | 0.62 | 0.12 |
| environment-hard-8 | 1.00 | 0.88 | 0.75 |
| environment-hard-10 | 1.00 | 0.88 | 0.62 |
| environment-hard-13 | 1.00 | 0.88 | 0.75 |
| environment-hard-9 | 0.67 | 0.64 | 0.83 |

- Both falsifiable targets hit 3/3 (`legal-29` at base 0.25 → P(3/3)≈1.6%; `legal-16` at A0 0.12 → ≈0.2%).
- **Manipulation → behavior → outcome chain is verified end-to-end:** `__source_file` appears in
  agent code on **45% of multi-file runs vs 2% on A1's own baseline** (~20×), with only 13%
  single-file spillover.
- **The null prediction also held:** `environment-hard-9` (crux = two-row header, not provenance)
  did not move (0.67 vs 0.64). The rule does what it claims, nothing more — which is what makes
  the rest credible.
- Rep scores 67.5 / 67.5 / 72.0: std 2.4, the most reproducible accuracy-positive arm in the
  entire program (next best: A0's 12.3, A1's 8.4).

**Caveats, stated plainly:** 3 reps; cross-pool comparison (A4 ran in its own pool @ 2259ebf2d,
controls @ 4af1e98da/9d60d01dc — golden parity held at every hop and the A0r4 sentinel showed no
offset, but same-pool is stronger); one regression flag — `archeology-hard-7` 0.00 vs A1 0.50
(n=3; not a multi-file task, no mechanism connects it; watch in validation).

**Decision: ADOPT pending validation.** A4 (= A1 render policy + `sourceProvenanceHint`) is the
shipped config candidate. Validation pool: A4 reps 4–8 (100 runs) → 8-rep verdict on identical
footing with A0/A1, plus the archeology-hard-7 watch.

## A5: killed

49.5 ± 4.0 — **−14.9 vs A1** using the same render. B2 was benign on the C8 base (−3.2, −10.4%
cost); stacked on A1's lean-interior render it is actively harmful, and it didn't even save money
here (0.0239, cost stacking did NOT materialize). Negative interaction: A1 already strips the
interior; adding a shape-rendered prior version per operator re-inflates blocks with near-duplicate
lines the agent must reconcile. Consistent with the day's theme — history channels don't pay, and
composition is not additive. **Do not revisit.**

## The full arc (what today established)

1. **C-knobs (C1–C8, both models, full-104):** render knobs reshuffle which coin-flips land;
   evidence-delivery, not capability. Hard-20: 0 never-pass / 18 flip / 2 always-pass.
2. **Rule B (B1/B2/B3):** no history channel buys accuracy; thought replay is a cache disaster.
3. **"Important versions" (4 definitions):** error-version, code-version, data-version, load-recipe
   — all dead. Data versions are write-once (0/1028); code rebuilds are a stuck-run symptom
   (within-task median +0.0); evidence visible ≥3 steps goes unacted (100 cases).
4. **Stats density (A2/A3):** trimming "waste" backfires monotonically — the stats block
   substitutes for exploration steps. Byte-level waste ≠ behavior-level waste.
5. **Rule A (8 reps):** +5.3 = 1.0×SE — not established; adopted as A_win on weak dominance.
6. **A4 provenance principle:** the one intervention grounded in reading traces against gold
   solutions, with a falsifiable per-task prediction — and the only one that cleared 2×SEM.
   Mechanism: per-file identity is a load-time fact erased by concat; deriving it from a
   name-suffix regex was a 0%-pass trap (n=26); DELTA's apparent +22pp on these tasks was purely
   a higher idiom-choice rate (44% vs 24%).

The lesson the program converged on: **generic context volume (history, stats, versions) does not
buy accuracy; one targeted, trace-derived semantic principle at the raw-data boundary does.**

## 8-rep validation (ORCHA4V, reps 4–8, 100 runs, 21:25–22:20)

**The 3-rep 71.7 partially regressed: A4 at 8 reps = 67.0 ± 8.2.** A4−A0 = +7.9 ≈ 1.5×SE-of-diff
— the *overall* accuracy claim does **not** clear 2×SEM. The 3-rep read was optimistic, as 3-rep
reads have been all day. What survives is sharper and better:

| task (8 reps each) | A4 | A1 | A0 |
|---|---|---|---|
| **legal-hard-29** | **1.00 (8/8)** | 0.25 | 0.25 |
| legal-hard-16 | 0.62 | 0.62 | 0.12 |
| environment-hard-8 | 0.62 | 0.88 | 0.75 |
| environment-hard-9 | 0.65 | 0.64 | 0.83 |
| environment-hard-10 | 0.75 | 0.88 | 0.62 |
| environment-hard-13 | 0.75 | 0.88 | 0.75 |

- **`legal-hard-29` is a real, decisive, mechanism-verified fix: 8/8 vs 2/8 in both controls.**
  P(8/8 | base 0.25) ≈ 1.5e-5. This is the task whose gold solution literally requires
  `__source_file`, and the one task where the causal chain (prompt → idiom in code → correct
  grouping → correct answer) is verified at every link.
- The other provenance-gold tasks did not move beyond noise (legal-16 ties A1; env tasks mixed).
  The rule reliably fixes the failure mode it names — exactly one task on this benchmark exhibits
  it fatally.
- Idiom adoption at 8 reps: **40% multi-file / 8% single-file** vs A1's 2%/0% — stable ~20×,
  targeted.
- `archeology-hard-7` regression (0.12 vs A1 0.50) has **no mechanistic link**: 0/8 A4 runs used
  the idiom there. Cross-pool dice, not spillover harm.
- **Cost: A4 is the cheapest arm in the entire program — $0.0228/task** (−4.6% vs A1, −5.4% vs
  A0), steps 7.0.

**Final verdict: ADOPT A4 as the shipped config — on honest grounds.** Weak dominance, again:
point-best accuracy among 8-rep arms (67.0 vs 64.4 vs 59.1, inside noise), strictly cheapest,
one certainty-grade task fix, no mechanistically-linked regression. The claim to publish is NOT
"+8 accuracy" — it is: *a trace-derived semantic principle at the raw-data boundary deterministically
fixes the failure mode it targets (8/8 vs 2/8) at negative cost, where 20+ generic render/history
interventions moved nothing.*

Open items: (1) data-layer `__source_file` auto-injection if 40% prompt adoption proves too low on
new workloads; (2) A6 candidate = two-row-header hint for the `env-9` family, same gold-vs-trace
method; (3) A5-style compositions are dead — test principles independently.

## Ops notes (hard-won today)

- `bun --watch` does NOT watch `prompts/code-mode.md` — restart `:3002` after prompt-file edits.
- `dataflow_agent.py` client default endpoint is `:3001` — always pass `agent_endpoint` in probes.
- System prompts are not logged in `react_steps.json` inputMessages — verify prompt manipulations
  behaviorally (e.g. `__source_file` in agent code).
- Flags stamp NESTED under `config.json:agent_settings`.

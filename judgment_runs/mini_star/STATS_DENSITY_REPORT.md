# Stats density on sources: A1 (full) vs A2 (anomaly-only) vs A3 (none)

**Run:** 2026-07-28 16:02–17:30, 140 runs (A2×3, A3×3, A0-sentinel×1 arms × 20 hard tasks),
P5, no retries, service `:3002` @ `9d60d01dc`, `src_dirty=False` throughout.
A1 + A0 reps 1–3 are the 14:25 run @ `4af1e98da`; the render is byte-identical across the two
SHAs for those configs (10 golden-parity snapshots unchanged), and `A0ControlReplicate4` is the
cross-sha sentinel — its 60.0 sits inside the A0 reps-1-3 spread (69.0/66.5/51.5), so **no
run-level offset detected** between vintages.

Question: the source stats block spends bytes announcing healthy defaults (`null=0` on 32.1% of
column entries, `distinct==rows` on 5.9%, whole block redundant on 22.4% of blocks, `Schema(…)`
echo on 91.6%). Does trimming it save money?

## Answer: NO — both trims backfired. The verbose stats block pays for itself.

| arm | acc (avg ± std) | $/task | steps | reasoning/task | uncached-in | cache% |
|---|---|---|---|---|---|---|
| **A1 full stats** | **67.3 ± 8.5** | **0.0217** | **6.8** | **6,559** | 10,311 | 83.9 |
| A2 anomaly-only stats | 61.2 ± 5.8 | 0.0236 | 7.4 | 7,199 | 10,975 | 83.7 |
| A3 no source stats | 66.5 ± 7.4 | 0.0277 | 8.2 | 8,324 | 13,510 | 82.3 |
| A0 control (no policy) | 61.8 ± 6.8 | 0.0211 | 6.9 | 6,897 | 9,582 | 85.1 |

The manipulation itself landed exactly as designed — verified per-class across all 3 A2 reps
(`null=0` → 0, all-unique → 0, sample-covered blocks → suppressed, Schema echo 92.6% → 46.3%,
9,207 residual chars = the correct conditional case) and A3 rendered **zero** stats blocks on
sources. The bytes really were removed. And the runs still got more expensive:

```
stats info:   full  →  anomaly  →  none
steps:         6.8       7.4       8.2      (monotone)
reasoning:   6,559     7,199     8,324      (monotone)
$/task:     0.0217    0.0236    0.0277      (monotone)
```

**Mechanism: the stats block substitutes for exploration.** Remove what it asserts (even the
"degenerate" `null=0` — which tells the agent it does NOT need to handle missing values) and the
agent re-derives those facts with extra probe steps and reasoning. A saved ~500 chars/context is
swamped by ~0.6–1.4 extra steps/task. This is the third independent replication of the pattern:
byte-trimming a LATEST context loses at the run level (fold-revisions and retire-probes died the
same way — mutation/removal loses to cache+behavior).

The subtlety worth keeping: "waste" measured at the byte layer (32% degenerate lines) is not waste
at the behavior layer. A `null=0` line is the *cheapest possible proof of completeness* — the
agent otherwise buys that proof with a step.

## Accuracy view (avg over reps, all inside noise)

A1−A0 with the 4-rep control: **+5.6pt at +2.9% cost** (was +5.0 at +3.3% with 3 controls) —
still inside the ±4–5pt floor (pooled std 8.5), still carried by 2 task flips (`biomedical-hard-8`,
`archeology-hard-7`), zero tasks worse. A2's 61.2 and A3's 66.5 are likewise inside noise; note
A2 < A3 makes no mechanistic sense, which is itself evidence these accuracy deltas are dice.
(The A3-rep1 "thrash" flagged mid-run was rep-level noise — reps 2–3 ran normal — but the
*arm-level* monotone step/cost trend above is 3-rep consistent and is the real signal.)

## Decisions

1. **A_win = A1 as configured** — full-density source stats, 12-row source cap, lean interior.
   Cheapest policy arm, highest avg accuracy, fewest steps.
2. **`statsDensity: "anomaly"` stays in the codebase, default off.** The render feature works
   exactly as specified (all 4 waste classes eliminated); the *policy* of using it on sources is
   what failed.
3. **Do not revisit source-stats trimming.** Three byte-trim rules have now died the same
   death; the render budget at the raw-data boundary is load-bearing.

## Trace sanity

All 7 arms: instant-fails 0, empties 0, no-response 0, quota 0. Steps 5.6–9.6 by rep,
cache 80.0–85.4%. Services flat (233 MB / ≤204 MB), single vintage per arm confirmed in every
sampled `config.json`.

## Roadmap (in motion)

- **Step 2 — running now:** A1-vs-A0 rep expansion to 8 reps/arm (orchA3, 180 runs, gated launch
  fired 17:30, ETA ~2h). Settles whether A1's +5.6 avg is real: 8 reps ≈ ±3pt SEM against the
  8.5 rep std. Verdict lands in `RULE_A_FINAL.md`.
- **Step 3 — combine round** (after orchA3): vs the 4-rep A0 control, 3 reps each, avg accuracy
  first then $/task:
  - `A_win` (= A1)
  - `A_win + sourceProvenanceHint` — the trace-grounded rule from the gold-solution deep dive:
    multi-file loaders must carry a `__source_file` column (per-file identity is a load-time fact
    erased by the final snapshot; suffix-derived grouping = 0% pass on `legal-hard-29` n=26, 0%
    on `legal-hard-16` without the idiom n=1). Patch drafted (config-gated prompt principle,
    byte-identical when off); **apply only after orchA3 drains** — prompts.ts hot-reloads `:3002`.
    Falsifiable: it must lift the 6 provenance-gold multi-file tasks, especially legal-29/16;
    if legal-29 doesn't move, the prompt form is insufficient → data-layer injection form.
  - optional `A_win + B2` (resultHistory=1: −10.4% cost, std 1.9, −3.2 inside noise).
- Closed permanently: B1, B3, B4/failure-ledger, all four "important version" definitions.

# A7 `Files read:` fact — paired test on two sampling budgets

**Run:** 2026-07-29 14:35–16:31. 4 configs × 3 reps × full 104 tasks = **1248 runs**, P30,
`:3002` @ `311ddd646`, fresh engine era (post-restart). Every A7 variant is paired with its **own
same-era control**, because the engine was restarted mid-day and cross-era comparison is invalid.

Accuracy = KramaBench's own score (native measures CSVs, `compute_scores.py` formula, validated
12/12 against `compute_scores.py --sut`).

| arm | config | accuracy | easy | hard | $/task | steps |
|---|---|---|---|---|---|---|
| D8 | LATEST 5k + code (control) | **71.3 ± 1.6** | 81.5 | 64.4 | 0.0173 | 6.1 |
| D8F | + `Files read:` fact | 71.2 ± 1.8 | 81.5 | 64.2 | **0.0154** | 5.9 |
| D12 | LATEST 1k + code + stats (control) | 63.8 ± 1.4 | 76.3 | 55.4 | 0.0173 | 6.4 |
| D12F | + `Files read:` fact | **68.8 ± 2.2** | 77.7 | **62.7** | 0.0162 | 6.2 |

## The headline: the fact pays only when the sampling budget is tight

| pair | Δ accuracy | SE-of-diff | verdict | Δ cost |
|---|---|---|---|---|
| D8F − D8 (5k) | **−0.1** | 1.39 | 0.07× SE — inside noise | **−11.0%** |
| D12F − D12 (1k) | **+5.0** | 1.51 | **3.30× SE — OUTSIDE noise** | **−6.4%** |

Per-rep: D8 `69.4/71.3/73.2` vs D8F `68.6/72.7/72.3`; D12 `65.8/62.4/63.3` vs D12F `69.1/71.3/66.0`.
The D12 pair does not overlap at all — every D12F rep beats every D12 rep.

**+5.0 pt at 3.3× SE, and −6.4% cheaper, is the largest clean accuracy win of the entire
program.** For comparison, every render knob measured before this landed inside ±5 pt noise, and
A4's prompt version of the same idea produced +8 vs A0 at only ~1.7× SE.

**Hard tasks carry it: 55.4 → 62.7 = +7.3 pt**, larger than the overall delta. That is the axis
the campaign has been unable to move all along.

## Mechanism: it substitutes for sampling, not for stats

The split is the finding. At 5k the agent already sees enough rows to infer what it loaded; the file
list is redundant and adds nothing (−0.1). At 1k it cannot, and being told what its own glob matched
is worth 5 points. So the fact is not a general accuracy lever — it is **compensation for a starved
render budget**.

Two observations sharpen this:

1. **The gain is NOT concentrated on multi-file tasks.** Summed per-task deltas over the 10
   multi-file tasks are only **+1.90** for D12F (and **−0.23** for D8F), while the other 94 tasks
   move **+3.4**. The fact renders on ~20% of runs, so a purely multi-file mechanism could not
   produce a 5-point overall gain — and it didn't.
2. So the honest mechanism is broader than "provenance for per-file grouping": knowing the file
   inventory appears to help the agent orient generally when it can see little data. `legal-hard-29`
   (the provenance-gold task) does improve, 0.33 → **1.00**, exactly as predicted — but it is one
   task of the +5.0, not the cause of it.

I want to flag that this weakens my earlier A4 story. I attributed A4's `legal-hard-29` result to a
specific per-file-identity mechanism. That mechanism is real and reproduces here, but the *bulk* of
the benefit in this paired test comes from tasks the mechanism does not explain. The narrow story was
too clean.

## Cost: the fact is free or better on both arms

D8F −11.0%, D12F −6.4%, both with fewer steps. It costs ~100 chars and removes exploration steps —
the same "cheap fact beats expensive bytes" shape as the load-quality profile.

## Vintage caveat (important)

Identical config, different engine era: **C8 = 69.0 ± 3.0 (era 1) vs D8 = 71.3 ± 1.6 (era 2)** —
a **+2.3** shift from restarting the engine alone. Every cross-era comparison in the C-series table
carries that much systematic uncertainty, which is exactly why this test was run paired. Do not read
D8/D8F/D12/D12F against C1-C12 rows.

Also note `C12 = 33.6 ± 26.7` in the sweep is **destroyed data**, not a result — its reps 2-3 were
killed by the engine death (39 and 103 of 104 answers empty). **D12 = 63.8** is its valid re-run.
C12's row should be struck from any summary.

## Integrity

All 12 arm-reps: 103-104 answers each, empty-response counts 0-4 (the normal heavy-task tail),
`Output Table:` present in 100% of traces (operators genuinely executing — the check that catches
the engine-death mode), 8/8 JVMs and port 8085 up throughout. Fact-bearing runs: 19-21 per rep on
the F arms, **0** on both controls, `__file_io__` never leaked as a column.

## Throughput vs concurrency (exclusive engine)

| P5 (shared) | P8 | P12 | P18 | P30 |
|---|---|---|---|---|
| 2.0/min | 1.9/min | 3.5/min | 8.7/min | **13.4/min** |

1248 runs in 1h56m. The inherited "P6 causes instant-fails" note is stale; 0 instant-fails at every
level tested.

## Recommendation

1. **Ship the `Files read:` fact.** It is free-to-cheaper on every arm and worth +5.0 pt where the
   render budget is tight. Default it on.
2. **Do not pair it with a 5k budget expecting accuracy** — there it buys only the 11% cost cut.
3. The interesting follow-up is the interaction, not the fact: if a cheap engine-side observation
   recovers 5 of the ~7.5 pt gap between 1k and 5k at 1k's lower cost, then **other cheap facts may
   substitute for sampling budget too**. That is a better research direction than more render knobs,
   and it is testable with the `file_io_telemetry` pattern already in place.
4. Re-run the D12F result at 6-8 reps before publishing the number — 3v3 at 3.3× SE is strong but
   thin, and this is the first result worth defending.

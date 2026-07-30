# P-series — the CODE budget, and why byte-trimming cannot save money

**Run:** 2026-07-30 04:54–06:59. 3 arms × 3 reps × full 104 tasks = **936 runs**, `:3004`
@ `afc64b980`, one engine, task-major interleave so all arms share engine age by
construction. Accuracy is KramaBench's own score (per-SUT measures CSVs, `compute_scores.py`
formula). Integrity: 103-104 answers per arm, `(empty response)` 0-3, `Output Table:` in
101-104 traces per arm, 0 engine recycles.

## The lever

Byte accounting (`byte_accounting.py`, ~100 traces/arm) found the one large render component
never placed under a budget — `max_operator_result_char_limit` clamps table ROWS only:

| arm | code | rows | stats | schema |
|---|---|---|---|---|
| D8 5k | 39.4% | 46.2% | — | 7.1% |
| D12 1k | **33.9%** | 20.8% | 27.7% | 9.1% |

At a 1k row budget code is the LARGEST component. It is also concentrated on settled
operators: 73.4% of code bytes sit on operators other than the last one.

A cap, not a drop, because code size is long-tailed (p50 286 B, p90 2,014 B, p99 6,081 B,
max 16,400 B). `property.codeMaxChars` keeps head+tail and elides the middle.

## Result: accuracy parity, cost REGRESSION

| arm | acc | per-rep | $/task | code B | prompt B |
|---|---|---|---|---|---|
| P0 control | 70.8 ± 2.7 | 68.2 / 74.5 / 69.7 | **0.0152** | 2,471 | 9,589 |
| P1 cap 800 | 70.4 ± 0.6 | 69.8 / 71.3 / 70.2 | 0.0154 **(+1.2%)** | 1,582 (−36%) | 8,526 (−11.1%) |
| P2 cap 400 | 69.2 ± 3.2 | 73.6 / 66.8 / 67.0 | 0.0164 **(+7.4%)** | 1,161 (−53%) | 8,361 (−12.8%) |

Accuracy: P1 −0.4 pt (0.23× SE), P2 −1.6 pt (0.68× SE). Both inside noise — parity, as
expected from the failure census.

The knob worked mechanically: 36-53% of code bytes and 11-13% of the whole prompt removed.
**And it cost more.**

## Why — the cost model is output-dominated

| arm | uncached in | cached in | output | total | out tok | reasoning tok |
|---|---|---|---|---|---|---|
| P0 | $0.00198 | $0.00103 | **$0.01223** | 0.01524 | 6,114 | 4,868 |
| P1 | $0.00181 | $0.00100 | $0.01261 | 0.01542 | 6,304 | 5,035 |
| P2 | $0.00175 | $0.00104 | $0.01358 | 0.01637 | 6,790 | 5,375 |

Three facts kill the input-trimming thesis:

1. **84-86% of input is cached** at $0.025/M — input bytes are already nearly free.
2. **Output is ~80% of total cost** at $2/M.
3. **Hiding code makes the agent reason more**: output +11%, reasoning +10%. It re-derives
   what the elided code did.

Trading ~1,000 input bytes for ~200 reasoning tokens loses at an 8× price ratio
($0.25/M in vs $2/M out). Uncached input did fall 12% exactly as designed; it simply does not
matter at this scale.

## The unifying rule for every cost result so far

| lever | mechanism | cost |
|---|---|---|
| A7 `Files read:` fact | −0.5 steps | **−11%** ✓ |
| LayoutNew (fact above `Code:`) | 6.4 → 5.9 steps | **−11%** ✓ |
| char-budget C9 / C10 / N3 / N5 | fewer input bytes | ~0 |
| code cap P1 / P2 | fewer input bytes, more reasoning | **+1 to +7%** ✗ |

**Only levers that reduce STEPS reduce cost. Input-byte trimming cannot, because input is
cached and output dominates.**

This corrects the earlier framing that "context tuning is a cost lever, not an accuracy
lever." Sharper: byte-shaping is **neither**. The two levers that ever paid did so by changing
agent *behaviour* — fewer turns — not by shrinking the prompt.

## Recommendation

1. **Do not default `codeMaxChars` on.** Ship it as a capability for when a context WINDOW is
   the binding constraint (it removes 36-53% of code bytes at accuracy parity), not as a cost
   measure.
2. **Stop building input-trimming knobs.** Five arms across two families (char budget ×4, code
   cap ×2) now agree. The remaining render components (rows 21-46%, stats 18-28%) sit behind
   the same cache+output wall.
3. **Aim at STEP COUNT.** Every win in the program reduced turns. Cheap engine-side facts that
   pre-empt an exploration step are the demonstrated shape (A7: ~100 chars, −0.5 steps, −11%).
4. If accuracy is the goal, it is not here at all: 93.2% of failures are clean wrong answers
   with 0.0 tool errors, ~13% precision misses, median relative error 40%.

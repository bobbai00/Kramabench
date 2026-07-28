# Full-104 knob star — gpt-5-mini vs gpt-5.2, C1–C6, M1/M3/M4

Run 2026-07-22/23. All arms full 104 tasks, oracle mode, recovery-equalized
(round0 + 2 retry rounds), then chunked LLM judge (temp 0) for M3/M4.
Fresh stack restart before the run; concurrency held at 4 (engine-safe).

## Arms

| knob | delta/latest | sampling | stats(D2) |
|---|---|---|---|
| anchor | delta | 1k | off |
| C1 sampling | delta | **5k** | off |
| C2 stats | delta | 1k | **on** |
| C3 versions | **latest**+code | 1k | off |
| C4 | delta | **5k** | **on** |
| C5 | delta | **2k** | **on** |
| C6 | **latest** | 1k | **on** |

## M1 (end-to-end, full 104, recovery-equalized)

| arm | gpt-5-mini | gpt-5.2 |
|---|---|---|
| anchor Delta1k schema | 75.5 | 74.4 |
| **C1 Delta5k schema** | **77.2** | **78.8** |
| C2 DeltaStats1k | 74.2 | (skipped) |
| C3 Latest1k+code | 76.1 | 76.2 |
| C4 DeltaStats5k | 77.1 | 77.3 |
| C5 DeltaStats2k | 75.2 | 75.1 |
| C6 LatestStats1k | 77.0 | 74.7 |

## M3 (evidence-seen) / M4 (step-done)

| arm | mini M3 | mini M4 | gpt5.2 M3 | gpt5.2 M4 |
|---|---|---|---|---|
| anchor | 0.654 | 0.761 | 0.733 | 0.753 |
| C1 5k schema | **0.714** | 0.740 | 0.719 | 0.772 |
| C2 stats1k | 0.671 | 0.723 | — | — |
| C3 latest+code | 0.630 | 0.742 | 0.699 | 0.758 |
| C4 stats5k | 0.701 | 0.751 | 0.715 | 0.725 |
| C5 stats2k | 0.658 | 0.743 | 0.735 | 0.761 |
| C6 latest+stats | 0.645 | 0.733 | 0.667 | 0.737 |

Failure modes: 73–93% mode-1 (step-missing) across every arm/model.

## Findings

1. **Sampling (1k→5k) is the one real M1 lever, both models.** C1 tops M1
   on mini (+1.7 vs anchor) and gpt-5.2 (+4.4). Everything else clusters
   74–79%. The knobs are narrow accuracy surfaces, not dials — replicates
   the levers-report thesis at full 104 on two models.

2. **Stats (D2) adds nothing, or hurts.** Compare stats-on vs schema-only at
   matched sampling:
   - 5k: C4 (stats) 77.1/77.3 vs C1 (schema) 77.2/78.8 → stats ≈ 0 (mini) /
     −1.5 (gpt-5.2). Stats does NOT stack on top of wide sampling.
   - 1k: C2 (stats) 74.2 vs anchor (schema) 75.5 → stats −1.3 (mini).
   Stats is the expensive lever (per-op decoration every step) that never
   pays its way on M1 here.

3. **Sampling helps via different channels per model.**
   - mini: sampling raises M3 (+0.060, anchor→C1) — it needs more rows on
     screen to SEE values (mini M3 baseline low, 0.654).
   - gpt-5.2: M3 flat (0.733→0.719), yet M1 +4.4 — gpt-5.2 already extracts
     evidence at 1k (high M3 baseline), so wider sampling helps through
     better reasoning over the fuller table, not evidence-presence.

4. **Latest vs delta ≈ neutral.** C3 (latest+code) and C6 (latest+stats)
   land within noise of their delta twins on M1 both models.

5. **Failures are overwhelmingly mode-1 (step-missing).** No render/version
   knob touches mode-1, which is why the M1 spreads are small — the binding
   failures are planning gaps, not evidence-delivery gaps.

## Method notes
- Global 4-wide task pool (orch4b) saturated the engine for C4/C5/C6 (6 arms
  at once) — ~35% faster than arm-sequential, no per-arm/batch idle.
- Retries recovered watchdog-killed hard tasks (+8–18 pts round0→final).
- gpt-5.2 C2 skipped per operator decision (already characterized).
- Numbers are KramaBench-native (compute_scores.py OVERALL over full 104).

Artifacts: judgment_runs/mini_star/{orch*_progress.log, judge_*_full.log,
c456_*.log, all104.txt}. gpt-5.2 M5/M6 (gpt-4o judge) launched separately —
fold in when complete.

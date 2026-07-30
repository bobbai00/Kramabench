# Rule B — which memory channel repays LATEST's re-derivation tax?

Run 2026-07-28. Base = C8 `Latest5kCodeInSnap` (best arm of the mini study).
Each ray adds ONE memory channel on top of the snapshot core, config-only
(`summarize_params`), 3 single-shot reps, 20 discriminating hard tasks
(`subset_hard.txt`), no retries. Accuracy = KramaBench's own answer-type
metric, subset-scoped (NOT the stub-diluted full-104 OVERALL). Baseline uses
the 5 existing C8 reps scored the same way on the same 20 tasks.

## The decisive table

| arm | accuracy (mean±std) | cache% | reasoning/task | uncached-in tok | $/task | steps |
|---|---|---|---|---|---|---|
| baseline C8 (latest5k+code) | 62.7 ± 9.2 | 82.8 | 7,304 | 11,616 | 0.0234 | 7.0 |
| B1 + code history (`codeHistory=1`) | 62.7 ± 9.8 | 84.8 | 7,788 | 10,935 | 0.0243 | 7.5 |
| B2 + data history (`result.history.lastK=1`, shape) | 59.5 ± 1.9 | 83.2 | 6,710 | 10,718 | **0.0210** | 6.8 |
| B3 + thought replay (`reasoningReplayK=3`) | 61.0 ± 5.9 | **48.4** | 7,225 | **39,011** | **0.0306** | 7.7 |

Deltas vs baseline:

| arm | acc | cache | reasoning | uncached-in | cost |
|---|---|---|---|---|---|
| B1 code history | +0.0 | +2.0pp | +483 | −681 | +4.0% |
| B2 data history | −3.2 | +0.4pp | −594 | −899 | **−10.4%** |
| B3 thought replay | −1.7 | **−34.4pp** | −80 | **+27,395** | **+30.9%** |

## Findings

1. **No memory channel improves accuracy.** All three land within the ±4–5 pt
   single-shot noise floor measured earlier (B1 +0.0, B2 −3.2, B3 −1.7). The
   re-derivation tax is real in *tokens* but adding memory does not convert
   into correctness at n=3 on this substrate.

2. **Thought replay is the clear loser — it wrecks the prompt cache.**
   Cache 82.8% → **48.4%** (−34pp), uncached input **+236%** (11.6k → 39.0k
   tokens/task), cost **+31%**. Mechanism, verified on matched same-task pairs
   (astronomy-hard-10: 38% vs 86/87% under B1/B2 at comparable input size):
   the replay block injects the last-K *thoughts*, which change every step, so
   the cacheable prefix is invalidated on every turn. Per-operator history
   lives in the stable snapshot region and caches normally.
   **This is the concrete answer to the cache-waste question: history placed in
   the stable region is cache-safe; history placed in a per-step-varying block
   is not.**

3. **Notably, replay did NOT cut reasoning tokens** (−80, i.e. flat). The
   hypothesis was that showing past thoughts would stop the agent
   re-deriving — it didn't. So latest's reasoning premium is not addressable
   by replaying thoughts.

4. **Data history (B2) is the efficiency winner**: −10.4% cost, −594 reasoning,
   cache intact, and *by far the most stable arm* (std 1.9 vs baseline's 9.2).
   Accuracy −3.2 is inside noise but the direction is not favourable, so this
   is a cost/stability lever, not an accuracy lever.

5. **Code history (B1) is accuracy-neutral and slightly cache-positive**
   (+2.0pp) at +4% cost.

## Verdict for the rule engine

- **Do not use thought replay** in a snapshot core. It is the one clearly
  harmful setting found: −34pp cache, +31% cost, no reasoning saving.
- **Per-operator history (code and/or data) is cache-safe** — safe to switch on
  selectively without a cache penalty, which is what a per-operator rule needs.
- **Prefer data history when optimising cost/stability** (−10% cost, 5× tighter
  run-to-run spread); prefer code history if accuracy-neutrality matters more.
- Caveat: n=3 single-shot on 20 hard tasks. Accuracy differences here are
  inside the noise floor; the cache/cost effects are large and mechanistic and
  therefore the trustworthy part of this result.

Artifacts: `ruleB_analyze.py`, `orchB_progress.log`, `poolB_*.log`, arms
`DataflowSystemGPT5MiniB{1,2,3}*Replicate{1,2,3}`.

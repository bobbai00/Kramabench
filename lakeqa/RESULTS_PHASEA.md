# LakeQA Phase A — first results (2026-07-13)

SUT: `DataflowSystemGPT52DeltaStats5kD2` (our best KramaBench config: DELTA,
5k cap, column stats + Output Table profile), oracle-style over the
stratified 24-task subset (`subset_24.json`), graded by LakeQA's own
gpt-5-mini judge. One recovery pass on the three infra-shaped failures
(mirroring the KramaBench recovery-first protocol, scaled down).

## Headline

- **Main sweep: 11/23 (48%)** + the pre-run smoke task failed → 11/24.
- **After recovery: 12/24 (50%)** — EQA000444 flipped to PASS on a clean
  rerun (first attempt churned to the step cap under 3-way parallel load
  with two heavy tasks, 26→12 steps); EQA000461 failed identically twice
  (deterministic renderer bug, below); EQA000232 exhausted the 26-step cap
  on BOTH attempts (20.9 then 96.4 min — chronic, slow executions on the
  89 MB source, not load-transient).
- **Cost: $5.76 final scratch total — ~$0.24/task** (8M+ input tokens,
  steps med 13, max 26). A LakeQA task costs ~5–8× a KramaBench oracle task:
  14 sources/task, 3–5 reasoning hops, bigger tables.
- The agent is **not out of its depth**: passes at every k (3/4/5), at
  d-5 breadth (21 files), on the 470 MB task (EQA000916, 46.6 min grind),
  and on a 23-step negative-number aggregation (EQA000471, −1184 exact).

## Per-task

| task | k-d | result | steps | cost | note |
|---|---|---|---|---|---|
| EQA000229 | 3-2 | fail | 9 | $0.07 | wrong value (101/114 vs 115) across two attempts |
| EQA000230 | 3-2 | PASS | 7 | $0.05 | judge accepts entity variant ("Cut Bank Penguin") |
| EQA000232 | 3-2 | fail ×2 | 26 | $0.62 | step-cap exhaustion both attempts (20.9 / 96.4 min) |
| EQA000301 | 3-3 | PASS | 20 | $0.27 | |
| EQA000302 | 4-2 | fail | 6 | $0.07 | **format**: `1.3%` vs `1.3` (judge enforces exact format) |
| EQA000303 | 4-3 | fail | 12 | $0.13 | **stopped short**: school entity given, count wanted |
| EQA000304 | 3-3 | PASS | 9 | $0.08 | |
| EQA000312 | 5-3 | fail | 17 | $0.83 | wrong value (Ward 1 vs Ward 5; 383 MB task) |
| EQA000377 | 3-4 | fail | 17 | $0.26 | gave up ("N/A") |
| EQA000379 | 3-4 | fail | 18 | $0.34 | wrong value (1871 vs 1869) |
| EQA000443 | 3-5 | fail | 13 | $0.20 | **stopped short**: school name given, number wanted |
| EQA000444 | 4-3 | fail→**PASS** | 26→~12 | $0.66 | recovered clean (4.7 min) |
| EQA000461 | 4-1 | fail ×2 | 3 | $0.01 | **renderer overflow bug** (below) |
| EQA000471 | 4-2 | PASS | 23 | $0.36 | 28.7 min; exact −1184 |
| EQA000490 | 4-1 | PASS | 14 | $0.16 | |
| EQA000656 | 4-4 | fail | 26 | $0.54 | step-cap exhaustion ("None") — un-recovered |
| EQA000726 | 4-5 | PASS | 11 | $0.16 | 21-file breadth |
| EQA000751 | 5-1 | fail | 13 | $0.20 | wrong value (2443 vs 10926); 79 min execution crawl |
| EQA000758 | 5-2 | PASS | 6 | $0.06 | |
| EQA000760 | 5-2 | PASS | 11 | $0.27 | |
| EQA000761 | 5-2 | PASS | 9 | $0.08 | |
| EQA000802 | 5-3 | PASS | 16 | $0.44 | 18 min |
| EQA000914 | 5-4 | fail | 6 | $0.08 | **stopped short/presentation** (below) |
| EQA000916 | 5-4 | PASS | 13 | $0.18 | 470 MB, 46.6 min |

## Failure taxonomy (13 fails pre-recovery)

1. **Presentation, not capability (4)** — the most fixable class.
   EQA000914 is the exhibit: the agent solved every hop correctly
   (Bexar County → Spurs → 1999 Finals vs the Knicks → New York County) and
   then answered with the whole chain, where the judge demands the single
   final value ("Manhattan"; extra fields = fail). EQA000443/EQA000303
   answered an intermediate hop's entity where the final hop's number was
   asked. EQA000302 appended `%` to a correct number. LakeQA's judge is far
   stricter on output format than KramaBench metrics — the adapter prompt
   currently carries none of LakeQA's answer-format emphasis. A one-line
   prompt addition ("report ONLY the single final value in the requested
   format") is the obvious Phase A.1 experiment.
2. **Step-cap / load churn (3)** — 26-step exhaustions (EQA000232/444/656).
   EQA000444 passed cleanly on rerun (recovery-fixable); EQA000232 exhausted
   the cap on both attempts — a real step-budget/execution-speed limit, not
   noise. KramaBench needed ~6 steps median; LakeQA needs 13+ and its heavy
   tasks want more than 25.
3. **Wrong value (4)** — real reasoning/aggregation misses
   (EQA000229/379/312/751), typically choosing a plausible-but-wrong
   filter or aggregation grain. These are the genuine capability gap.
4. **Gave up (1)** — EQA000377 answered "N/A" after 17 steps.
5. **Renderer overflow (1, deterministic)** — see bug.

## BUG (dataflow-agent follow-up): unbounded render on pathological frames

EQA000461 dies identically on every attempt at step ~2 with
`litellm.ContextWindowExceededError`: **the request reached 1,325,861
tokens** (limit 272k; litellm log shows 4 oversized requests across the two
attempts: 1.32M ×2, 1.01M ×2 — deterministic, not transient). Both caps were
applied per the agent-service log (`maxOperatorResultCharLimit: 5000`,
`maxOperatorResultCellCharLimit: 3000`), yet the composed request grew ~5 MB.
Trigger: the task's first step creates five loaders including Wikipedia
prose and tab-parsed variants of comma-delimited .txt dumps — frames with
pathological shapes (single mega-column / mis-split columns). Some rendered
section must scale with an uncapped dimension (per-column stats/schema lines
are the prime suspect — every KramaBench table was ≤ ~40 well-formed
columns, so this path was never exercised). Needs a renderer unit test with
these exact frames; fix belongs in `agent-service` summarize, budgeted at
the section level.

This is external validity doing its job: a failure mode structurally
impossible on KramaBench's clean lake, surfaced by the first 24 LakeQA
tasks.

## What Phase A establishes

- **Adoption works end-to-end**: same artifact set, same tooling
  (`kb.py cost/traces/...` run unchanged on `LakeQA_<SUT>`), LakeQA's own
  judge, $6 and ~2h for a 24-task arm. Cheap enough for A/B arms.
- **The reasoning half of LakeQA is within reach** (~50% with zero
  LakeQA-specific tuning) but decidedly unsaturated — headroom in both
  directions (presentation fixes ↑, wrong-values are honest misses).
- **Ready next steps**, in increasing cost: (a) answer-format prompt line +
  rerun the presentation fails; (b) identical-config twin rerun to
  calibrate LakeQA's noise floor (required before any A/B claim); (c) the
  Delta-vs-Latest pair on this subset (does the KramaBench C3 separation
  replicate out-of-distribution?); (d) renderer overflow fix; (e) Phase B
  search tools over the full mini lake.

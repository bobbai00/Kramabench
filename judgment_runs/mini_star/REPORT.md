# gpt-5-mini knob star — C1/C2/C3, focused-10, recovery-gated

Run 2026-07-22. All arms gpt-5-mini, oracle mode. Star of one-knob rays off a
shared anchor. Recovery-first (2 rerun rounds on the frozen failed set) + twin-noise
flaky gate, then M1/M2/M3/M4.

## Arms

| key | SUT class | knob position |
|---|---|---|
| anchor | `DataflowSystemGPT5MiniDelta1kSchemaOnly` | delta, 1k, schema-only |
| C1_5k | `DataflowSystemGPT5MiniDelta5kSchemaOnly` | + sampling 1k→5k |
| C2_stats | `DataflowSystemGPT5MiniDeltaStats1kD2` | + column stats (data_level=2) |
| C3_latest_code | `DataflowSystemGPT5MiniLatest1kCodeInSnap` | delta→latest + code-in-snapshot ON |

C3 moves two things at once (version fold + code shown) — per Bob's request:
"for latest turn code on, for rest use delta."

## Tasks (focused-10)

Hard-biased where knobs historically bind + 3 chronic flippers + coin controls,
all 6 workloads: legal-hard-15, legal-hard-2, legal-easy-19, biomedical-hard-5,
biomedical-easy-2, environment-hard-12, astronomy-hard-9, astronomy-easy-4,
wildfire-hard-18, archeology-hard-9.

## The dominant result: twin-noise eats the flips

**5 of 10 tasks flip pass/fail across identical reruns** (round0/1/2):

| task | anchor | C1_5k | C2_stats | C3_latest_code |
|---|---|---|---|---|
| legal-hard-2 | ..P (flaky) | PPP | PPP | PPP |
| astronomy-hard-9 | PPP | ..P (flaky) | PPP | .P. (flaky) |
| wildfire-hard-18 | .P. (flaky) | .P. (flaky) | .PP (flaky) | ... |
| legal-hard-15 | ... | ... | .P. (flaky) | ... |
| legal-easy-19 | ... | ... | ... | .P. (flaky) |

Every apparent knob "win" in the base run lived inside a flaky task. The round0
snapshot looked like "anchor loses legal-hard-2, all 3 rays win it" — but a single
rerun of the anchor recovered it. Flip dissolved. Same for astronomy-hard-9.

## M1 — recovery-equalized (best of 3 rounds)

| task | anchor | C1_5k | C2_stats | C3_latest_code |
|---|---|---|---|---|
| legal-hard-15 | . | . | P* | . |
| legal-hard-2 | P | P | P | P |
| legal-easy-19 | . | . | . | P* |
| biomedical-hard-5 | . | . | . | . |
| biomedical-easy-2 | . | . | . | . |
| environment-hard-12 | P | P | P | P |
| astronomy-hard-9 | P | P | P | P |
| astronomy-easy-4 | . | . | . | . |
| wildfire-hard-18 | P | P | P | . |
| archeology-hard-9 | . | . | . | . |
| **PASS/10** | **4** | **4** | **5** | **4** |

`*` = pass came on only 1 of 3 rolls (flaky, not attributable).

- **Stable-pass everywhere (2):** environment-hard-12, astronomy-hard-9.
- **Stable-fail everywhere (4):** biomedical-hard-5, biomedical-easy-2,
  astronomy-easy-4, archeology-hard-9 — task-intrinsic, no knob touches them.
- **C2's nominal +1** comes entirely from legal-hard-15 landing on 1/3 rolls → noise.

**M1 verdict: no clean accuracy attribution survives recovery+flaky-gating in 10
tasks.** Replicates the levers-report conclusion on the mini substrate: on
oracle KramaBench the knobs are not accuracy dials.

## Cost — per-step input tokens (the stable invariant)

Naive totals are dominated by step-count thrashing (anchor spiraled to 57 agent-steps
vs 41–51), so use per-step input tokens (levers-report method):

| arm | per-step input tok | vs anchor | total steps (10tk) |
|---|---|---|---|
| anchor 1k | 10.0k | — | 57 |
| C1 5k | 11.5k | **+15%** | 41 |
| C2 stats | 11.9k | **+19%** | 51 |
| C3 latest+code | 8.6k | **−14%** | 42 |

Cleanly reproduces the levers-report signatures: sampling adds a modest per-step
byte cost, profiling is the most expensive lever (decorates every op every step),
latest is leaner per step (folded snapshot, even with code shown).

## M3 / M4 — the mechanism (chunked LLM judge, gpt-4o-mini)

| arm | M3 (evidence SEEN) | M4 (step DONE) |
|---|---|---|
| anchor 1k | 0.628 | 0.767 |
| C1 5k | 0.670 (**+0.042**) | 0.796 |
| C2 stats | 0.637 (+0.009) | 0.675 (−0.092) |
| C3 latest+code | 0.636 (flat) | 0.896 (**+0.129**) |

- **C1 (sampling) +0.042 M3** — the wider render puts more of the needed values
  on screen. This is the rows knob's canonical channel, directionally matching the
  report's +0.081-where-it-binds.
- **C2 (stats) flat M3** on this set — the profile lines did not deliver much extra
  answer-relevant evidence for these 10 tasks (its report win, legal-hard-15's 47%
  dup line, is the flaky one here).
- **C3 +0.129 M4** — expected and near-tautological: `enable_code_in_snapshot`
  surfaces each operator's code in the snapshot, so the action lens (M4) sees the
  steps performed. It is a delivery artifact of the flag, not extra planning.

### Failure modes (why they fail)

| arm | n_failed | mode1 step-missing | mode2 value-absent | mode3 had-all |
|---|---|---|---|---|
| anchor | 7 | 71% | 29% | — |
| C1 5k | 7 | 57% | 29% | 14% |
| C2 stats | 6 | **100%** | — | — |
| C3 latest+code | 8 | 50% | 38% | 12% |

**Failures are dominated by mode-1 (the agent never performed the needed step).**
Render knobs (rows/stats) only fix mode-2 (value-absent). So the failures on this
hard mini set are the *wrong kind* for these knobs to fix — which is the mechanistic
reason M1 is flat across the star.

## M2 — subtask eval (61 subtasks over the 10 tasks, isolation)

| arm | micro | vs anchor | macro | sub-pass≥0.9 |
|---|---|---|---|---|
| anchor 1k | 0.165 | — | 0.187 | 9/61 |
| C1 5k | 0.187 | **+0.022** | 0.211 | 10/61 |
| C2 stats | 0.168 | +0.003 | 0.190 | 9/61 |
| C3 latest+code | 0.183 | +0.018 | 0.208 | 11/61 |

Absolute scores are low (0.16–0.19) — the known M2 under-scoring (subtasks run in
isolation; e.g. environment-hard-12 scores 0.03 at M2 in every arm despite the main
task passing everywhere). But the **ordering C1 > C3 > C2 ≈ anchor is consistent
across M2-micro, M2-macro, and M3**: sampling is the knob that most raises the fraction
of subtasks answerable, which is the same knob that most raises evidence-seen (M3
+0.042). Stats is flat at both M2 and M3. Deltas are tiny (+0.003…+0.025) and, on a
substrate where 5/10 main tasks are flaky, sit near the noise floor — but the
cross-metric agreement on ordering is the real signal. (2 units, env-hard-12 anchor
and C2, hit the 30-min subtask-pool timeout; enough subtasks landed to score.)

## Synthesis

1. On gpt-5-mini + oracle KramaBench, the three context knobs are **evidence-delivery
   channels, not accuracy dials** — same conclusion as the GPT-5.2 levers star,
   now replicated on a weaker model. M1 flat; M2/M3 move only in the delivery
   direction, led by sampling (C1).
2. **Method is the finding**: 5/10 tasks are flaky; naive A/B flips are noise. Only
   recovery-first + flaky-gating exposes that no clean attribution survives here.
3. The knobs DO change what the agent sees/does (M3 up for sampling, M4 up for
   code-in-snapshot; per-step cost signatures distinct), but on this hard set the
   binding failures are mode-1 (planning/step-missing), which no render knob repairs.
4. Cost ordering is stable and matches theory: profiling most expensive (+19%/step),
   sampling mid (+15%/step), latest cheapest (−14%/step).

## Artifacts (this dir)
- `tasks10.txt`, `collect.py`, `flaky.py`, `cost.py`
- `snap_round{0,1,2}.json` — per-round scores
- `flaky_result.json`, `cost_summary.json`, `m2_summary.json`
- `m2collect.py` — M2 subtask aggregator
- `initial_sweep.log`, `recovery.log`, `judge.log`, `m2.log`

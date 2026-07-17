# ProbePrompt reruns — probing-issue tasks under the permanent probe prompt (2026-07-17)

Three fresh-control SUTs, config-identical to the C1/C2/C3 arms, NEW scratch
dirs (base folders untouched): `Delta5kSchemaOnlyProbePrompt`,
`Latest3kSchemaOnlyProbePrompt`, `DeltaStats3kD2ProbePrompt`. The only delta
vs their bases is the agent-service prompt vintage: the raw-probe principles
+ worked-example beats (dataflow-agent acf87127f → 57bd2fd0a). Task set: 9
probing-issue tasks + 2 clean controls, single roll each (recovery rounds NOT
yet run — chronic caveat applies to every pass/fail flip below).

## Before → after (score / steps / answer)

**Delta5kSchemaOnly → +ProbePrompt**
| task | old | new |
|---|---|---|
| astronomy-hard-9 | 0, 8st, '19' | 0, 11st, '1' |
| environment-hard-9 | 0, 5st | **0.8**, 4st |
| wildfire-hard-17 | 0.76, 17st, 3317.4 | **1.0, 11st, 4830.9 = GOLD** |
| archeology-hard-1 | 0, '0.2422' | 0, '10665.8' (regime fixed) |
| archeology-hard-5 | 0, '36828.7' | 0, '29207.5' |
| archeology-easy-8 | 0, '1' | 0, '82' (closer) |
| environment-hard-8 | 1 | 0 (chronic flip down) |
| environment-hard-11 | 1 | 0 (chronic flip down) |
| controls | pass | pass, +0–1 st |

**Latest3kSchemaOnly → +ProbePrompt**
| task | old | new |
|---|---|---|
| astronomy-hard-9 | 0, **26st $0.451, no answer** | 0, 12st $0.161, '3' |
| environment-hard-9 | 0.36 | **0.8** |
| wildfire-hard-17 | 0, **26st $0.189, no answer** | **0.82**, 22st, 3775.5 |
| archeology-hard-1 | 0, **26st $0.236, no answer** | 0, **7st $0.041, 8477.86** |
| archeology-hard-2 | 0, 19st | 0, 7st |
| environment-hard-8 | 1, 12st | 0, 18st (chronic flip down) |
| environment-hard-11 | 0 | **1** (chronic flip up) |
| controls | pass | pass, +1–2 st |

**DeltaStats3kD2 (best arm) → +ProbePrompt**
| task | old | new |
|---|---|---|
| astronomy-hard-9 | **1**, 11st $0.197, 24 | 0, 5st $0.075, '37' |
| environment-hard-9 | **1.0**, 8st | 0.0, 5st |
| wildfire-hard-17 | **1.0**, 10st, 4826.9 | 0.76, 7st, 3322.0 |
| archeology-hard-1 | 0, 14st, 380.8 | 0, 8st, **8477.86** |
| archeology-hard-5 | 0, '0.0260' | 0, **'64819.9'** (gold 66158.4) |
| archeology-easy-8 | 0, '872' | 0, '55' (gold 52) |
| environment-hard-8 | 1 | 1 |
| environment-hard-11 | 1 | 0 (chronic flip down) |
| controls | pass | pass, +1 st |

## Durable findings (single-roll pass flips are NOT among them)

1. **The prompt drives the behavior.** New traces open with `raw_*` preview
   operators and carry explicit `deleteOperator` cleanup (e.g. the stats-arm
   archeology-hard-1 trace: previews first, 6 deletes). The example beat
   transferred.
2. **Latest churn deaths eliminated, 3/3.** Every old 26-step no-answer on
   this arm now terminates with an answer (12/22/7 steps; those three tasks'
   cost $0.876 → $0.377, −57%). The probe procedure appears to break the
   mint-new-op spiral — a behavioral effect beyond parse quality, and the
   single strongest result of the rerun.
3. **Answer-variance collapse on archeology-hard-1: all three arms now
   produce the IDENTICAL 8477.8648** (pre-probe: 380.8 / 71,980 / 0.24 /
   no-answer scattered). 1.2% from gold 8577.53; the residual is the
   interpolate-at-two-points semantics, not the parse. archeology-hard-5
   similarly jumps from degenerate 0.026 to 64,820 (gold 66,158) on the best
   arm; easy-8 moves toward gold on all arms. The probe prompt converts
   scattered parse-garbage into stable near-gold answers on the dirty-header
   family.
4. **Controls: +0–2 steps, no losses** — the probing habit's overhead on
   clean tasks is small.
5. **HONEST negative to adjudicate:** subset pass counts moved 2→1 (5k),
   1→1 (Latest), and **5→1 on the best arm** — it dropped astronomy-hard-9,
   environment-hard-9, wildfire-hard-17, environment-hard-11 in these rolls.
   All four are chronic flippers, and the drops came with SHORTER
   trajectories (5–7 steps vs 8–11): hypothesis — probe-then-conclude can
   produce premature confidence on the stats arm. This is exactly what
   recovery rounds + a twin pair must adjudicate before any verdict; a
   single roll each way on chronic tasks decides nothing (twin noise flips
   9–12 tasks between identical configs).

## Next

1. Symmetric recovery (2× `rerun-failed --all-failed` equivalent on this
   subset) for the three ProbePrompt arms, then re-tally.
2. Full-benchmark ProbePrompt fresh-control arm (+ twin) vs the pre-probe
   base — required before any aggregate claim about the permanent prompt.
3. If the premature-confidence hypothesis survives recovery, tune the
   verify clause (principle 3) rather than reverting the probe habit.

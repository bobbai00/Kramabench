# Rank-4 paired analysis: DataflowSystemGPT52LatestStats3kD2SmallTableControl vs DataflowSystemGPT52LatestStats3kD2ProbeRetire

Shared task directories: 104 (control 104, treatment 104)

## Configuration check

Tasks with configs in both arms: 104
Settings keys differing: {'probe_retirement_config': 104}
Unexpected diffs (should be empty): (none)

## Rule activation (treatment arm)

Tasks with >=1 rendered activation: 14 / 104
Sum of final-step signature counts: 21

| Task | Agent steps | Steps w/ signature | Final-step signatures |
| --- | ---: | ---: | ---: |
| `archeology-hard-1` | 10 | 1 | 1 |
| `astronomy-hard-7` | 25 | 8 | 3 |
| `astronomy-hard-8` | 21 | 7 | 2 |
| `biomedical-easy-2` | 5 | 1 | 1 |
| `biomedical-easy-9` | 5 | 1 | 1 |
| `biomedical-hard-1` | 8 | 5 | 2 |
| `biomedical-hard-5` | 5 | 1 | 1 |
| `biomedical-hard-8` | 7 | 3 | 2 |
| `environment-hard-8` | 12 | 4 | 2 |
| `environment-hard-9` | 16 | 4 | 2 |
| `legal-hard-1` | 12 | 6 | 1 |
| `wildfire-hard-17` | 14 | 10 | 1 |
| `wildfire-hard-18` | 5 | 1 | 1 |
| `wildfire-hard-20` | 6 | 2 | 1 |

Control-arm signature leak (must be []): []

## Accuracy (pass = answer-type metric >= 0.9)

| Outcome | Tasks |
| --- | ---: |
| Both pass | 71 |
| Control only | 9 |
| Treatment only | 3 |
| Both fail | 21 |

Passes: control 80/104 (76.9%), treatment 74/104 (71.2%)

Accuracy divergences (independent trajectories — attribution needs the trace):

- `astronomy-easy-4`: control-only
- `astronomy-easy-6`: control-only
- `astronomy-hard-8`: control-only
- `astronomy-hard-9`: control-only
- `biomedical-hard-5`: treatment-only
- `biomedical-hard-7`: control-only
- `environment-hard-11`: control-only
- `environment-hard-12`: control-only
- `environment-hard-7`: control-only
- `legal-easy-19`: treatment-only
- `legal-hard-1`: treatment-only
- `legal-hard-14`: control-only

## Fair paired usage (tasks with stats.json in both arms)

Paired usage tasks: 103

| Measure | Control | Treatment | Treatment − Control |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $5.035086 | $5.085101 | $0.050015 (+0.99%) |
| input_tokens | 6,997,884 | 7,076,622 | +78,738 (+1.13%) |
| cached_tokens | 6,079,104 | 6,086,784 | +7,680 (+0.13%) |
| output_tokens | 168,813 | 163,407 | -5,406 (-3.20%) |
| total_tokens | 7,166,697 | 7,240,029 | +73,332 (+1.02%) |
| num_steps | 774 | 804 | +30 (+3.88%) |
| Uncached input | 918,780 | 989,838 | +71,058 (+7.73%) |
| Cache hit (control) | | | 86.9% |
| Cache hit (treatment) | | | 86.0% |

## Activation-split cost

| Cohort | n | Cost C→T | Uncached input | Steps C→T |
| --- | ---: | --- | --- | --- |
| Activated | 14 | $1.1057 → $1.6037 (**+45.0%**) | +74.6% | 130 → 165 |
| Non-activated | 89 | $3.9294 → $3.4814 (−11.4%) | −11.4% | 644 → 639 |

Caveats that make the +45% an over-read:

- **Selection bias**: activation conditions (probe settled ≥2 events, discovery
  encoded, healthy downstream) are more likely to be reached in LONGER
  trajectories — the cohort conditions on treatment-run length. The −11.4% on
  non-activated (cross-day control pairing) shows the day-to-day variance floor.
- **Same-step activated cases isolate the pure render effect**, and it is
  small: `biomedical-hard-1` 9/9 steps Δ$0.0000 (2 retirements),
  `wildfire-hard-20` 7/7 −$0.0009, `biomedical-easy-2` 6/6 +$0.0112,
  `biomedical-hard-5` 6/6 +$0.0305. One-time mid-prompt cache break ≈ cancels
  the trimmed bytes.
- No flapping: per-step retirement counts latch monotonically in 13/14 tasks
  (one transient in `biomedical-hard-8`). The step ballooning
  (`astronomy-hard-7` 14→26, `environment-hard-9` 6→17) is post-retirement
  answer-path iteration that never touches the retired probes — and
  `environment-hard-9`'s extra steps bought a treatment-only PASS.

## Accuracy-flip attribution

Control-only flips (9): only ONE (`astronomy-hard-8`) is an activated task;
the other eight are non-activated (rule rendered nothing; contexts
rule-untouched) and include chronic flippers from the delta_vs_latest_3k audit
(`astronomy-easy-4`, `astronomy-easy-6`, `environment-hard-7`,
`environment-hard-12`, `biomedical-hard-7`). The 9:3 imbalance is unlucky but
within the flip base rate seen across matched reruns of this benchmark.

**`astronomy-hard-8` — plausible rule-attributable regression.** Control
answered exactly gold (`[6.1655e-07, 5.1206e-07]`); treatment returned
degenerate `[0.0, 0.0]` RMSEs. The retired probes are the CDF
variable-inventory listers (`swarm_zvars_inventory`,
`swarm_acc_inventory_full`); retirement rendered from step 14 and the
model-building/eval steps (15–22) ran with the inventories hidden. A zero RMSE
indicates a constant/wrong variable — the exact "model later needs the probe
to distinguish sources" falsification mode: the selection LOOKED settled
(chosen variable encoded downstream) but re-opened when results came back
degenerate, and the fact line (1–3 matched values) does not carry the full
inventory. The task is also a chronic both-direction flipper, so variance
cannot be excluded; a checkpoint probe at step 14 would settle it. Treated as
a hard-stop signal per the audit's gate.

Counter-note: `biomedical-hard-5` — the canonical open-source-selection
negative control — was ACTIVATED and produced a treatment-only PASS; the
encoded-downstream guard held where it was designed to.

## Verdict against the acceptance gate

| Gate | Result |
| --- | --- |
| 1. No attributable accuracy regression | **AT RISK** — 1 plausible attributable failure (`astronomy-hard-8`, re-opened source selection); 8/9 other flips are non-activated variance |
| 2. Same-answer cases keep the same dataflow | PASS on inspected same-step cases |
| 3. Cache-aware `cost_usd` decreases | **FAIL** — +0.99% aggregate (flat), no win anywhere |
| 4. Steps/tool churn don't offset savings | FAIL on activated cohort (selection-biased, but no positive signal) |

**Rank 4 as implemented does not graduate.** The guard machinery works as
designed (clean latching, no control leakage, the biomedical-hard-5 guard
held), but there is no cost win to buy — retired probe tables are small
relative to the context, the mid-task render change spends a cache break that
cancels the trim, and the one plausible accuracy casualty is exactly the
audit's anticipated failure mode (a "settled" discovery that re-opens under
downstream failure). If pursued further, the fact line should carry the FULL
small inventory (not just matched values) for list-shaped probes, and the
retirement should apply only at a cache-stable boundary.

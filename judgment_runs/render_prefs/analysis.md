# Write-time render prefs (DELTA 5k): treatment vs stats-on vs schema-only

Shared tasks: 104. Comparators are July-6 vintage (code caveat).

## Accuracy (pass = metric >= 0.9)

| Arm | Passes | Rate |
| --- | ---: | ---: |
| stats-on (Delta5kD2) | 81/104 | 77.9% |
| schema-only | 78/104 | 75.0% |
| render-prefs | 77/104 | 74.0% |

vs stats-on: stats-on-only 8 ['astronomy-easy-4', 'biomedical-hard-7', 'environment-hard-13', 'environment-hard-8', 'legal-hard-1', 'legal-hard-15', 'wildfire-hard-16', 'wildfire-hard-17']; treatment-only 4 ['biomedical-easy-2', 'environment-hard-11', 'legal-hard-22', 'wildfire-hard-12']

vs schema-only: schema-only-only 6 ['astronomy-easy-4', 'biomedical-hard-7', 'environment-hard-13', 'environment-hard-8', 'legal-hard-15', 'wildfire-hard-16']; treatment-only 5 ['biomedical-easy-2', 'environment-hard-10', 'environment-hard-11', 'environment-hard-18', 'environment-hard-9']

## Knob usage (treatment)

Tasks with >=1 declaration: 103/104; pref-bearing calls: 880/974 create/modify calls (90%)
Declaration histogram: {'outputSummary=minimal': 552, 'showOutputStatistics=True': 414, 'showOutputStatistics=False': 229, 'outputSummary=standard': 327, 'showOutputStatistics=None': 0, 'outputSummary=None': 0}
deleteOperator calls: treatment 9 vs stats-on comparator 0

## Paired cache-aware usage

### render-prefs vs stats-on (103 paired tasks)

| Measure | stats-on | render-prefs | Δ |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $6.1112 | $5.9225 | $-0.1887 (-3.09%) |
| input_tokens | 7,431,921 | 7,328,704 | -103,217 (-1.39%) |
| cached_tokens | 5,900,032 | 6,087,552 | +187,520 (+3.18%) |
| output_tokens | 171,280 | 191,799 | +20,519 (+11.98%) |
| num_steps | 688 | 688 | +0 (+0.00%) |
| Uncached input | 1,531,889 | 1,241,152 | -290,737 (-18.98%) |

### render-prefs vs schema-only (103 paired tasks)

| Measure | schema-only | render-prefs | Δ |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $6.3191 | $5.9225 | $-0.3966 (-6.28%) |
| input_tokens | 7,020,637 | 7,328,704 | +308,067 (+4.39%) |
| cached_tokens | 5,435,136 | 6,087,552 | +652,416 (+12.00%) |
| output_tokens | 185,240 | 191,799 | +6,559 (+3.54%) |
| num_steps | 708 | 688 | -20 (-2.82%) |
| Uncached input | 1,585,501 | 1,241,152 | -344,349 (-21.72%) |

## Flip attribution (vs stats-on)

6 of the 8 stats-on-only flips are chronic direction-flippers across this
month's matched reruns (`astronomy-easy-4`, `environment-hard-8`,
`environment-hard-13`, `legal-hard-15`, `wildfire-hard-17`,
`biomedical-hard-7`). All 8 tasks used lean declarations somewhere — but so
does 63% of all operator traffic, so presence is not attribution. The two
non-chronic flips (`legal-hard-1`, `wildfire-hard-16`) leaned on answer-path
operators (a join, a chi-square) — plausible over-economizing, unproven. The
cross-vintage confound (comparators predate the small-table renderer rule)
cannot be excluded in either direction; a fresh code-matched control
(`DataflowSystemGPT52DeltaStats5kD2FreshControl`, launched 2026-07-12) is the
decisive comparison.

## Interim verdict

**First intervention in the program to reduce cache-aware cost.**

- vs stats-on: −3.09% cost at IDENTICAL total steps (688 = 688), with the
  signature inverted relative to every failed mutation experiment: raw input
  DOWN (−1.4%) while cached tokens UP (+3.2%) — uncached input −19.0%. The
  append-only design carries no cache tension, as constructed.
- vs schema-only: −6.28% cost — cheaper than BOTH poles while sitting between
  them informationally (schema-only pays for its missing stats in steps +20
  and uncached +22%).
- Adoption: 103/104 tasks declare; 90% of create/modify calls carry prefs;
  63% choose `minimal`; 36% turn stats off per-version; 9 deleteOperator
  calls (comparator: 0) — the hygiene principle also landed.
- Output tokens +12% (the declarations + slightly longer thoughts): ~$0.02,
  dwarfed by the input savings.
- Accuracy 74.0% vs 77.9% stats-on / 75.0% schema-only: −4 / −1 net, mostly
  chronic flippers; awaiting the fresh control for a code-matched read.

The label harvest (per-version state → declared level) is recorded in every
trace's toolCall inputs — the training set for distilling the write-time
policy into the offline selector.

## FINAL verdict — code-matched control (2026-07-12)

`DataflowSystemGPT52DeltaStats5kD2FreshControl` (current code, flag off,
same full+2-recovery protocol):

| | FreshControl | RenderPrefs | Δ |
| --- | ---: | ---: | ---: |
| Pass@0.9 | **84/104 (80.8%)** | 77/104 (74.0%) | **−6.8 pts** (9 vs 2 flips) |
| Cache-aware cost | **$5.4545** | $5.9225 | **+8.58%** |
| Steps | 699 | 688 | −11 |
| Uncached input | 1,219,392 | 1,241,152 | +1.78% |

**The interim "first cost win" is RETRACTED — it was entirely the vintage
gap.** The July-6 comparator was both weaker (81 vs 84 passes) and more
expensive ($6.11 vs $5.45) than the same configuration under current code;
measured against its true control, the treatment loses on both axes.

What the code-matched pair actually establishes:

1. **The mechanism is cache-benign as designed** (uncached +1.8%, steps
   −11) — the append-only construction holds. The cost regression is NOT
   cache churn; it is the feature's own overhead: the prompt
   fragment + example variant every step, and — in DELTA specifically — the
   declared params re-rendering inside every event's Action block for the
   rest of the trajectory (880 declarations × the remaining events), plus
   +12% output tokens.
2. **Self-chosen starvation is still starvation.** With 63% of versions
   declared `minimal`, the 9:2 flip imbalance is directional: the agent
   over-economizes when the prompt frames context as a budget. E1 taught
   this for imposed leanness; this run shows the agent's own frugality
   judgment is also poorly calibrated against task needs.
3. **Side finding:** current-code Delta-5k-D2 at 80.8% is the strongest
   Delta arm ever recorded — the permanent renderer rules (small-table
   suppression et al.) improved the baseline materially since July 6.

## Concrete fixes before a v2 arm

- **Strip pref params from the DELTA Action render** (renderToolCall): they
  are view metadata, not workflow semantics — re-rendering them in every
  event is pure declaration-carrying cost.
- **Recalibrate the prompt**: drop the budget framing; make `minimal` the
  marked choice for provably-settled ops only (e.g. "prefer standard while
  any downstream decision may depend on this data").
- Consider defaulting `showOutputStatistics` guidance to "leave on" — 36%
  stats-off likely removed join-hygiene signal (several flips are join/key
  tasks).

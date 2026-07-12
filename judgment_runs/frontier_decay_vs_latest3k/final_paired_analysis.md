# Final Paired Analysis: Latest 3k Control vs Frontier Decay

Generated 2026-07-10 after the full run and two `--all-failed` recovery rounds.

| Arm | SUT |
| --- | --- |
| Control | `DataflowSystemGPT52LatestStats3kD2SmallTableControl` |
| Treatment | `DataflowSystemGPT52LatestStats3kD2FrontierDecay` |

Pass means the answer-type-specific score is `>= 0.9`:

- exact numeric/string: `success`;
- approximate numeric: `rae_score`;
- exact list: `f1`;
- approximate list: `f1_approximate`;
- approximate string: `llm_paraphrase`.

Scores come directly from each final `evaluation.json`; usage comes from
`stats.json.cost_usd` and its token fields. The primary cost comparison uses
only tasks with usage artifacts in both arms.

Canonical accuracy command:

```bash
python bobflow_context_learning/analyze/compare_arms.py \
  DataflowSystemGPT52LatestStats3kD2SmallTableControl \
  DataflowSystemGPT52LatestStats3kD2FrontierDecay
```

## Coverage

The run universe contains 104 shared tasks. The current workload definitions
also contain newer `environment-hard-2` and `environment-hard-3`, but neither
arm ran them, so they are outside this comparison.

| Artifact | Control | Treatment |
| --- | ---: | ---: |
| Task directories / evaluations | 104 | 104 |
| Answers / ReAct traces / `stats.json` | 103 | 101 |
| Missing execution artifacts | `astronomy-hard-11` | `astronomy-hard-7`, `astronomy-hard-11`, `astronomy-hard-12` |

All missing executions have a zero-score watchdog evaluation. The three tasks
are both-fail outcomes, so they create no directional accuracy flip. Cost is
available for 101 paired tasks.

Control completed `astronomy-hard-7` and `astronomy-hard-12` but failed both;
their combined control cost is `$0.371138`. Assigning zero cost to treatment's
missing executions would therefore create a false treatment saving.

## Accuracy

| Outcome | Tasks |
| --- | ---: |
| Both pass | 76 |
| Control only | 4 |
| Treatment only | 3 |
| Both fail | 21 |

| Measure | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| Passes | 80 / 104 | 79 / 104 | -1 task |
| Pass rate | 76.9% | 76.0% | -1.0 percentage point |

Accuracy divergences:

| Direction | Task | Control answer | Treatment answer |
| --- | --- | --- | --- |
| Treatment only | `biomedical-hard-5` | `1.2775` | `2.6563` |
| Control only | `environment-hard-13` | `11` | `12` |
| Control only | `environment-hard-8` | `54.03` | `51.41` |
| Treatment only | `environment-hard-9` | two beach names | three correct beach names |
| Treatment only | `legal-easy-19` | `0.199` | `0.523` |
| Control only | `wildfire-hard-12` | `No` | `Yes` |
| Control only | `wildfire-hard-17` | `4830.9` | no response |

These are independent trajectories. The outcome matrix alone does not
attribute any flip to frontier decay.

## Fair Paired Usage

Totals over the 101 tasks with `stats.json` in both arms:

| Measure | Control | Treatment | Treatment - Control |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $4.663948 | $4.681166 | +$0.017218 (+0.37%) |
| Mean cost/task | $0.046178 | $0.046348 | +$0.000170 |
| Input tokens | 6,633,128 | 6,096,138 | -536,990 (-8.10%) |
| Cached input | 5,770,112 | 5,034,752 | -735,360 (-12.74%) |
| Uncached input | 863,016 | 1,061,386 | +198,370 (+22.99%) |
| Cache hit | 87.0% | 82.6% | -4.4 points |
| Output tokens | 153,136 | 138,761 | -14,375 (-9.39%) |
| Total tokens | 6,786,264 | 6,234,899 | -551,365 (-8.12%) |
| Agent steps | 734 | 721 | -13 (-1.77%) |
| Mean steps/task | 7.27 | 7.14 | -0.13 |

Frontier decay reduces raw input, but the treatment receives substantially
more uncached input and is slightly more expensive on the fair paired set.

For completeness, unmatched available totals are `$5.035086` over 103 control
tasks and `$4.681166` over 101 treatment tasks. They are not comparable because
the treatment total omits the two control executions costing `$0.371138`.

## Workloads

Accuracy uses all 104 evaluations:

| Workload | Tasks | Control pass | Treatment pass | BP | C only | T only | BF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Archeology | 12 | 4 | 4 | 4 | 0 | 0 | 8 |
| Astronomy | 12 | 8 | 8 | 8 | 0 | 0 | 4 |
| Biomedical | 9 | 8 | 9 | 8 | 0 | 1 | 0 |
| Environment | 20 | 16 | 15 | 14 | 2 | 1 | 3 |
| Legal | 30 | 27 | 28 | 27 | 0 | 1 | 2 |
| Wildfire | 21 | 17 | 15 | 15 | 2 | 0 | 4 |

Paired usage columns show `control / treatment`:

| Workload | Usage n | Cost | Input | Uncached input | Output | Steps | Cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Archeology | 12 | $0.574931 / $0.492662 | 976,639 / 722,729 | 85,631 / 109,609 | 19,225 / 13,825 | 100 / 82 | 91.2% / 84.8% |
| Astronomy | 9 | $0.851502 / $0.829545 | 865,560 / 880,934 | 192,792 / 207,782 | 28,313 / 24,866 | 79 / 84 | 77.7% / 76.4% |
| Biomedical | 9 | $0.308905 / $0.402582 | 525,468 / 548,272 | 70,172 / 103,728 | 7,602 / 10,233 | 60 / 65 | 86.6% / 81.1% |
| Environment | 20 | $1.104755 / $0.862118 | 1,527,042 / 1,060,144 | 195,458 / 197,040 | 37,834 / 26,161 | 152 / 122 | 87.2% / 81.4% |
| Legal | 30 | $0.782560 / $0.837759 | 1,174,229 / 1,189,441 | 111,061 / 133,697 | 28,725 / 29,931 | 175 / 179 | 90.5% / 88.8% |
| Wildfire | 21 | $1.041295 / $1.256500 | 1,564,190 / 1,694,618 | 207,902 / 309,530 | 31,437 / 33,745 | 168 / 189 | 86.7% / 81.7% |

Environment's raw reduction is offset by Biomedical, Legal, and especially
Wildfire. Every treatment workload has a lower cache-hit rate.

## Both-Pass Cost

All 76 both-pass tasks have usage in both arms.

| Direction | Tasks | Gross saving |
| --- | ---: | ---: |
| Control cheaper | 40 | $0.520031 |
| Treatment cheaper | 36 | $0.429351 |
| Equal | 0 | $0 |

Treatment is net `$0.090680` more expensive across both-pass tasks.

| Both-pass usage | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| Cost | $3.147913 | $3.238593 | +$0.090680 (+2.88%) |
| Input | 4,352,613 | 4,117,607 | -235,006 (-5.40%) |
| Cached input | 3,747,328 | 3,400,320 | -347,008 |
| Uncached input | 605,285 | 717,287 | +112,002 (+18.50%) |
| Output | 102,349 | 99,163 | -3,186 |
| Steps | 506 | 511 | +5 |

Using the existing material-gap filter (`>= $0.005` and `>= 10%` relative to
the cheaper arm), control is materially cheaper in 24 cases for `$0.491378`
gross savings; treatment is materially cheaper in 15 for `$0.399438`.

## Same-Behavior Proxy

Within the 76 both-pass tasks:

- 73 have the same normalized answer;
- 30 have the same step count;
- 31 have the same coarse final shape;
- 24 satisfy all three conditions.

Coarse shape means operator count, link count, and operator-type multiset. It
does not prove identical code.

| Strict cohort usage | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| Tasks | 24 | 24 | 0 |
| Cost | $0.430931 | $0.417677 | -$0.013254 (-3.08%) |
| Input | 748,585 | 742,113 | -6,472 |
| Cached input | 667,904 | 660,480 | -7,424 |
| Uncached input | 80,681 | 81,633 | +952 |
| Output | 12,347 | 11,374 | -973 |
| Steps | 115 | 115 | 0 |

Treatment is cheaper in 14 strict cases for `$0.031051` gross savings; control
is cheaper in 10 for `$0.017797`.

Only one strict treatment trace, `wildfire-hard-20`, actually contains the
frontier-decay rendering signature. Both arms answer `0.0465` with seven steps
and a five-operator/three-link shape:

| `wildfire-hard-20` | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| Cost | $0.035665 | $0.043255 | +$0.007590 |
| Input | 62,233 | 55,674 | -6,559 |
| Cached input | 53,632 | 41,472 | -12,160 |
| Uncached input | 8,601 | 14,202 | +5,601 |

This confirms the central aggregate result: frontier decay can shorten the raw
prompt while losing enough prefix-cache reuse to increase billed cost. Because
the other 23 strict traces do not fire the rule, their small net saving cannot
be attributed to frontier decay.

## Conclusion

The final recovered treatment is one pass lower and `$0.017218` more expensive
on the fair 101-task cost set. It reduces raw input by 8.1%, but cache hit falls
4.4 points and uncached input rises 23.0%.

These results reject the current overlay as a cost-saving default under this
prompt/cache layout. They do not reject the lifecycle eligibility principle:
the direct payload audit proves that it removes settled context. A follow-up
must make the rendering cache-stable or use checkpointed same-trajectory probes
before changing eligibility breadth.

# GPT-5.2 Latest 3k Baseline

Generated 2026-07-10 from
`system_scratch/DataflowSystemGPT52LatestStats3kD2`.

## Coverage and accuracy

Pass means the task's answer-type-specific metric is at least `0.9`, matching
`bobflow_context_learning/analyze/compare_arms.py`.

| Measure | Baseline |
| --- | ---: |
| Historical task directories | 104 |
| Evaluation artifacts | 104 |
| Answer-aware passes | 77 / 104 (74.0%) |
| Complete `stats.json` / ReAct traces | 103 / 104 |
| Missing execution artifact | `astronomy-hard-11` |

The current workload JSONs now contain 106 tasks. The historical baseline does
not contain the newer `environment-hard-2` and `environment-hard-3`, so the new
run must be compared on the 104 shared tasks unless those two baseline tasks are
run separately.

## Usage

Usage totals cover the 103 tasks with `stats.json`; the missing task is not
silently assigned zero cost.

| Measure | Value |
| --- | ---: |
| Cache-aware cost | $4.648098 |
| Mean cost per covered task | $0.045127 |
| Agent steps | 729 |
| Input tokens | 6,187,400 |
| Cached input tokens | 5,296,640 (85.6% of input) |
| Uncached input tokens | 890,760 |
| Output tokens | 154,454 |
| Reasoning tokens | 0 |
| Total tokens | 6,341,854 |

## Workload breakdown

Accuracy uses all historical tasks. Cost and token columns use only tasks with
`stats.json`, hence astronomy has usage coverage 11/12.

| Workload | Tasks | Passes | Usage n | Cost | Steps | Input | Cached | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Archeology | 12 | 5 (41.7%) | 12 | $0.402064 | 72 | 628,348 | 539,392 | 10,857 |
| Astronomy | 12 | 4 (33.3%) | 11 | $1.376548 | 134 | 1,391,400 | 1,135,104 | 52,099 |
| Biomedical | 9 | 8 (88.9%) | 9 | $0.383230 | 68 | 589,910 | 501,504 | 10,054 |
| Environment | 20 | 14 (70.0%) | 20 | $0.891250 | 138 | 1,203,305 | 1,043,328 | 30,622 |
| Legal | 30 | 28 (93.3%) | 30 | $0.773205 | 171 | 1,144,117 | 1,025,792 | 27,616 |
| Wildfire | 21 | 18 (85.7%) | 21 | $0.821801 | 146 | 1,230,320 | 1,051,520 | 23,206 |
| **Total** | **104** | **77 (74.0%)** | **103** | **$4.648098** | **729** | **6,187,400** | **5,296,640** | **154,454** |

## Small-table stats confound

The permanent renderer rule suppresses stats only when a table has fewer than
five rows and is provably complete. Historical ReAct files do not retain the
backend's exact `OperatorInfo.truncated === false` flag, so exact eligibility
cannot be reconstructed. A conservative trace-level proxy required:

- `Output Table: N rows` with `N < 5` and a stats block;
- every logical row index visible, with no row-gap marker;
- no `[truncated]` marker; and
- at most 50 columns, so the renderer did not project columns.

| Trace-derived quantity | Count |
| --- | ---: |
| ReAct traces scanned | 103 |
| Traces with at least one proxy-eligible result | 85 (82.5%) |
| Unique task/operator results | 142 |
| Repeated prompt renderings | 440 |
| Row-count distribution over unique results | 0: 13, 1: 107, 2: 6, 3: 11, 4: 5 |
| Stats text across repeated renderings | 147,013 characters |
| Repository `chars / 4` token estimate | about 36,753 tokens |
| Estimated share of baseline input | about 0.59% |

Before the truncation-marker exclusion, 87 traces, 144 task/operator results,
and 451 renderings contained a sub-five-row table plus stats. The excluded cases
were `astronomy-easy-5/tle_raw` and `legal-hard-2/metro_html`.

Therefore the permanent rule is broad across tasks but small in direct prompt
volume. The approximately 36.8k-token estimate is an upper bound on its direct
baseline-context reduction, not expected billed-token savings: most repeated
text may be cached, and removing text can change later prefix-cache alignment.
Any materially larger saving in a same-step, same-workflow treatment cohort is
evidence for frontier decay or behavior change rather than this rule alone.

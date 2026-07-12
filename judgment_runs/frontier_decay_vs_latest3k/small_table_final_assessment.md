# Permanent Small-Table Stats Suppression: Final Assessment

Generated 2026-07-10 from:

- pre-rule historical run: `DataflowSystemGPT52LatestStats3kD2`
- recovered small-table-only control: `DataflowSystemGPT52LatestStats3kD2SmallTableControl`

The frozen `final_control_scratch/` copy is byte-for-byte identical to the live
control directory. Both arms use GPT-5.2, Latest context, a 3,000-character
result cap, StatsD2, column stats, 25 maximum steps, reflection, and parallel
tool calls. Frontier decay is off in both (`missing` historically, explicit
`null` in the new control).

## Bottom line

The permanent rule is a good fundamental rule because its justification is
information equivalence: if every cell of a complete table is visible, its
column statistics are deterministic functions of already-visible data. It
does not depend on the operator being a source, intermediate, numeric, or
one-column operator.

The benchmark is consistent with accuracy safety but is not a causal proof.
Accuracy rose by 3 passes, while cost also rose because the new independent run
took 45 more steps. The direct removed-text estimate is only about 0.6%-0.7%
of input-token occurrences, so neither the accuracy change nor the much larger
aggregate token change should be attributed to the rule.

A proof review found one missing gate: visualization rows replace HTML/JSON
payloads with `<skipped: visualization content>`. The implementation now
excludes `isViz`, and its targeted regression test passes. This closes the only
identified case where row counts could agree while actual cell content was not
visible.

## Shared-task comparison

Pass means the answer-type-specific metric is at least `0.9`, matching
`bobflow_context_learning/analyze/compare_arms.py`.

| Measure | Historical | New control | Control - historical |
| --- | ---: | ---: | ---: |
| Shared evaluated tasks | 104 | 104 | - |
| Passes | 77 (74.0%) | 80 (76.9%) | +3 (+2.9 points) |
| Paired usage tasks | 103 | 103 | - |
| Cache-aware cost | $4.648098 | $5.035086 | +$0.386988 (+8.3%) |
| Agent steps | 729 | 774 | +45 (+6.2%) |
| Input tokens | 6,187,400 | 6,997,884 | +810,484 (+13.1%) |
| Cached input | 5,296,640 | 6,079,104 | +782,464 (+14.8%) |
| Uncached input | 890,760 | 918,780 | +28,020 (+3.1%) |
| Output tokens | 154,454 | 168,813 | +14,359 (+9.3%) |
| Total tokens | 6,341,854 | 7,166,697 | +824,843 (+13.0%) |
| Cache hit rate | 85.6% | 86.9% | +1.3 points |

`astronomy-hard-11` lacks `stats.json` and `react_steps.json` in both arms, so
it is included in accuracy but excluded symmetrically from usage. No missing
task is assigned zero cost.

Accuracy outcomes are 73 both-pass, 20 both-fail, 7 new-control-only, and 4
historical-only. The task-level cost median actually favors the new control by
$0.000858, and it is cheaper on 60 of 103 tasks, but a few long trajectories
make its aggregate more expensive. Among the 73 both-pass tasks it still takes
29 more steps and costs $0.258162 more. This is trajectory variance, not a
credible estimate of the renderer rule's cost.

## Direct activation

The recovered control's ReAct prompts were parsed for the rule signature:

- `Output Table: N rows` with `N < 5`;
- all `N` row indices visible, without a row-gap or cell-truncation marker;
- a typed `Schema` line; and
- no `Column Schema and stats:` block.

Because frontier decay is disabled, this signature isolates permanent
small-table suppression, subject to the trace not retaining every backend flag.

| Activation measure | New control |
| --- | ---: |
| Completed ReAct traces | 103 |
| Tasks with a signature | 88 (85.4%) |
| Distinct task/operator pairs | 156 |
| Distinct task/operator/table results | 165 |
| Repeated prompt renderings | 571 |
| Repeated rendered columns | 1,229 |

The 165 distinct results are mostly simple, but not uniformly so:

| Characteristic | Count |
| --- | ---: |
| Row counts | 0: 14, 1: 125, 2: 6, 3: 13, 4: 7 |
| One-column results | 81 |
| At most three columns | 145 |
| Maximum columns | 19 |
| DataProcessing results | 135 |
| DataLoading results | 30 |
| Schema type mentions | numeric 232, string 122, bool 4, datetime 3, list 3, binary 1 |

Thus the common characteristic is **complete, directly inspectable content**,
not integer columns or intermediate topology. The rule reaches both sources and
derived operators and includes strings and structured types. Most activations
are one-row aggregates, but making that a requirement would unnecessarily miss
safe 0-, 2-, 3-, and 4-row cases.

The historical pre-rule proxy measured 147,013 characters of stats text
across 440 eligible repeated renderings, approximately 36,753 token occurrences
using the repository's `chars / 4` approximation (0.59% of historical input).
Scaling that observed text density to the recovered control gives a sensitivity
band of approximately 45,000-47,700 removed token occurrences:

- about 45,000 when scaled by repeated rendered-column count;
- about 47,700 when scaled by rendering count.

That is approximately 0.64%-0.68% of the control's raw input tokens. It is an
order-of-magnitude direct-context estimate, not a billed-token estimate:
renderings repeat in cached prefixes, and changing a prefix can change cache
alignment. It is also far smaller than the observed +810,484 input-token run
difference, confirming that independent trajectories dominate the aggregate.

## Accuracy regressions

The four historical-only cases do not show a mechanism in which omitted tiny
table statistics hid decisive information:

| Task | New-control failure | Suppressed tiny results |
| --- | --- | --- |
| `archeology-hard-7` | Used a latitude/longitude box instead of Euclidean distance, yielding 294 instead of 274. | Final one-row count only. |
| `biomedical-hard-5` | Took the median of `Log2_variant_per_Mbp` without exponentiating, yielding 1.2775 instead of 2.6563. | Exact 1-/3-row sheet-name lists and final one-row result. |
| `legal-easy-19` | Divided by `Number of Fraud Reports` instead of `Reports with $ Loss`, yielding 0.199 instead of 0.523. | Final one-row result only. |
| `wildfire-hard-14` | Included `Unhealthy for Sensitive Groups Days` in a different definition of generally unsafe days, yielding 0.42 instead of 0.64/0.65. | Final one-row correlation only. |

In each case the semantic choice is already encoded in the processing code
before the final tiny result is rendered. The full rows remain visible, so the
removed statistics contain no unique fact that would repair the transform.
This is useful negative evidence, but four stochastic regressions are not a
formal non-inferiority test.

## Static proof gates

The current predicate suppresses only the full stats block while retaining the
shape, TSV rows, typed schema, format hints, coercion diagnostics, and structural
hints. Its existing gates are:

1. `resultMode === "table"`.
2. `detail !== "shape"`.
3. Reported total rows are strictly below five; five is protected.
4. The renderer holds exactly the reported total number of rows.
5. The backend explicitly reports `truncated === false`.
6. No renderer row cap removed rows.
7. No wide-table projection removed columns (maximum 50 visible columns).
8. No nested cell contains `...[truncated]...`.
9. The result is not a visualization row whose HTML/JSON payload is hidden.

Tests cover the four-row positive case and negative cases for the five-row
boundary, backend truncation, unknown completeness, local row sampling,
non-table modes, held/reported row-count disagreement, nested cell truncation,
hidden visualization payloads, shape-only rendering, and wide-column
projection. The zero-row case is also pinned: stats are suppressed only while
its typed schema remains visible. The focused test command passes all 44 tests:

```bash
cd agent-service
bun test src/agent/tools/result-formatting.test.ts
```

## Recommendation

Keep the rule with the visualization exclusion. It is the strongest kind of
static cost rule available here: suppress only information that is
recomputable from an exact visible representation, and fail closed whenever
completeness or visibility is uncertain. Do not add type, source/intermediate,
or terminal-topology restrictions; the recovered activations show those are
correlations, not the fundamental safety condition.

For a causal dollar estimate, instrument rendered stats characters/tokens at
the formatter boundary or replay identical checkpoints with the block toggled.
Independent full benchmark runs are suitable for regression detection, but not
for estimating a sub-1% static context effect.

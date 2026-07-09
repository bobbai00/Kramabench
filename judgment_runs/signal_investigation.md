# Signal Investigation: Accuracy vs Cost

Scope: 12 GPT-5.2 dataflow arms:

- Context mode: `Latest`, `Delta`
- Result context limit: `3k`, `5k`, `7k`
- Schema/statistics: `StatsD2`, `SchemaOnly`

Pass threshold: score `>= 1.0`, matching `kb.py compare`.

## Selection Principles

### Accuracy signal

Keep cases where the richer arm passes and the leaner arm fails because the
richer arm had useful additional information.

Richer side assumptions for this pass:

| Dimension | Richer arm | Leaner arm |
| --- | --- | --- |
| Context mode | `Delta` | `Latest` |
| Sampling/context size | larger k | smaller k |
| Schema/statistics | `StatsD2` | `SchemaOnly` |

Reject cases where the richer arm wins for unrelated reasons, such as a unit
conversion bug, random extra exploration, or a different interpretation not
grounded in extra visible information.

### Cost signal

Keep cases where both arms pass, produce the same answer, and build the same or
near-same logical pipeline, but the richer arm costs more because it carries
unneeded context.

This is stricter than "both pass and one is cheaper". A candidate should have:

- same final answer,
- same or near-same operator/link structure,
- same task interpretation,
- cost gap explainable by larger input context, lower cache hit, stats text, or
  edit/history context.

## Candidate Mining

I compared all one-dimension pairs among the 12 arms.

Analyzer command:

```bash
python scripts/analyze_signal_cases.py --out-dir judgment_runs/signal_analyzer
```

Generated files:

- `judgment_runs/signal_analyzer/summary.json`
- `judgment_runs/signal_analyzer/accuracy_cases.csv`
- `judgment_runs/signal_analyzer/cost_cases.csv`
- `judgment_runs/signal_analyzer/top_accuracy_cases.csv`
- `judgment_runs/signal_analyzer/top_cost_cases.csv`
- `judgment_runs/signal_analyzer/manual_validation.md`
- `judgment_runs/signal_analyzer/manual_validation.csv`

| Candidate type | Sample | Stats/info | Mode |
| --- | ---: | ---: | ---: |
| Richer passes, lean fails | 70 | 38 | 44 |
| Both pass, same answer, same workflow-shape proxy, richer costs more | 69 | 29 | 33 |

The workflow-shape proxy is only a filter. Manual inspection is still required
because operator IDs, titles, and code style differ across runs.

Latest analyzer run:

| Metric | Count |
| --- | ---: |
| One-dimension A/B pairs | 24 |
| Shared task-pair rows | 2,473 |
| Accuracy flips, any direction | 252 |
| Accuracy flips where richer arm wins | 152 |
| Accuracy principle matches by rule | 106 |
| Cost candidates: same answer + same final shape | 767 |
| Cost principle matches by rule | 133 |
| Cost principle + same steps | 78 |
| Cost principle + same steps + code similarity >= 0.2 | 49 |
| Cost principle + same steps + code similarity >= 0.5 | 17 |

Accuracy principle matches by dimension:

| Dimension | Flip candidates | Richer wins | Rule matches | Reverse wins |
| --- | ---: | ---: | ---: | ---: |
| Sample/context size | 118 | 70 | 38 | 48 |
| Stats/info | 64 | 38 | 38 | 26 |
| Context mode | 70 | 44 | 30 | 26 |

Cost principle matches by dimension:

| Dimension | Same-answer same-shape candidates | Rule matches | Same steps | Strict: same steps + code sim >= 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Sample/context size | 390 | 70 | 321 | 10 |
| Stats/info | 195 | 30 | 163 | 4 |
| Context mode | 182 | 33 | 147 | 3 |

Manual validation batch:

| Signal | Audited | Accepted | Rejected | Main implication |
| --- | ---: | ---: | ---: | --- |
| Accuracy | 91 | 7 | 84 | Rule-based richer-wins counts are optimistic; most are transform/verification/source-selection false positives, but two clean sample-visibility cases now exist. |
| Cost | 90 | 51 | 39 | Same-pipeline cost cases are more reliable, but output-shape, transform, churn, and cache-effect differences still need filtering. |

See `judgment_runs/signal_analyzer/manual_validation.md` for case-level
evidence and `manual_validation.csv` for joinable labels.

Additional manual-audit result: accepted accuracy cases are rare but not zero.
The clearest accepted stats signals so far are Boston Harbor
`environment-hard-12` and biomedical `biomedical-hard-5`, where stats/schema
context exposed useful structural information (`Case_excluded` in the biomedical
case). We now have two accepted sample-size accuracy cases:
`legal-hard-15`, where two cross-state micropolitan rows sum to the exact answer
gap, and `wildfire-easy-3`, where the 7k trace exposes duplicate county rows for
the same state and leads to state-geometry dissolve while the 3k trace keeps one
county per state. Most audited sample wins still reject for source selection,
verification, file parsing, rate-definition, denominator choice, target
construction, or tool-recovery differences.

Latest batch result: `environment-hard-13` was rejected because the lean arm
computed the correct `11` before finalizing stale `12`; `astronomy-hard-8` was
rejected because the split/target construction differed from the gold script.
For cost, `wildfire-hard-16` is a clean same-pipeline sample-cost case, while
`legal-hard-24` is same-shape but cache-dominated and should not be counted as
pure sample compaction.

Newest batch result: `environment-hard-11` was rejected because the lean arm
collapsed repeated `Enterococcus` columns and used only Broadway; the grouped
header was visible. `astronomy-easy-4` was rejected as an evaluator-tolerance
case because the accepted rich answer still does not follow the gold
minima-period computation. For cost, `wildfire-easy-3` is a clean same-pipeline
stats-cost case, while `legal-hard-22` is same-pipeline but cache-dominated.

Batch 8 result: `environment-hard-12` mode accuracy is parser/layout rather
than Delta history, and `legal-hard-1` info accuracy mixes deduplication with
population-source selection. Two sample-cost candidates,
`environment-hard-13` and `environment-hard-12`, were rejected because the
richer runs took longer construction paths with repeated edits/tool-call churn
rather than just carrying larger samples through equivalent pipelines.

Batch 9 result: four more high-ranked candidates were rejected. Accuracy:
`astronomy-hard-10` is no-response/operator churn despite a plausible final
correlation workflow, and `environment-hard-10` is repeated-column parser logic
visible even in schema-only context. Cost: `astronomy-hard-9` is extra
construction work around the OMNI spec loader, and `environment-hard-14` is
cache-sensitive with different rainfall/exceedance aggregation, not clean
sample-size cost.

Batch 10 result: accuracy false positives continue to come from mechanisms
outside the proposed context dimensions: `astronomy-hard-9` is OMNI
path/parser/tool recovery, and `biomedical-hard-5` is B-APM sheet source
selection. Cost had one reject (`legal-hard-6`, repeated unused cleaning
operator and non-equivalent final construction) and one accepted stats-cost case
(`legal-hard-18`, same formula and answer with stats arm carrying more context).

Batch 11 result: `astronomy-hard-10` stats/info accuracy was rejected as
schema-only convergence/operator churn, and `environment-hard-9` mode accuracy
was rejected as beach parser/name normalization. Two sample-cost matches were
accepted: `environment-hard-10` preserves the all-station Wollaston correlation
pipeline, and `wildfire-easy-3` preserves the state-dissolve/NIFC-overlap
pipeline while the richer sample-size arms carry larger contexts.

Batch 12 result: accuracy matches again rejected for non-protocol mechanisms:
`astronomy-hard-7` is no-response/operator churn before modeling, and
`environment-hard-8` is generic beach parser/layout loss. Cost matches added
two accepts: `astronomy-easy-5` as same-pipeline stats-cost, and
`legal-hard-24` as same-pipeline sample-cost.

Batch 13 result: two context-mode accuracy matches were rejected as parser
layout rather than history effects: `astronomy-hard-9` differs in OMNI AP
parsing, and `environment-hard-10` differs in handling repeated Wollaston
station columns. Two sample-cost cases were accepted:
`environment-hard-15` and `biomedical-easy-9` both preserve the final logical
pipeline and answer while the larger-sample arm carries more input context.

Batch 14 result: two more accuracy matches were rejected. `environment-hard-9`
mode is beach parser/name normalization, and `wildfire-hard-17` info is
numeric-vs-string `NWS ID` normalization where the leading-zero clue was already
visible. Cost added two accepts: `legal-hard-7` as Delta-history cost, and
`wildfire-easy-3` as same-pipeline sample-size cost.

Batch 15 result: `biomedical-hard-7` mode was rejected as sheet-header
verification, and `astronomy-hard-9` sample was rejected as OMNI AP
parser/lag-transform logic. Two same-answer cost candidates were rejected:
`legal-hard-14` and `biomedical-easy-6` both have cache/extra-step dominated
cost gaps rather than clean context-size savings.

Batch 16 result: `legal-hard-15` mode was rejected as dedup/verification rather
than context mode, and `astronomy-easy-6` sample was rejected as endpoint-rate
versus average-step-rate logic. Cost added one accept (`legal-hard-30`
stats-cost, same fraud-vs-identity-theft comparison pipeline) and one reject
(`legal-hard-7` sample, dominated by extra edits and a tool error).

Batch 17 result: `environment-hard-9` info was rejected as beach parser/name
layout, and `biomedical-hard-7` sample was rejected as sheet-header
verification. Cost added one accept (`astronomy-easy-5` stats-cost, same TLE
48445 altitude-change pipeline) and one reject (`legal-hard-22`, equivalent
formula but cache/output driven rather than clean stats context).

Batch 18 result: `environment-hard-10` info was rejected as Wollaston
datasheet versus water-body-testing source/transform selection, and
`legal-hard-1` mode was rejected as MSA dedup/aggregation verification rather
than context mode. Cost added one reject (`legal-hard-30` mode, cache-driven
with nearly equal total tokens) and one accept (`wildfire-hard-20` 7k-vs-5k,
same 2008 NOAA Pareto pipeline with larger 7k input context).

Batch 19 result: two more mode accuracy matches were rejected as transform
errors: `legal-hard-18` omits the identity-theft category share in the Latest
arm, and `astronomy-easy-6` uses per-interval rate averaging rather than
endpoint period rates. Cost added one accept (`legal-hard-14`, same New England
MSA top-five pipeline with larger 7k context) and one reject
(`wildfire-hard-4`, extra diagnostics/raw-line parsing churn rather than pure
sample-size cost).

Batch 20 result: two stats/info accuracy matches were rejected:
`astronomy-hard-10` is SP3 parser/truncation because schema-only uses only the
first 600 lines per SP3 file for its final altitude series, and
`wildfire-hard-17` is RAWS identifier mapping/verification rather than hidden
stats. Cost added one reject (`legal-hard-30`, cache-driven despite equivalent
fraud-vs-identity-theft logic) and one accept (`wildfire-hard-20`, same 2008
NOAA Pareto pipeline with stats carrying larger context).

Batch 21 result: `environment-hard-11` info was rejected as repeated-column
beach parser coverage, and `biomedical-hard-5` mode was rejected as
metadata-vs-B-APM source-sheet selection. Cost added one accept
(`legal-hard-24`, same top-state/top-MSA pipeline with Delta carrying larger
context) and one reject (`environment-hard-18`, different monthly-vs-annual
trend aggregation).

Batch 22 result: `legal-hard-15` mode was rejected as cross-state MSA
dedup/aggregation and `environment-hard-8` info as beach parser/source
omission. Cost added two rejects: `legal-hard-8` sample has extra operator
rewrite/output, and `legal-hard-22` mode is cache/input-accounting rather than
clear Delta-history context.

Batch 23 result: `biomedical-easy-2` info was accepted as a clean
Case_excluded/stats accuracy signal, while `legal-hard-15` mode was rejected as
duplicate-MSA aggregation. Cost added one accept (`biomedical-hard-3`, same
lowest-APP-Z-score/age pipeline with stats carrying larger context) and one
reject (`astronomy-easy-5`, different TLE event definition despite same answer).

Batch 24 result: `environment-easy-2` info was rejected as rounded comparison
semantics and `environment-hard-10` sample as repeated-column parser coverage.
Cost added two accepts: `biomedical-easy-6` as same-pipeline stats-cost, and
`wildfire-hard-4` as Delta-history/context cost for the same NIFC
cost-per-acre pipeline.

Batch 25 result: `wildfire-hard-17` info was rejected as RAWS identifier
mapping/verification and `biomedical-hard-5` mode as source-sheet selection.
Cost added one reject (`legal-hard-22` 3k stats-vs-schema, cache/input
accounting) and one accept (`legal-hard-22` 7k-vs-3k stats, same-pipeline
sample-size cost).

Batch 26 result: `biomedical-easy-2` info was accepted as another clean
Case_excluded/stats accuracy signal, while `environment-hard-7` sample was
rejected as percent scaling. Cost added one accept (`environment-hard-20`,
same rainfall-min-city/top-beach pipeline with stats carrying more context)
and one reject (`legal-hard-22` mode, cache/input accounting).

Batch 27 result: `environment-hard-12` info was accepted as repeated-column
structure exposed by stats/schema context, while `environment-easy-2` mode was
rejected as rounded comparison semantics. Cost added two accepts:
`wildfire-hard-20` same NOAA Pareto pipeline with stats carrying larger context,
and `environment-easy-4` same Wollaston open-percentage pipeline with 7k
carrying larger context.

Batch 28 result: `environment-hard-7` info was rejected as percent scaling and
`wildfire-hard-17` sample as RAWS identifier mapping/verification. Cost added
one reject (`archeology-easy-10`, same two-op pipeline but cache/accounting
dominates) and one accept (`legal-hard-22` 7k-vs-5k schema-only, same formula
with larger 7k context).

Batch 29 result: `environment-hard-11` info was rejected as repeated-column
parser coverage and `legal-hard-18` mode as formula omission. Cost added one
reject (`wildfire-hard-11`, same acres-per-capita pipeline but cache/accounting
dominates) and one accept (`wildfire-easy-2`, same geospatial pipeline with
larger 7k context).

Batch 30 result: `legal-hard-15` info was rejected as MSA filter-scope/dedup
logic, and `environment-hard-13` sample as threshold/missing-value logic. Cost
added two accepts: `legal-easy-25` same military-branch median-loss pipeline
with stats carrying more context, and `legal-easy-11` same Other-report-share
pipeline with 7k carrying larger context.

Batch 31 result: `biomedical-hard-5` mode was rejected as B-APM source-sheet
selection and `environment-hard-13` mode as threshold/censor handling. Cost
added one accept (`environment-easy-5`, same four-region summer-rainfall
pipeline with 7k carrying larger context) and one reject (`archeology-easy-10`,
same mean-population pipeline but cache accounting dominates the dollar gap).

Batch 32 result: `astronomy-easy-4` sample was rejected as evaluator/paraphrase
behavior and `biomedical-hard-7` sample as header verification. Cost added two
sample-size accepts: `environment-hard-20` same rainfall-min-city/top-beach
pipeline with 5k carrying more context, and `legal-easy-11` same
Other-report-share pipeline with 7k carrying larger context.

Batch 33 result: `legal-hard-15` mode was rejected as duplicate-row
dedup/aggregation and `biomedical-hard-5` sample as `Case_excluded` filter
logic. Cost added one reject (`wildfire-easy-1`, repeated loader/debug churn)
and one accept (`legal-easy-25`, same military-branch median-loss pipeline with
5k carrying larger context).

Batch 34 result: `environment-hard-9` info was rejected as beach-name/parser
layout and `wildfire-hard-18` sample as evaluator/wording. Cost added two
accepts: `legal-hard-7` same relative-growth pipeline with Delta carrying extra
history, and `legal-hard-29` same MSA fraud-share pipeline with 7k carrying
larger context.

Batch 35 result: both `environment-easy-2` accuracy matches were rejected as
rounding/comparison semantics rather than stats or sample visibility. Cost
rejected `astronomy-hard-10` mode as output/accounting rather than Delta
history, and `wildfire-easy-1` sample as loader/debug churn rather than clean
sample-size cost.

## First Accepted Cost Signals

### Stats cost: `environment-easy-1`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Question: What percentage of 2013 Massachusetts beach samples exceeded
bacterial standards, causing temporary closures?

| Field | Rich: StatsD2 | Lean: SchemaOnly |
| --- | ---: | ---: |
| Answer | 4.796 | 4.796 |
| Cost | $0.018956 | $0.010501 |
| Total tokens | 23,862 | 22,686 |
| Cached tokens | 16,640 | 20,608 |
| Steps | 4 | 4 |
| Final operators | 2 | 2 |
| Final links | 1 | 1 |

Pipeline:

- load `water-body-testing-2013.csv`
- compute `Violation == "yes"` percentage over all rows

Mechanism: same logical pipeline and same result. The stats arm carries slightly
more input and has a much lower effective cache hit, so its cache-aware cost is
higher. This is a clean stats-cost example.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52DeltaStats3kD2/environment-easy-1/`
- `system_scratch/DataflowSystemGPT52Delta3kSchemaOnly/environment-easy-1/`

### Sampling/context-size cost: `environment-easy-5`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Question: Which region had the highest total rainfall during summer 2020?

| Field | Rich: 7k | Lean: 5k |
| --- | ---: | ---: |
| Answer | Ashburnham | Ashburnham |
| Cost | $0.057041 | $0.029144 |
| Total tokens | 44,740 | 40,910 |
| Cached tokens | 19,328 | 32,256 |
| Steps | 4 | 4 |
| Final operators | 5 | 5 |
| Final links | 4 | 4 |

Pipeline:

- load Boston, Chatham, Amherst, and Ashburnham monthly precipitation CSVs
- select `Year == 2020`, sum `Jun + Jul + Aug`, sort descending

Mechanism: same logical pipeline and same result. The 7k arm carries more input
context and has much lower cache reuse. The extra context does not change the
answer.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52Delta7kSchemaOnly/environment-easy-5/`
- `system_scratch/DataflowSystemGPT52Delta5kSchemaOnly/environment-easy-5/`

### Context-mode cost: `wildfire-hard-11`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats5kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats5kD2`

Question: Which state had the highest acres burned per capita?

| Field | Rich/history: Delta | Lean: Latest |
| --- | ---: | ---: |
| Answer | Wyoming | Wyoming |
| Cost | $0.015827 | $0.010339 |
| Total tokens | 19,976 | 18,499 |
| Cached tokens | 15,872 | 16,896 |
| Steps | 4 | 4 |
| Final operators | 2 | 2 |
| Final links | 1 | 1 |

Pipeline:

- load `Wildfire_Acres_by_State.csv`
- compute `Total Acres Burned / Population`, sort descending

Mechanism: same logical pipeline and same answer. Delta keeps more history/input
context; Latest reaches the same result with less cache-aware cost.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52DeltaStats5kD2/wildfire-hard-11/`
- `system_scratch/DataflowSystemGPT52LatestStats5kD2/wildfire-hard-11/`

## Accuracy Audit Status

### Stats accuracy: `environment-hard-12`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Question: Which Boston Harbor beach had the highest number of failed samples
when there was no rainfall in the preceding three days?

| Field | Rich: StatsD2 | Lean: SchemaOnly |
| --- | ---: | ---: |
| Answer | Wollaston Beach | Carson Beach |
| Ground truth | Wollaston Beach | Wollaston Beach |
| Cost | $0.165667 | $0.070423 |
| Steps | 5 | 5 |
| Tool calls | 21 | 11 |
| Final operators | 20 | 11 |

Preliminary mechanism: StatsD2 exposes enough schema/profile detail to make the
agent build per-beach, per-site long-form operators for repeated
`Tag`/`Enterococcus` columns. SchemaOnly uses a more generic parser and counts
only the visible repeated `Enterococcus` columns after basic cleanup, producing
Carson instead of Wollaston.

Status: accepted as a strong stats-accuracy candidate, but it is not a cost
candidate because the pipelines differ substantially.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52DeltaStats7kD2/environment-hard-12/`
- `system_scratch/DataflowSystemGPT52Delta7kSchemaOnly/environment-hard-12/`

### Rejected sampling/context-size accuracy: `biomedical-hard-7`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Question: How many are the significant genes by acetylproteomics?

| Field | Rich: 7k | Lean: 5k |
| --- | ---: | ---: |
| Answer | 16 | 15 |
| Ground truth | 16 | 16 |
| Cost | $0.022504 | $0.014780 |
| Steps | 8 | 5 |
| Final operators | 4 | 2 |

Trace evidence:

- Lean reads `D-SE-acetyl` with pandas default header inference. The visible
  result is 15 rows under column header `BRD8`, and it returns 15.
- Rich also sees that default-header output, but then loads the sheet with
  `header=None`. The raw output shows 16 rows, including row 0 `BRD8`, then it
  counts 16 significant genes.

Decision: reject for the sampling-accuracy bucket. The lean arm already saw the
decisive clue: `BRD8` appeared as the column header with 15 visible data rows.
The richer arm succeeded by doing a better raw-sheet verification with
`header=None`, so the mechanism is verification/header handling rather than
hidden sample visibility.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52LatestStats7kD2/biomedical-hard-7/`
- `system_scratch/DataflowSystemGPT52LatestStats5kD2/biomedical-hard-7/`

### Accepted sampling/context-size accuracy: `legal-hard-15`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Trace evidence:

- Rich visible context includes `LaGrange, GA-AL Micropolitan Statistical Area`
  and `Lebanon-Claremont, NH-VT Micropolitan Statistical Area`.
- Lean filters to `Metropolitan Statistical Area` and drops those cross-state
  micropolitan rows.
- The dropped values are `453` and `242`; their sum `695` exactly equals the
  answer gap between `243377` and `242682`.

Decision: accept for the sampling-accuracy bucket. This is the first audited
case that matches the strict mechanism: the richer sample exposes decisive
rows/values that the leaner context omits.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52Delta7kSchemaOnly/legal-hard-15/`
- `system_scratch/DataflowSystemGPT52Delta3kSchemaOnly/legal-hard-15/`

### Accepted sampling/context-size accuracy: `wildfire-easy-3`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Trace evidence:

- 7k visible context shows multiple Iowa county rows, including `Scott` and
  `Story`, with the same `adm1_id=USA-20230119-19`.
- 7k workflow dissolves county geometries with `groupby('adm1_name')` and
  shapely `unary_union`.
- 3k visible context shows only the first/last sample rows around `Lancaster`
  and `Story`, then the workflow uses `drop_duplicates('adm1_id')`.
- 7k returns `['California', 'Nevada']`; 3k returns a nine-state list.

Decision: accept for the sampling-accuracy bucket. This is a second audited
case where larger visible context changes a data-compaction decision that is
fundamental to correctness.

Raw artifacts:

- `system_scratch/DataflowSystemGPT52LatestStats7kD2/wildfire-easy-3/`
- `system_scratch/DataflowSystemGPT52LatestStats3kD2/wildfire-easy-3/`

### Rejected context-mode accuracy: `environment-hard-8`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats5kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats5kD2`

Question: What percentage of failed Boston Harbor beach samples had rainfall
within 24 hours before sampling?

| Field | Rich: Delta | Lean: Latest |
| --- | ---: | ---: |
| Answer | 54.03 | 51.56 |
| Ground truth | 54.03 | 54.03 |
| Cost | $0.315930 | $0.078135 |
| Steps | 10 | 8 |

Decision: reject for the context-mode accuracy bucket. Latest uses a brittle
generic block parser over raw beach files and leaks rainfall-like values into
Enterococcus. Delta builds and refines per-file tidy operators, including
multi-station beaches. That explains the result, but it is not clean evidence
that Delta context itself supplied the decisive missing information.

## Rejected Accuracy False Positive

### `environment-hard-7`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

The richer arm passes (`10.87`) and the lean arm fails (`0.11`), but manual
inspection shows the lean pipeline computed a fraction and failed to multiply by
100. This is a unit conversion bug, not a missing-information or sampling
visibility signal.

Decision: reject for the accuracy-information bucket.

## Next Investigation Targets

Prioritize:

1. More sampling/context-size accuracy cases, especially ones where the failed
   arm's visible result is truncated before the decisive row/value.
2. More same-pipeline cost cases for stats and sample size, using simple tasks
   where final operators are two or three nodes and steps are identical.
3. Context-mode cost cases where Delta and Latest build the same final DAG but
   Delta retains additional edit/history context.

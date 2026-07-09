# Manual Trace Audit: LatestStats5kD2 vs DeltaStats5kD2

Pair:

- A: `DataflowSystemGPT52LatestStats5kD2`
- B: `DataflowSystemGPT52DeltaStats5kD2`

Pass threshold: score `>= 1.0`, matching `kb.py compare`.

This is the first manual audit batch. The purpose is to explain mechanisms from
KramaBench artifacts, not to use an LLM judge as authority.

## Pair Summary

From `./kb.py compare --sut DataflowSystemGPT52LatestStats5kD2 DataflowSystemGPT52DeltaStats5kD2 --top 20`:

| Metric | A: Latest | B: Delta |
| --- | ---: | ---: |
| Tasks with results | 104 | 103 |
| Passing tasks | 74 | 79 |
| Pass rate | 71% | 77% |
| Total cost | $4.99 | $6.11 |
| Total tokens | 7,275,051 | 7,603,201 |

Shared-task buckets:

| Bucket | Count |
| --- | ---: |
| Both pass | 70 |
| A only pass | 4 |
| B only pass | 9 |
| Both fail | 20 |
| Both pass, A cheaper | 55 |
| Both pass, B cheaper | 15 |

## Audit Records

### environment-hard-8

Category: `B_only`

Question: What percentage of samples that failed the swimming standard at
Boston Harbor beaches had rainfall within 24 hours prior to sampling?

| Field | A: Latest | B: Delta |
| --- | ---: | ---: |
| Score | 0 | 1 |
| Answer | 51.56 | 54.03 |
| Ground truth | 54.03 | 54.03 |
| Cost | $0.078135 | $0.315930 |
| Steps | 8 | 10 |
| Total tokens | 76,934 | 245,131 |
| Cached tokens | 53,248 | 149,376 |

Primary reason: `wrong_transform`

Secondary reasons: `context_mode`, `verification`

First divergence: A created a generic `extract_samples` parser that assumed
regular 4-column blocks and adjacent repeated station fields. B instead built
per-file tidy operators and then refined the multi-station beaches separately.

Evidence:

- A final workflow has a generic `extract_samples` operator that loops over
  column blocks using positions like `date_col=start`, `rain1_col=start+1`,
  `ent_col=start+3`, and an adjacent `site_name=start+4`/`ent2=start+5`
  fallback.
- B final workflow has separate tidy operators for simple beaches and
  multi-station beaches, then concatenates the normalized rows before filtering
  `enterococcus > 104` and computing `(rain_1d > 0).mean() * 100`.
- B spent more steps and tokens, but corrected the schema-specific reshaping
  that A never repaired.

Counterfactual: A likely succeeds if it parses each beach file using the actual
per-file header/station layout, or verifies the failed-sample denominator and
per-beach contribution before returning the final percentage.

Confidence: high

Raw artifacts:

- `system_scratch/DataflowSystemGPT52LatestStats5kD2/environment-hard-8/`
- `system_scratch/DataflowSystemGPT52DeltaStats5kD2/environment-hard-8/`

### legal-hard-24

Category: `both_pass_A_cheaper`

Question: For the state with the highest number of reports across identity
theft, fraud, and other reports, which metropolitan area has the highest number
of identity theft reports?

| Field | A: Latest | B: Delta |
| --- | ---: | ---: |
| Score | 1 | 1 |
| Answer | Los Angeles-Long Beach-Anaheim, CA Metropolitan Statistical Area | Los Angeles-Long Beach-Anaheim, CA Metropolitan Statistical Area |
| Cost | $0.047727 | $0.152115 |
| Steps | 8 | 18 |
| Total tokens | 67,902 | 248,301 |
| Cached tokens | 59,136 | 222,336 |

Primary reason: `operator_churn`

Secondary reasons: `fewer_steps`, `smaller_context`

First divergence: B repeatedly rebuilt alternate state-report cleaning and
top-state operators before converging, while A used a direct state-total to
MSA-identity-theft path.

Evidence:

- A final workflow has 7 operators and 5 links:
  `state_it`, `state_all`, `msa_it_raw`, `state_all_clean`,
  `top_state_all`, `msa_it_clean`, `top_msa_it_in_top_state`.
- B final workflow has 16 operators and 21 links, including repeated alternate
  branches such as `state_reports_clean`, `top_state_total_reports`,
  `state_name_lookup`, `state_reports_clean2`, `top_state_name2`,
  `state_reports_fixed`, `top_state_fixed`, `state_fixed_reheader`,
  and `top_state_total_reports_including_it`.
- Both reach the correct answer, but B pays for redundant exploration and a
  larger retained DAG/edit trail.

Counterfactual: B would likely approach A's cost if obsolete state-cleaning
branches were pruned from context or if the run converged earlier on the correct
header/top-state interpretation.

Confidence: high

Raw artifacts:

- `system_scratch/DataflowSystemGPT52LatestStats5kD2/legal-hard-24/`
- `system_scratch/DataflowSystemGPT52DeltaStats5kD2/legal-hard-24/`

### wildfire-easy-2

Category: `both_pass_B_cheaper`

Question: Which NIFC geographic area intersects with the most US states? Give
the abbreviation of the geographic area.

| Field | A: Latest | B: Delta |
| --- | ---: | ---: |
| Score | 1 | 1 |
| Answer | EACC | EACC |
| Ground truth | EACC | EACC |
| Cost | $0.083017 | $0.033938 |
| Steps | 10 | 6 |
| Total tokens | 122,193 | 59,109 |
| Cached tokens | 97,792 | 50,304 |

Primary reason: `fewer_steps`

Secondary reasons: `operator_churn`, `smaller_context`

First divergence: A deduplicated state records by `adm1_id`, then repeatedly
rewrote `area_state_intersections` and added a `states_gdf` geometry-normalizing
operator. B dissolved county-level geometries into state polygons once, then
joined NIFC areas to dissolved states and counted distinct states.

Evidence:

- A has 9 agent steps and 9 `createOrModifyOperator` calls. It modifies
  `area_state_intersections` four times and ends with 6 operators:
  `usa_states`, `nifc_areas`, `states_adm1`, `states_gdf`,
  `area_state_intersections`, `area_state_counts`.
- B has 5 agent steps and 5 `createOrModifyOperator` calls. It ends with 5
  operators: `usa_states`, `nifc_areas`, `states_dissolve`,
  `area_state_intersections`, `area_state_counts`.
- B's input context is smaller as a consequence of fewer edits:
  58,315 input tokens vs A's 120,299 input tokens.

Counterfactual: A would likely match B's cost if it chose the dissolve-by-state
plan immediately and avoided repeated intersection rewrites and the extra
geometry-normalization operator.

Confidence: high

Raw artifacts:

- `system_scratch/DataflowSystemGPT52LatestStats5kD2/wildfire-easy-2/`
- `system_scratch/DataflowSystemGPT52DeltaStats5kD2/wildfire-easy-2/`

## Interim Pattern

The initial pattern is not simply "Delta is better" or "Latest is cheaper":

- Delta can improve accuracy when the task requires iterative correction of
  schema-specific transformations (`environment-hard-8`).
- Latest can be much cheaper when Delta accumulates redundant operators and
  alternative branches (`legal-hard-24`).
- Delta can also be cheaper when it converges to the right DAG earlier and keeps
  fewer edits in context (`wildfire-easy-2`).

This supports treating context mode, sampling/data level, and statistics/schema
as mostly orthogonal experiment dimensions, but the manual trace evidence shows
that their effects interact through operator churn, verification behavior, and
schema-specific transformation choices.

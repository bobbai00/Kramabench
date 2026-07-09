# Manual Validation of Signal Matches

Scope: first validation batch over analyzer matches from
`judgment_runs/signal_analyzer/`.

Method:

- No LLM judge was used.
- Each case was checked against raw artifacts under `system_scratch/<SUT>/<task>/`.
- Evidence came from `ground_truth.json`, `answer.json`, `evaluation.json`,
  `stats.json`, `workflow.json`, and `react_steps.json` / `inputMessages`.
- The goal was to decide whether a rule-based match really follows the two
  protocol principles:
  - accuracy: richer context gives useful additional information that changes a
    wrong decision into a correct one;
  - cost: same final result and same/near-same logical pipeline, but one arm
    costs more because it carries unnecessary context/history/stats/sample.

## Batch Summary

This batch is intentionally biased toward high-impact and suspicious matches,
not a random sample.

| Signal | Audited | Accepted | Rejected | Unclear |
| --- | ---: | ---: | ---: | ---: |
| Accuracy principle | 91 | 7 | 84 | 0 |
| Cost principle | 90 | 51 | 39 | 0 |

Main finding: the analyzer's accuracy count is optimistic. Many "richer wins"
are not true information-availability wins; they are wrong-transform,
verification, timeout, or unit-conversion cases. Cost matches are more reliable
when the final pipeline is simple and the code is visibly equivalent. For
Delta-mode cost, extra steps/edit history can itself be the unnecessary context,
so requiring identical step count would be too strict.

Batch 2 update: cost candidates remain much stronger than accuracy candidates,
but cache-aware pricing matters. Two apparent cost wins were rejected because
the cheaper arm did not actually carry less input context; it simply had more
cached input tokens.

Accuracy batch 2 update: six more apparent richer-wins cases were rejected.
Boston Harbor tasks repeatedly expose a parser/layout issue: the winning arm
often builds beach-specific parsers while the losing arm tries a generic
repeated-block parser. That is useful engineering evidence, but it is not a
clean signal for stats, sample size, or latest/delta context mode.

Batch 3 update: one more stats accuracy case was accepted
(`biomedical-hard-5`, exclusion handling). Most additional accuracy matches
still fail the protocol for transform/source-selection reasons. Cost matches
remain useful, but larger gaps often come from operator churn rather than
context compaction.

Batch 4 update: the first clean sample-visibility accuracy case was accepted:
`legal-hard-15`. The 7k arm exposed two cross-state micropolitan rows that the
3k arm dropped, and those two rows explain the exact answer gap. Most other
sample-size accuracy candidates still reject for trend definition,
aggregation/join semantics, target construction, denominator choice, or
verification/recovery depth. The cost batch added clean stats/sample cost cases,
but also several cache-effect rejects.

Batch 5 update: a second clean sample-visibility accuracy case was accepted:
`wildfire-easy-3`. The 7k trace visibly exposes duplicate county rows for the
same state, causing the agent to dissolve counties into state geometries; the 3k
trace keeps one county per state. A same-step sample-cost case
(`wildfire-hard-20`) was accepted, while `legal-hard-18` was rejected because
cache accounting dominates the cost gap. Another context-mode accuracy candidate
(`environment-hard-9`) was rejected as beach-label normalization.

Batch 6 update: two more accuracy candidates were rejected after raw-trace
inspection: `environment-hard-13` is verification/stale-finalization, and
`astronomy-hard-8` is target/split construction. `wildfire-hard-16` was accepted
as a same-pipeline sample-cost case; `legal-hard-24` was rejected because the
dollar gap is mostly cache accounting despite similar workflow shape.

Batch 7 update: two more stats/info accuracy candidates were rejected.
`environment-hard-11` is repeated-column beach parser/layout logic, not hidden
stats; `astronomy-easy-4` is an evaluator-tolerance issue because the accepted
rich answer still does not follow the gold minima-period computation. For cost,
`wildfire-easy-3` is a clean same-pipeline stats-cost case, while
`legal-hard-22` is a same-pipeline mode case rejected as cache-effect.

Batch 8 update: `environment-hard-12` mode and `legal-hard-1` info accuracy
were both rejected. The former is parser/layout handling where Latest keeps
only Carson rows; the latter combines dedup/aggregation and population-source
selection. Two high-gap sample-cost candidates, `environment-hard-13` and
`environment-hard-12`, were rejected as operator churn / non-identical
construction rather than pure sample cost.

Batch 9 update: four high-ranked remaining candidates were rejected.
`astronomy-hard-10` sample accuracy is a no-response/convergence case, not a
hidden sample row. `environment-hard-10` info accuracy is repeated-column
parser/layout logic. For cost, `astronomy-hard-9` is dominated by extra spec
loader edits in the stats arm, while `environment-hard-14` is cache-sensitive
and uses different rainfall/exceedance aggregation logic.

Batch 10 update: `astronomy-hard-9` stats accuracy was rejected as
path/parser/tool recovery around the OMNI spec and AP column, and
`biomedical-hard-5` sample accuracy was rejected as source selection
(`mmc7` B-APM sheet vs metadata). For cost, `legal-hard-6` was rejected as
operator churn / non-equivalent construction, while `legal-hard-18` was
accepted as a same-formula stats-cost case: both compute the same 91,000 answer
but the stats arm carries more context and extra turns.

Batch 11 update: two more accuracy candidates were rejected.
`astronomy-hard-10` stats/info is convergence/operator churn in the schema-only
arm, and `environment-hard-9` mode is beach parser/name normalization rather
than Delta history. Two sample-cost candidates were accepted:
`environment-hard-10` and `wildfire-easy-3` both preserve the final logical
pipeline and answer while the richer sample-size arm carries larger context.

Batch 12 update: `astronomy-hard-7` sample accuracy was rejected because the 3k
arm never built the modeling operator, and `environment-hard-8` mode accuracy
was rejected as another beach parser/layout issue. Two cost matches were
accepted: `astronomy-easy-5` stats cost and `legal-hard-24` sample cost both
preserve the final logical pipeline and answer while the richer arm carries
larger context.

Batch 13 update: two more context-mode accuracy candidates were rejected as
parser/layout issues: `astronomy-hard-9` uses different OMNI AP parsing, and
`environment-hard-10` drops repeated Wollaston station columns in the Latest
arm. Two sample-cost matches were accepted: `environment-hard-15` and
`biomedical-easy-9` preserve the final logical pipeline and answer while the
larger-sample arm carries more input context.

Batch 14 update: `environment-hard-9` mode accuracy was rejected as beach-name
normalization and repeated-station parsing, and `wildfire-hard-17` info
accuracy was rejected as ID normalization around leading-zero `NWS ID` values
rather than clean stats. Two cost matches were accepted: `legal-hard-7` as
Delta-history cost and `wildfire-easy-3` as sample-size cost.

Batch 15 update: two more accuracy matches were rejected: `biomedical-hard-7`
is sheet-header verification, and `astronomy-hard-9` sample is OMNI AP parser /
lag-transform logic. Two same-answer cost candidates were also rejected because
the dollar gaps are cache/extra-step dominated rather than clean context-size
cost: `legal-hard-14` and `biomedical-easy-6`.

Batch 16 update: `legal-hard-15` mode accuracy was rejected because the winning
Delta run deduplicates duplicate cross-state MSA rows after inspecting an
intermediate table, while Latest sums raw duplicates; that is transform /
verification, not clean context-mode information. `astronomy-easy-6` sample
accuracy was rejected as endpoint-rate versus average-step-rate logic. For
cost, `legal-hard-30` was accepted as a same-pipeline stats-cost case, while
`legal-hard-7` sample was rejected because extra edits/tool error dominate the
sample-size cost story.

Batch 17 update: two more accuracy candidates were rejected.
`environment-hard-9` info is beach-name/parser layout rather than clean stats,
and `biomedical-hard-7` sample is the same sheet-header verification issue.
For cost, `astronomy-easy-5` was accepted as a same-pipeline stats-cost case,
while `legal-hard-22` was rejected as cache/output driven despite equivalent
final logic.

Batch 18 update: two more high-ranked accuracy candidates were rejected.
`environment-hard-10` info is source/transform selection around Wollaston
datasheet versus water-body testing samples, and `legal-hard-1` mode is
dedup/aggregation verification rather than latest/delta context. For cost,
`legal-hard-30` mode was rejected as cache-driven, while `wildfire-hard-20`
7k-vs-5k was accepted as a clean same-pipeline sample-size cost case.

Batch 19 update: two context-mode accuracy candidates were rejected for
ordinary transform errors. `legal-hard-18` Latest omits the 2024
identity-theft category share, and `astronomy-easy-6` Latest averages
per-interval rates instead of using endpoint period rates. For cost,
`legal-hard-14` was accepted as same-pipeline sample-size cost, while
`wildfire-hard-4` was rejected because the 5k arm pays for extra diagnostics
and a different raw-line parsing path rather than just larger sample context.

Batch 20 update: two stats/info accuracy candidates were rejected.
`astronomy-hard-10` is SP3 parser/truncation, where the schema-only arm uses
only the first 600 lines per SP3 file for the final altitude series.
`wildfire-hard-17` is RAWS identifier mapping / verification, not hidden stats.
For cost, `legal-hard-30` was rejected as cache-driven despite equivalent
logic, while `wildfire-hard-20` was accepted as a same-pipeline stats-cost
case.

Batch 21 update: two more accuracy candidates were rejected.
`environment-hard-11` is repeated-column beach parser coverage, and
`biomedical-hard-5` mode is metadata-vs-B-APM source-sheet selection. For cost,
`legal-hard-24` was accepted as a same-pipeline Delta-context cost case, while
`environment-hard-18` was rejected because the two arms use different trend
aggregation definitions.

Batch 22 update: `legal-hard-15` mode was rejected as cross-state MSA
dedup/aggregation, and `environment-hard-8` info was rejected as beach
parser/source omission. For cost, `legal-hard-8` sample was rejected because the
7k arm has extra operator rewrite/output rather than pure sample-size cost, and
`legal-hard-22` mode was rejected as cache/input-accounting rather than clear
Delta-history context.

Batch 23 update: `biomedical-easy-2` info was accepted as a clean
Case_excluded/stats signal: excluding two serous rows marked excluded changes
the mean from `68.1` to `68.5`. `legal-hard-15` mode was again rejected as
duplicate MSA aggregation rather than context mode. For cost,
`biomedical-hard-3` info was accepted as a same-pipeline stats-cost case, while
`astronomy-easy-5` sample was rejected because the TLE altitude-change event
definition differs.

Batch 24 update: `environment-easy-2` info was rejected as rounded comparison
semantics, and `environment-hard-10` sample was rejected as repeated-column
parser coverage rather than sample visibility. Two cost cases were accepted:
`biomedical-easy-6` as same-pipeline stats-cost and `wildfire-hard-4` as
Delta-history/context cost with the same final wildfire cost-per-acre pipeline.

Batch 25 update: `wildfire-hard-17` info was rejected as RAWS identifier
mapping/verification, and `biomedical-hard-5` mode was rejected as B-APM
source-sheet selection rather than Delta context. For cost, `legal-hard-22`
3k stats-vs-schema was rejected as cache/input accounting, while `legal-hard-22`
7k-vs-3k stats was accepted as same-pipeline sample-size cost.

Batch 26 update: `biomedical-easy-2` info was accepted again as a clean
Case_excluded/stats signal, while `environment-hard-7` sample was rejected as
percent scaling (`10.87` vs `0.11`). For cost, `environment-hard-20` info was
accepted as same-final-pipeline stats-cost, and `legal-hard-22` mode was
rejected as cache/input accounting rather than Delta-history context.

Batch 27 update: `environment-hard-12` info was accepted as repeated-column
structure exposed by stats/schema context, while `environment-easy-2` mode was
rejected as rounded comparison semantics. Cost added two accepts:
`wildfire-hard-20` as same-pipeline stats-cost and `environment-easy-4` as
same-pipeline sample-size cost.

Batch 28 update: `environment-hard-7` info was rejected as percent scaling, and
`wildfire-hard-17` sample was rejected as RAWS identifier mapping/verification
rather than sample visibility. For cost, `archeology-easy-10` info was rejected
as cache/accounting despite identical code shape, while `legal-hard-22`
7k-vs-5k schema-only was accepted as same-pipeline sample-size cost.

Batch 29 update: `environment-hard-11` info was rejected as repeated-column
parser coverage, and `legal-hard-18` mode was rejected as formula omission
around the 2024 identity-theft share. For cost, `wildfire-hard-11` mode was
rejected as cache/accounting, while `wildfire-easy-2` sample was accepted as
same-pipeline sample-size cost.

Batch 30 update: `legal-hard-15` info was rejected as MSA filter-scope/dedup
logic, and `environment-hard-13` sample was rejected as threshold/missing-value
logic rather than sample visibility. Cost added two accepts: `legal-easy-25` as
same-pipeline stats-cost and `legal-easy-11` as same-pipeline sample-size cost.

Batch 31 update: `biomedical-hard-5` mode was rejected as B-APM source-sheet
selection, and `environment-hard-13` mode was rejected as threshold/censor
handling. For cost, `environment-easy-5` was accepted as same-pipeline
sample-size cost, while `archeology-easy-10` was rejected as cache-sensitive
despite identical two-operator logic.

Batch 32 update: `astronomy-easy-4` sample was rejected as an evaluator/paraphrase
issue, and `biomedical-hard-7` sample was rejected as header verification. Cost
added two sample-size accepts: `environment-hard-20` and `legal-easy-11` both
preserve the final logical pipeline while the larger sample setting carries
more context and extra construction steps.

Batch 33 update: `legal-hard-15` mode was rejected as duplicate-row
dedup/aggregation, and `biomedical-hard-5` sample was rejected as
`Case_excluded` filter logic rather than sample visibility. For cost,
`wildfire-easy-1` sample was rejected as loader/debug churn, while
`legal-easy-25` sample was accepted as same-pipeline sample-size cost.

Batch 34 update: `environment-hard-9` info was rejected as beach-name/parser
layout, and `wildfire-hard-18` sample was rejected as evaluator/wording. Cost
added two accepts: `legal-hard-7` as Delta-history cost and `legal-hard-29` as
same-pipeline sample-size cost.

Batch 35 update: both `environment-easy-2` accuracy matches were rejected as
rounding/comparison semantics, not stats or sample visibility. Cost rejected
`astronomy-hard-10` mode as output/accounting rather than Delta history, and
`wildfire-easy-1` sample as loader/debug churn rather than clean sample-size
cost.

## Accuracy Validations

### ACCEPT: `info / environment-hard-12`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Result:

- Rich answer: `Wollaston Beach`
- Lean answer: `Carson Beach`
- Ground truth: `Wollaston Beach`

Protocol judgment: ACCEPT as `schema_stats`.

Fundamental reason: the stats/data-rich run builds per-beach long-form parsers
for repeated `Tag` / `Enterococcus` columns. The schema-only run builds a single
generic parser and misses the nonstandard header/repeated-column structure.

Evidence:

- Rich creates per-beach operators such as `wollaston_long`, `carson_long`, and
  `pleasure_bay_castle_island_long`, promotes the embedded header row, then
  unpivots repeated Enterococcus station columns.
- Lean uses one generic `failed_counts_no_rain` operator, looks for literal
  column names after default CSV loading, and chooses `Carson Beach`.

### REJECT: `mode / environment-hard-8`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Result:

- Rich answer: `54.03`
- Lean answer: `51.56`
- Ground truth: `54.03`

Protocol judgment: REJECT for mode-accuracy. Label: `wrong_transform`.

Fundamental reason: the failure is a parser/verification error, not clear
evidence that Delta context supplied decisive extra information.

Evidence:

- Lean uses a generic raw block parser, `extract_samples`, and its preview shows
  rainfall-like values leaking into `enterococcus`.
- Rich uses separate beach-specific tidy operators and handles repeated
  `Enterococcus` columns by position.

### REJECT: `sample / biomedical-hard-7`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Result:

- Rich answer: `16`
- Lean answer: `15`
- Ground truth: `16`

Protocol judgment: REJECT for sampling-accuracy. Label: `verification`.

Fundamental reason: the lean arm already saw the decisive clue: default pandas
header inference made `BRD8` the column header and showed 15 data rows. The rich
arm made a better verification choice by loading `header=None`; the win is not
because a larger sample exposed a hidden row.

Evidence:

- Lean finalizes from `acetyl_sheet`, whose visible output is 15 rows under
  column header `BRD8`.
- Rich adds `acetyl_sheet_raw` with `header=None`, sees 16 rows including
  `BRD8`, and returns 16.

### REJECT: `sample / environment-easy-2`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Result:

- Rich answer: `[2003, 2011, 2015, 2018, 2020, 2021, 2022, 2023]`
- Lean answer: `[2003, 2011, 2018, 2020, 2021, 2022, 2023]`

Protocol judgment: REJECT for sampling-accuracy. Label: `wrong_transform`.

Fundamental reason: both arms load all 2002-2023 files. The difference is
comparison semantics: rich compares raw yearly rates to the rounded average;
lean compares rounded yearly rates to the rounded average, dropping 2015.

Evidence:

- Rich `fresh_beach_above_avg_years` filters `exceedance_rate > round(avg, 2)`.
- Lean `years_above_avg` filters `exceed_rate_2dp > avg_exceed_rate_2dp`.

### REJECT: `info / astronomy-hard-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Result:

- Rich answer: `['Proton_flux_>30_Mev', -0.193]`
- Lean answer: `No response from agent`

Protocol judgment: REJECT for info-accuracy. Label: `operator_churn`.

Fundamental reason: the lean run times out after repeated SP3 parsing and
downstream non-execution. Timeout/no response is not enough evidence that stats
provided decisive information.

Evidence:

- Rich parses SP3 directly in `sp3_raw` and produces `best_corr`.
- Lean lists SP3 files then repeatedly modifies `sp3_positions`; downstream
  `joined` / `best_corr` repeatedly show not-executed behavior and the final
  response is empty.

### REJECT: `sample / environment-hard-7`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Result:

- Rich answer: `10.87`
- Lean answer: `0.11`
- Ground truth: `10.87`

Protocol judgment: REJECT for sampling-accuracy. Label: `wrong_transform`.

Fundamental reason: both compute the same fraction, but lean fails to multiply
by 100 for percentage points.

Evidence:

- Both compute `high_rate = 0.151236` and `low_rate = 0.042489`.
- Rich returns `(high_rate - low_rate) * 100 = 10.87`.
- Lean returns `high_rate - low_rate = 0.11`.

### SUPERSEDED: `sample / wildfire-easy-3`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Result:

- Rich answer: `['California', 'Nevada']`
- Lean answer: several incorrect states.

Protocol judgment: SUPERSEDED by the later Batch 5 audit, which accepted this
case as `sampling_visibility`.

Fundamental reason: this early read over-attributed the win to geospatial
cleanup. The later raw-trace check found the decisive earlier divergence: 7k
visible context exposes duplicate county rows for the same state, while 3k sees
only first/last rows and keeps one county geometry per state.

Evidence:

- Old observation still true but incomplete: Rich adds geometry cleaning
  operators such as `fix_state_geoms`, `fix_state_geoms2`, `area_polygons`, and
  `fix_area_geoms`.
- Superseding evidence is recorded in Batch 5 and in `manual_validation.csv`.

### REJECT: `sample / biomedical-hard-7` (DeltaStats 7k vs 3k)

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Result:

- Rich answer: `16`
- Lean answer: `15`
- Ground truth: `16`

Protocol judgment: REJECT for sampling-accuracy. Label: `verification`.

Fundamental reason: the lean arm already saw the decisive clue. It loaded
`D-SE-acetyl` and the visible output showed 15 rows under column header `BRD8`;
that means pandas treated the first gene as a header. The richer arm succeeded
because it performed a raw-sheet verification with `header=None`, not because a
larger sample exposed a hidden downstream row.

Evidence:

- Lean creates `acetyl_sig` with default `pd.read_excel(...,
  sheet_name='D-SE-acetyl')`; the observation shows column `BRD8` and rows
  `DHX15` through `PMS2`, then `acetyl_sig_count` returns `15`.
- Rich first gets `0` from a wrong `Gene`-column assumption, then creates
  `acetyl_d_se_acetyl_load` with `header=None`; the observation shows 16 rows
  including row 0 `BRD8`, and the final count is `16`.
- Direct pandas check confirms `header=None` gives shape `(16, 1)` while default
  header inference gives shape `(15, 1)`.

### REJECT: `info / environment-hard-12` (LatestStats vs LatestSchemaOnly)

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Result:

- Rich answer: `Wollaston Beach`
- Lean answer: `Constitution Beach`

Protocol judgment: REJECT for stats-accuracy. Label: `wrong_transform`.

Fundamental reason: the difference is parser/layout handling. The stats arm
uses shared rainfall columns and both Enterococcus replicates; the schema-only
arm uses a positional repeated-block parser that misaligns rainfall for later
beaches.

Evidence:

- Rich `no_rain_fail_counts` returns Wollaston 59, Tenean 40, Constitution 23.
- Lean `beach_datasheets_tidy_fix2` leaves Wollaston with `rain_3day = NaN`,
  and final `boston_no_rain_fail_top_v2` returns Constitution 11.

### REJECT: `info / environment-hard-8`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Result:

- Rich answer: `54.03`
- Lean answer: `50.58`

Protocol judgment: REJECT for stats-accuracy. Label: `wrong_transform`.

Fundamental reason: the stats arm recovers by reloading the CSVs with
`header=None` and melting fixed Enterococcus columns. The schema-only arm also
gets a large long table but infers site headers as `Tag` / `Enterococcus`, which
changes the failed/rain count.

Evidence:

- Rich `tidy_enterococcus_all` has 21,986 rows and `failed_rain_pct` returns
  `54.03`.
- Lean `beach_samples_long` has 21,986 rows but final
  `failed_samples_rain_pct` reports 1,643 failed and 831 rain, returning
  `50.58`.

### REJECT: `info / astronomy-hard-8`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT for stats-accuracy. Label: `wrong_transform`.

Fundamental reason: both arms identify and use `a_cal`; the final difference is
the model-evaluation split, not schema/statistics information.

Evidence:

- Rich `rmse_models` uses a chronological 70/30 split with `n_train = 14` and
  `n_test = 7`.
- Lean `rmse_models` uses an 80/20 split with `ntest_kp = 5` and
  `ntest_pdyn = 5`.

### REJECT: `mode / environment-hard-12`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: REJECT for context-mode accuracy. Label: `wrong_transform`.

Fundamental reason: Delta parses each beach file separately. Latest concatenates
all datasheets and applies a generic repeated-block parser that misaligns later
beach rainfall and Enterococcus columns. That is not a stale-state or
delta-history mechanism.

Evidence:

- Rich `top_beach_norain` returns Wollaston with fail count 129.
- Lean `datasheets_tidy` produces 87,641 rows with malformed later-beach
  alignment; final `no_rain_fail_counts` returns Carson 21 and Wollaston 1.

### REJECT: `mode / environment-hard-9`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: REJECT for context-mode accuracy. Label: `wrong_transform`.

Fundamental reason: Delta parses Enterococcus values per source file and site.
Latest maps date text into numeric Enterococcus values, so the failure is parser
logic rather than context mode.

Evidence:

- Rich `beaches_meeting_100pct` returns exactly Castle Island, City Point, and
  Pleasure Bay.
- Lean `tidy_samples` shows Carson `enterococcus` values such as 27, 20, and 13
  coming from date strings like "August 27"; the final output wrongly marks all
  eight beaches as zero-failure.

### REJECT: `sample / wildfire-hard-17`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT for sampling-accuracy. Label:
`tool_error_recovery`.

Fundamental reason: the rich arm recovers from null joins by verifying station
ID matching. The lean arm abandons that join and reinterprets station ID as
elevation. Both saw the relevant schema/description; this is recovery and
verification, not a row hidden beyond the smaller sample.

Evidence:

- Rich builds `raws_station_id_map`, `noaa_station_id_map`, and
  `matched_station_elevations` with 759 matched rows, then returns average
  elevation `4830.9`.
- Lean `station_elev_joined` has 779 rows with all `elevation_ft = NaN`, then
  `noaa_station_elevation_mean` treats `station_verified_in_psa` as feet and
  returns `203996.6`.

## Cost Validations

### ACCEPT: `info / environment-easy-1`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: same logical pipeline and answer; stats/data context adds
cost without changing the outcome.

Evidence:

- Both load `water-body-testing-2013.csv`.
- Both compute percentage of `Violation == "yes"` and answer `4.796`.
- Both have 4 steps, 2 operators, and 1 link.
- Rich cost: `$0.018956`; lean cost: `$0.010501`.
- Rich cached tokens: `16,640`; lean cached tokens: `20,608`.

### ACCEPT: `info / biomedical-easy-2`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: both discover the Excel sheet, load the same metadata sheet,
filter serous non-excluded cases, and average `Age`. Stats adds context cost.

Evidence:

- Both answer `68.5`.
- Both use operators equivalent to `xlsx`, `meta`, and `serous_age`.
- Rich input tokens: `42,430`; lean input tokens: `35,805`.
- Rich cost: `$0.031928`; lean cost: `$0.018830`.

### ACCEPT: `info / biomedical-hard-1`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: both follow the same Excel multi-sheet analytical path and
compute the same Spearman correlation. Rich carries stats/data-level context on
large intermediate tables and one extra construction step.

Evidence:

- Both answer `0.4765`.
- Both final workflows have 10 operators and 7 links.
- Rich input tokens: `104,744`; lean input tokens: `51,751`.
- Rich cost: `$0.066049`; lean cost: `$0.044874`.

### ACCEPT: `sample / environment-easy-5`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: same four-file rainfall pipeline and same answer. The 5k
arm carries larger previews; the extra context is unnecessary.

Evidence:

- Both load Boston, Chatham, Amherst, and Ashburnham precipitation files.
- Both select `Year == 2020`, sum `Jun + Jul + Aug`, and answer `Ashburnham`.
- Both have 4 steps, 5 operators, and 4 links.
- Rich final-step input: `11,554`; lean final-step input: `9,207`.
- Rich cost: `$0.049370`; lean cost: `$0.022398`.

### ACCEPT: `sample / environment-easy-5` (Delta 7k vs 5k)

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: same four-loader rainfall DAG, same task interpretation,
same answer. The 7k arm carries larger result context and has lower cache reuse;
the extra context is unnecessary for this task.

Evidence:

- Both load Boston, Chatham, Amherst, and Ashburnham monthly precipitation CSVs.
- Both filter `Year == 2020`, sum `Jun + Jul + Aug`, sort descending, and answer
  `Ashburnham`.
- Both have 4 steps, 5 operators, and 4 links.
- Rich input tokens: `43,990`; lean input tokens: `40,228`.
- Rich cached tokens: `19,328`; lean cached tokens: `32,256`.
- Rich cost: `$0.057041`; lean cost: `$0.029144`.

### ACCEPT: `info / wildfire-easy-15`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: both build the same July wildfire correlation pipeline and
answer `No`. The stats/data-level arm carries more context without changing the
result.

Evidence:

- Both load wildfire data, filter July records, and compute the correlation
  between `hec` and `wind_med`.
- Both have 5 steps and produce the same yes/no answer.
- Rich cost: `$0.026123`; lean cost: `$0.018959`.
- Rich input tokens: `38,476`; lean input tokens: `30,194`.

### ACCEPT: `sample / archeology-easy-6`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: ACCEPT.

Fundamental reason: both build the same `worldcities` pipeline: filter southern
and western hemisphere cities, pick the maximum population city, and answer Sao
Paulo. The 5k arm uses less context with the same step count.

Evidence:

- Both answer Sao Paulo.
- Both have 4 steps.
- Rich input tokens: `26,240`; lean input tokens: `24,985`.
- Rich cost: `$0.018908`; lean cost: `$0.012333`.

### ACCEPT: `sample / astronomy-easy-2`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: ACCEPT.

Fundamental reason: both load the same Swarm A files and compute the peak
density ratio; the smaller result context preserves the answer while reducing
input context.

Evidence:

- Both answer `7.52`.
- Both have 4 steps.
- Rich input tokens: `33,872`; lean input tokens: `27,513`.
- Rich cost: `$0.022313`; lean cost: `$0.017110`.

### ACCEPT: `sample / environment-hard-7`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: ACCEPT.

Fundamental reason: both compute the same EJ-population beach violation-rate
difference and answer `10.87`. The join/aggregation code differs slightly, but
the logical computation is equivalent enough for a cost signal.

Evidence:

- Both load 2023 beach tests and EJ population data.
- Both filter marine beaches, join by community, compare `>50%` EJ-population
  areas against `<25%`, and return the percentage-point difference.
- Both have 5 steps.
- Rich input tokens: `38,385`; lean input tokens: `35,641`.
- Rich cost: `$0.029704`; lean cost: `$0.022281`.

### ACCEPT: `mode / wildfire-hard-11`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats5kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: ACCEPT.

Fundamental reason: same simple final pipeline, but Delta carries more
context/history and costs more without changing the result.

Evidence:

- Both load `Wildfire_Acres_by_State.csv`.
- Both compute `Total Acres Burned / Population` and answer `Wyoming`.
- Both have 4 steps, 2 operators, and 1 link.
- Delta final-step input: `7,164`; Latest final-step input: `6,195`.
- Delta cost: `$0.015827`; Latest cost: `$0.010339`.

### ACCEPT: `mode / environment-hard-13`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: ACCEPT.

Fundamental reason: same final logical DAG shape and same computation, but Delta
keeps edit history from repeated fixes to the tidy operator. This matches the
Delta-cost principle: extra versions/history of essentially the same operator
raise cost while preserving the outcome.

Evidence:

- Both answer `11`.
- Both final workflows have 3 operators and 2 links: load Constitution Beach,
  clean/extract three Enterococcus stations, count 2024 mixed-standard days.
- Delta has 9 steps, 7 tool calls, 1 tool error, and modifies
  `constitution_beach_tidy` 5 times.
- Latest has 5 steps, 3 tool calls, and no tool errors.
- Delta cost: `$0.105699`; Latest cost: `$0.020484`.

### REJECT: `sample / archeology-easy-10`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT under strict same-pipeline cost.

Fundamental reason: both answer `Singapore`, but the final result shape differs:
rich returns the full sorted ranking; lean returns only the top row.

Evidence:

- Rich outputs all country averages.
- Lean applies `.head(1)`.
- This is close but not clean enough for the strict cost protocol.

### REJECT: `mode / archeology-easy-10`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: REJECT under strict same-pipeline cost.

Fundamental reason: same issue as above: both answer `Singapore`, but one final
operator returns the full ranking and the other returns only the top row.

### REJECT: `mode / legal-hard-23`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats3kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT.

Fundamental reason: both answer `District of Columbia`, but the cleaning/join
logic differs materially enough that this is not the same-pipeline cost signal.

Evidence:

- Rich keeps component columns and uses an outer join with fill-zero.
- Lean filters ranked/DC/PR rows differently and uses an inner join.
- Final tables differ in column structure.

### REJECT: `mode / environment-easy-1`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats3kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`cache_effect`.

Fundamental reason: both use the same simple violation-percentage pipeline, but
Latest is cheaper mainly because its cached token count is much higher, not
because it carries less input context.

Evidence:

- Both answer `4.796`.
- Delta input tokens: `23,584`; Latest input tokens: `23,731`.
- Delta cached tokens: `16,640`; Latest cached tokens: `21,376`.
- Delta cost: `$0.018956`; Latest cost: `$0.011404`.

### REJECT: `sample / wildfire-easy-8`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sampling-cost signal. Label: `cache_effect`.

Fundamental reason: both answer `Lightning` with the same cause aggregation,
but the 3k arm does not have smaller input context in the trace. Its cost win is
mostly a cache effect.

Evidence:

- Both filter fires with `gt_100 == 1`, group by cause, and answer `Lightning`.
- Rich input tokens: `22,165`; lean input tokens: `23,247`.
- Rich cached tokens: `15,872`; lean cached tokens: `21,504`.
- Rich cost: `$0.017220`; lean cost: `$0.010229`.

## Batch 3 Validations

### ACCEPT: `info / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as `schema_stats`.

Fundamental reason: the stats/data-rich arm used the `Case_excluded` column to
drop excluded serous cases before computing the median. The schema-only arm
filtered only `Histologic_type == "Serous"` and included excluded cases.

Evidence:

- Rich filters `Case_excluded != "Yes"`; lean does not.
- Stats context exposes `Case_excluded` distribution (`No=144`, `Yes=9`).
- Raw recompute: serous including excluded cases gives `2.4241`; excluding the
  two excluded serous rows gives `2.6563`.

### REJECT: `mode / legal-hard-1`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats3kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT for context-mode accuracy. Label: `wrong_transform`.

Fundamental reason: Delta wins because it deduplicates multi-state MSA rows
before averaging. Latest groups and sums duplicate identity-theft reports by
normalized key, doubling large metros such as New York and Boston. That is a
deduplication/aggregation transform difference, not a latest-vs-delta context
mechanism.

Evidence:

- Delta `metro_gt1m` has 76 rows and 55 distinct keys, then
  `metro_gt1m_dedup` reduces it to 55 rows and returns `12964.8727`.
- Latest `metro_id_theft_join` shows doubled reports, e.g. New York
  `135638` and Boston `39858`, then averages 54 rows and returns `19293.1852`.

### REJECT: `sample / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT for sampling accuracy. Label:
`wrong_source_selection`.

Fundamental reason: the 3k arm sees that `mmc7` has a `B-APM subtypes` sheet,
but it never loads that table. It uses `Log2_variant_per_Mbp` from the metadata
file instead. The 7k arm reloads `B-APM subtypes` directly and joins it to
metadata on `idx`. This is source selection and verification, not a row hidden
beyond the smaller sample.

Evidence:

- Lean `mmc7_sheets` lists `B-APM subtypes`, then `mmc1_meta` filters 14 serous
  rows and returns median `2.4241`.
- Rich `b_apm_table_reload` loads 95 rows from `B-APM subtypes`, joins 12
  serous rows from metadata, and returns `2.6563`.

### REJECT: `info / archeology-hard-7`

Protocol judgment: REJECT. Label: `wrong_transform`.

Fundamental reason: the rich arm counts cities within Euclidean radius
`dlat^2 + dlng^2 <= 0.1^2`; the lean arm counts a rectangular box
`abs(lat) <= 0.1 and abs(lng) <= 0.1`. The difference is geometry
interpretation, not stats/schema visibility.

Evidence:

- Rich answer: `274`.
- Lean answer: `294`.

### REJECT: `info / legal-easy-19`

Protocol judgment: REJECT. Label: `wrong_transform`.

Fundamental reason: both arms see the same fraud/loss table. Rich divides the
selected loss-band sum by `Reports with $ Loss`, while lean divides by
`Number of Fraud Reports`.

Evidence:

- Band sum is `516,308`.
- `516,308 / 987,520 = 0.523`; `516,308 / 2,600,678 = 0.199`.

### REJECT: `info / wildfire-hard-12`

Protocol judgment: REJECT. Label: `verification`.

Fundamental reason: both traces derive the same median start day-of-year series.
The lean arm converts a small slope (`0.404` DOY/year) into `Yes`; the rich arm
correctly treats it as insufficient evidence and answers `No`.

### REJECT: `sample / astronomy-easy-6`

Protocol judgment: REJECT. Label: `wrong_transform`.

Fundamental reason: the rich arm computes endpoint delta over elapsed period;
the lean arm computes the mean of per-interval derivatives. Both are visible
from the traces, and the difference is rate definition rather than sample
visibility.

Evidence:

- Endpoint method gives quiet/storm rates `[0.0193, -0.0020]`.
- Interval-mean method gives `[0.0138, 0.0007]`.

### REJECT: `sample / astronomy-hard-9`

Protocol judgment: REJECT. Label: `tool_error_recovery`.

Fundamental reason: the rich arm loads OMNI data with whitespace-delimited
`read_csv`; the lean arm uses `read_fwf`, misreads DOY/hour columns, and then
keeps editing time parsing. That is file-loading/tool recovery, not sample
visibility.

Evidence:

- Rich raw OMNI stats show DOY `1..366` and hour `0..23`, final lag `24`.
- Lean misread makes the same columns look like `0..9`, creates 330 AP rows,
  and returns lag `19`.

### ACCEPT: `mode / legal-hard-6`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: ACCEPT as `delta_history_cost`.

Fundamental reason: both use the same stats/sample settings and answer
`1.1413`, but Delta accumulates repeated cleaner variants and event history
while Latest keeps a compact current-dataflow context and solves in fewer steps.

Evidence:

- Delta cost: `$0.057143`; Latest cost: `$0.018079`.
- Delta has 9 steps and repeated `csn_top3_clean` rewrites; Latest solves in
  5 steps.

### ACCEPT: `sample / environment-hard-18`

Protocol judgment: ACCEPT as `sampling_cost`.

Fundamental reason: both arms compare 2020-2023 annual exceedance/rainfall trend
and answer `True`. The 3k arm has much lower input context with the same step
count.

Evidence:

- Rich cost: `$0.091091`; lean cost: `$0.054811`.
- Rich step-2 input: `17,642`; lean step-2 input: `11,410`.

### REJECT: `info / legal-hard-6`

Protocol judgment: REJECT as a stats-cost signal. Label: `operator_churn`.

Fundamental reason: both answer `1.1413`, but the stats-rich arm repeatedly
rewrites `csn_top3_clean` and then bypasses the cleaner with a direct raw ratio.
The gap is construction churn, not stats context.

### REJECT: `sample / legal-hard-6`

Protocol judgment: REJECT as a sample-cost signal. Label: `operator_churn`.

Fundamental reason: both use StatsD2 and the visible table fits both 5k/7k
limits. The 7k arm repeatedly rewrites the cleaner while the 5k arm proceeds to
the ratio after one cleaner.

### REJECT: `info / wildfire-easy-2`

Protocol judgment: REJECT as a stats-cost signal. Label: `operator_churn`.

Fundamental reason: both answer `EACC`, but the stats-rich arm repeatedly
rewrites geospatial intersection operators and inserts extra geometry handling.
The cost gap is mostly construction churn.

### REJECT: `sample / astronomy-easy-1`

Protocol judgment: REJECT as a sample-cost signal. Label:
`tool_error_recovery`.

Fundamental reason: both answer `15`, but the 7k arm debugs an unused failed MAE
parsing branch and emits much more output. This is recovery/verbosity, not
sample compaction.

## Batch 4 Validations

### REJECT: `sample / environment-hard-18`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT for sampling accuracy. Label: `wrong_transform`.

Fundamental reason: both arms have enough data to answer the task. The 7k arm
compares annual 2020-2023 trend directions, while the 5k arm compares every
month-to-month change across the 48-month series. The divergence is trend
definition, not sample visibility.

### REJECT: `sample / environment-hard-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT for sampling accuracy. Label: `wrong_transform`.

Fundamental reason: the 5k arm computes the correlation over all Wollaston
station rows. The 3k arm first computes that same correlation, then replaces it
with a daily-mean correlation after intersecting with Boston Harbor/EJ dates.
This is aggregation/join semantics, not hidden rows.

### REJECT: `sample / wildfire-hard-17`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT for sampling accuracy. Label: `verification`.

Fundamental reason: the 5k arm stops after direct station joins fail and
averages all RAWS elevations. The 7k arm keeps probing identifier mappings.
This is recovery/verification depth, not a row hidden by the sample cap.

### REJECT: `sample / astronomy-hard-8`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT for sampling accuracy. Label: `wrong_transform`.

Fundamental reason: the 5k arm uses features at hour `t` and target shifted by
`-3h` with a 70/30 split. The 3k arm builds target values through a `t+3h` join
and uses an 80/20 split. This is supervised-target and split construction, not
sample visibility.

### ACCEPT: `sample / legal-hard-15`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as `sampling_visibility`.

Fundamental reason: the larger sample exposes two cross-state micropolitan rows
that the smaller-sample arm does not keep in its chosen filter. Those rows are
decisive for the answer.

Evidence:

- Rich visible context includes `LaGrange, GA-AL Micropolitan Statistical Area`
  and `Lebanon-Claremont, NH-VT Micropolitan Statistical Area`.
- Lean filters to `Metropolitan Statistical Area` and drops those
  micropolitan rows.
- The dropped values are `453` and `242`; their sum `695` exactly equals the
  answer gap between `243377` and `242682`.

### REJECT: `sample / biomedical-easy-2`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT for sampling accuracy. Label: `wrong_transform`.

Fundamental reason: the 5k arm computes a raw row-level serous mean. The 7k arm
filters `Case_excluded != "Yes"` and deduplicates by
`Proteomics_Participant_ID`. The difference is exclusion/deduplication logic.

### REJECT: `sample / biomedical-easy-2`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT for sampling accuracy. Label: `wrong_transform`.

Fundamental reason: the 3k arm sees the relevant fields
`Proteomics_Participant_ID`, `Case_excluded`, `Histologic_type`, and `Age`, but
still computes a raw row-level mean instead of a non-excluded participant mean.

### REJECT: `sample / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: REJECT for sampling accuracy. Label: `wrong_transform`.

Fundamental reason: the 5k arm uses the wrong denominator by summing Fraud,
Identity Theft, and Other as all reports. The 7k arm divides Bank Account/New
Accounts by the Identity Theft total. This is denominator choice, not hidden
sample visibility.

### ACCEPT: `sample / environment-easy-5`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: ACCEPT as `sampling_cost`.

Fundamental reason: both build the same four-loader JJA rainfall pipeline and
answer `Ashburnham`. The larger sample cap only increases per-step input
context.

Evidence:

- Rich step inputs are about `13.6k` and `14.0k`.
- Lean step inputs are about `10.1k` and `10.4k`.

### REJECT: `sample / environment-easy-5`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a sampling-cost signal. Label: `cache_effect`.

Fundamental reason: both pipelines answer `Ashburnham`, but the apparent cost
win is dominated by cached-token differences rather than a clean context-size
reduction. Rich cached `19.3k` of `44.0k` input tokens; lean cached `27.6k` of
`33.3k`.

### ACCEPT: `info / biomedical-hard-3`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as `schema_stats_cost`.

Fundamental reason: both build the same APM/meta minimum `APP_Z_score` age
pipeline and answer `60`. The stats arm carries larger column-stat context
without changing the logical solution.

### REJECT: `sample / archeology-easy-3`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sampling-cost signal. Label: `cache_effect`.

Fundamental reason: both produce the same rank-parsing answer `3.1333`, but the
cost gap is too cache-sensitive for clean sample-size attribution.

### REJECT: `sample / legal-hard-6`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sampling-cost signal. Label: `cache_effect`.

Fundamental reason: the lean arm is cheaper despite using more input/output
tokens because its cached-input share is much higher. The workflows also use
different cleaning logic, so this is not a clean same-pipeline sample-cost
case.

### ACCEPT: `info / wildfire-hard-16`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as `schema_stats_cost`.

Fundamental reason: both build the equivalent NOAA chi-square pipeline and
answer `[6.326, 0.787]`. The stats arm consistently carries larger per-step
inputs from column statistics.

## Batch 5 Validations

### ACCEPT: `sample / wildfire-easy-3`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: ACCEPT as `sampling_visibility`.

Fundamental reason: the 7k visible sample exposes multiple county rows for the
same state, so the agent correctly dissolves county geometries into one state
geometry. The 3k visible sample only shows the first and last rows, then the
agent keeps one county geometry per state with `drop_duplicates('adm1_id')`.

Evidence:

- 7k trace shows Iowa county rows `Scott` and `Story` sharing
  `adm1_id=USA-20230119-19`.
- 7k workflow uses `groupby('adm1_name')` with shapely `unary_union`.
- 3k trace shows `Lancaster` and `Story` around the truncated sample and does
  not show the duplicate Iowa clue.
- 3k workflow uses `drop_duplicates('adm1_id')`, keeping one county geometry
  per state.
- 7k returns `['California', 'Nevada']`; 3k returns a nine-state list with max
  count `2`.

### REJECT: `mode / environment-hard-9`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats5kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT for context-mode accuracy. Label:
`wrong_transform`.

Fundamental reason: Delta normalizes the two Pleasure Bay stations into
`Pleasure Bay Beach`, while Latest keeps `Pleasure Bay Beach (Broadway)` and
`Pleasure Bay Beach (Flagpole)` as separate final labels. The result difference
is beach-label normalization, not a clean latest-vs-delta context mechanism.

### ACCEPT: `sample / wildfire-hard-20`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as `sampling_cost`.

Fundamental reason: both build the same NOAA 2008 Pareto pipeline and answer
`0.0465` with the same step/tool count. The 7k arm carries more context.

Evidence:

- Both load NOAA wildfire rows, filter `start_year == 2008`, use
  `prim_threatened_aggregate`, sort descending, find the smallest prefix
  reaching 90% of total, and divide by all 2008 fires.
- Both have 7 total steps, 6 tool calls, 5 final operators, and no tool errors.
- 7k input tokens: `56,590`; 3k input tokens: `48,475`.
- 7k per-agent-step input average: about `7.8k`; 3k: about `6.8k`.

### REJECT: `sample / legal-hard-18`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: REJECT as a sampling-cost signal. Label: `cache_effect`.

Fundamental reason: both solve the same `91000` estimation pipeline, but the
cost gap is dominated by cache accounting rather than clean sample-size cost.

Evidence:

- Both use the same settings except result-context size (`7000` vs `5000`) and
  both answer `91000`.
- Both load the same three legal CSVs and compute
  `2007 total * 2024 identity-theft share * 2024 age-40+ share`.
- Rich cost: `$0.072463`; lean cost: `$0.049904`.
- Rich uncached input tokens: `20,188`; lean uncached input tokens: `7,766`.
- Rich also makes one extra tool call (`10` vs `9`).

## Batch 6 Validations

### REJECT: `info / environment-hard-13`

Pair:

- Rich: `DataflowSystemGPT52LatestStats5kD2`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT for stats/info accuracy. Label: `verification`.

Fundamental reason: the schema-only arm already produced the correct
`mixed_days_2024 = 11` in two intermediate operators, then finalized a stale
wrong path that counted `12`. Stats were not the decisive missing information.

Evidence:

- Rich answer is `11`; lean final answer is `12`.
- Lean `cb_2024_mixed_days_v2` and `cb_2024_mixed_days_clean` both output
  `mixed_days_2024 = 11`.
- Lean final `cb_2024_mixed_from_flags` counts `12` because
  `2024-05-22` has `South_present=true`, `South_meets=false`,
  `South_value=NaN`.
- Rich uses a null-aware count: `any_not = meets.eq(False) & vals.notna()`.

### REJECT: `mode / astronomy-hard-8`

Pair:

- Rich/history: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean/latest: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT for context-mode accuracy. Label: `wrong_transform`.

Fundamental reason: Delta uses the gold target/split construction; Latest uses
a different supervised-learning target and train/test split. The trace does not
show Delta context as the decisive mechanism.

Evidence:

- Gold solution uses `shift(-3)` for the acceleration target and a chronological
  70/30 split.
- Delta `forecast_dataset` uses `df['y'] = df['a_along_mean'].shift(-3)` and
  `rmse_models` uses `floor(0.7 * n)`.
- Latest `rmse_models` builds `t_target = time + 3h`, merges on time, and uses
  `split = int(n * 0.8)`.

### ACCEPT: `sample / wildfire-hard-16`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as `sampling_cost`.

Fundamental reason: both build the same NOAA Jan-Mar known-cause chi-square
pipeline and answer `[6.326, 0.787]`; the 7k arm carries larger context with the
same work shape.

Evidence:

- Both have 6 total steps, 5 tool calls, 4 final operators, and no tool errors.
- Both filter Jan/Feb/Mar fires with known non-`U` cause, crosstab
  `region x cause`, and run `scipy.stats.chi2_contingency`.
- 7k input tokens: `47,099`; 5k input tokens: `39,681`.
- 7k per-agent-step input average: about `7.6k`; 5k: about `6.4k`.

### REJECT: `sample / legal-hard-24`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a sampling-cost signal. Label: `cache_effect`.

Fundamental reason: both build the same state/MSA identity-theft pipeline and
answer the same MSA, but the dollar gap is mostly cache accounting.

Evidence:

- Both have 5 total steps, 9 tool calls, 7 final operators, and no tool errors.
- Both compute the top state by fraud/other + identity-theft reports, then
  choose the top identity-theft MSA in that state.
- 5k input tokens: `40,267`; 3k input tokens: `36,800`.
- 5k uncached input tokens: `17,099`; 3k uncached input tokens: `7,360`.

## Batch 7 Validations

### REJECT: `info / environment-hard-11`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: REJECT for stats/info accuracy. Label: `wrong_transform`.

Fundamental reason: the schema-only arm collapses duplicate `Enterococcus`
columns and uses only the Broadway station. The stats-rich arm preserves both
Pleasure Bay stations, Broadway and Flagpole. The grouped two-row header was
visible in the lean trace, so this is parser/layout logic rather than hidden
stats.

Evidence:

- Rich answer is `0.37`; lean answer is `0.40`.
- Lean `pbci_tidy` selects `Date`, `rain_1d`, and the first duplicate
  `Enterococcus` column, then filters only `enterococcus_broadway > 104`.
- Rich names `Broadway_Enterococcus`, `Flagpole_Enterococcus`, and
  `CastleIsland_Enterococcus`, then fails Pleasure Bay when Broadway OR
  Flagpole exceeds `104`.
- Broadway-only failures average to about `0.40`; Broadway-or-Flagpole failures
  average to about `0.37`.

### REJECT: `info / astronomy-easy-4`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: REJECT for stats/info accuracy. Label: `evaluator_issue`.

Fundamental reason: the accepted rich answer does not follow the gold period
calculation either. The apparent accuracy flip depends on evaluator tolerance,
not decisive stats.

Evidence:

- Gold solution computes the average period from successive minima:
  `np.mean(np.diff(minima_years))`.
- Rich computes `11.25` by averaging periods from both maxima and minima.
- Lean computes `11.5` from maxima spacing.
- Rich receives `llm_paraphrase=1`; lean receives `llm_paraphrase=0`, despite
  both using the same extrema years.

### ACCEPT: `info / wildfire-easy-3`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as `schema_stats_cost`.

Fundamental reason: both build the same state-dissolve/NIFC-intersection/count
pipeline and answer `['California', 'Nevada']`; the stats/data-level arm simply
carries larger input context.

Evidence:

- Both have 7 total steps, 6 tool calls, 5 final operators, and no tool errors.
- Both dissolve `usa.gpkg` counties to state polygons, intersect with NIFC
  Geographic Areas, and count distinct GACC areas per state.
- Rich step input tokens: `58,282`; lean step input tokens: `47,377`.
- Cached step tokens are equal at `40,832`, so the cost gap tracks extra
  non-cached input context.

### REJECT: `mode / legal-hard-22`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: REJECT as a context-mode cost signal. Label: `cache_effect`.

Fundamental reason: both build the same Bank Account/New Accounts proportion
pipeline with nearly identical input totals. The cost gap comes from cache
accounting, not visible Delta-history work.

Evidence:

- Both answer `0.0555`, with 7 total steps, 6 tool calls, 5 final operators,
  and no tool errors.
- Both divide Bank Account/New Accounts reports by total Identity Theft reports.
- Delta input tokens: `54,670`; Latest input tokens: `54,076`.
- Delta uncached input tokens: `17,934`; Latest uncached input tokens: `6,204`.

## Batch 8 Validations

### REJECT: `mode / environment-hard-12`

Pair:

- Rich/history: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean/latest: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`parser_layout`.

Fundamental reason: Delta returns the correct `Wollaston Beach` because it
parses each beach file separately. Latest concatenates raw beach tables, then
its parser only recovers Carson rows, so the failure is table-layout parsing and
missing verification rather than clear Delta lineage/history context.

Evidence:

- Latest `parsed_samples` output has 1,128 rows and all are `Carson`.
- Latest final `no_rain_fail_counts` only contains `Carson: 15`.
- Delta has per-file parsers including `wollaston_tidy`, then ranks Wollaston
  first.

### REJECT: `info / legal-hard-1`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the rich run deduplicates joined MSAs and uses the 387-row
U.S. MSA population table. The lean run averages duplicate joined rows and also
extracts a broader population source that includes Puerto Rico, so the outcome
mixes aggregation and source-selection differences.

Evidence:

- Rich `msa_joined` has 76 rows, 55 distinct MSA keys, and 21 duplicate-key
  rows; `msa_joined_dedup` reduces this to 55 rows.
- Rich final average is `12964.8727`, matching gold.
- Lean final join has 77 rows over 56 distinct keys, does not deduplicate, and
  returns `13612.7273`.

### REJECT: `sample / environment-hard-13`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `11`, but the 7k run costs more because it
takes a longer construction path with a tool error and repeated edits, not
because an otherwise identical pipeline simply carries a larger sample.

Evidence:

- 7k: 9 total steps, 7 tool calls, 1 tool error, and repeated
  `constitution_beach_tidy` edits.
- 5k: 5 total steps, 3 tool calls, 0 tool errors.
- Both final workflows have 3 final operators, but the code paths differ.

### REJECT: `sample / environment-hard-12`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `Wollaston Beach`, but the 5k run uses more
steps/tool calls and repeatedly edits beach parsers. The final parsers and
beach labels also differ, so this is not a clean same-pipeline sample-cost
example.

Evidence:

- 5k: 9 total steps, 23 tool calls, 19 final operators.
- 3k: 6 total steps, 19 tool calls, 19 final operators.
- 5k repeatedly edits `pleasure_bay_castle_island_tidy`; 3k reaches the same
  answer through a different parser/labeling path.

## Batch 9 Validations

### REJECT: `sample / astronomy-hard-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`operator_churn`.

Fundamental reason: the 5k run answers `['Proton_flux_>30_Mev', -0.193]`, but
the 3k run does not fail because a decisive data row is hidden. Its final
workflow contains the same kind of Swarm altitude, OMNI2, Sat_Density, and
correlation operators, but downstream operators remain not executed and the
agent exits with no final response after exhausting its step budget.

Evidence:

- 5k: 10 total steps, 13 tool calls, final answer matches gold.
- 3k: 26 total steps, 66 tool calls, no final response.
- 3k repeatedly edits `corr_table` / `corr_answer`; raw trace contains hundreds
  of `(not executed)` observations even though `corr_table` computes Pearson
  correlations and `corr_answer` formats the expected result.

### REJECT: `info / environment-hard-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_layout`.

Fundamental reason: the rich run reshapes all four Wollaston
`Tag`/`Enterococcus` station pairs and gets the gold `0.206`. The lean run
drops duplicate column names and computes only the first station pair, producing
`0.222`. The repeated columns are visible in the schema-only trace, so the
difference is parser/layout logic rather than missing statistics.

Evidence:

- Rich `wollaston_clean_2023` output has 7,585 rows from all four station
  pairs; `wollaston_corr` returns `0.2060014116405839`.
- Lean `wollaston_tidy` output has 1,892 rows and `wollaston_corr` returns
  `0.22234248078925745`.
- Lean workflow explicitly uses `df.loc[:, ~df.columns.duplicated(keep='first')]`,
  discarding repeated `Tag` / `Enterococcus` columns.

### REJECT: `info / astronomy-hard-9`

Pair:

- Rich: `DataflowSystemGPT52LatestStats5kD2`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a stats/info cost signal. Label:
`operator_churn`.

Fundamental reason: both answer lag `24`, but the stats arm costs more because
it takes a longer construction path, repeatedly editing the OMNI spec loader.
The cost gap is not clean evidence that stats context alone is unnecessary.

Evidence:

- Stats arm: 12 total steps, 12 tool calls, 7 final operators.
- Schema-only arm: 5 total steps, 10 tool calls, 7 final operators.
- Final workflows both parse OMNI2 AP, compute TLE semi-major-axis change, scan
  lags 0-48, and select lag 24.

### REJECT: `sample / environment-hard-14`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label: `cache_effect`.

Fundamental reason: both answer `Marine`, but this is not a clean larger-sample
cost case. The 7k arm has fewer steps and lower total input than 3k, yet costs
more because much less input is cached. The final workflows also aggregate the
rainfall/exceedance relationship differently.

Evidence:

- 7k: 4 total steps, 57,728 input tokens, 19,712 cached tokens, cost
  `$0.097586`.
- 3k: 7 total steps, 78,076 input tokens, 68,096 cached tokens, cost
  `$0.057620`.
- 7k aggregates exceedance by beach type and averages rainfall by region group;
  3k groups exceedance by community and joins rainfall by community before
  correlation.

## Batch 10 Validations

### REJECT: `info / astronomy-hard-9`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`tool_error_recovery`.

Fundamental reason: the rich run returns lag `24`, but the lean run fails
through OMNI format/path handling and downstream tool errors. The decisive
difference is not that stats supplied hidden domain information; the schema-only
run tries the wrong `.txt` spec path, reads the fixed-width OMNI data into the
wrong shape, uses AP column `16`, then times out.

Evidence:

- Rich `omni2_ap_hourly` uses OMNI column `49` for AP and `omni2_spec` falls
  back from `.txt` to `.text`.
- Lean `omni2_spec` raises `FileNotFoundError` for `omni2.txt`.
- Lean trace ends with `Error: The operation timed out` after errors in
  `omni2_ap_hourly`, `tle_43180_parsed`, `may1_30_join_base`, and
  `best_lag_r2`.

### REJECT: `sample / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`source_selection`.

Fundamental reason: the rich run eventually uses the `B-APM subtypes` sheet in
`mmc7` and joins it to the metadata, producing `2.6563`. The lean run uses
`Log2_variant_per_Mbp` directly from the metadata table for serous tumors,
producing `2.4241`. This is a source-selection/interpretation difference, not
evidence that the decisive sample row was outside the 5k context.

Evidence:

- Rich final `serous_median_variants_per_mbp_v2` joins
  `b_apm_table_reload` (95 rows) to `meta_table_clean` (153 rows).
- Lean final path filters `mmc1_meta` to 14 serous rows and computes the median
  directly from metadata `Log2_variant_per_Mbp`.
- Lean loaded `mmc7_sheets`, so the external source was not absent from the run.

### REJECT: `sample / legal-hard-6`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `1.1413`, but the 7k run repeatedly edits an
unused cleaning operator and its final ratio operator reads raw columns
directly. The 3k run uses a clean parsed table. The cost gap is therefore edit
churn and non-equivalent construction, not simply carrying a larger sample.

Evidence:

- 7k: 9 total steps, 7 tool calls, repeated `csn_top3_clean` edits, and error
  strings in the trace.
- 3k: 5 total steps, 3 tool calls, no error strings.
- 7k final `credit_card_ratio_2024_2020` consumes the raw loading operator;
  3k final ratio consumes `it_reports_clean`.

### ACCEPT: `info / legal-hard-18`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both runs answer `91000` using the same logical formula:
2007 total reports multiplied by the 2024 Identity Theft share and the 2024
age-40-plus share, rounded to the nearest thousand. The stats arm carries more
context and takes two extra turns without changing the result.

Evidence:

- Rich: 7 total steps, 61,020 input tokens, 40,832 cached tokens, cost
  `$0.072463`.
- Lean: 5 total steps, 35,592 input tokens, 28,544 cached tokens, cost
  `$0.038301`.
- Both final workflows load report count, report categories, and
  identity-theft-by-age tables, compute the 40+ age share, compute the 2024
  Identity Theft share, and apply both shares to 2007 total reports.

## Batch 11 Validations

### REJECT: `info / astronomy-hard-10`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`operator_churn`.

Fundamental reason: the stats arm correctly returns
`['Proton_flux_>30_Mev', -0.193]`, but the schema-only arm does not fail
because a decisive statistic is hidden. It creates many overlapping SP3,
Sat_Density, and correlation operators, never converges, and gives no final
answer.

Evidence:

- Rich: 11 total steps, 15 tool calls, 12 final operators, and `best_corr`
  returns `Proton_flux_>30_Mev` with Pearson `-0.19344252820628005`.
- Lean: 26 total steps, 64 tool calls, 27 final operators, and answer
  `No response from agent`.
- Lean trace repeatedly edits SP3/Sat_Density/correlation variants; the final
  workflow contains multiple unused alternatives such as `best_corr`,
  `best_corr_v2`, `sp3_raw_swa`, `sp3_raw_sw1`, and sample/probe operators.

### REJECT: `mode / environment-hard-9`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats3kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`parser_layout`.

Fundamental reason: Delta succeeds by loading each Boston Harbor beach
datasheet separately and normalizing site labels into beach names. Latest uses
a generic parser over concatenated sheets, extracts site-level labels, and
returns extra beaches. The divergence is parser/layout/name normalization, not
lineage/history context.

Evidence:

- Delta final answer is the gold set:
  `Castle Island Beach`, `City Point Beach`, `Pleasure Bay Beach`.
- Latest returns eight labels such as `Carson Beach, South Boston`,
  `Constitution Beach, East Boston`, and `Wollaston Beach, Quincy`.
- Delta `pleasure_bay_castle_island_tidy` explicitly maps repeated
  `Enterococcus` station columns to `Pleasure Bay Beach` and
  `Castle Island Beach`; Latest `tidy_samples` derives names from concatenated
  column labels.

### ACCEPT: `sample / environment-hard-10`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both runs answer `0.206` using the same logical pipeline:
parse all four Wollaston `Tag`/`Enterococcus` station pairs, filter to the
EJ/Boston Harbor target beach, and compute rainfall/Enterococcus Pearson
correlation over 7,585 rows. The 7k arm carries larger context and takes two
extra parser-edit turns, but the final result and logic are preserved.

Evidence:

- 7k: 9 total steps, 99,024 input tokens, cost `$0.079915`.
- 3k: 7 total steps, 63,453 input tokens, cost `$0.047530`.
- Both final correlation operators return `0.2060014116405839` from 7,585
  Wollaston rows.

### ACCEPT: `sample / wildfire-easy-3`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `['California', 'Nevada']` through the same
state-dissolve/NIFC-area-overlap/count pipeline. The richer 7k arm carries
larger per-step input context and costs more without changing the final
logical workflow.

Evidence:

- Both final workflows have 6 operators and 5 links.
- Both dissolve `usa.gpkg` county geometries by `adm1_name`, dissolve NIFC
  areas by `GACCName`, count distinct GACC overlaps per state, and select the
  max-count states.
- 7k step input tokens: `67,888`; 3k step input tokens: `45,902`.

## Batch 12 Validations

### REJECT: `sample / astronomy-hard-7`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`operator_churn`.

Fundamental reason: the 5k arm builds the density-prediction/RMSE workflow and
answers `1.211e-13`. The 3k arm never reaches the modeling operator; it spends
25 agent steps repeatedly recreating the six OMNI2/GOES/Sat_Density loaders and
exits with no response. This is convergence failure, not a hidden sample row.

Evidence:

- 5k final workflow has 9 operators including `rmse_result`.
- 3k final workflow has only 6 data-loading operators and no links.
- 3k has 150 tool calls, all `createOrModifyOperator`, and answer
  `No response from agent`.

### REJECT: `mode / environment-hard-8`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats3kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`parser_layout`.

Fundamental reason: Delta parses each Boston Harbor beach file separately and
uses all relevant station Enterococcus columns, producing the gold `54.03`.
Latest repeatedly tries generic parsers over concatenated sheets and ends with
a deduplicated first-column parser, producing `51.62`. The divergence is parser
layout, not Delta history.

Evidence:

- Delta `harbor_long` output has 21,986 rows and `failed_with_rain` returns
  `54.03`.
- Latest `failed_samples` has 1,081 rows and returns `51.618871415356146`.
- Latest `beaches_tidy` selects only `Date`, `1-Day Rain`, and the first
  `Enterococcus` after dropping duplicate columns.

### ACCEPT: `info / astronomy-easy-5`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both runs answer `2` using the same logical pipeline:
parse TLE pairs for satellite 48445, estimate altitude at each TLE epoch,
filter 2024, and count consecutive changes within 12 hours where altitude
changes by more than 1,000 meters. The stats arm carries larger context and
extra edits without changing the final result.

Evidence:

- Stats arm: 7 total steps, 78,815 input tokens, cost `$0.078289`.
- Schema-only arm: 6 total steps, 52,528 input tokens, cost `$0.049280`.
- Both final `tle_48445_major_changes_2024` operators return count `2` from
  951 altitude rows.

### ACCEPT: `sample / legal-hard-24`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both runs answer
`Los Angeles-Long Beach-Anaheim, CA Metropolitan Statistical Area` with the same
logical pipeline: compute the state with the highest total reports across all
types, then select that state's MSA with the highest identity-theft reports.
The 7k arm carries much larger context and takes extra turns.

Evidence:

- 7k: 8 total steps, 79,726 input tokens, cost `$0.066454`.
- 3k: 5 total steps, 36,800 input tokens, cost `$0.041398`.
- Both final workflows have 7 operators and 6 links, load the same state/MSA
  report sources, identify California as the top state, and select the same Los
  Angeles MSA with 71,624 identity-theft reports.

## Batch 13 Validations

### REJECT: `mode / astronomy-hard-9`

Pair:

- Rich/history: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean/latest: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`parser_layout`.

Fundamental reason: Delta answers the gold lag `24`, while Latest answers `15`,
but the first meaningful divergence is OMNI2 parsing, not preserved Delta
history. Delta reads `omni2_2024.dat` as whitespace-delimited columns and uses
column 49 for AP. Latest reads raw text lines and extracts AP from an ad hoc
character offset, then reports a very low best `r2`.

Evidence:

- Delta `omni2_2024_ap` constructs datetime from columns 0-2 and reads AP from
  `cols[49]`; `lag_r2_scan` returns `lag_hours=24`, `r2=0.6627`, `n=121`.
- Latest `omni2_ap_may` extracts AP with `first_int(s, 18)`, and
  `best_lag_r2` returns `best_lag_hours=15`, `best_r2=0.0025`, `n_pairs=131`.
- Latest also leaves an `omni2_spec` loader for a missing `omni2.txt` path.

### REJECT: `mode / environment-hard-10`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats5kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`parser_layout`.

Fundamental reason: Delta returns the gold correlation `0.206`; Latest returns
`0.222`. The difference is that Delta reshapes the repeated Wollaston
`Tag`/`Enterococcus` station blocks, while Latest drops duplicate columns and
keeps only the first station block. This is a repeated-column parser/layout
issue, not a clean latest-vs-delta context-mode effect.

Evidence:

- Delta `wollaston_clean_2023` unpivots four station pairs `(4,5)`, `(6,7)`,
  `(8,9)`, and `(10,11)`, then computes correlation over all parsed station
  rows.
- Latest `beaches_clean` calls `df2.loc[:, ~df2.columns.duplicated()]` and
  returns only `Date`, `rain_3d`, `Tag_1`, and `Enterococcus_1`.
- Both load the same Wollaston and EJ population sources, so the answer gap is
  explained by parser construction rather than context history.

### ACCEPT: `sample / environment-hard-15`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `Damon Pond Beach (DCR)` with the same logical
pipeline: load 2020-2023 water-body testing files, filter Fresh beaches,
require measurements in all four years, compute average `Violation` exceedance,
and select the highest-rate beach. The 5k arm has the same step/tool counts but
carries more input context.

Evidence:

- Both final workflows have 8 operators, 7 links, 7 total steps, and 8 tool
  calls.
- 5k: 73,065 input tokens, 57,344 cached, cost `$0.052163`.
- 3k: 57,586 input tokens, 52,224 cached, cost `$0.032439`.
- Per-step input grows to 12,914 in 5k versus 9,401 in 3k.

### ACCEPT: `sample / biomedical-easy-9`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `-0.008003975` by loading the same workbook,
settling on `F-SS-phospho`, and computing the mean `FDR.phos` difference
between `CBX3` rows and all other genes. The 7k arm carries more input context
and one extra exploratory step, but the final logical pipeline is equivalent.

Evidence:

- Both final workflows have 9 operators and one final link into the
  FDR-difference operator.
- 7k: 77,374 input tokens, 62,208 cached, 9 total steps, cost `$0.054605`.
- 3k: 58,514 input tokens, 52,736 cached, 8 total steps, cost `$0.036406`.
- Both final processing operators select `FDR.phos` and compute
  `mean(CBX3) - mean(non-CBX3)`.

## Batch 14 Validations

### REJECT: `mode / environment-hard-9`

Pair:

- Rich/history: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean/latest: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`parser_layout`.

Fundamental reason: Delta returns the gold set
`['Castle Island Beach', 'City Point Beach', 'Pleasure Bay Beach']`, while
Latest returns only `City Point Beach, South Boston` and
`Pleasure Bay Beach, South Boston`. The divergence is not Delta history; it is
beach-name normalization and repeated-station parsing. Delta splits the
combined Pleasure Bay/Castle Island datasheet into separate harbor beach names.
Latest keeps source labels with suffixes and computes Enterococcus by taking a
columnwise minimum across station columns.

Evidence:

- Gold solution reads the Boston Harbor beach datasheets separately and melts
  repeated `Tag` / `Enterococcus` location columns.
- Delta `beaches_100pct_harbor` explicitly maps
  `Pleasure Bay and Castle Island` back to the separate harbor list entries.
- Latest `beach_samples` extracts all `Enterococcus` columns but then uses
  `ent_df.min(axis=1)`, which can mark a beach passing if any station is below
  threshold instead of checking all sampled station measurements.

### REJECT: `info / wildfire-hard-17`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the stats arm returns the gold `4830.9` by converting RAWS
`NWS ID` to numeric before joining to NOAA `station_verified_in_psa`. The
schema-only arm also discovers `NWS ID`, but string-matches it and misses rows
with leading zeros such as `041018`. Because that leading-zero clue is visible
in the schema-only trace, this is an ID-normalization / verification failure,
not clean proof that stats supplied missing information.

Evidence:

- Numeric `NWS ID` matching finds 759 RAWS station rows and average elevation
  `4830.8524`.
- String `NWS ID` matching finds only 437 rows and average elevation
  `5134.3867`.
- Examples of numeric-matched but string-missed values include `041018`,
  `040508`, `040429`, and `041213`.
- Both traces show `Station ID` and `WX ID` have zero overlap; the decisive
  choice is numeric normalization of `NWS ID`.

### ACCEPT: `mode / legal-hard-7`

Pair:

- Rich/history: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean/latest: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: ACCEPT as a Delta-history cost signal. Label:
`delta_history_cost`.

Fundamental reason: both answer `Bank Account` with the same logical pipeline:
load the identity-theft-by-year CSV, clean year/category/report-count rows,
pivot 2020 and 2024 by theft type, and compute relative growth. Delta carries
extra edit history and error recovery without changing the final result.

Evidence:

- Both final workflows have 3 operators and 2 links.
- Delta has 6 agent steps, 5 tool calls, one tool error, and edits
  `csn_top3_id_theft_clean` three times.
- Latest has 4 agent steps, 3 tool calls, no tool errors, and a compact final
  `it_clean -> it_growth` pipeline.
- Delta cost is `$0.033400`; Latest cost is `$0.015565`.

### ACCEPT: `sample / wildfire-easy-3`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `['California', 'Nevada']` with equivalent
state-dissolve plus NIFC geographic-area overlap counting. The 7k arm carries
larger input context and one extra state-geometry edit while preserving the
final logical workflow.

Evidence:

- Both final workflows have 5 operators and 4 links.
- 7k: 65,734 input tokens, 53,504 cached, 7 total steps, cost `$0.046418`.
- 5k: 48,109 input tokens, 42,624 cached, 6 total steps, cost `$0.028888`.
- 7k uses spatial join with distinct `(state, GACC)` overlaps; 5k dissolves
  GACC polygons and uses overlay intersections. Both implement the same
  state-to-NIFC-area count semantics and return the same max states.

## Batch 15 Validations

### REJECT: `mode / biomedical-hard-7`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats5kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`verification`.

Fundamental reason: Delta returns the gold count `16`; Latest returns `15`.
The difference is sheet-header verification, not Delta history. The gold script
explicitly adds one because the acetylproteomics sheet's first item is treated
as a header under default pandas loading. Delta loads `D-SE-acetyl` with
`header=None` and counts the header-like first-column value. Latest loads the
sheet with default header inference and stops with the 15-row sheet view.

Evidence:

- Gold solution: `num_genes = len(dfs[acetyl_sheet]) + 1`.
- Delta final `acetyl_sheet_d` uses `pd.read_excel(..., header=None)`, then
  `acetyl_sig_gene_count` counts distinct non-empty first-column values.
- Latest final workflow has only loaders and no processing link; it answers
  from the default-loaded `D-SE-acetyl` sheet length.

### REJECT: `sample / astronomy-hard-9`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`parser_layout`.

Fundamental reason: 7k returns the gold lag `24`; 5k returns `46`. The
divergence is OMNI AP parsing and lag construction, not sample visibility. The
7k workflow parses AP from whitespace column 49 and evaluates AP shifted
against the TLE semi-major-axis change. The 5k workflow reads raw text lines,
uses whitespace field 7 as AP, joins only same-time drag/AP rows first, and then
shifts the already joined AP series.

Evidence:

- Gold solution uses fixed-width OMNI columns where `ap` is column 49 and
  shifts AP forward by each lag before joining to drag.
- 7k `omni2_ap_hourly` extracts `cols[49]`, and `drag_ap_lag_scan` shifts AP
  by lag relative to each drag timestamp.
- 5k `omni_ap_hourly` parses `ap = float(parts[7])`, which is not the AP
  column, then `best_lag` applies `df['ap'].shift(lag)` after the same-time
  join.

### REJECT: `mode / legal-hard-14`

Pair:

- Rich/history: `DataflowSystemGPT52DeltaStats7kD2`
- Lean/latest: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`cache_effect`.

Fundamental reason: both answer the same top five New England MSAs, but this is
not a clean Delta-history cost case. Both have the same step/tool/operator
counts and similar total input. Delta costs more mostly because far less input
is cached, and the final code differs in how New England is identified.

Evidence:

- Both final workflows have 4 operators, 3 links, 5 total steps, and 4 tool
  calls.
- Delta input tokens: 37,186; cached: 23,424; cost `$0.044325`.
- Latest input tokens: 35,628; cached: 31,744; cost `$0.028032`.
- Delta extracts source state from file names and groups by MSA; Latest
  extracts state abbreviations from the MSA string and deduplicates
  `(msa, reports)`.

### REJECT: `sample / biomedical-easy-6`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `IA` from the same metadata-sheet pipeline:
load `UCEC_CPTAC3_meta_table_V2.1`, filter `Age > 70`, and take the
`FIGO_stage` mode. But the 5k arm does not consistently carry larger per-step
context; it costs more because of one extra step and substantially lower cache
hit.

Evidence:

- 5k: 33,363 input tokens, 22,400 cached, 6 total steps, cost `$0.030385`;
  per-step input average `5,403`.
- 3k: 30,344 input tokens, 27,648 cached, 5 total steps, cost `$0.015506`;
  per-step input average `5,884`.
- Both final workflows have 3 operators, one link, and equivalent
  `Age > 70` / `FIGO_stage.value_counts()` logic.

## Batch 16 Validations

### REJECT: `mode / legal-hard-15`

Pair:

- Rich/history: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean/latest: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta answers the gold `243377`; Latest answers `592134`.
The difference is deduplication of cross-state MSAs that appear once per state
file. Delta builds an intermediate `cross_state_msa` table, observes 94 raw
cross-state rows, then groups to 43 distinct MSA names before summing. Latest
uses one direct operator and sums the duplicated rows. This is a transform /
verification difference, not evidence that Delta mode itself supplied unique
additional information.

Evidence:

- Gold solution concatenates state MSA files, flags cross-state areas, and
  calls `drop_duplicates()` before summing reports.
- Delta `cross_state_msa` output has 94 rows; `cross_state_msa_dedup` reduces
  them to 43 distinct areas and sums to `243377`.
- Latest `cross_state_msa_reports_2024` sums raw filtered rows directly and
  returns `592134`.
- Recomputing from source files gives raw cross-state sum `593524` and deduped
  distinct-area sum `243377`.

### REJECT: `sample / astronomy-easy-6`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`wrong_transform`.

Fundamental reason: 7k returns `[0.0193, -0.0020]`; 3k returns
`[0.0138, 0.0007]`. The difference is the rate definition. The 7k workflow and
gold script compute one endpoint rate per period as `(last semi-major axis -
first semi-major axis) / elapsed days`. The 3k workflow computes successive
per-step rates and averages them, which is a different estimator.

Evidence:

- Gold solution chooses start/end records in each quiet/storm file and computes
  the endpoint decay rate.
- 7k `quiet_decay_rate` and `storm_decay_rate` compute
  `(df['a_km'].iloc[-1] - df['a_km'].iloc[0]) / dt_days`.
- 3k `rates` computes `da / dt_day` for each row-to-row interval and returns
  `rate.mean()`.

### ACCEPT: `info / legal-hard-30`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `No` with the same logical pipeline: load all
2024 state MSA fraud files and identity-theft files, clean MSA report-count
rows, join comparable MSA entries, and test whether any identity-theft count
exceeds fraud. The stats arm preserves the same result and workflow shape while
carrying larger and less-cached context.

Evidence:

- Both final workflows have 5 operators, 4 links, 6 total steps, and 5 tool
  calls.
- Stats arm: 50,068 input tokens, 34,688 cached, cost `$0.050681`.
- Schema-only arm: 44,835 input tokens, 38,144 cached, cost `$0.034890`.
- Both final compare operators compute a boolean
  `identity_theft_reports > fraud_reports` over joined MSA report rows.

### REJECT: `sample / legal-hard-7`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `Bank Account` with the same relative-growth
pipeline over 2020 and 2024 identity-theft categories. But the 7k arm's extra
cost is not cleanly attributable to larger sample context: it has extra cleaner
edits, one tool error, and two extra agent steps.

Evidence:

- Both final workflows have 3 operators and 2 links.
- 7k has 6 agent steps, 5 tool calls, one tool error, and edits
  `csn_top3_id_theft_clean` three times.
- 5k has 4 agent steps, 3 tool calls, no tool errors, and one cleaner edit.
- 7k cost is `$0.033400`; 5k cost is `$0.017713`.

## Batch 17 Validations

### REJECT: `info / environment-hard-9`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_layout`.

Fundamental reason: the stats arm returns the gold set by writing
beach-specific parsers and splitting the combined
`pleasure_bay_and_castle_island` file into `Pleasure Bay Beach` and
`Castle Island Beach`. The schema-only arm uses one generic parser and keeps the
combined label `Pleasure Bay / Castle Island`. This is beach-name normalization
and parser layout, not clean evidence that stats supplied decisive missing
information.

Evidence:

- Gold solution reads the named Boston Harbor datasheets and melts repeated
  `Tag` / `Enterococcus` location columns per beach.
- Stats arm final workflow has separate operators such as
  `pleasure_bay_samples`, where Broadway/Flagpole rows are assigned to
  `Pleasure Bay Beach` and Castle Island rows to `Castle Island Beach`.
- Schema-only `all_beaches_long` assigns the entire combined file to
  `Pleasure Bay / Castle Island`, so its final answer cannot match the gold
  separated beach names.

### REJECT: `sample / biomedical-hard-7`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`verification`.

Fundamental reason: 5k returns `16`; 3k returns `15`. The mechanism is the same
sheet-header issue observed in earlier audits. The gold script adds one because
default pandas loading treats the first acetylproteomics gene (`BRD8`) as a
header. The 5k arm verifies the sheet with `header=None`; the 3k arm
default-loads `D-SE-acetyl` and counts only the remaining rows.

Evidence:

- Gold solution: `num_genes = len(dfs[acetyl_sheet]) + 1`.
- 5k `acetyl_sheet_d` uses `pd.read_excel(..., header=None)` and counts
  distinct first-column values.
- 3k `acetyl_sig` uses default `pd.read_excel(..., sheet_name='D-SE-acetyl')`
  and `acetyl_sig_count` counts distinct values in the default-loaded first
  column.

### ACCEPT: `info / astronomy-easy-5`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `2` with equivalent final TLE 48445
altitude-change logic: parse TLE pairs, estimate altitude at each epoch, and
count 2024 changes over 1000m within 12 hours. The stats arm carries more input
and output context and performs one extra edit while preserving the result.

Evidence:

- Both final workflows have 3 operators and 2 links.
- Stats arm: 48,338 input tokens, 1,712 output tokens, 6 total steps, cost
  `$0.043443`.
- Schema-only arm: 36,833 input tokens, 885 output tokens, 5 total steps, cost
  `$0.026448`.
- Both final counting operators return event count `2`.

### REJECT: `info / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: REJECT as a stats/info cost signal. Label: `cache_effect`.

Fundamental reason: both answer `0.0555` with equivalent logic: divide the
`Bank Account` / `New Accounts` identity-theft count by the Identity Theft
total reports. However, this is not clean stats-cost compaction. Both runs have
the same step/tool/operator counts and nearly identical input totals; the stats
arm costs more mainly because of lower cache hit and longer outputs.

Evidence:

- Both final workflows have 5 operators, 4 links, 7 total steps, and 6 tool
  calls.
- Stats arm: 54,670 input tokens, 36,736 cached, 1,125 output tokens, cost
  `$0.053563`.
- Schema-only arm: 54,737 input tokens, 44,544 cached, 890 output tokens, cost
  `$0.038093`.
- Final formulas are equivalent despite different operator decomposition.

## Batch 18 Validations

### REJECT: `info / environment-hard-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_source_selection`.

Fundamental reason: the stats arm returns the gold answer `0.206` by following
the gold-like Wollaston datasheet path: parse repeated `Tag` / `Enterococcus`
station columns and correlate adjusted Enterococcus against `3-Day Rain`. The
schema-only arm returns `0.288` because it builds a different source pipeline:
filter `water-body-testing-2023` to Boston Harbor marine samples, then join
those samples to Wollaston rainfall by date. This is source and transform
selection, not clean evidence that column stats supplied the decisive missing
information.

Evidence:

- Gold solution ultimately computes correlation from
  `wollaston_beach_datasheet.csv` after preparing the repeated station columns.
- Stats workflow has `harbor_ej90_long` and `harbor_corr`, which reshape the
  Wollaston Enterococcus station columns and compute the Pearson correlation.
- Schema-only workflow has `boston_harbor_enterococci_2023`,
  `boston_harbor_ej_with_rain`, and `ej_corr`, joining water-body testing rows
  to Wollaston daily rain before computing the correlation.
- Final workflow shapes differ materially: stats has 8 operators / 6 links;
  schema-only has 11 operators / 10 links.

### REJECT: `mode / legal-hard-1`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta answers `12964.8727`; Latest answers `13902.2400`.
The first meaningful divergence is not stale history or better Delta lineage.
Delta adds `msa_joined_dedup`, dropping duplicate matched MSAs by `msa_key`
before averaging reports. Latest inner-joins the normalized MSA tables and
averages `reports` directly. This is deduplication / aggregation verification,
not a clean latest-vs-delta context signal.

Evidence:

- Gold script calls `df_reports.drop_duplicates()` and
  `df_population.drop_duplicates()` before merging and averaging.
- Delta workflow: `msa_joined` filters matched MSAs over 1M population, then
  `msa_joined_dedup` drops duplicates by `msa_key`, then
  `avg_reports_over_1m` averages `reports`.
- Latest workflow: `metro_it_matched` joins on normalized keys and
  `avg_reports_over_1m` directly averages matched rows with `pop_2023 > 1M`.
- Both arms use stats/data level 2, sample 5k, and differ only in
  `context_mode`; the observed success mechanism is a transform choice.

### REJECT: `mode / legal-hard-30`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `No` with equivalent fraud-vs-identity-theft
MSA comparison pipelines and the same 6 agent steps. Delta is more expensive,
but not because it carries visibly more history or does more work. Its total
input is slightly lower than Latest, and the cost gap is mainly from lower
cached input.

Evidence:

- Delta: 50,068 input tokens, 1,264 output tokens, 34,688 cached tokens, cost
  `$0.050681`, 6 steps.
- Latest: 50,798 input tokens, 1,252 output tokens, 44,672 cached tokens, cost
  `$0.036066`, 6 steps.
- Delta final logic: clean fraud and identity-theft MSA tables, join, and test
  whether identity-theft reports exceed fraud reports.
- Latest final logic: clean fraud and identity-theft MSA tables, join/sum by
  metro area, and test for identity-theft reports greater than fraud reports.

### ACCEPT: `sample / wildfire-hard-20`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `0.0465` with equivalent 2008 NOAA Pareto
logic: filter to 2008, use `prim_threatened_aggregate` as residential houses
damaged, sort descending, and compute the minimal fraction of fires accounting
for at least 90% of total damage. The workflows have the same shape and step
count, while the 7k arm carries larger input context through the same pipeline.

Evidence:

- Both final workflows have 5 operators and 3 links.
- Both runs take 7 agent steps and use the same data files.
- 7k: 56,590 input tokens, 801 output tokens, cost `$0.046743`.
- 5k: 49,708 input tokens, 903 output tokens, cost `$0.032498`.
- Per-step input is consistently higher in the 7k run after loading:
  7k steps 3-6 use 7,525 / 8,355 / 9,403 / 9,595 input tokens, while 5k uses
  6,444 / 7,374 / 8,226 / 8,399.

## Batch 19 Validations

### REJECT: `mode / legal-hard-18`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta answers `91000` by applying the intended formula:
2007 total reports times the 2024 identity-theft share of all reports times the
2024 age-40+ share among identity-theft reports. Latest answers `520000`
because it applies only the age-40+ identity-theft share to all 2007 reports,
omitting the identity-theft category share. This is a formula omission, not a
latest-vs-delta context mechanism.

Evidence:

- Gold solution multiplies `total_2007_reports * identity_theft_percentage *
  id_theft_over_40`.
- Delta `it_2007_est_40plus` computes `it_share_2024 = it_2024 / total_2024`,
  then `total_2007 * it_share_2024 * share_40plus`.
- Latest `estimate_2007_age40plus` computes `total_2007 * share_40plus` and
  never uses the category-level identity-theft share.
- Both arms have stats/data level 2 and 5k sample; only context mode differs,
  but the observed divergence is a formula choice.

### REJECT: `mode / astronomy-easy-6`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta answers `[0.0193, -0.0020]` by computing endpoint
period rates `(a_last - a_first) / delta_days`, matching the gold approach.
Latest answers `[0.0138, 0.0007]` because it averages successive per-interval
`da/dt` values within each period. This is the same rate-definition issue seen
in earlier sample audits, not a context-mode effect.

Evidence:

- Gold script finds first and last records for the quiet/storm windows and
  computes one average rate from the endpoint altitude change.
- Delta has separate `quiet_decay_rate` and `storm_decay_rate` operators using
  `a0`, `a1`, and elapsed days.
- Latest `rates` computes `out['a_km'].diff() / epoch.diff()` and returns the
  mean of those per-step rates.
- Delta takes 5 steps; Latest takes 3 steps, but the accuracy gap is the rate
  formula, not context history.

### ACCEPT: `sample / legal-hard-14`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer the same New England top-5 MSA list with the
same 4-operator / 3-link final workflow shape: load New England states, load
state MSA identity-theft CSVs, clean rows and report counts, group/sort
metropolitan areas, and take the top five. The 7k arm carries larger input
context and uses one extra agent step while preserving the logical pipeline.

Evidence:

- Both answer the gold top-five list exactly.
- Both final workflows have 4 operators and 3 links.
- 7k: 37,186 input tokens, 1,153 output tokens, 23,424 cached tokens, 5 steps,
  cost `$0.044325`.
- 5k: 27,292 input tokens, 1,180 output tokens, 21,632 cached tokens, 4 steps,
  cost `$0.030211`.
- The final top-five operators both group by `metropolitan_area`, sum reports,
  sort descending, and take `head(5)`.

### REJECT: `sample / wildfire-hard-4`

Pair:

- Rich: `DataflowSystemGPT52Latest5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `2065.1`, but this is not clean sample-size
cost. The 5k arm spends an extra diagnostic step repeatedly inspecting raw
lines, then its final computation uses raw-line parsing operators. The 3k arm
uses parsed tab-separated loaders for the final calculation, with raw-line
inspection operators left as diagnostics. The cost gap is extra diagnostics,
longer outputs, and data-path churn rather than simply carrying larger samples
through the same pipeline.

Evidence:

- 5k: 82,180 input tokens, 2,268 output tokens, 10 steps, cost `$0.063477`.
- 3k: 78,904 input tokens, 1,450 output tokens, 9 steps, cost `$0.049518`.
- 5k step trace includes repeated `suppression_costs_raw` /
  `human_caused_acres_raw` inspection and safe first-120-lines debugging.
- 5k final `cost_per_acre` consumes `suppression_costs_clean` and
  `human_caused_acres_clean` derived from raw-line parsers; 3k final
  `cost_per_acre` consumes cleaned parsed loaders.
- Both workflows have 8 operators / 5 links, but the active final data path
  differs.

## Batch 20 Validations

### REJECT: `info / astronomy-hard-10`

Pair:

- Rich: `DataflowSystemGPT52LatestStats5kD2`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_truncation`.

Fundamental reason: the stats arm answers `['Proton_flux_>30_Mev', -0.193]`
by parsing full SP3 PL47 files, computing hourly altitude changes, joining to
OMNI2/Sat_Density, and ranking correlations. The schema-only arm answers
`['Lat_Angle_of_B_GSE', 0.606]` because its final `sp3_L47_positions` input is
derived from `sp3_text`, which loads only the first 600 lines of each SP3 file
for inspection. The failure is file parser/truncation and verification, not a
clean stats-information mechanism.

Evidence:

- Gold solution parses all matching SP3 files and reads `PL47` position lines.
- Stats workflow `sp3_pos` reads all matching SP3 files and parses `PL47`
  records before `altitude_hourly`, `corr_wide`, `best_corr`, and `final_pick`.
- Schema-only workflow `sp3_text` loads the first 600 lines from each SP3 file;
  `sp3_L47_positions` then parses `PL47` rows only from that truncated text.
- Both arms eventually inspect/recognize the `PL47` record format, but the
  schema-only final correlation is based on a truncated altitude series.

### REJECT: `info / wildfire-hard-17`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the stats arm reaches the gold answer `4830.9` after
probing multiple RAWS identifier columns and finally treating NOAA
`station_verified_in_psa` as an `NWS ID`. The schema-only arm first tries
`Station ID` and `WX ID`, gets no usable mapping, and then falls back to the
mean elevation of all RAWS stations, returning `3317.4`. This is identifier
mapping and verification behavior, not clean evidence that stats supplied the
decisive missing information.

Evidence:

- Gold script converts RAWS `NWS ID` to numeric and joins NOAA
  `station_verified_in_psa` values against it.
- Stats trace includes overlap/profile operators for `Station ID`, `WX ID`,
  `NESS ID`, last-six-digit checks, and then `avg_elev_via_nwsid`.
- Schema-only trace includes `noaa_station_elevations_by_wxid`, then
  `noaa_station_elev_avg_fixed`, but finalizes from
  `noaa_station_elevation_answer`, the mean over all RAWS station elevations.
- Both arms have the same context mode and sample size; the observed difference
  is join-key discovery and fallback behavior.

### REJECT: `info / legal-hard-30`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info cost signal. Label: `cache_effect`.

Fundamental reason: both answer `No` with equivalent fraud-vs-identity-theft
MSA comparison logic and the same 6-operator / 5-link final workflow shape.
This is not clean stats-cost compaction: the stats arm has fewer total input
tokens than schema-only, and the dollar gap mainly comes from lower cache hit
and longer output.

Evidence:

- Stats arm: 41,184 input tokens, 2,024 output tokens, 32,512 cached tokens,
  cost `$0.049202`, 5 steps.
- Schema-only arm: 52,483 input tokens, 1,361 output tokens, 47,488 cached
  tokens, cost `$0.036106`, 7 steps.
- Stats workflow builds path manifests then reads each state file inside
  `id_theft_2024` / `fraud_2024`.
- Schema-only workflow concatenates state files up front, cleans both sides,
  joins on `(state_file, metro_area)`, and checks `idtheft_reports >
  fraud_reports`.

### ACCEPT: `info / wildfire-hard-20`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `0.0465` with the same 2008 NOAA Pareto
pipeline: filter to 2008, use `prim_threatened_aggregate` as residential house
damage, sort descending, compute cumulative share, and return the proportion
of fires needed to reach 90% of total damage. The stats arm preserves the same
logical pipeline and step count while carrying much larger input context.

Evidence:

- Both final workflows have 5 operators and 3 links.
- Both traces have 7 agent steps and the same loader -> 2008 filter -> damage
  ranking -> 90% proportion structure.
- Stats arm: 68,457 input tokens, 900 output tokens, cost `$0.043091`.
- Schema-only arm: 50,468 input tokens, 717 output tokens, cost `$0.030418`.
- Final counting operators both sort by `prim_threatened_aggregate` and compute
  the minimal prefix reaching 90% of total damage.

## Batch 21 Validations

### REJECT: `info / environment-hard-11`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_layout`.

Fundamental reason: the stats arm returns `0.37` by parsing both Pleasure Bay
station blocks, Broadway and Flagpole, and flagging a failed sample if either
Enterococcus series exceeds 104. The schema-only arm returns `0.40` because it
keeps only the first `Tag` / `Enterococcus` pair, Broadway, and computes the
mean rainfall over Broadway failures only. This is repeated-column parser
coverage, not clean evidence that stats supplied decisive missing information.

Evidence:

- Gold script melts all repeated `Tag` / `Enterococcus` location columns, then
  excludes Castle Island Playground.
- Stats `beach_clean` explicitly creates `broadway_enterococcus` and
  `flagpole_enterococcus`; `fail_rain_avg` uses
  `(broadway > 104) | (flagpole > 104)`.
- Schema-only `pb_clean` renames only the first repeated `Tag` /
  `Enterococcus` pair to Broadway; `pb_failed_avg_rain` filters only
  `enterococcus_broadway > 104`.
- Both arms are Latest/3k; the difference is parser handling of repeated
  columns, not a clean stats-only clue.

### REJECT: `mode / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`source_selection`.

Fundamental reason: Delta answers `2.6563` by loading the `B-APM subtypes`
sheet from `mmc7`, joining serous samples from `mmc1` metadata by `idx`, and
converting `Log2_variant_per_Mbp` to linear variants/Mbp. Latest answers
`2.4241` because it uses `Log2_variant_per_Mbp` directly from the metadata
sheet instead of joining the B-APM subtype table. This is source-sheet
selection, not a latest-vs-delta context effect.

Evidence:

- Gold solution reads `mmc7.xlsx`, sheet `B-APM subtypes`, and joins serous
  sample IDs from the clinical metadata.
- Delta has `mmc7_apm`, `serous_samples`, and `serous_variants_per_mbp`
  operators, with a join on `sample_id = idx`.
- Latest has only `mmc1_meta`, `serous_log2_vpm`, and
  `serous_median_variants_per_mbp`, pulling `Log2_variant_per_Mbp` from the
  metadata table.
- Both arms are schema-only/3k; only context mode differs, while the mechanism
  is source choice.

### ACCEPT: `mode / legal-hard-24`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: ACCEPT as a context-mode cost signal. Label:
`delta_context_cost`.

Fundamental reason: both answer `Los Angeles-Long Beach-Anaheim, CA
Metropolitan Statistical Area` with equivalent top-state then top-MSA logic:
combine state identity-theft and fraud/other report counts, pick the state with
the largest total, then select the highest identity-theft MSA for that state.
Both workflows have the same shape and step count, while the Delta arm carries
larger input/output context.

Evidence:

- Both final workflows have 7 operators and 6 links.
- Both traces have 8 agent steps.
- Delta: 79,726 input tokens, 2,341 output tokens, 67,200 cached tokens, cost
  `$0.066454`.
- Latest: 70,705 input tokens, 1,831 output tokens, 60,672 cached tokens, cost
  `$0.053809`.
- Final operators both compute state totals, take `top_state`, clean MSA
  identity-theft rows, and select the top MSA for California.

### REJECT: `sample / environment-hard-18`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`different_transform`.

Fundamental reason: both answer `True`, but their final trend definitions are
not equivalent. The 7k arm builds a monthly series, imputes monthly missing
values, then compares annual means of monthly precipitation and monthly
exceedance rates. The 5k arm computes annual fresh-beach exceedance rates and
annual rainfall sums per station before averaging by year. Same answer and
same DAG shape are not enough here; the transform differs materially.

Evidence:

- Gold solution computes annual fresh exceedance rates and sums June/July/August
  rainfall per selected city before comparing year-to-year directions.
- 7k `fresh_exceed_monthly`, `rain_monthly_long`, `imputed_series`, and
  `trend_compare` operate over monthly values and annual means.
- 5k `fresh_exceed_by_year`, `rain_long_imputed`, `rain_annual_by_year`, and
  `trend_compare` operate over annual exceedance rates and annual rainfall
  sums.
- Both have 12 operators, 11 links, and 4 steps, but the final logic is not the
  same pipeline.

## Batch 22 Validations

### REJECT: `mode / legal-hard-15`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta answers `243377` by deduplicating cross-state MSAs
before summing reports, while Latest answers `592134` by directly summing
duplicate state-file rows. This is aggregation/verification, not decisive
latest-vs-delta context information.

Evidence:

- Gold solution drops duplicates before summing cross-state MSA report counts.
- Delta builds `cross_state_msa`, then `cross_state_msa_dedup`, grouping by
  `(metro_area, state_codes)` before summing.
- Latest `cross_state_msa_2024_total` filters cross-state rows and sums
  `reports` directly.

### REJECT: `info / environment-hard-8`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_layout`.

Fundamental reason: the stats arm parses all named Boston Harbor beach
datasheets and returns `54.03`. The schema-only arm's long-form builder accepts
`pleasure_bay_castle_island` as a function input but omits it from the
concatenated frames, returning `54.20`. The failure is source/parser omission,
not clean column-statistics information.

Evidence:

- Gold solution iterates all named Boston Harbor beach datasheets and counts
  failed Enterococcus samples with prior-day rain.
- Stats final parser includes the Boston Harbor beach tables needed for the
  gold percentage.
- Schema-only `harbor_samples_long` concatenates constitution, tenean,
  city_point, malibu, carson, wollaston, and m_street, but not
  `pleasure_bay_castle_island`.

### REJECT: `sample / legal-hard-8`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `True` by comparing the Miami MSA report count
between the Florida state MSA file and the national 2024 metropolitan file, but
the 7k arm has an extra step and rewrites the final comparison operator. The
cost gap is not purely larger sample context.

Evidence:

- 7k: `$0.035510`, 41,837 input tokens, 1,238 output tokens.
- 5k: `$0.023149`, 29,833 input tokens, 718 output tokens.
- Both final workflows have 3 operators and 2 links.
- 7k reloads the national file with `cp1252`, creates `miami_counts`, then
  rewrites it to use the national file's `Unnamed: 1` area column.
- 5k performs the same count comparison directly in `miami_counts_compare`.

### REJECT: `mode / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `0.0555` with equivalent Bank Account/New
Accounts over total Identity Theft logic and the same workflow shape. Delta is
more expensive because it carries modestly more input and has lower cached
input, but there is no visible extra edit/history mechanism.

Evidence:

- Delta: 5 operators, 4 links, `$0.038093`, 54,737 input tokens, 44,544 cached
  input tokens.
- Latest: 5 operators, 4 links, `$0.026010`, 49,074 input tokens, 46,208
  cached input tokens.
- Delta computes `prop_bank_new_accounts` from cleaned identity-theft type rows
  and the report-type total.
- Latest computes the same numerator and denominator in separate operators and
  divides them in `proportion`.

## Batch 23 Validations

### ACCEPT: `info / biomedical-easy-2`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info accuracy signal. Label:
`schema_stats`.

Fundamental reason: the task asks for serous tumor samples analyzed in the
study. The stats arm uses `Case_excluded != 'Yes'` before averaging serous
patient ages and returns `68.5`. The schema-only arm averages all serous rows
and returns `68.1`.

Evidence:

- Gold solution filters `Case_excluded == 'No'` and
  `Histologic_type == 'Serous'`.
- Direct data check: all serous rows have mean age `68.0769`; excluding the two
  serous rows marked `Case_excluded == 'Yes'` leaves 12 rows with mean `68.5`.
- Rich `serous_age` filters both `Histologic_type == 'Serous'` and
  non-excluded cases.
- Lean `serous_age_mean` filters only `Histologic_type` containing serous.
- The rich trace includes column stats showing `Case_excluded` has `No=144`
  and `Yes=9`, making the exclusion cue explicit.

### REJECT: `mode / legal-hard-15`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta answers `243377` by dropping duplicate cross-state
MSA rows before summing. Latest answers `593524` by directly summing repeated
state-file rows. This is deduplication/aggregation logic, not evidence that
Delta context supplied decisive extra state.

Evidence:

- Gold solution concatenates state MSA identity-theft files, flags cross-state
  MSAs, drops duplicate rows, then sums reports.
- Delta final workflow cleans rows, filters cross-state MSAs, then
  `drop_duplicates(subset=['msa'])` before summing.
- Latest final workflow filters cross-state MSAs and sums `reports` directly
  without deduplication.

### ACCEPT: `info / biomedical-hard-3`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `60` with the same logical pipeline: load
metadata, load the `B-APM subtypes` sheet, find the minimum `APP_Z_score`, then
join/filter metadata to retrieve `Age`. The stats arm carries much larger input
context while preserving the final work.

Evidence:

- Both traces have 8 agent steps.
- Both final workflows have 5 operators and 3 links.
- Stats arm: 79,955 input tokens, 955 output tokens, cost `$0.051080`.
- Schema-only arm: 62,033 input tokens, 1,072 output tokens, cost `$0.039095`.
- Final code differs mostly in naming/order (`mmc7` vs `apm_subtypes`) but uses
  the same `idx` and `APP_Z_score` lookup followed by `Age` retrieval.

### REJECT: `sample / astronomy-easy-5`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a sample-size cost signal. Label:
`different_transform`.

Fundamental reason: both answer `2`, but their event definitions are not the
same. The 7k arm counts an epoch if any later TLE within 12 hours differs by
more than 1000m; the 3k arm counts consecutive altitude differences within 12
hours, matching the gold loop more closely. Same answer and same DAG shape are
not enough for a clean sample-size cost signal.

Evidence:

- Gold solution loops over consecutive TLE heights and checks
  `delta_a > threshold` and `delta_t <= 12`.
- 7k `major_altitude_changes` scans a forward 12-hour window and uses
  `nanmax(abs(alt[i+1:j+1] - alt[i]))`.
- 3k `tle_48445_altitude_events_2024` uses `.diff()` on consecutive epoch and
  altitude values.
- Both have 3 operators, 2 links, and 5 steps, but the final transform differs.

## Batch 24 Validations

### REJECT: `info / environment-easy-2`

Pair:

- Rich: `DataflowSystemGPT52LatestStats5kD2`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the answer gap is from comparison semantics. The stats arm
compares unrounded yearly freshwater exceedance rates to the overall average
and includes 2015. The schema-only arm rounds yearly rates and the average to
two decimals before comparing, so 2015 ties at `0.04` and is dropped. This is
not a clean stats-information effect.

Evidence:

- Gold solution compares each year's unrounded rate against the overall
  violations/samples rate.
- Direct data check: 2015 has rate `0.041895`; overall weighted average is
  `0.040487`, and both round to `0.04`.
- Rich `fresh_beach_avg_and_years` uses
  `exceed_rate > avg_rate`.
- Lean `freshwater_avg_and_above` uses
  `exceedance_rate.round(2) > avg_r`.

### REJECT: `sample / environment-hard-10`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`parser_layout`.

Fundamental reason: the 7k arm returns `0.206` by parsing every repeated
Wollaston `Tag` / `Enterococcus` station pair into long form. The 5k arm
returns `0.222` because it drops duplicate column names and keeps only the first
pair. This is repeated-column parser coverage, not evidence that a decisive row
was outside the smaller sample.

Evidence:

- Gold solution melts all repeated `Tag` / `Enterococcus` columns from the
  Wollaston datasheet.
- 7k `beach_clean` scans adjacent `Tag` + `Enterococcus` pairs and concatenates
  them.
- 5k `beaches_clean` does `df2.loc[:, ~df2.columns.duplicated()]` and uses
  only `Tag_1` / `Enterococcus_1`.

### ACCEPT: `info / biomedical-easy-6`

Pair:

- Rich: `DataflowSystemGPT52LatestStats3kD2`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `IA` using the same workflow: list workbook
sheets, load the UCEC metadata sheet, filter patients with `Age > 70`, and
count `FIGO_stage`. The stats arm carries more context while preserving the
same final work.

Evidence:

- Both workflows have 3 operators and 1 link.
- Stats arm: 42,532 input tokens, 545 output tokens, cost `$0.026016`.
- Schema-only arm: 30,344 input tokens, 425 output tokens, cost `$0.015506`.
- Final code in both arms uses `pd.to_numeric(Age) > 70` and
  `FIGO_stage.value_counts()`.

### ACCEPT: `mode / wildfire-hard-4`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: ACCEPT as a context-mode cost signal. Label:
`delta_history_cost`.

Fundamental reason: both answer `2065.1` with the same final logic: parse raw
NIFC human-caused acres and suppression-cost tables, join by year, compute
`cost_per_acre`, and sort descending. Delta keeps more interaction/history
context and has one extra step, making it more expensive while preserving the
pipeline.

Evidence:

- Both final workflows have 7 operators and 4 links.
- Delta: 7 steps, 60,038 input tokens, 2,189 output tokens, cost `$0.056685`.
- Latest: 6 steps, 46,476 input tokens, 1,657 output tokens, cost `$0.045664`.
- Both final `cost_per_acre` operators divide total suppression cost by total
  human-caused acres after parsing raw rows.

## Batch 25 Validations

### REJECT: `info / wildfire-hard-17`

Pair:

- Rich: `DataflowSystemGPT52LatestStats5kD2`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the stats arm returns `4830.9` after probing RAWS
identifier columns and joining NOAA `station_verified_in_psa` to zero-padded
RAWS `NWS ID`. The schema-only arm returns `2858.3` after trying `Station ID`
and `WX ID` variants and falling back to the wrong elevation source. This is
identifier mapping and verification, not clean hidden stats information.

Evidence:

- Gold solution converts RAWS `NWS ID` to numeric and filters by NOAA
  `station_verified_in_psa`.
- Rich builds `stations_keymatch_check`, `raws_keys_preview`, then
  `stations_elev_by_nwsid`, normalizing both sides to six-digit IDs.
- Lean builds several `Station ID` / `WX ID` matching diagnostics and a
  `noaa_elev_mean` fallback, but does not use the gold-like NWS-ID join for
  the final answer.

### REJECT: `mode / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`source_selection`.

Fundamental reason: Delta answers `2.6563` by loading the `B-APM subtypes`
sheet from `mmc7`, joining serous metadata by `idx`, and converting
`Log2_variant_per_Mbp`. Latest answers `2.4241` by computing from
`Log2_variant_per_Mbp` in the metadata table. This is source-sheet selection,
not a latest-vs-delta context effect.

Evidence:

- Gold solution reads `mmc7.xlsx`, sheet `B-APM subtypes`.
- Delta has `mmc7`, `serous_samples`, `serous_idx_map`, and
  `serous_variants_per_mbp` operators.
- Latest has diagnostics for `mmc7` sheets but finalizes with
  `variants_per_mbp_serous` over `mmc1`.

### REJECT: `info / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `0.0555` with equivalent final logic and the
same step/DAG shape. The stats arm is more expensive, but the gap is driven by
modest input growth plus lower cached input rather than a clear stats-context
payload.

Evidence:

- Both workflows have 5 operators and 4 links and 7 agent steps.
- Stats arm: 48,319 total input tokens in `stats.json`, 926 output tokens,
  cost `$0.041881`.
- Schema-only arm: 45,745 input tokens, 922 output tokens, cost `$0.031675`.
- Summed step usage shows stats has about 2k more input but about 4k fewer
  cached input tokens.

### ACCEPT: `sample / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `0.0555` with equivalent Bank Account/New
Accounts over Identity Theft total logic, the same 7 steps, and the same
5-operator/4-link shape. The 7k arm carries larger input/output context.

Evidence:

- 7k: 54,670 input tokens, 1,125 output tokens, cost `$0.053563`.
- 3k: 48,319 input tokens, 926 output tokens, cost `$0.041881`.
- Both load `2024_CSN_Report_Type.csv` and
  `2024_CSN_Identity_Theft_Reports_by_Type.csv`, extract the total report
  denominator and Bank Account/New Accounts numerator, then divide.

## Batch 26 Validations

### ACCEPT: `info / biomedical-easy-2`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info accuracy signal. Label:
`schema_stats`.

Fundamental reason: this repeats the clean `Case_excluded` mechanism. The
stats arm excludes cases where `Case_excluded == 'Yes'` before averaging serous
patient ages and returns `68.5`. The schema-only arm averages all serous rows
and returns `68.1`.

Evidence:

- Gold solution filters `Case_excluded == 'No'` and
  `Histologic_type == 'Serous'`.
- Rich `serous_age` filters non-excluded serous rows before computing age mean.
- Lean `serous_age` filters only `Histologic_type == 'Serous'`.
- Direct data check from the previous accepted pair showed all serous rows
  average to `68.0769`, while excluding the two serous excluded rows leaves
  mean `68.5`.

### REJECT: `sample / environment-hard-7`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`wrong_transform`.

Fundamental reason: both arms compute the same high-EJ minus low-EJ marine
exceedance-rate difference, but the 7k arm returns percentage points while the
3k arm returns the raw fraction. This is a percent-scaling transform error, not
sample visibility.

Evidence:

- Gold solution multiplies each group rate by `100`.
- Direct data check: high-EJ rate is `0.151236`, low-EJ rate is `0.042489`;
  the fraction gap is `0.108747`, and the percentage-point gap is `10.8747`.
- Rich `exceedance_diff` computes `mean() * 100` for both groups and returns
  `10.87`.
- Lean `exceedance_rate_diff` returns `hi_rate - lo_rate` without multiplying
  by `100`, producing `0.11`.

### ACCEPT: `info / environment-hard-20`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `['Bucks Creek', 'Pleasant Street',
'Forest Street']` with the same logical pipeline: compute 2015 summer rainfall
for Amherst, Chatham, Boston, and Ashburnham; pick the least-rain city; then
rank that city's beaches by violation rate. The stats arm carries much larger
context and takes two extra turns while preserving the final workflow shape.

Evidence:

- Both final workflows have 8 operators and 7 links.
- Stats arm: 6 steps, 79,801 input tokens, 1,272 output tokens, cost
  `$0.051620`.
- Schema-only arm: 4 steps, 44,503 input tokens, 1,192 output tokens, cost
  `$0.041547`.
- Both final operators rank beaches by violation rate, violation count, and
  indicator level after selecting the least-rain city.

### REJECT: `mode / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `0.0555` with equivalent Bank Account/New
Accounts over Identity Theft total logic, the same 7 steps, and the same
5-operator/4-link shape. Delta is costlier because of modestly higher input,
longer output, and lower cached-input usage, not visible extra edit/history
context.

Evidence:

- Delta: 45,745 input tokens, 922 output tokens, cost `$0.031675`.
- Latest: 43,574 input tokens, 783 output tokens, cost `$0.022704`.
- Summed step usage shows Delta has about 1.5k more input, 137 more output
  tokens, and about 2.2k fewer cached input tokens.
- Both workflows load the same two CSN tables, extract the same numerator and
  denominator, and divide.

## Batch 27 Validations

### ACCEPT: `info / environment-hard-12`

Pair:

- Rich: `DataflowSystemGPT52LatestStats5kD2`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info accuracy signal. Label:
`schema_stats`.

Fundamental reason: the stats arm answers `Wollaston Beach` by re-headering
each Boston Harbor datasheet and unpivoting all repeated `Enterococcus`
station columns. The schema-only arm answers `Carson` because its generic
parser drops duplicate header names and keeps only the first repeated
`Enterococcus` column. Here the decisive information is the repeated-column
table structure.

Evidence:

- Gold solution melts all `Tag` / `Enterococcus` columns for each beach before
  counting no-rain failures.
- Rich `fail_counts_no_rain` loops through every `Enterococcus` column and
  concatenates per-beach long-form samples.
- Lean `parsed_samples` runs
  `data = data.loc[:, ~pd.Index(data.columns).duplicated(keep='first')]`,
  discarding repeated station columns before counting failures.

### REJECT: `mode / environment-easy-2`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta includes 2015 by comparing unrounded yearly
freshwater exceedance rates to the unrounded average. Latest drops 2015 because
it rounds yearly rates and the average to two decimals before comparison. This
is rounding/comparison semantics, not a latest-vs-delta context effect.

Evidence:

- Gold solution compares unrounded rates.
- Delta `freshwater_exceedance_above_avg_years` uses
  `exceedance_rate > avg_rate`.
- Latest `freshwater_avg_and_above` uses
  `exceedance_rate.round(2) > avg_r`.
- 2015's true rate is just above the average but both round to `0.04`.

### ACCEPT: `info / wildfire-hard-20`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `0.0465` with the same 2008 NOAA
`prim_threatened_aggregate` Pareto pipeline: filter 2008 incidents, sort by
residential houses threatened/damaged, compute cumulative share, and find the
incident proportion needed to reach 90%. The stats arm carries larger context.

Evidence:

- Both final workflows have 5 operators and 3 links.
- Both traces have 7 steps.
- Stats arm: 64,745 input tokens, 1,139 output tokens, cost `$0.041151`.
- Schema-only arm: 50,189 input tokens, 1,013 output tokens, cost `$0.031453`.

### ACCEPT: `sample / environment-easy-4`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `86` with the same 2019-2023 Wollaston Beach
open-percentage pipeline: load five yearly water-quality files, concatenate,
filter Quincy/Wollaston rows, and compute the percent of samples with
`Violation == 'NO'`. The 7k arm carries much larger input context.

Evidence:

- Both workflows have 8 operators and 7 links.
- Both traces have 6 steps.
- 7k: 83,194 input tokens, 902 output tokens, cost `$0.047942`.
- 3k: 61,560 input tokens, 928 output tokens, cost `$0.036252`.
- Final code in both arms computes `round(100 * met / total)`.

## Batch 28 Validations

### REJECT: `info / environment-hard-7`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: both arms compute the same high-EJ minus low-EJ marine
exceedance-rate gap, but the stats arm returns percentage points and the
schema-only arm returns the raw fraction. This is percent scaling, not missing
stats information.

Evidence:

- Gold solution multiplies each group rate by `100`.
- Rich `ej_exceedance_rate_diff` computes `(high_rate - low_rate) * 100` and
  returns `10.87`.
- Lean `ej_rate_diff` computes `gt_50 - lt_25` and rounds the fraction,
  returning `0.11`.

### REJECT: `sample / wildfire-hard-17`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the 5k arm returns `4830.9` after probing several RAWS
identifier columns and joining NOAA `station_verified_in_psa` to RAWS `NWS ID`.
The 3k arm returns `3317.4` after trying `WX ID` / `Station ID` joins and then
falling back to the mean elevation of all RAWS stations. This is ID mapping and
verification, not sample visibility.

Evidence:

- Gold solution filters RAWS rows by `NWS ID`.
- Rich builds overlap/profiling operators, then finalizes with
  `avg_elev_via_nwsid`.
- Lean has `station_elevations` keyed on `WX ID` and final
  `publicview_raws_elev_avg`, which averages all station elevations.

### REJECT: `info / archeology-easy-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52Delta7kSchemaOnly`

Protocol judgment: REJECT as a stats/info cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `Singapore` with the same two-operator pipeline:
load `worldcities.csv`, drop null populations, group by country, and sort by
mean population. The stats arm is costlier, but the extra input/output is small
and the cache profile differs, so this is not a clean stats-context cost case.

Evidence:

- Both workflows have 2 operators and 1 link.
- Both traces have 4 steps.
- Stats arm: 28,430 input tokens, 234 output tokens, cost `$0.023192`.
- Schema-only arm: 27,563 input tokens, 197 output tokens, cost `$0.015512`.
- Summed step cache is lower for the stats arm (`14,336` vs `16,128` cached
  input tokens).

### ACCEPT: `sample / legal-hard-22`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `0.0555` with equivalent Bank Account/New
Accounts over Identity Theft total logic, the same 7 steps, and the same
5-operator/4-link workflow shape. The 7k arm carries materially larger input
context.

Evidence:

- 7k: 54,737 input tokens, 890 output tokens, cost `$0.038093`.
- 5k: 44,873 input tokens, 810 output tokens, cost `$0.026969`.
- Both workflows load the same CSN report-type and identity-theft type tables,
  extract the same numerator and denominator, and divide.

## Batch 29 Validations

### REJECT: `info / environment-hard-11`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_layout`.

Fundamental reason: the stats arm answers `0.37` by using both Pleasure Bay
station columns, Broadway and Flagpole, and taking the maximum Enterococcus
value for the beach before averaging one-day rainfall on failing samples. The
schema-only arm answers `0.40` because it keeps only the first repeated
`Tag` / `Enterococcus` pair. This is parser coverage of repeated columns, not a
clean stats-only clue.

Evidence:

- Gold solution melts repeated `Tag` / `Enterococcus` columns, then excludes
  Castle Island Playground.
- Rich `pb_failures` computes `max(ent_broadway, ent_flagpole)` for Pleasure
  Bay.
- Lean `pb_clean` selects only `Date`, `1-Day Rain`, `Tag`, and
  `Enterococcus`, which resolves to the first repeated pair.

### REJECT: `mode / legal-hard-18`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats7kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta returns `91000` by multiplying 2007 total reports by
both the 2024 identity-theft category share and the 2024 age-40+ share within
identity-theft reports. Latest returns `520000` because it applies only the
age-40+ share to all 2007 reports. This is a formula omission, not a
latest-vs-delta context effect.

Evidence:

- Gold solution uses
  `total_2007_reports * identity_theft_percentage * id_theft_over_40`.
- Delta has separate `identity_theft_total_2007` and
  `identity_theft_2007_age40plus_est` operators.
- Latest `age40plus_2007_estimate` multiplies 2007 total reports directly by
  `share_age40plus_2024`.

### REJECT: `mode / wildfire-hard-11`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `Wyoming` with the same two-operator
acres-per-capita ranking pipeline and the same 4 steps. Delta is costlier, but
the workflow is identical and the gap is driven by slightly larger input/output
plus lower cached input, not visible Delta-history context.

Evidence:

- Both workflows have 2 operators and 1 link.
- Delta: 22,676 input tokens, 296 output tokens, cost `$0.020038`.
- Latest: 21,557 input tokens, 288 output tokens, cost `$0.011114`.
- Summed cached input is lower for Delta (`11,520`) than Latest (`14,080`).

### ACCEPT: `sample / wildfire-easy-2`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats5kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `EACC` with the same state-dissolve /
NIFC-intersection pipeline: load USA and NIFC geometries, dissolve state
polygons, spatially join intersections, and count distinct states per NIFC
area. The 7k arm carries much larger input/output context and takes one extra
step.

Evidence:

- Both workflows have 5 operators and 4 links.
- 7k: 7 steps, 78,121 input tokens, 1,021 output tokens, cost `$0.044964`.
- 5k: 6 steps, 58,315 input tokens, 794 output tokens, cost `$0.033938`.
- Final `area_state_counts` operators group by `GACCAbbreviation` and count
  distinct states.

## Batch 30 Validations

### REJECT: `info / legal-hard-15`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the stats arm keeps all cross-state statistical areas after
deduping and returns `243377`. The schema-only arm filters only rows containing
`Metropolitan Statistical Area`, excluding cross-state micropolitan areas and
returning `242682`. This is filter-scope and dedup logic, not clean stats
information.

Evidence:

- Gold concatenates all State MSA Identity Theft CSVs, parses state segments,
  deduplicates, and sums all cross-state rows.
- Rich `it_msa_2024_cross_state` keeps rows with more than one distinct state
  abbreviation.
- Lean `msa_idt_2024_clean` filters to `Metropolitan Statistical Area`, so
  cross-state micropolitan rows are removed before the sum.

### REJECT: `sample / environment-hard-13`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the 7k arm counts mixed-standard 2024 days for Constitution
Beach correctly at `11`. The 3k arm strips censor markers and uses
`~lt(104)`, so missing or non-meeting values can create an extra discordant day
and it returns `12`. This is threshold and missing-value handling, not sample
visibility.

Evidence:

- Gold groups Constitution Beach 2024 station values and counts dates with a
  failing Enterococcus value but fewer than all stations failing.
- Rich parses the repeated North/Middle/South station columns and counts mixed
  days.
- Lean strips `<` and `>` from censored values and treats negated comparisons as
  failure indicators, which changes the mixed-day count.

### ACCEPT: `info / legal-easy-25`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a stats/info cost signal. Label:
`schema_stats_cost`.

Fundamental reason: both answer `U.S. Space Force` with the same embedded-header
military-branch `Median Fraud Loss` maximum pipeline and the same
3-operator/2-link workflow shape. The stats arm carries about 9.3k more input
tokens and takes one extra step.

Evidence:

- Rich promotes the embedded `Military Status` header row, parses
  `Median Fraud Loss`, filters military branches, and takes the max.
- Lean promotes the same header structure, filters `U.S.` branch rows, parses
  `Median Fraud Loss`, and takes the max.
- Rich: 39,700 input tokens, 870 output tokens, cost `$0.026013`.
- Lean: 30,450 input tokens, 626 output tokens, cost `$0.018506`.

### ACCEPT: `sample / legal-easy-11`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `No` with equivalent
`Other / (Fraud + Identity Theft + Other)` logic over 2001-2024 and the same
3-operator/2-link workflow shape. The 7k arm carries about 11.8k more input
tokens and takes one extra step.

Evidence:

- Both workflows clean the CSN report-type table, parse yearly counts for
  Fraud, Identity Theft, and Other, then check whether `Other` exceeds half of
  the total in any year.
- Rich: 39,838 input tokens, 988 output tokens, cost `$0.027504`.
- Lean: 28,072 input tokens, 483 output tokens, cost `$0.016778`.

## Batch 31 Validations

### REJECT: `mode / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`source_selection`.

Fundamental reason: Delta returns `2.6563` by loading `mmc7` B-APM subtypes,
joining serous sample ids, and converting `Log2_variant_per_Mbp` to linear
variant counts. Latest returns `2.4241` by computing from the metadata table's
`Log2_variant_per_Mbp` column. This is source-sheet selection, not a
latest-vs-delta context effect.

Evidence:

- Gold solution reads `mmc7.xlsx`, sheet `B-APM subtypes`, then joins serous
  ids from the clinical metadata.
- Delta has `mmc7_apm`, `serous_samples`, and `serous_variant_rates`.
- Latest has no final `mmc7` data path; it derives `serous_log2var` directly
  from `mmc1_meta`.

### REJECT: `mode / environment-hard-13`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest7kSchemaOnly`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta returns `11` by checking station values with both
`lt(104)` and `ge(104)`. Latest returns `12` because it treats `<`-tagged
values as zero and uses `~meets` to represent non-meeting samples. This is
threshold/censor handling, not context mode.

Evidence:

- Gold melts repeated Constitution Beach station columns and counts dates where
  at least one station meets the standard and another does not.
- Delta `constitution_beach_2024_mixed_days` computes `any_meets` and
  `any_not` from parsed station values.
- Latest `days_mixed_2024` uses `meets.any(axis=1) & (~meets).any(axis=1)`,
  which changes handling for censored or missing values.

### ACCEPT: `sample / environment-easy-5`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `Ashburnham` with the same four-file June,
July, and August 2020 rainfall-sum pipeline and the same 5-operator/4-link
workflow shape. The 7k arm carries about 11.7k more input tokens with the same
step count.

Evidence:

- Both workflows load Boston, Chatham, Amherst, and Ashburnham precipitation
  CSVs and sum `Jun + Jul + Aug` for `Year == 2020`.
- 7k: 48,345 input tokens, 843 output tokens, cost `$0.038345`.
- 3k: 36,682 input tokens, 717 output tokens, cost `$0.027662`.
- Both take 4 total steps.

### REJECT: `sample / archeology-easy-10`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a clean sample-size cost signal. Label:
`cache_effect`.

Fundamental reason: both answer `Singapore` with the same two-operator
mean-population-by-country pipeline and the same 4 steps. The 7k arm has only
about 3.7k more input tokens, while cached input drops by about 2.6k, making the
dollar gap cache-sensitive rather than clean sample-size cost.

Evidence:

- Both workflows load `worldcities.csv`, drop rows with missing population,
  group by `country`, and sort by mean city population.
- 7k: 28,430 input tokens, 18,944 cached tokens, cost `$0.023192`.
- 3k: 24,718 input tokens, 21,504 cached tokens, cost `$0.012538`.
- Output tokens and workflow shape are nearly identical.

## Batch 32 Validations

### REJECT: `sample / astronomy-easy-4`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`evaluator_issue`.

Fundamental reason: both arms build the same SILSO sunspot peak/trough pipeline
and produce the same average period (`11.25`) with the same maxima and minima
years. The rule match is caused by the paraphrase evaluator accepting one
wording and rejecting the other, not by larger sample visibility.

Evidence:

- Both workflows load `SN_y_tot_V2.0.csv`, split semicolon-delimited rows,
  filter 1960-2020, use `scipy.signal.find_peaks`, and compute max/min cycle
  periods.
- Both final answers list maxima `1968, 1979, 1989, 2000, 2014` and minima
  `1964, 1976, 1986, 1996, 2008`.
- Evaluation differs only in `llm_paraphrase` (`1` for 7k, `0` for 3k).

### REJECT: `sample / biomedical-hard-7`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`verification`.

Fundamental reason: 7k reloads the `D-SE-acetyl` sheet with `header=None` and
counts the header-like `BRD8` row, returning `16`. 3k uses pandas default header
inference and counts `15` rows under the `BRD8` header. This is sheet-header
verification, not hidden sample visibility.

Evidence:

- Gold solution explicitly adds one to the default-read row count because the
  first item is treated as a header.
- Rich has both `acetyl_sheet` and `acetyl_sheet_raw`, then counts from the raw
  `header=None` table.
- Lean only uses `acetyl_data` with default header inference.

### ACCEPT: `sample / environment-hard-20`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `['Bucks Creek', 'Pleasant Street',
'Forest Street']` with the same final rainfall-min-city then beach-pollution
pipeline and the same 8-operator/7-link shape. The 5k arm carries about 38k
more input tokens and splits construction across two more steps.

Evidence:

- Both workflows load four precipitation CSVs plus `water-body-testing-2015`.
- Both compute the least-rainfall city for summer 2015, then rank beaches in
  that city by violation/pollution metrics.
- 5k: 79,801 input tokens, 1,272 output tokens, 6 steps, cost `$0.051620`.
- 3k: 41,748 input tokens, 1,240 output tokens, 4 steps, cost `$0.041027`.

### ACCEPT: `sample / legal-easy-11`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `No` with equivalent
`Other / (Fraud + Identity Theft + Other)` logic over 2001-2024 and the same
3-operator/2-link workflow shape. The 7k arm carries about 9.4k more input
tokens and one extra step.

Evidence:

- Both workflows clean the CSN report-type table, parse Fraud, Identity Theft,
  and Other counts, and check whether Other exceeds half the total.
- 7k: 39,838 input tokens, 988 output tokens, 6 steps, cost `$0.027504`.
- 3k: 30,460 input tokens, 536 output tokens, 5 steps, cost `$0.017062`.

## Batch 33 Validations

### REJECT: `mode / legal-hard-15`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52LatestStats3kD2`

Protocol judgment: REJECT as a context-mode accuracy signal. Label:
`wrong_transform`.

Fundamental reason: Delta returns `243377` by deduplicating cross-state
statistical-area rows before summing. Latest returns `593524` by summing
duplicate rows from multiple state files directly. This is dedup/aggregation
verification, not a latest-vs-delta context effect.

Evidence:

- Gold concatenates all state files, parses cross-state areas, then drops
  duplicates before summing.
- Delta `it_msa_2024_clean` deduplicates `(metropolitan_area, reports)` before
  `it_msa_2024_cross_state_total`.
- Latest `cross_state_total_2024` filters cross-state patterns and sums
  `reports` without a duplicate-removal step.

### REJECT: `sample / biomedical-hard-5`

Pair:

- Rich: `DataflowSystemGPT52LatestStats7kD2`
- Lean: `DataflowSystemGPT52LatestStats5kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`wrong_transform`.

Fundamental reason: 7k returns `2.6563` by filtering `Case_excluded != Yes`
before computing the serous median. 5k returns `2.4241` by computing over serous
metadata rows without the exclusion filter. Since both arms have stats, this is
filter/exclusion logic rather than hidden sample visibility.

Evidence:

- Gold filters excluded cases before selecting serous samples.
- Rich `serous_tmb_median` applies both `Case_excluded != Yes` and
  `Histologic_type == Serous`.
- Lean `variants_per_mbp_serous` filters only `Histologic_type == Serous`.

### REJECT: `sample / wildfire-easy-1`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest5kSchemaOnly`

Protocol judgment: REJECT as a clean sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `7805421` with a NOAA three-month
`Acres Burned` rolling-sum pipeline, but the 7k arm repeatedly rewrites the
loader and adds extra raw-preview/debug steps. The cost gap is construction
churn plus larger context, not clean sample-size cost.

Evidence:

- Both final workflows load `noaa_wildfires_monthly_stats.csv`, skip the
  metadata preamble, and compute a 3-month rolling sum after 2000.
- 7k has repeated `noaa_monthly` edits plus `noaa_raw_preview`, and takes
  8 steps.
- 5k has fewer loader/debug edits and takes 6 steps.

### ACCEPT: `sample / legal-easy-25`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer `U.S. Space Force` with equivalent
military-branch `Median Fraud Loss` maximum pipelines and the same
3-operator/2-link workflow shape. The 5k arm carries about 5.8k more input
tokens and takes one extra step.

Evidence:

- Both workflows load the 2024 military-consumer report CSV, extract the
  military-branch section, parse `Median Fraud Loss`, and sort descending.
- 5k: 36,297 input tokens, 1,071 output tokens, 6 steps, cost `$0.027307`.
- 3k: 30,450 input tokens, 626 output tokens, 5 steps, cost `$0.018506`.

## Batch 34 Validations

### REJECT: `info / environment-hard-9`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats5kD2`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`parser_layout`.

Fundamental reason: the stats arm returns separate `Castle Island Beach`,
`City Point Beach`, and `Pleasure Bay Beach` labels. The schema-only arm keeps
the combined `Pleasure Bay / Castle Island` label. This is beach-name/parser
layout, not clean stats information.

Evidence:

- Gold prepares each Boston Harbor datasheet and handles the
  Pleasure Bay/Castle Island file as separate sample locations.
- Rich `standardize_pleasure_castle` emits separate `Pleasure Bay Beach` and
  `Castle Island Beach` rows.
- Lean's final answer preserves `Pleasure Bay / Castle Island` as a combined
  beach label.

### REJECT: `sample / wildfire-hard-18`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats7kD2`
- Lean: `DataflowSystemGPT52DeltaStats3kD2`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`evaluator_issue`.

Fundamental reason: both arms conclude that aggressive suppression is associated
with longer fires and fewer threatened buildings after weather controls. The 7k
answer explicitly says this means suppression does not make fires end faster;
the 3k answer omits that phrasing. This is evaluator/wording behavior, not
sample visibility.

Evidence:

- Rich answer: longer-lasting fires, fewer threatened buildings, does not make
  fires end faster.
- Lean answer: fires lasting longer, fewer threatened buildings after weather
  controls.
- Both workflows use NOAA wildfire data, weather controls, and regression
  estimates for duration and building impact.

### ACCEPT: `mode / legal-hard-7`

Pair:

- Rich: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: ACCEPT as a Delta-history cost signal. Label:
`delta_history_cost`.

Fundamental reason: both answer `Bank Account` with the same
2020-vs-2024 identity-theft relative-growth pipeline and the same
3-operator/2-link workflow shape. Delta has one extra cleaner edit/step and
about 8.3k more input tokens.

Evidence:

- Both workflows load the top-three identity-theft reports CSV, clean year/type
  report rows, and compute relative growth from 2020 to 2024.
- Delta: 37,217 input tokens, 1,155 output tokens, 6 steps, cost `$0.028077`.
- Latest: 28,914 input tokens, 804 output tokens, 5 steps, cost `$0.019318`.

### ACCEPT: `sample / legal-hard-29`

Pair:

- Rich: `DataflowSystemGPT52Delta7kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta5kSchemaOnly`

Protocol judgment: ACCEPT as a sample-size cost signal. Label:
`sampling_cost`.

Fundamental reason: both answer
`Washington-Arlington-Alexandria, DC-VA-MD-WV Metropolitan Statistical Area`
with the same all-state MSA fraud-share pipeline and the same 6-operator/5-link
workflow shape. The 7k arm carries about 32.7k more input tokens and two extra
construction steps.

Evidence:

- Both workflows concatenate all State MSA Fraud and Other CSVs, parse report
  counts, compute state totals, filter states with at least 5 metro/micro areas,
  and choose the highest within-state share.
- 7k: 78,024 input tokens, 1,236 output tokens, 8 steps, cost `$0.044982`.
- 5k: 45,302 input tokens, 1,051 output tokens, 6 steps, cost `$0.036335`.

## Batch 35 Validations

### REJECT: `info / environment-easy-2`

Pair:

- Rich: `DataflowSystemGPT52DeltaStats3kD2`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a stats/info accuracy signal. Label:
`wrong_transform`.

Fundamental reason: the stats arm includes `2015` after comparing rates on a
percent scale rounded to two decimals. The schema-only arm rounds fractional
rates and the average to two decimals before comparing, making 2015 tie the
rounded average and drop out. This is rounding/scale semantics, not stats
information.

Evidence:

- Gold computes yearly freshwater exceedance rates and compares them to the
  historical average rate.
- Rich `freshwater_exceedance_above_avg_years` compares rounded percent values.
- Lean `years_above_avg` compares `exceed_rate_2dp` to a rounded fractional
  average (`0.xx`), dropping 2015.

### REJECT: `sample / environment-easy-2`

Pair:

- Rich: `DataflowSystemGPT52Delta5kSchemaOnly`
- Lean: `DataflowSystemGPT52Delta3kSchemaOnly`

Protocol judgment: REJECT as a sample-size accuracy signal. Label:
`wrong_transform`.

Fundamental reason: 5k compares unrounded freshwater exceedance rates and
includes `2015`; 3k rounds fractional yearly and average rates to two decimals
before comparing and drops `2015`. This is comparison semantics, not sample
visibility.

Evidence:

- Rich `freshwater_exceedance_above_avg_years` filters
  `exceedance_rate > avg_rate`.
- Lean `years_above_avg` filters `exceed_rate_2dp > avg_exceed_rate_2dp`.
- Both arms load the same water-body-testing files and use the same Fresh beach
  filter.

### REJECT: `mode / astronomy-hard-10`

Pair:

- Rich: `DataflowSystemGPT52Delta3kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a context-mode cost signal. Label:
`shorter_outputs`.

Fundamental reason: both answer `['Proton_flux_>30_Mev', -0.193]` with the same
OMNI2/Sat_Density/SP3 altitude-change correlation pipeline. Delta has fewer
steps and nearly the same input tokens, but emits about 579 more output tokens.
The cost gap is output/accounting, not Delta-history context.

Evidence:

- Both workflows parse SP3 Swarm-A positions, compute hourly altitude change,
  join OMNI2 and Sat_Density hourly variables, and scan Pearson correlations.
- Delta: 89,210 input tokens, 2,561 output tokens, 7 steps, cost `$0.074237`.
- Latest: 88,386 input tokens, 1,982 output tokens, 8 steps, cost `$0.065697`.

### REJECT: `sample / wildfire-easy-1`

Pair:

- Rich: `DataflowSystemGPT52Latest7kSchemaOnly`
- Lean: `DataflowSystemGPT52Latest3kSchemaOnly`

Protocol judgment: REJECT as a clean sample-size cost signal. Label:
`operator_churn`.

Fundamental reason: both answer `7805421` with the NOAA three-month
`Acres Burned` rolling-sum pipeline, but the 7k arm repeatedly rewrites the
loader and adds raw-preview debugging before the final operator. The cost gap
is construction churn, not clean sample-size cost.

Evidence:

- Both workflows load `noaa_wildfires_monthly_stats.csv`, skip the preamble, and
  compute a rolling 3-month `Acres Burned` maximum.
- 7k has repeated `noaa_monthly` edits plus `noaa_raw_preview`, and takes
  8 steps.
- 3k has one raw preview plus the final rolling operator, and takes 6 steps.

## Implications for the Analyzer Counts

Current rule-based counts:

- Accuracy principle matches by rule: `106`
- Cost principle matches by rule: `133`
- Strict cost subset with same steps and code similarity >= 0.5: `17`

Manual validation suggests:

- Accuracy needs stricter filtering. Many rule matches are not real
  information-availability wins.
- Cost needs two subtypes:
  - pure same-work cost: same steps or near-identical workflow/code, useful for
    stats/sample costs;
  - Delta-history cost: same final logical DAG, but more edits/tool calls/error
    recovery in the rich Delta run.

Recommended next analyzer refinement:

1. For accuracy, add rejection labels for timeout/no-response, unit conversion,
   rounding/comparison semantics, and verification-only wins.
2. For cost, split counts into:
   - `pure_context_cost`: same answer, same final DAG shape, same/near-same code,
     similar step count;
   - `delta_history_cost`: same answer, same final DAG shape, rich Delta has
     materially more tool calls or repeated edits to the same operator.

# Final Accuracy Flip Audit

Audited 2026-07-10.

## Method

Compared final artifacts for:

- Control (`C`): `DataflowSystemGPT52LatestStats3kD2SmallTableControl`
- Treatment (`T`): `DataflowSystemGPT52LatestStats3kD2FrontierDecay`

Pass means the answer-type-specific metric is at least `0.9`: `success` for
numeric/string exact, `rae_score` for numeric approximate, `f1` for list
exact, and `llm_paraphrase` for string approximate. Across 104 shared tasks,
there are seven final flips: three T-only passes and four C-only passes.

For every flip I inspected `answer.json`, `evaluation.json`, `workflow.json`,
and every agent input/tool call in `react_steps.json`. A frontier-decay
exposure requires a table with more than four rows whose prompt rendering
retains schema but drops column stats and reduces the row sample. Stats-free
tables with fewer than five rows are the shared small-table rule, not frontier
decay.

## Result

| Task | Winner | C answer / score | T answer / score | First T decay | Classification |
| --- | --- | --- | --- | --- | --- |
| `biomedical-hard-5` | T | `1.2775` / 0 | `2.6563` / 1 | None | No exposure; C wrong transform |
| `environment-hard-13` | C | `11` / 1 | `12` / 0 | None | No exposure; T null-handling bug |
| `environment-hard-8` | C | `54.03` / 1 | `51.41` / 0 | A7, after bug | Post-divergence exposure; T lossy reshape |
| `environment-hard-9` | T | two wrong/missing names / 0 | three gold names / 1 | None | No exposure; C generic parser loses a site |
| `legal-easy-19` | T | `0.199` / 0 | `0.523` / 1 | None | No exposure; C wrong denominator |
| `wildfire-hard-12` | C | `No` / 1 | `Yes` / 0 | None | No exposure; T decision-rule noise |
| `wildfire-hard-17` | C | `4830.9` / 1 | no response / 0 | A5, before branch divergence | Exposed but evidence points to unrelated diagnostic/churn failure |

No flip supplies positive evidence that frontier decay caused the accuracy
change. Five have no treatment exposure at all, one is exposed only after the
accuracy-relevant transform is already wrong, and one has temporally prior
exposure but retains the relevant information elsewhere in full context.
The prompt scan found three decayed renderings across two operators in
`environment-hard-8`, and 35 renderings across five operators in
`wildfire-hard-17`; the other five flips had zero.

## Case Evidence

### `biomedical-hard-5` - T-only pass

- No T prompt contains a frontier-decayed table.
- C A4 `serous_variant_per_mbp` takes the median of raw
  `Log2_variant_per_Mbp` and returns `1.277478143`; it neither applies
  `2 ** Log2_variant_per_Mbp` nor filters `Case_excluded == 'No'`.
- T A3/A6 filters included serous tumors, applies the exponentiation, and
  returns `2.656346749406481`.

Classification: **no exposure / unrelated C wrong transform**, high
confidence.

### `environment-hard-13` - C-only pass

- No T prompt contains a frontier-decayed table.
- C A4 requires a non-null point to count as not meeting:
  `meets.eq(False) & vals.notna()`, yielding 11 mixed days.
- T A3 uses `(~meets).any(axis=1)`. Because comparisons against missing values
  are false, inversion counts a missing point as a non-meeting point and
  inflates the result to 12.

Classification: **no exposure / unrelated T null-handling bug**, high
confidence.

### `environment-hard-8` - C-only pass

- The accuracy-relevant bug is introduced in T at A5: `beach_samples_wide` pivots
  to one row per file/beach/date with `aggfunc='first'`. Datasheets have
  repeated Enterococcus columns for multiple sampling sites, so this collapses
  site-level samples.
- T A6 sees that wrong table in full context with stats: 10,956 rows. It then
  derives only 1,062 failed samples and eventually reports 51.41.
- The first decay is later, at A7: `beach_samples` (`79,428 x 5`) loses stats
  after `beach_samples_wide` already exists. At A8, `beach_samples_wide` also
  decays only after `failed_samples` exists. The final A7 operation consumes
  the still-full `failed_samples`, not either decayed ancestor.
- C A13 preserves every duplicated Enterococcus column as a separate
  `site_idx`, producing 21,986 samples and 2,071 failures; A14 returns 54.03.

Classification: **post-divergence exposure / unrelated T lossy reshape**, high
confidence.

### `environment-hard-9` - T-only pass

- No T prompt contains a frontier-decayed table.
- C A1 concatenates heterogeneous beach sheets and C A3 generically reads only
  one measurement column (`c5`). It does not separate Castle Island from the
  combined Pleasure Bay/Castle Island sheet; its final table has only City
  Point and Pleasure Bay as passing, both with unwanted location suffixes.
- T A1 loads each sheet separately. T A2 explicitly maps Pleasure Bay from
  columns 5/7 and Castle Island from column 9, returning the three gold names.

Classification: **no exposure / unrelated C parsing/topology choice**, high
confidence.

### `legal-easy-19` - T-only pass

- No T prompt contains a frontier-decayed table.
- C A2 divides the five `$1-$500` bins by `Number of Fraud Reports`, producing
  `0.198528...`.
- T A2 uses the task-relevant `Reports with $ Loss` denominator (`987,520`),
  producing `0.523`.

Classification: **no exposure / unrelated C denominator choice**, high
confidence.

### `wildfire-hard-12` - C-only pass

- No T prompt contains a frontier-decayed table.
- Both arms derive a trend from yearly median fire-start day. T A2 computes a
  slope of `0.404142...` days/year but defines `shifted` as `m != 0`, so any
  numerical nonzero trend becomes `Yes`.
- C A2 uses a materiality rule, `abs(slope) > 0.5`, and returns the gold `No`.

Classification: **no exposure / unrelated interpretation and decision-rule
noise**, high confidence.

### `wildfire-hard-17` - C-only pass

- The arms are semantically aligned through A4: load NOAA fires and RAWS,
  derive 779 unique `station_verified_in_psa` IDs, then try the wrong RAWS
  `Station ID` join.
- At T A5, before the first paired branch-choice divergence, `noaa_wildfires`
  is genuinely decayed (`6,658 x 37`, schema and endpoint sample retained,
  stats removed). C A5 tries `WX ID`; T A5 retries `Station ID`.
- The removed information is not needed for the join diagnosis. In the same T
  A5 prompt, `noaa_station_ids` remains fully rendered with stats (779 IDs,
  min 20,107, max 482,106), `raws` remains fully rendered with stats and a
  visible `NWS ID` schema field, and the failed join explicitly shows 779 null
  elevations.
- T A7 nevertheless writes diagnostics for only `WX ID` and `Station ID`,
  omitting the visible `NWS ID`. C A8 tests five candidate columns and finds
  `NWS ID` gives 759 matched rows / 758 distinct stations; C A9-A10 then returns
  4830.9.
- T instead builds many speculative ID transforms. Later decays affect old
  mapping branches only after their consumer reports match counts of 3, 0,
  and 0. At A25 T has a wrong interim mean (`3458.759...`), starts a rounding
  operator, and reaches the step limit without a final response.

Classification: **exposed before divergence, but unrelated diagnostic omission
and operator churn**, medium-high confidence. This is the only flip where
temporal exposure prevents a categorical no-effect claim, but the missing
information is redundant and the observed failure mechanism does not match
what decay removed. The direct counterfactual is to test the still-visible
`NWS ID`, not to restore NOAA source statistics.

## Causal Summary

| Causal bucket | Tasks |
| --- | ---: |
| No frontier-decay exposure | 5 |
| Decay only after first accuracy-relevant divergence | 1 |
| Decay before divergence, evidence supports unrelated failure | 1 |
| Plausibly caused by frontier decay | 0 |

These seven independent-run flips should therefore be treated as transform,
interpretation, and search variance rather than an observed frontier-decay
accuracy cost. `wildfire-hard-17` remains the one case worth a checkpointed
same-prefix replay if stronger causal evidence is required.

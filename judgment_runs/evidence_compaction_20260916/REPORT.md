<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements. See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership. The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied. See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Compact evidence: 104-task Terra-medium evaluation

Final reported results: one original attempt per task and configuration. No retry
scores or retry costs are included in the primary tables. Further retries are
stopped at the user's request. See the excluded-attempt accounting below.

## What was evaluated

Lossless compaction after the existing shared sample/evidence budget selector:

- Combine selected null/non-null/distinct-count sentences into shorter tables.
- Remove a Data count only when an actually displayed Filter Flow fact covers
  the exact same measurement, coverage, column and result version.
- Share repeated upstream summaries within one rendered observation group.
  The earlier full summary must be visible and match the upstream version.

No extra samples or facts fill the space saved. Samples, schema, errors, range
facts, value facts and the selection policy are unchanged. Stored observations
remain self-contained; sharing applies only to their model-facing rendering.
No old `dataLevel`, `flowLevel`, `columnStats` or `inspectResult` is reintroduced.

The implementation was evaluated at root commit
`1718a1307bd819110751a61f0a4f64f42c246194`, using KramaBench harness commit
`0f443ff954a4bac19792bb724a89671875f8a877`. Later test/report commits do not
change the evaluated runtime. Branch: `feat/evidence-compaction` in both repos.

## Frozen protocol

The three new evidence arms use `gpt-5.6-terra`, medium reasoning, native Python V2 operators,
batching with explicit `observe`, DELTA context, profiling enabled, 25 steps,
a 2,000-character observation budget and a 3,000-character cell limit. Parallel
tool calls, code-in-snapshot, thought replay and optional pull tools are off.
The exposed tools are `dataflow`, `deleteOperator` and `inspectError`.

Only the Data/Flow exposure flags differ. Agent services are isolated on
3017 / 3018 / 3019; the Texera backend is shared and was not restarted or
modified during this campaign. Maximum concurrency is six tasks, two per arm,
with at most one task per arm/domain to avoid native result-file collisions.

Full SUT names follow:
`DataflowSystemTerraCompactEvidenceFull{DataOnly,FlowOnly,Combined}20260916Rep1`.
Smoke namespaces replace `Full` with `Smoke` and are not included in full-run
scores or costs.

The Native row is the saved historical
`DataflowSystemTerraNativeCampaignObserveOnlyBatchParent20260916Rep1` first
pass: native V1 operators, batching + observe, no Data/Flow evidence and no
column statistics. It also uses Terra-medium, DELTA, 25 steps and the same
2,000-character result budget. Its source is
`95e3b12b462d3f198e3148e482453e9939621186`; its harness is
`4b91499b3cdd94f482f8afccc8e9ed73063102ec`.

This is **not a same-version, evidence-only ablation against Native**. The
three new rows are mutually matched, but the Native row also differs in
operator catalog, tool/error surface and implementation revision. Its old
L1/L1 settings supplied ordinary schema/source structure, not the V2 evidence
channels. There is no new, matched 104-task V2 no-evidence run; the six-task
earlier V2 pilot is not substituted for a full run.

The native `kb.py tasks` execution path and native per-task evaluator produce
the answers, primary metrics and token cost records. Overall scores use the
repository's `compute_scores_for_system`: support-weighted primary metric
means over all 104 tasks, including partial credit. A score percentage is not
the same as the exact-pass proportion.

Exact pass means completed with primary score >= 1. The requested comparison
uses only `round == "first"`: never the latest or best of several attempts.
Do not automatically rerun failed or partially correct tasks. The read-only
analysis rejects recovery rows, duplicate tasks and incomplete task sets.
Additional attempts that occurred before this clarification remain archived
and are excluded; they are disclosed separately below.

The native latest-measures CSV is incomplete when tasks are launched singly.
Consequently aggregation uses the preserved per-task native
`evaluation.json`, not that partial CSV. No custom judge replaces the benchmark
scorer, and no post-hoc re-scoring changes the original evaluations.

### Input readiness

All 104 canonical tasks remain in each denominator. `astronomy-hard-7` is
blocked before model dispatch because six required files for the specified
October 2016 dates are absent. Similar filenames cover different dates and
were not substituted. Its native score is zero; its missing model-cost record
is not treated as measured zero-token usage. The same exclusion from model
dispatch applies to all four reported arms.

The qualified input/proof inventory covers 802 resolved files. The existing
weather dataset was linked into this worktree after verifying its source
checksum; no task data was fabricated or changed.

## First-pass results

| Arm | Native score / 104 | Exact passes | Mean cost / executed task | Total model cost |
| --- | ---: | ---: | ---: | ---: |
| Native (historical V1; no evidence) | 69.78% | 68/104 | $0.02816 | $2.900720 |
| Data-only | 76.19% | 75/104 | $0.03372 | $3.473659 |
| Flow-only | 72.35% | 71/104 | $0.03459 | $3.562919 |
| Data + flow | 73.32% | 72/104 | $0.03472 | $3.575805 |

Each arm has 103 actual model dispatches and one input-readiness-blocked task.
All dispatched first-pass attempts have complete captured usage. Mean model
cost is total recorded model cost divided by **103 dispatched tasks**, not
104 and not just successful tasks. Failure costs remain included.

Costs are native `stats.json` token estimates at frozen campaign rates:
$2/M uncached input, $0.20/M cached input and $12/M output. Reasoning tokens
are part of output, not charged twice. These are not provider invoices and
exclude native judge charges and backend compute.

| Arm | Input tokens | Cached input | Output tokens | Cached input fraction |
| --- | ---: | ---: | ---: | ---: |
| Native (historical V1) | 3,788,721 | 3,330,807 | 109,894 | 87.91% |
| Data-only | 5,931,619 | 5,570,619 | 136,461 | 93.91% |
| Flow-only | 5,952,838 | 5,559,486 | 138,693 | 93.39% |
| Data + flow | 5,937,186 | 5,554,826 | 141,677 | 93.56% |

## Excluded additional attempts

The initial supervisor scheduled extra retries after the first pass. The user
clarified that only one repetition was wanted; further admissions were paused
and the retry supervisor was then terminated. At the pause, 62 extra attempt
records were finished and six were in flight. Those six had finished by the
stop check. No second recovery round was launched.

There are **68 excluded additional attempt records**, of which **65 dispatched
a model**, with **$2.838580** in complete recorded model-token cost. This is
additional actual estimated spend, **not included in the first-pass table**.
Neither their answers nor their scores replace any first-pass result.
The original first-pass files were preserved before retries; the report reads
those archives, not the subsequently updated current task directories.

This disclosure does not authorize future retries. The campaign policy is
single-pass reporting and no further failed-task reruns.

## Qualification and trace audit

Tests were implemented first: six new regressions failed before the production
change, then passed. Final agent test suite: 1,070 passed, zero failed,
7,221 assertions and eight snapshots; TypeScript typecheck passed. The focused
modern native-campaign and report suites have 38 passing tests, including six
new read-only analysis tests. Historical fixture
constructors were updated to reject retired legacy settings rather than
re-enabling those settings.

Real-engine gates passed 18 cases: eight compaction cases, eight existing
evidence cases and two error-diagnostic cases. These cover all four flag
combinations in DELTA and LATEST, forward-referenced DAG dependencies, cached
observe, hidden successful results and immutable stored observations.

Separate real-model smoke: six exact passes from two tasks across three arms.
The complete first-pass audit checks actual captured model input, not a
reconstructed prompt: 309 dispatched runs, 1,550 model requests, 1,241
append-only transitions, 2,448 immutable tool-result replays, 1,468 observations,
7,408 retained sample chunks, 440 compact count tables, 2,976 exact column
counts and 47 valid same-frame shared references. All checks passed; every
attempt's usage sums and cache-aware cost reconcile.

### Direct representation savings

These compare the exact selected strings with their equivalent pre-compaction
sentences. They do not replay the unmodified agent and are not counterfactual
token bills. Character counts are UTF-16 before container indentation.

| Arm | Equivalent expanded observation characters | Count/overlap savings | Shared-upstream savings | Total saved |
| --- | ---: | ---: | ---: | ---: |
| Data-only | 591,776 | 90,537 | 0 | 15.30% |
| Flow-only | 631,007 | 0 | 2,029 | 0.32% |
| Data + flow | 685,883 | 81,673 | 1,907 | 12.19% |

Data + flow's count/overlap savings split into 80,363 grouped-count characters
and 1,310 exact-overlap characters. Most savings come from compact count
formatting, not from repeated upstream summaries. There is no concurrent
uncompacted 104-task control in this campaign, so this run does not establish
a causal dollar saving or accuracy change versus the old renderer.

## Bounded manual trace findings

These are explanations of saved native outcomes, not replacement scores.
Cases were selected from first-pass exact-pass flips and the largest both-pass
cost differences; the first attempts remain available under `_attempts` after
retries.

- `biomedical-easy-2`: Data-only returns 68.5; Combined returns 68.1. Both
  select serous tumor samples, but only Data-only subsequently filters
  `Case_excluded == 'No'`, matching the native gold program. This is a missing
  semantic filter, not a lost compact count.
- `biomedical-hard-7`: Combined returns 16; Data-only returns 15. Both see the
  same complete 15-row table with column name `BRD8`. Combined rescans with
  `header: false` and counts the first gene; Data-only stops. The gold program
  explicitly accounts for that first row. Flow includes extra source/parse
  context, but this one trace pair does not prove it caused the correction.
- `archeology-easy-11`: Combined filters primary capitals, selects the largest
  per country, and returns 17.4274. Data-only eventually switches to primary
  capitals but drops the per-country population selection, returning 17.1667.
- `environment-hard-9`: both pass, but Data-only uses ten model calls versus
  Combined's five ($0.100903 versus $0.047992). Data-only repairs a rejected
  `Path()` use, a missing `file` column and an invalid callback signature.
  Extra repair turns and generated code dominate the gap.
- `environment-hard-10`: both return 0.206. Combined uses eight model calls
  versus Data-only's five ($0.079231 versus $0.036613). It performs additional
  intermediate inspections and submits output names in the `op` field before
  repairing them to `op: 'udf'` with explicit IDs. Its output-token count is
  3,510 versus 1,342.

The promising next cost experiment is to prevent malformed operator/function
requests and unnecessary verification turns while preserving important source
and header checks. A single shorter, correct trajectory can save more than
further compressing already-cached evidence. Do not infer from these examples
that Flow is generally harmful or that Data-only is statistically superior;
the tables select only the original replicate per arm.

## Execution cleanup caveat

The original supervisor paused on an already-completed task whose workflow
retained an older `Running` history record after a timed-out execution.
Safe resumption preserved all finished attempts and kept the same source,
model, evidence flags, limits and backend. It permitted quarantine only after
verifying ownership, an AVAILABLE agent and a terminal latest execution.
Unresolved historical records were neither deleted nor forcibly marked done.
Source, input, capture and currently-active-execution checks remained mandatory.

Eight first-pass workflow/agent resources remain recorded as pending cleanup
with verified terminal latest executions and unresolved historical records.
They were retained rather than forcibly deleted or marked finished. Isolated
services were left available for inspection; the shared backend was unchanged.
These cleanup exceptions are separate from score and trace-capture validity.

## Artifacts

Campaign root on the evaluation machine:
`/tmp/evidence-compaction-campaign.ul85kx`.

Raw task artifacts remain at `system_scratch/<SUT>/<task>/`, with previous
attempts at `system_scratch/<SUT>/_attempts/<task>/<round>/`. Files include
native `evaluation.json`, `stats.json`, actual `react_steps.json` input messages,
`snapshots.json`, workflow, settings, resource journals and qualification proofs.
Shared raw logs and credentials are not included in the committed report.

Committed report artifacts:

- [first_pass_tasks.csv](first_pass_tasks.csv): 416 original task metrics and
  costs, with SHA-256 bindings to each native evaluation and statistics file.
  Empty cost cells mean no recorded model cost, not fabricated zero usage.
- [summary.json](summary.json): native per-domain/overall scores, exact passes,
  token totals and per-executed-task costs.
- [trace_audit_summary.json](trace_audit_summary.json): complete first-pass
  context/cache checks and exact representation savings for the three new arms.
- [provenance.json](provenance.json): source pins, qualified settings, input
  blocker, test totals and excluded-attempt accounting.
- [analyze.py](analyze.py): read-only reproduction, using the repository's
  official aggregation function. It does not call a model or re-score an answer.

From the KramaBench root:

```bash
python judgment_runs/evidence_compaction_20260916/analyze.py
python -m unittest discover -s tests -p 'test_compact_evidence*.py'
python -m unittest discover -s tests -p test_native_campaign.py
```

All 416 exported primary metrics were checked against their original native
`evaluation.json` files, and all 412 dispatched costs against `stats.json`.
The analysis tests also reproduce the committed summary exactly, retain partial
credit and blocked tasks in the score denominator, and refuse to turn unknown
costs into zero.

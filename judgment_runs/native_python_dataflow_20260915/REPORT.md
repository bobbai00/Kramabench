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

# Python-native dataflow: completed Luna/Terra pilot

2026-09-15. **14/14 official attempts complete, scored and comparison-eligible.**
One task, one attempt per arm/model; this is a mechanism pilot, not an overall
KramaBench accuracy estimate. No scored attempt was replaced or rerun.

Batching + explicit `observe` is the promising result here: it answered correctly
on both models and cost 24.7% less on Luna and 33.8% less on Terra than the
native all-results reference. The Python framework and evidence work passed
the functional gates, but **data-only, flow-only and combined evidence did not
improve accuracy**. Do not promote the evidence treatment or claim a general
cost/accuracy improvement from these measurements.

## 1. Official answers and cache-aware cost

Task: `environment-easy-3`, numeric-exact, official expected answer **268**.
The question asks how many beaches had a higher bacterial exceedance rate in
2013 than in 2012, excluding beaches without 2012 samples.
Official `success` is 1 for 268 and 0 for 267; the frozen pass threshold is
`>= 0.9`. All costs below are the finalized `stats.json.cost_usd` token estimates
in USD, including cached-input discounts. Steps count model steps, including
the final answer, not the initial user message or backend executions.

| Arm | Luna answer / score | Luna USD | Steps | Terra answer / score | Terra USD | Steps |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Native all-results/no-stats (`V1Reference`) | 267 / 0 | 0.004128 | 8 | 268 / 1 | 0.049167 | 12 |
| Batching + observe (`BatchParent`) | 268 / 1 | 0.003109 | 4 | 268 / 1 | 0.032548 | 4 |
| Python, evidence/collection off (`V2Control`) | 267 / 0 | 0.011342 | 13 | 268 / 1 | 0.031557 | 4 |
| Python, collection on/exposure off (`CollectorOnly`) | 267 / 0 | 0.002469 | 4 | 268 / 1 | 0.017218 | 3 |
| Python + data evidence (`DataOnly`) | 267 / 0 | 0.002930 | 4 | 267 / 0 | 0.024455 | 3 |
| Python + flow evidence (`FlowOnly`) | 267 / 0 | 0.002291 | 3 | 267 / 0 | 0.023779 | 3 |
| Python + both (`Combined`) | 267 / 0 | 0.002439 | 3 | 267 / 0 | 0.031571 | 4 |

Total cost of these 14 scored attempts: **$0.239003**. This excludes development,
failed qualification attempts and model-smoke calls; it is not total project
spend or a provider invoice. There is no LLM judge cost for numeric-exact scoring.

Terra's collector-only run is the cheapest correct Terra run, but it exposed
no profile facts and used a different, shorter plan. That observation is not
evidence that collecting hidden statistics reduces model cost. Luna's Python
control is unusually expensive because of repeated invalid tool arguments
(section 4); evidence arms being cheaper than that run is not a robust saving.

## 2. Frozen comparison and provenance

Both **`gpt-5.6-luna` and exact `gpt-5.6-terra` used medium reasoning**. The live
gateway route was checked, not inferred from the SUT name; `tara` was not
substituted. Settings and named arms are in the
[registered protocol](../../docs/native-python-pilot.md) and
[arm definitions](../../systems/native_python_system.py).

All scored attempts used DELTA, native mode, 25 maximum steps, 2,000 result
characters, TSV, legacy column statistics off, no code snapshot/thought replay
or compaction, the same two oracle-selected input files, and isolated fresh
workflows/CU registrations. The oracle supplies file names, not the answer or
gold solution. Batch arms disabled provider parallel calls; V1 enabled them.
LATEST was functionally tested separately, not scored in this pilot.

| Arm family | Actual agent source | Port | Collection / exposure |
| --- | --- | ---: | --- |
| V1Reference | `bdd6b84037bd32818684e291dcdf5933b7a23466` | 3013 | Legacy all-results; no column statistics |
| BatchParent | `41cd8ae97e30edfec90c847886ed4068dd0fa9b0` | 3012 | V1 batch + explicit observe; no new profiles |
| All V2 | `1c12044149da5a17c5c2f32c940c962762583133` | 3011 | Control off/off; collector on/off; data on/data; flow on/flow; combined on/both |

Measured harness: `1929be6841a9a7ff2ad9b7c87d2abdc7168d64c5`.
These are execution-time SHAs; the report commit necessarily comes later.
The V2 branch descends from the batch parent. V1Reference is a **controlled
live reference on the pre-batching V1 source with isolated-launch support**,
not an old historical result copied into this table.

All services shared the candidate Texera master on 8085, updated/restarted with
the user's explicit permission through `bin/local-dev.sh`. Engine distribution
was built at `ab892c8c15225c7c6ec659b351cb6127bb388dd1`; later changes were
agent/harness-side. Admission bound the actual 410-JAR classpath, Python worker
interpreter/protos, source trees and live process identity before/after tests
and attempts. Thus this does not compare old and new engine performance.

The serial run order was Terra control/data/flow/combined, Luna
control/data/flow/combined, Terra collector/parent/reference, then Luna
collector/parent/reference. No fixed sampling seed or repeated trials were
used. V2 qualification calls also warmed shared prompt caches. Both order and
cache state limit causal comparisons; do not treat these runs as independent
replications of a population accuracy result.

Input hashes (SHA-256):

- `water-body-testing-2012.csv` (1,467,124 bytes): `e98422d708150a37b0265b808d256a052ffa54d0593c1af3ab99d7d10e9d3ff0`.
- `water-body-testing-2013.csv` (1,469,864 bytes): `911d1ef8bf290494ace9ff1785b5e9789a50de21606aaef8d4b45a7ab695becb`.

## 3. Tokens, latency and collector measurements

Input includes cached tokens; output includes reasoning tokens. Reasoning is
shown for diagnosis and is **not charged a second time**. The final column
prices every input token at the uncached rate as a diagnostic counterfactual;
it is not actual spend and does not simulate a new uncached rollout.

| Model / arm | Input | Cached input | Output | Reasoning | Wall s | All-input-uncached USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna / V1Reference | 60,699 | 51,302 | 1,019 | 530 | 78.393 | 0.013363 |
| Luna / BatchParent | 29,397 | 20,791 | 810 | 186 | 76.920 | 0.006851 |
| Luna / V2Control | 185,240 | 167,826 | 3,752 | 495 | 109.606 | 0.041550 |
| Luna / CollectorOnly | 46,460 | 43,688 | 867 | 259 | 87.541 | 0.010332 |
| Luna / DataOnly | 48,213 | 43,854 | 984 | 183 | 73.941 | 0.010823 |
| Luna / FlowOnly | 35,078 | 31,868 | 843 | 204 | 66.226 | 0.008027 |
| Luna / Combined | 35,125 | 31,758 | 942 | 204 | 71.151 | 0.008155 |
| Terra / V1Reference | 103,137 | 92,215 | 740 | 161 | 77.207 | 0.215154 |
| Terra / BatchParent | 31,179 | 21,961 | 810 | 215 | 76.660 | 0.072078 |
| Terra / V2Control | 48,375 | 44,445 | 1,234 | 338 | 79.550 | 0.111558 |
| Terra / CollectorOnly | 33,894 | 33,001 | 736 | 98 | 68.969 | 0.076620 |
| Terra / DataOnly | 34,151 | 31,646 | 1,093 | 227 | 74.442 | 0.081418 |
| Terra / FlowOnly | 34,649 | 31,875 | 988 | 153 | 68.920 | 0.081154 |
| Terra / Combined | 49,428 | 45,127 | 1,162 | 351 | 79.496 | 0.112800 |

Frozen rates per million tokens, recorded in each `stats.json`: Luna
input/cache-read/output = $0.20/$0.02/$1.20; Terra = $2.00/$0.20/$12.00.
Cache-creation rates were $0.25/$2.50 respectively, but observed creation tokens
were zero. Rate provenance is `systems.cost_utils._FALLBACK` (token estimate).
The raw trace helper's `in` includes a 168-token initial user step; use the
official stats and model-step usage, not that aggregate, for billed-token math.

Batching need not reduce output: Terra's parent emitted 810 tokens versus 740
for the reference, while reducing model steps from 12 to 4 and input tokens
from 103,137 to 31,179. The Python control emitted 1,234 tokens in four steps.
Richer arguments and plan choices still matter after eliminating round trips.
Cache-aware Terra V2Control cost is slightly below BatchParent, whereas its
all-input-uncached counterfactual is higher. The apparent saving is not robust
to cache accounting.

Every backend request in the next table has explicit successful compilation
and runtime phase outcomes: **67/67 compile, 67/67 execute**, no unknown or
unfinished requests in the closed journals. Argument rejection happens before
this denominator and is reported separately.

| Model / arm | Backend requests | Rejected edits | Observed collection ms | Profile bytes | Unique / partial productions |
| --- | ---: | ---: | ---: | ---: | --- |
| Luna / V1Reference | 6 | 0 | n/a | n/a | n/a |
| Luna / BatchParent | 5 | 0 | n/a | n/a | n/a |
| Luna / V2Control | 6 | 9 | 0 | 0 | 0 / 0 |
| Luna / CollectorOnly | 4 | 0 | 867.516 | 170,292 | 11 / 4 |
| Luna / DataOnly | 5 | 0 | 875.134 | 201,159 | 13 / 4 |
| Luna / FlowOnly | 3 | 0 | 862.203 | 177,860 | 10 / 4 |
| Luna / Combined | 3 | 0 | 884.935 | 201,949 | 13 / 4 |
| Terra / V1Reference | 11 | 0 | n/a | n/a | n/a |
| Terra / BatchParent | 5 | 0 | n/a | n/a | n/a |
| Terra / V2Control | 5 | 0 | 0 | 0 | 0 / 0 |
| Terra / CollectorOnly | 3 | 0 | 849.465 | 198,648 | 11 / 4 |
| Terra / DataOnly | 3 | 0 | 863.070 | 193,855 | 12 / 4 |
| Terra / FlowOnly | 3 | 0 | 863.798 | 179,706 | 10 / 4 |
| Terra / Combined | 5 | 0 | 863.873 | 190,073 | 12 / 4 |

Collection milliseconds sum the observed `ProfileAccumulator.add` timers,
deduplicated by workflow/operator/port/producing execution. They exclude other
collector setup/finalization, encoding, transport and rendering costs, and are
not whole-attempt CPU or wall overhead. Profiles can be partial under the
200 ms collection budget and bounded distinct storage. Bytes describe observed
profile payloads, not model-visible bytes or total network traffic. All eight
collection-enabled runs had partial productions; no claim of complete global
distributions follows. Legacy collection measurements are unavailable, not zero.

The recorder's `backend_termination_verified:false` is deliberately retained:
the journal alone cannot prove backend terminal state. Separate owned-resource
cleanup records verified termination/removal and stopped the isolated service
for all 14 attempts; see [qualification evidence](qualification.json).

## 4. What the traces explain

### Wrong grouping, not an arithmetic or engine failure

The failing plans group/join by **Beach Name alone**. That merges distinct
locations. An offline audit of the actual inputs found 14 beach names occurring
in multiple locations. Grouping by name yields **267**, while grouping by
`Community Code` + `Beach Name` yields **268**. Community Code determines County
Code in these files; no missing entity keys, missing Violation values, or
within-community beach-name case variants explain the difference. The gold
solution's composite location/name key has no concatenation collisions here.
See [reproducible semantic audit](semantic_audit.json) and
[the existing solution](../../solutions/environment/environment-easy-3.py).

This audit was performed after the attempts and never supplied to the models.
A count difference of one does not mean only one beach was misgrouped.
Nor do separate per-column distinct counts prove the right semantic entity key.

Terra V2Control previewed both files, observed both parsed scans, then submitted
an eight-operator chain using Community Code + Beach Name and observed only the
final count (model steps 1/2/3, answer at step 4). Terra BatchParent also used
that composite key. Terra CollectorOnly previewed and then used a nine-operator
full-data batch with the same key, comparing fractions directly inside Filter
instead of introducing two rate columns. It did **not** use a UDF. Its shorter
correct plan explains a plausible cost mechanism without invoking hidden data
evidence. Luna BatchParent likewise used the composite key and answered 268.

### Enabled evidence is not necessarily delivered evidence

Counts below are operator observation frames containing at least one selected
`data.*` or `flow.*` fact whose full text was found in a **later recorded model
input** (user/tool role). A frame may contain several facts. These are not
counts of stored profiles or snapshots. Legacy references are outside this
V2-frame measure.

| V2 arm | Luna data / flow frames | Terra data / flow frames |
| --- | --- | --- |
| V2Control | 0 / 0 | 0 / 0 |
| CollectorOnly | 0 / 0 | 0 / 0 |
| DataOnly | 3 / 0 | 0 / 0 |
| FlowOnly | 0 / 4 | 0 / 3 |
| Combined | 1 / 4 | 2 / 5 |

Terra DataOnly previewed ten raw lines from each file, then built the entire
ten-operator computation and observed only its final count. Complete small
preview tables and the one-row count suppressed redundant data profiles;
parsed full-data scans/intermediates were not observed. **No additional data
fact reached the model**, despite the enabled treatment. Its cheaper wrong
answer cannot demonstrate that statistics helped or harmed reasoning.

Terra FlowOnly observed parsed scans, then an eight-operator whole calculation
and final count. It saw scan provenance/contracts and final aggregate facts,
but not the intermediate grouping/join observations. A one-to-one join of
already name-grouped tables would not prove that name was the correct entity
key in the first place. Flow correctness under declared keys is different from
semantic correctness of those keys.

Terra Combined did observe both parsed inputs before choosing the wrong key.
Those frames were 1,984 and 1,983 characters. Selected data facts covered
**Beach Name and Beach Type Description counts**; Community Code, County Code
and Violation profiles were omitted for budget. Scan provenance, samples and
contracts competed for the same budget. The selector ranks columns used by
the current operator higher and breaks other ties lexically; a Scan has no
declared future grouping key. This is a concrete relevance/budget limitation,
not proof that more arbitrary statistics would solve the task.

Luna DataOnly did see Community/County counts as well as the Beach Name counts
after the parsed scans, yet still grouped by name. Thus merely making these
individual profiles visible is not sufficient. Its final comparison-table
profile arrived after that grouping decision. All witnesses, selected facts,
budget omissions and step IDs are in [results.json](results.json).

For all eight control/data/flow/combined runs the first agent step's
`inputMessages` array has the same canonical JSON SHA-256,
`24d8844ae73db9ca686a30b21bee0b9ae3cabb999a314a8d2fc2e168ea670719`.
This hashes that array only, **not** a whole provider request/tool schema/system
prompt. First actions already differ before new evidence exists. With one
stochastic rollout per condition, later plan differences cannot all be
attributed to evidence exposure.

### Luna's repeated invalid Derive arguments

In Luna V2Control, step 3 puts subsequent intended operators inside a Derive
function string/columns list. Validation reports that `/columns/2` and
`/columns/3` must be objects. Steps 4–11 repeatedly submit malformed nested
payloads, with further pointers to required `as`, `inputs` and `dtype` fields.
There are **nine rejected edits**, visible compact diagnostics, no
`inspectError` call, and no successful repair of that operator. Step 12 falls
back to a Pandas UDF, still using Beach Name alone; step 13 answers 267.

This is a tool-argument usability/repair failure, not a failed backend compile.
The trace's outer batch `isError:false` does not mean every contained edit was
accepted. Per-operator receipts and the validation-stage counts are required.
The failed attempts and their token costs remain included in the control row.

## 5. Implemented scope and qualification

The candidate has 19 opt-in Python-native operators with no SQL Compute in V2,
the shared dependency scheduler, batched edits and explicit observation,
ordinary structured profiles, version-bound data/flow selection, and mandatory
revision-pinned error inspection. Existing V1 remains available. Exact callback,
dtype, empty-input, error and shape contracts are in the parent repository's
`docs/native-python-dataflow-plan.md` and checkpoint ledger.

Recorded checks at the final candidate source:

- Agent service: `bun test` **1,129 pass, 0 fail**; `bun run typecheck` passes.
- Harness: `.venv/bin/python -m unittest discover -s tests` **184 pass**.
- Matching real-worker fixtures: **six suites / 20 mode-setting cases pass**,
  covering Python callbacks, core dependencies/revisions, profiles, evidence,
  shape operators and diagnostics in DELTA/LATEST. Fixtures were grouped for
  concurrent correctness testing; their durations are not speed benchmarks.
- Actual Terra-medium four-turn model smoke passes in both DELTA and LATEST:
  reverse-ordered dependent batch observing only its answer; cached explicit
  observation without a new execution; hidden-success callback failure followed
  by `inspectError`; upstream repair plus a new consumer of the existing DAG in
  one reverse-ordered batch. Actual later inputs witness content/error delivery.

Live qualification uncovered and fixed empty/truncated native context previews,
missing message-layout readback, and LATEST's lack of a completed-action signal.
LATEST now keeps the current dataflow plus one bounded current-task last-action
receipt (no parameters, rows or diagnostic body), and its prompt says visible
completed results can support a direct answer. This does not replay the whole
trajectory; DELTA is unchanged. Earlier failed smoke artifacts are preserved.

**Remaining LATEST limitation:** its cached turn made six identical cached
reads (five redundant), versus one in DELTA. All preserved the same producing
version/rows and caused no extra backend execution. Before official dispatch,
the smoke audit was amended to count redundant read-only calls separately from
functional failure; changed targets/edits/versions still fail, and mutation
turns still require one batch. The old failed audit was not reclassified;
fresh smoke attempts passed the amended, documented protocol. This establishes
functional behavior, not optimal LATEST cost or a representative reasoning win.

Compact test summaries and source-proof hashes are in
[qualification.json](qualification.json). Private full proof/attempt records are
under `/tmp/native-python-live-master.i9nfol/campaign-5/worker-gates/` and
`campaign-6/`. Test resources and ports 3011/3012/3013 were cleaned up; the
shared candidate master remains running on 8085. Other services were not
stopped for cleanup.

## 6. Recommendation and reproducibility

Keep batching + `observe` as the framework foundation. Keep V2/evidence opt-in
until a follow-up addresses the observed failures and a held-out evaluation
shows a repeatable benefit. This pilot does not justify a paper claim that
data/flow evidence improves accuracy or universally reduces cost.

The next bounded design work should focus on:

1. Making the grouping/entity decision visible: use operator contracts and
   existing ordinary profiles to prioritize relevant input columns and explain
   which keys survive grouping/joining. Consider a bounded upstream-summary
   path to an explicitly observed descendant. Do not expose unobserved tables
   wholesale or equate a mechanically valid join with a semantically valid key.
2. Simpler, less deeply nested function-slot payloads and a bounded repair
   policy after repeated argument errors. Error inspection is available, but
   this Luna trace did not use it; inspect/repair behavior needs evaluation.
3. A small preselected held-out task set and repeated matched trials, balancing
   cache warm-up/order. Separate delivery rate, decision changes, functional
   validity, cost and answer quality. No beach-specific statistic, gold rule,
   automatic answer correction or all-104-task campaign was added here.

These are proposals, not changes applied to the frozen scored attempts.

For each row the raw directory is
`system_scratch/DataflowSystem{Luna|Terra}PythonPilot{Arm}20260915Rep1/environment-easy-3/`.
Inspect `verdict.json`, `evaluation.json`, `stats.json`, `attempt.json`,
`pilot_metrics.json`, `react_steps.json`, `snapshots.json` and the saved workflow.
The committed report omits full transcripts, table samples and credentials;
`results.json` preserves compact timelines, actual-input witnesses, official
statistics and SHA-256 hashes of the principal raw artifacts.

To regenerate the two analytical JSON files, from the harness root:

```bash
.venv/bin/python judgment_runs/native_python_dataflow_20260915/analyze.py
```

This only reads existing attempts and data and writes the two report JSON
files. It never calls a model/scorer, changes a verdict, rewrites an attempt
or starts a service. Missing attempts remain pending rather than becoming
zero-cost failures/wins. Full runtime proof stays private; its hash-bound
sanitized summary is committed separately.

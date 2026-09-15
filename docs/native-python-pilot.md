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

# Python-native pilot: registered arms and measurement contract

Date: 2026-09-15. Status: **local setup, not a completed evaluation**.
No new task, real LLM, official score or cost/accuracy comparison is recorded
by this checkpoint. The parent Dataflow Agent plan is
`docs/native-python-dataflow-plan.md`, on `feat/native-python-dataflow`,
descended from batching/observe commit `41cd8ae97`.

This isolated harness starts at `260ceb82c1cd45e64e73d7b3f5e10405ebf7a225`.
Other workspaces and old pilot result namespaces are unchanged. The explicit
CU passthrough follows the same isolation pattern as the separate evidence
pilot's `4e8c35f` / `732a77a`, without merging its experimental evidence flag
or changing the old arms' defaults.

## Registered treatments

Each SUT name is `DataflowSystemLunaPythonPilot<Arm>20260915Rep1`.
`kb.py systems` lists all seven. Definitions are in
[`systems/native_python_system.py`](../systems/native_python_system.py).

| Arm | Service port | Catalog/tool mode | Collect profiles | dataLevel | flowLevel |
| --- | --- | --- | --- | --- | --- |
| V1Reference | 3013 | Historical V1, individual tools | Legacy absent | 1 | 1 |
| BatchParent | 3012 | Actual batching parent, batch | Legacy absent | 1 | 1 |
| V2Control | 3011 | V2, batch | false | 0 | 0 |
| CollectorOnly | 3011 | V2, batch | true | 0 | 0 |
| DataOnly | 3011 | V2, batch | true | 1 | 0 |
| FlowOnly | 3011 | V2, batch | true | 0 | 1 |
| Combined | 3011 | V2, batch | true | 1 | 1 |

V1's old L1/L1 anchor includes schema/source structure but no column statistics;
it is **not** the V2 evidence-on treatment. V2 keeps its mandatory typed
structural schema at L0/L0, and the two levels toggle selected evidence.
`columnStats=false` stays fixed: V2 uses its own bounded profile selector, not
the legacy column-statistics appendage. Collection is explicitly independent
of exposure. No new arm sends the prototype `nativeFlowEvidence` flag.

Shared settings: `gpt-5.6-luna`, native mode, DELTA, 25 steps, TSV,
2,000 result characters, 3,000 characters per cell, 240-second tool timeout,
10-minute execution timeout, `resultSelection=all`, `attemptReflection=true`,
no code-in-snapshot, no thought replay, no automatic/static compaction,
no session/versioned layout, no optional recall/resume/general-inspect tools.
The V2 catalog's required `inspectError` remains available independently of
the legacy general-inspect toggle. Provider parallel calls are enabled only
for V1Reference; every batch arm disables them. Runtime routing/driver must
still be recorded from the actual service.

Treatment-changing overrides fail instead of silently reusing a named SUT.
Only output directory, agent endpoint, and computing-unit ID may vary. Every
pilot instance requires `computing_unit_id=<positive int>` or
`NATIVE_PYTHON_PILOT_CUID`. This bypasses legacy first-unit discovery and
survives fresh workflow/agent setup. It does not create, claim ownership of,
or terminate that CU: orchestration must create its own registration and
clean up only its resources. Separate services share one compatible Texera
backend. These defaults do not start any services or restart shared workers.

## First-run protocol and gates

Use one attempt of **`environment-easy-3`** per arm, serially for this pilot.
The exact ID exists in `workload/environment.json`; its answer type is
`numeric_exact`. Keep the harness's existing oracle-file-subset policy
(`use_truth_subset=true`) fixed across arms. The two source files are the
2012 and 2013 water-body testing CSVs. Hash the actual resolved task/data files
and keep the gold answer out of model input. This is a plumbing/mechanism
pilot, not a success-rate estimate or a target for gold-specific tuning.

Before spending a model call:

1. Pass the matching-engine correctness matrix and the separate real-model
   smoke for dependent batching, explicit observation and error repair.
2. Resolve each **actual listening service process and its source/build**;
   freeze full source SHAs, effective settings, backend/worker identity and
   data hashes in a run manifest. A port-to-worktree hint, or a V1 rerun on
   new code, does not establish historical provenance. The parent is
   `41cd8ae97`; the precise original V1 revision still needs live verification.
3. Confirm isolated CUs/workflows, free result namespaces, required execution
   measurements and complete artifact capture. Do not use a watchdog rerun or
   overwrite to replace an unsuccessful first attempt. A missing/aborted
   response is not proof that the backend execution stopped.
4. Run through the existing KramaBench harness and official evaluator. For
   this numeric-exact task, select official `success`, with the frozen
   answer-type-aware threshold **>=0.9**. For later tasks use the metric mapped
   to their answer type; a missing expected metric is unscored, not the maximum
   of unrelated metrics. `kb.py compare`'s max-metric/strict-1.0 convention is
   different and must be labeled separately.

The guarded orchestration, official scoring/report and real-input audit are
**still pending**, so arm registration alone is not a launch-ready pilot.

## Trace and execution measurements

`kb.py react_metrics` now also counts native typed tools and batch inputs:
operator-attempt distribution, attempted per-ID edits, batch sizes (including
zero-operator pulls), requested/accepted observations, observation errors,
partial-batch outcomes, dependency blocks, rejection stages, error-inspection
calls, accepted/refused native deletions and accepted repairs after recorded
edit rejections. Missing/misaligned
receipts are unreported, not successful. Final native operator types come from
the saved DAG. The historical `final_ops_react` name is only an attempt-based
estimate; use `wf_ops` for actual final DAG size.

Accepted edits are **not executions**. These helpers intentionally do not infer
compile/runtime pass rates or runtime repairs from mutation receipts. Cached
observation can make no execution request, and independent sub-DAGs can cause
several requests for one batch.

The parent repo adds `benchmark/e2e/execution-recorder.ts`, a test-only loopback
proxy scoped to explicitly registered pilot CUs. All reference/candidate
services can route through it with `WORKFLOW_EXECUTION_SERVICE_ENDPOINT`;
unset `EXECUTION_ENDPOINT_TEMPLATE` to avoid bypassing it. The fixed backend
still authorizes forwarded Bearer tokens. The recorder changes no operator
arguments or response bodies and follows no redirects. It records one event
per initiated execution HTTP request, with workflow/CU/request/actual execution
identity, counts/selected target IDs, HTTP status, request duration and explicit
phase measurements. It retains no authorization header, source code, table
cells, console or raw error text. Target IDs are bounded with an omission flag.

The new engine `phaseMetrics` schema has `version:1`, `compilation`,
`compilationDurationMs`, and `runtimeStartAttempted`. Compilation measures only
the strict `WorkflowCompiler.compile` call, not cache preparation, runtime
setup or Python callback preflight. The latter remains a separate submission
gate. The start flag means the coordinator start RPC was attempted, not that
it succeeded. Missing/invalid phases and ambiguous infrastructure errors are
unknown, not passes or compiler failures. Report known passes/failures,
not-attempted and unknown counts **with denominators**; zero attempts has no
pass rate. Request-duration sums are not total wall time or collector CPU time.

The recorder currently exposes detached copies in memory. The pilot runner
must persist them as `execution_requests.jsonl`, bind them to the correct arm
and preserve partial records on failure. Local recorder tests use a controlled
HTTP backend and are not Texera worker or LLM runs.

## Required pilot artifacts and cost report

Keep `ground_truth.json`, `answer.json`, `evaluation.json`, `stats.json`,
`config.json`, `react_steps.json` with actual `inputMessages`, `workflow.json`,
execution-request records, and version-bound profile/collector metadata under
`system_scratch/<SUT>/<task>/`. A dated sanitized report belongs under
`judgment_runs/native_python_dataflow_<date>/`, naming root and harness SHAs.

Reuse `load_cost_stats`, `step_token_rows`, `react_metrics` and official
scoring helpers. Primary cost is `stats.json.cost_usd` with cache discounts,
not total tokens. Freeze and publish the price schedule/source used; this is
a token-based estimate, not a provider invoice. Missing pricing/usage or an
unexplained zero with nonzero usage must be flagged, not counted as a free win.
The existing generic harness's price-error fallback and port-based provenance
hints require these explicit validation gates before this pilot is released.

Report input/cached/output/reasoning tokens, steps, batch/observe/inspection
counts, churn, partial failures and repairs, compile/runtime request outcomes,
collection overhead and elapsed time separately. De-duplicate collector work
by producing result identity, not snapshot occurrence. Audit when each fact
was actually present in model input before attributing a better decision to
it. No savings/accuracy claim is justified by the local setup tests.

## Local verification

`python -m unittest discover -s tests -q`: 19 pass. This includes eight new
V2/client/arm tests and seven new native-trace tests, plus the four unchanged
legacy pilot tests. `kb.py systems` resolves all seven new SUT classes.
No network/service setup or model call is part of these harness tests. New
client/system options are keyword-only, preserving legacy positional arguments.

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
10-minute execution timeout, a fixed 1,800-second client turn budget,
`resultSelection=all`, `attemptReflection=true`,
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

The one-attempt path, official numeric scoring, persistent recorder, scoped
phase reporting, source-pinned agent launcher, allocation journal, verified
owned-resource cleanup and collector aggregation are implemented. The owned
single-arm runner now composes them with frozen inputs and post-drain metric
finalization; its orchestration is locally tested. The production backend/
worker and prior-live-gate qualification callback, matching worker/model runs,
official pilot and final comparison report are **still pending**. A real
setup-only lifecycle smoke passed earlier; this is not a launch-ready pilot.

## Isolated agent launch and shutdown

`utils.pilot_launch.launch_agent_service` runs only
`bin/local-dev.sh up agent-service --skip-build`, with an explicit non-default
port and an exclusive private `TEXERA_LOCAL_DEV_DIR`. It never adopts an
occupied port, installs packages, redirects an old deployment state directory,
or starts/restarts shared JVM services. A full expected root SHA is required.
The private `launch.json` binds clean runtime/launcher tree hashes before and
after launch, boot identity, a newly started listener PID/start time, its actual
entry point, health and the execution-recorder route. Process environments are
compared locally for those routing fields, not copied into artifacts.

`verify_service_launch` rechecks the live process, source, route and health.
`stop_agent_service` additionally requires an empty agent list and the original
launcher state directory before running only `down agent-service`. It verifies
the owned process exited and the port became free, and writes a separate
private stop record. It does not infer backend termination or delete workflows
and computing units. Ambiguous/failed launches are not automatic stop targets.

The CLI is `python -m utils.pilot_launch launch|verify|stop --help`.
Keep the calling orchestration process alive through the run: a successful
launch record is a point-in-time check, not a guarantee that an external tool
runtime will preserve a background child after the command ends. The local
smoke found exactly that boundary: a first one-shot launch disappeared before
the next command and verification refused it. A fresh contained
launch/verify/empty-stop invocation passed on port 3011 without any agent,
workflow, CU, model call or backend execution. The recorder forwarded zero
requests. No other service was stopped.

This proof covers tracked runtime source and launcher files, not immutable
installed dependencies: lock files are hashed, while the resolved shared
`node_modules` directory is recorded but its contents are not fingerprinted.
It also does not qualify the shared backend/worker, task/data, price schedule
or real-model behavior. Those checks still belong to the outer guard.

## Single-attempt execution and failure artifacts

`native_python_pilot.run_pilot_task` uses the existing KramaBench `Executor`
and `Evaluator`, filtering exactly `environment-easy-3` and keeping the oracle
file policy. It does not use caches, watchdog reruns or recovery rounds. New
arms use `NativePilotSystem`; ordinary `kb.py tasks` calls without the required
guard fail before setup or a model dispatch. The guard runs before setup,
immediately before dispatch and after the attempt. The production guard is
not supplied yet; tests use controlled guards and a mocked agent, not real
provenance or Texera qualification. Every guard return must be a dictionary
with `qualified` exactly `True`. `False`, a missing field, a truthy string or
integer, and a plain boolean cannot admit work. Postflight rejection preserves
the raw answer but makes the attempt ineligible for treatment comparison.

Dataset preparation no longer allocates a throwaway agent for these arms.
Each task reserves a fresh directory exclusively, including unsuccessful or
empty first attempts. `empty_turn_retries=0` overrides the legacy retry
environment, and the fixed 1,800-second turn budget is recorded. There is no
fresh-setup/reset fallback. Existing non-pilot arms retain their setup/retry
policy.

The client supports an event callback. Accepted step/state/complete/error
events are flushed and fsynced into private `events.jsonl` immediately; init
payloads and raw delegates are omitted. Each finalization attempts bounded
REST reads of the full trace, effective agent settings/identity, workflow and
every step snapshot. Snapshot identities are checked. Malformed or unavailable
REST traces fall back to the received step journal and remain explicitly
partial. `attempt.json` records dispatch/completion/qualification separately,
resource IDs, capture status and sanitized exception types. The full raw
trace and snapshots are private run artifacts, not sanitized publications.

Additional artifacts are `capture.json`, `snapshots.json`, canonical
`response.json`, `verdict.json`, and helper-derived `pilot_metrics.json`.
The last file contains trace/token accounting and optional scoped execution
and producing-result collector measurements (with the limits described below).
Failed setup/preflight is unscored because no model
attempt was made; dispatched unsuccessful attempts remain in the record.
Official raw scores and eligibility for the treatment comparison are separate.
For this numeric task the evaluator no longer constructs an unused pipeline
LLM judge; the official `success` metric and >=0.9 threshold are unchanged.

A model's final answer that was actually received is retained even if the
completion frame or input-message capture is missing. That does not establish
complete usage, a complete input trace, backend termination or treatment
qualification. Synthetic error/stop steps are not parsed as answers.

The pilot's destructor/`cleanup()` intentionally do not delete resources.
The pending outer runner must verify terminal backend state and clean up only
its own agent/workflow/CU, recording unresolved IDs. Aborting a socket is not
termination evidence. A crash during setup still needs per-allocation journaling
in the outer runner; the current finalization journal covers exceptions and
interrupts it can catch, not SIGKILL or power loss before an ID is persisted.

## Accounting and provenance corrections

- The server re-broadcasts its last ReAct step when marking `isEnd=true`.
  The Python WebSocket client formerly added that step's usage twice. It now
  replaces usage by step ID, and the pilot separately deduplicates the saved
  trace. Historical artifacts are **not rewritten**; comparisons with old
  recorded costs require auditing this accounting difference.
- `price_schedule()` captures the existing harness rates and their source
  before dispatch. Pricing that attempt uses the frozen schedule, including
  cache reads, without charging reasoning again. These are token estimates,
  not verified current public prices or provider invoices. Cross-arm schedule
  equality must still be enforced by the production manifest guard.
- Complete `cost_usd` requires complete valid per-step input/output/cache
  usage and a completed turn. Partial known usage is retained with
  `observed_cost_usd`, while whole-attempt cost is null. Missing cache usage,
  missing pricing, impossible counts and unexplained zero charges do not
  become a free result. Raw per-step usage remains in the trace.
- `kb.py load_cost_stats` preserves unknown/partial costs as null. Cost,
  compare, Venn and case-metric commands exclude them from cheaper-run
  decisions and identify their known-cost coverage. Accuracy overlap is not
  discarded simply because one cost is unknown. The CLI's existing strict
  score convention is not changed to the pilot's >=0.9 convention.
- Generic service provenance now resolves the actual local listening PID,
  its entry point, on-disk Git revision, source dirtiness and process start
  identity. Unresolved cases stay unknown. This is **not proof of loaded
  source**; it explicitly returns `loaded_revision_verified=false` until the
  outer runner verifies a source-pinned launch and pre/post stability.

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

The parent's `execution-recorder-run.ts` persists an exclusive private
`execution_requests.jsonl`: a startup header, a fsynced start **before** each
forward, a finish afterward, and a footer only after draining. Its read-only
`/__recorder/status` identifies the recorder and outstanding requests. A failed
start write prevents forwarding; a failed finish write preserves the actual
backend response but prevents further admissions. SIGINT/SIGTERM drain; a
SIGKILL test preserves an unfinished start. Neither a missing response nor a
closed recorder proves backend termination.

`utils.execution_journal.execution_report` requires the exact workflow, CU and
recorder instance IDs. It deduplicates identical events, separates other
workflows/CUs, and flags conflicting identities, malformed/truncated data,
orphan finishes and inconsistent footers. Missing evidence is distinct from a
valid closed zero-request journal. It validates measured phases instead of
trusting saved outcome labels or heuristic engine error text. Each phase has
passed/failed/not-attempted/unknown counts; `pass_rate` uses
`passed / (passed + failed)`, with `known_attempted` and `known_coverage`
alongside it. No attempts means a null rate, not 100%.

`run_pilot_task(..., execution_journal_path=..., recorder_id=...)` attaches this
report to `pilot_metrics.json`, scoped to the actual attempt resource IDs.
Both optional arguments must be provided together. Without them, measurement
status is missing and rates are null. A production guard must enforce the
required recorder; optional reporting arguments are not an admission bypass.
Capture after an attempt can see an open journal: the outer runner must retain
the complete journal. `finalize_pilot_measurements(system)` refreshes only
`pilot_metrics.json` after drain, using its original journal path and recorder
ID. Missing bindings, open or invalid/foreign journals are refused without
replacing the initial report. No model, executor, evaluator, guard or cleanup
is called by this refresh; answer, score, trace and token-cost artifacts remain
unchanged. The refreshed stage is `after_recorder_drain`, not proof of complete
phase coverage or backend termination. Repeating the refresh is idempotent.
Request
duration and compiler-duration sums are observed measurements, not wall time
or collector cost. Local HTTP fixtures are not Texera worker or LLM runs.

## Frozen inputs and owned single-arm orchestration

`utils.pilot_inputs.freeze_pilot_inputs` binds the actual official workload,
exact in-memory task, the two ordered prompt file paths, matching harness/
worker file bytes, prompt hash and the existing frozen harness price schedule.
`verify_pilot_inputs` repeats the checks and rejects changes. File reads detect
path/stat drift while hashing. Manifests contain hashes, paths, byte counts and
pricing, not raw rows, prompt text or gold answers. A passing input check sets
`runtime_qualified: false`: the caller must prove that the supplied execution
directory is the actual worker working directory.

Only the pilot sorts expanded file paths before building the prompt. Legacy
file expansion remains unchanged. `prepare_pilot_task` is shared by the
official attempt and read-only input preparation, so a `-tiny` workload cannot
silently replace the exact official task in one path but not the other.

`utils.pilot_session.run_owned_pilot` takes one registered arm, separately
pinned agent and recorder worktrees/SHAs, workload/dataset/execution paths,
fresh private session output, optional task output (default
`system_scratch/<SUT>`), and a **mandatory** `qualification` callback. Its
order is:

1. Check clean agent/recorder/harness source, frozen arm port, free first-attempt
   namespace and frozen local inputs; require runtime/prior-live-gate approval
   **before** authentication or resource allocation.
2. Journal a fresh owned LOCAL CU; start a candidate recorder for that CU;
   verify the exact child, source hashes, route, persistence and idle status;
   launch only the pinned agent through its supported isolated launcher.
3. Run the existing `run_pilot_task` once. Each guard independently rechecks
   input bytes/prompt/pricing, harness and recorder source, the launched agent
   and recorder health before invoking the mandatory runtime qualification.
4. Drain that recorder, refresh derived measurements, verify owned-resource
   capture/ownership/terminal executions before cleanup, then stop only the
   still-matching empty agent service. A wrong or ineligible answer is not
   labeled a successful session. Exceptions preserve private IDs/artifacts;
   the only unconditional shutdown action is draining the owned recorder.

Historical agent worktrees use the same separately pinned **candidate recorder**
as V2; they do not need to contain the new recorder files. The callback receives
`(context, system, stage, info)`, with system/info absent at `before_resources`.
It must supply actual matching-engine/worker, reference-source, effective
settings, ownership and prior correctness/model evidence. No implementation of
that production callback or permissive CLI is included here. A fixture returning
`{"qualified": True}` remains a mock, not runtime qualification. Keep the outer
Python caller alive throughout the session and start it freshly from the pinned
harness. Tracked source and interpreter identity are recorded; installed Python
or Bun dependency contents are not independently fingerprinted by this helper.

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
Generic historical price-error fallbacks are not sufficient evidence. The
pilot's explicit unknown-cost handling and listener-source resolution still
require the production manifest/launch validation gates before release.

Report input/cached/output/reasoning tokens, steps, batch/observe/inspection
counts, churn, partial failures and repairs, compile/runtime request outcomes,
collection overhead and elapsed time separately. De-duplicate collector work
by producing result identity, not snapshot occurrence. Audit when each fact
was actually present in model input before attributing a better decision to
it. No savings/accuracy claim is justified by the local setup tests.

## Allocation receipts and verified cleanup

Pilot setup now reserves an exclusive `resource_allocations.jsonl` inside the
attempt directory. It fsyncs an intent before each workflow/agent creation and
the returned ID before the next allocation. The separately created local CU
has its own `cu_allocations.jsonl`; a supplied CU reference is not ownership.
Names carry a fresh journal nonce. Neither names nor receipts alone authorize
deletion. Journals are mode 0600, exclude tokens/delegates, and reject retries,
changed identities and malformed/truncated records. A crash between server
creation and receipt remains an unresolved intent, never an automatic retry.

`DataflowAgent.setup(on_resource=...)` is opt-in, synchronous, and uses bounded
no-redirect requests with explicit CU and resource names. The pilot always uses
it; legacy clients keep their existing transport/discovery defaults. Setup now
fails before API calls if its attempt bundle is missing. Model turns remain
single-attempt, and the pilot destructor still performs no cleanup.

`utils.pilot_cleanup.cleanup_owned_resources` requires complete artifacts,
matching allocation journals, the same source-pinned idle agent/delegate,
current resource ownership and names, the exact local CU/backend, a closed
scoped recorder, and terminal database executions matching every request.
Unknown outcomes or another workflow's recorder traffic block deletion. The
recorder footer must retain the startup route, CU whitelist and request limit.
It records intent before each mutation, deletes the agent, deletes only the
owned workflow, terminates only the owned **local CU registration**, and
verifies each absence before proceeding. A malformed list is not absence.
Local CU termination does not stop the shared JVM. No broad service stop,
workflow delete or task rerun is a cleanup fallback.

Failures preserve the unresolved IDs and partial cleanup report. This is one
verified cleanup attempt, **not automatic recovery/resumption** after a
partially completed delete. Investigate uncertain allocations/executions or
partial cleanup before proceeding; do not blindly invoke it again. The outer
production guard and runner integration remain required.

### Live setup-only smoke

From the harness, use a fresh private directory outside `system_scratch`:

```bash
.venv/bin/python -u -m utils.pilot_resource_smoke \
  --worktree /absolute/path/to/native-python-dataflow \
  --sha <full-candidate-root-SHA> --port 3011 \
  --output /private/fresh/setup-smoke
```

This uses the client's local Texera/CU-manager endpoints and development login
defaults. It creates a new local registration for the existing backend,
starts a private loopback recorder, launches only the candidate agent through
`bin/local-dev.sh`, exercises actual pilot setup/capture, then drains, verifies
owned cleanup, and stops the empty private service. The orchestration process
stays alive throughout; do not assume detached children outlive a tool runner.
No `serve_query`, `/message`, workflow execution or official task is invoked.

Live check on 2026-09-15 passed at root `762e80e96`: private service PID
2722192 on 3011, workflow 4156, CU 73, agent `agent-bqtik7-1`. All three owned
resources were cleaned and service termination verified. Fresh independent
workflow/CU list reads confirm absence. The shared master PID 2114172 and its
start time/cwd remained unchanged. Actual captured trace/workflow were empty;
the recorder closed with zero requests, zero unfinished requests and no issues.
Compiler/runtime pass rates are **null**, not 100%. Metadata and empty workflow
capture are retained in `/tmp/native-python-c7-resources.ZQNqmc/run`; this text
is the durable sanitized record. The final cross-workflow refusal was then
added as a failing-first regression; that case was mocked, not exercised live.

This is resource-lifecycle qualification only: no Python worker, operator,
real model trace, accuracy or token-cost result is implied.

## Collector measurement semantics

`utils.collector_measurements.collector_report` now replaces the placeholder
in `pilot_metrics.json`. It reads decoded, version-bound snapshot profiles,
matches the producing execution to the workflow/CU-scoped execution journal,
and de-duplicates `(workflow, operator, output port, producing execution)`.
Retrieving the same cached result again adds no collector work; actual reruns
remain separate. Changed bindings/measurements or reused worker producer IDs
are conflicts, not opportunities to sum or select a maximum.

Reported `observed_collection_ms` sums worker `ProfileAccumulator.add()`
timers. It excludes profile snapshot/encoding, transport and agent rendering;
it is neither total wall time nor CPU time. Cells processed and worker-profile
payload bytes are also observed component totals, not total network traffic.
Partial, unavailable, unbound and unattributed productions remain explicit.
Even complete snapshot capture can miss failed/unreturned materializations,
so `whole_attempt_overhead_complete` remains false. Collection-off/no profiles
is labeled `disabled_by_request`; effective settings still need live admission
verification. No sample rows, column values or custom statistics are copied
into the cost report.

## Real-model smoke protocol (prepared; not yet run live)

`utils.native_model_smoke.run_model_smoke` accepts an already-owned fresh
`NativePilotSystem`-compatible session, an exclusive private output directory,
`context_mode` (`delta` or `latest`), and a **mandatory** qualification callback:

```python
report = run_model_smoke(
    owned_system,
    output_directory="/private/fresh/model-smoke-delta",
    context_mode="delta",
    guard=qualified_live_session,
)
```

The callback receives `(system, stage, actual_agent_info)`. It must verify the
live source/service, matching engine/worker and owned resource identities and
return `{"qualified": True}` only after those checks. The production callback
and outer-runner wiring are **not supplied by this fixture**; a stub returning
true is not qualification. This interface is intentionally not an unguarded
CLI. The caller also owns journaled allocation, final recorder drain and
verified cleanup; this helper performs none of them.

Run independently on fresh DELTA and LATEST sessions with `gpt-5.6-luna` and
`vercel-tool-use`. Required effective settings: native V2/batch, no parallel
tool calls, `messageLayout: native`, profile collection on, data/flow levels
1/1, column stats off, result/cell limits 2000/3000, `maxSteps: 25`,
`maxResultRows: 0`, code snapshot/thought replay off and context window 0.
The helper checks those settings before and after each turn, uses a client
ceiling of 1800 seconds and disables empty-turn retries.

Four directive-driven turns test:

1. One reverse-ordered dependent UDF → Filter → Project batch; observe only
   its answer, verify executed rows and model JSON, and forbid unobserved source
   canary rows in actual input. The small input is generated by the provided
   Python UDF; no data lake or benchmark task is used.
2. `operators: []` with explicit source observation; unchanged producing
   identity/rows and newly visible canary values, absent before the pull. The
   execution journal still provides the independent test for whether any
   backend request occurred.
3. Deliberate callback failure with `observe: []`, followed by model-called
   `inspectError`. Check the actual execution/source-line diagnostic, cleared
   stale downstream rows, error visibility before inspection and inspection
   text in a later model input. The model tool returns text directly, unlike
   the REST inspection endpoint's object response.
4. Repair an upstream predicate while adding a downstream consumer of the
   existing DAG in the same reverse-ordered batch. Check changed downstream
   result identity, cleared error, actual rows and model JSON.

Visibility proof joins the tool event's real snapshot observation to a **later
recorded `inputMessages`** entry on the same turn. System/assistant echoes and
an earlier input or final snapshot alone are insufficient. Raw service traces,
snapshots, sanitized info, prompt, dispatch/transport state and audit witnesses
are retained in each turn directory. Failure stops the fixture without a
model retry, resource deletion, shared restart or backend cancellation. These
strict directive tests are functional qualification, not representative cost
measurements or proof that stats improve reasoning.

## Local verification

`.venv/bin/python -m unittest discover -s tests -q`: **160 pass**, including the original
19 client/arm/native-trace and legacy pilot tests. New regressions cover actual
listener resolution with controlled process fixtures, single-attempt policy,
final-step rebroadcast accounting, durable partial capture, unknown/frozen
cost, setup/dispatch/postflight failure, malformed trace/snapshot responses,
answer-versus-transport status, and the real Executor/Evaluator path with a
mocked agent. The earlier 21 tests cover source-pinned launch/empty shutdown,
durable-journal denominators, missing/conflicting evidence and the measured
report through the actual Executor/Evaluator path with a mocked agent.
The additional 39 tests cover collector accounting and report integration,
journaled setup, scoped cleanup and setup-only orchestration. Regression tests
were run failing before fixes for mismatched recorder scope, malformed absence
responses and cross-workflow cleanup. The newest 15 tests cover all four
model-smoke audits, input-timing/echo/leak/stale-result failures, full mocked
orchestration, qualification/model/mode rejection and interrupted-turn capture
without retry/cancellation. The latest 28 tests cover frozen inputs (10), strict
guard-return rejection (3), post-drain refresh without rerun/rescore (3), and
owned session ordering/failure/source drift plus live-recorder admission with
controlled fixtures (12). The new guard regressions failed before the fix.
They make **no model calls**. `kb.py systems`
resolves all seven new SUT classes.

These tests make no Texera service setup or model calls. They are not live
worker E2E tests or real model traces. New client/system options remain
keyword-only, preserving legacy positional arguments. New/scoped Python files
pass Ruff formatting and F-rule checks; no dependency was installed into the
shared benchmark venv.

Additional development checks (not benchmark results): the parent reports
1,120 Bun tests and typecheck passing; its 15 recorder tests include a real
child-process interruption. A three-response loopback HTTP fixture wrote a
real Bun journal which this Python reader consumed: two compiler passes/one
failure; one runtime pass/one failure/one not attempted; 9 ms of synthetic
compiler measurements. The separate real agent-service lifecycle smoke is
described above. No Texera worker or provider was called in these checks.

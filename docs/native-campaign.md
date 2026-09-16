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

# Native full campaigns

## ObserveOnly replacement, 2026-09-16 Rep1

The 2026-09-15 campaign is superseded. Preserve its classes and artifacts for
historical inspection, but do not resume its drivers, administrative amendment
scripts, or configurations. Start all 104 tasks anew in each of these eight
namespaces:
`DataflowSystem{Luna|Terra}NativeCampaignObserveOnly{BatchParent|DataOnly|FlowOnly|Combined}20260916Rep1`.

The replacement changes no frozen model, reasoning effort, evidence level,
token budget, retry policy, pricing rate, or scoring rule. Both services must
instead run separately source-pinned revisions with `inspectResult` removed.
The V1 BatchParent remains based on the original no-V2-evidence implementation,
but its manifest pins the inspector-removal commit rather than the historical
`41cd8ae` revision. Old classes still require the historical revision.

Create a fresh manifest with `campaign: "NativeCampaignObserveOnly20260916Rep1"`,
the integrated harness SHA, and the newly qualified service launch records and
SHAs. Requalify engine, gateway and input bindings; do not reuse an old manifest
by editing its campaign label. The schema and endpoint/CU environment overrides
below are otherwise unchanged. Set the explicit endpoint variables when using
fresh ports, and preserve the old services' pending resource journals.

Before every model dispatch and after capture, admission reads the existing
`GET /api/agents/:id/system-info`. It requires an enabled `dataflow` tool and a
valid prompt/tool registry, and rejects `inspectResult` anywhere in the prompt,
tool names, descriptions or schemas, including disabled registry entries.
`config.json` retains the exact public prompt and registry descriptors,
enabled/all tool names, and their canonical JSON SHA-256 under
`admission.tool_surface` and `postflight.tool_surface`. This is the effective
registry plus public descriptors/prompt, not a model-wire parameter-schema
capture: the existing API may report `{}` for native-tool `inputSchema`.
Registry membership, descriptions and prompt still provide the absence check.
A missing endpoint or invalid surface blocks model dispatch. The deprecated
`enableInspectTool` setting may be absent or false;
true is rejected and is not a substitute for removing the actual tool.

The admission guard is read-only and fails closed for each affected attempt.
It has no global stop-file latch. A supervisor detecting qualification/capture
failure can stop later native admissions only within its polling interval;
preserve any already-running attempts. Captured terminal `agent_error` or
`interrupted` outcomes with partial costs remain ordinary failed tasks eligible
for the common recovery policy, not automatic reasons for a global halt.

The canonical domain workload loader (not the shadowing `*-tiny.json` loader)
is shared by all campaign classes. All 104 questions, oracle-file lists and
format hints remain unchanged. New attempt metadata and per-attempt pricing
identify the new campaign; its recovery hook rejects archived/current metadata
from the superseded campaign before moving any artifacts.

Use the native command with a new name and a fresh output namespace:

```bash
campaign_sut=DataflowSystemLunaNativeCampaignObserveOnlyCombined20260916Rep1
NATIVE_CAMPAIGN_ROUND=first python kb.py run --sut "$campaign_sut" \
    --parallel --isolate --watchdog-min 35
```

The recovery, capture, cost and cleanup rules below still apply. Run both
explicit recovery rounds with the same strict-primary-metric threshold for
every new arm; do not select old-campaign attempts as either first attempts or
recoveries.

## Historical campaign, 2026-09-15 Rep1

The eight classes use the existing `kb.py run`, `tasks`, `rerun-failed`, and
official evaluator. Each class is named
`DataflowSystem{Luna|Terra}NativeCampaign{Arm}20260915Rep1`.

| Arm | Service | Catalog | Collection | Data / flow |
| --- | --- | --- | --- | --- |
| BatchParent | 3012, original `41cd8ae97e30edfec90c847886ed4068dd0fa9b0` | V1 | absent | legacy 1 / 1 |
| DataOnly | 3011, frozen candidate | V2 | on | 1 / 0 |
| FlowOnly | 3011, frozen candidate | V2 | on | 0 / 1 |
| Combined | 3011, frozen candidate | V2 | on | 1 / 1 |

The model IDs are exactly `gpt-5.6-luna` and `gpt-5.6-terra`, both through
the verified gateway at medium reasoning. Settings match the corresponding
pilot: DELTA, batch tools, no parallel tool calls, 25 steps, 2,000 result
characters, no code snapshots, no legacy column statistics or thought replay.
The original V1 data/flow levels do not expose V2 evidence. Pilot namespaces
and behavior are unchanged.

## Manifest and operational settings

The supervisor qualifies the existing engine, launches the two services with
`launch_agent_service`, and freezes a manifest before running any tasks.
Services may point directly at `http://127.0.0.1:8085`; the launch record's
historical `recorder_url` field then contains that backend URL. This campaign
does **not** claim recorder-based backend request/compile/runtime metrics.
It still captures model input messages, tool traces, all available snapshots,
profiles, final workflows, raw events, and token usage.

```python
from utils.native_campaign import file_binding
from utils.pilot_qualification import gateway_route

manifest = {
    "version": 1,
    "campaign": "NativeCampaign20260915Rep1",
    "harness_sha": "<full committed harness SHA>",
    "computing_unit_id": 123,
    "services": {
        "BatchParent": {
            "endpoint": "http://127.0.0.1:3012",
            "launch_record": "/absolute/path/batch/launch.json",
            "git_sha": "41cd8ae97e30edfec90c847886ed4068dd0fa9b0",
        },
        "V2": {
            "endpoint": "http://127.0.0.1:3011",
            "launch_record": "/absolute/path/v2/launch.json",
            "git_sha": "<full committed candidate SHA>",
        },
    },
    "gateway": {model: gateway_route(model)
                for model in ("gpt-5.6-luna", "gpt-5.6-terra")},
    "runtime": {
        "endpoint": "http://127.0.0.1:8085",
        "pid": 12345,
        "start_ticks": 123456,
        "boot_id": "<from qualified engine runtime evidence>",
    },
    "bindings": [file_binding(path) for path in frozen_paths],
    "task_inputs": {},
}
```

`frozen_paths` includes all six workload JSONs, all six format-hint JSONs,
resolved task input files at both harness and worker locations, launch records,
and readiness/runtime proof files. `file_binding()` records the actual hash
and filesystem identity. Admission rechecks every identity and hashes the
current task's files, including wildcard matches; harness and worker bytes
must agree. Missing inputs fail admission before model dispatch and produce
an empty answer with failure artifacts, retaining the full 104-task denominator.

Optional `task_inputs` maps a task ID to its frozen list of relative prompt
paths. Use it only for documented input-name corrections shared by every arm.
It changes neither task JSON nor ground truth. Bind the corrected files and
record the correction evidence in the manifest. Unresolved references must
remain unresolved rather than being replaced with inferred datasets.

The guard checks manifest stability, harness HEAD and source cleanliness,
loaded service revisions, gateway model/reasoning/container identity, the
existing master's PID/start/boot identity, CU routing in the agent, and
requested versus effective settings before dispatch. Source/input changes
require a newly qualified manifest; treatment changes require new SUT names.

```bash
export NATIVE_CAMPAIGN_MANIFEST=/absolute/path/campaign-manifest.json
export NATIVE_CAMPAIGN_CUID=123
export NATIVE_CAMPAIGN_BATCH_ENDPOINT=http://127.0.0.1:3012
export NATIVE_CAMPAIGN_V2_ENDPOINT=http://127.0.0.1:3011
export KB_MAX_PARALLEL=2
```

The explicit positive CU ID is mandatory: there is no first-CU discovery.
The endpoint variables default to `http://localhost:3012` / `:3011` and must
match the manifest exactly. Cached scoring allocates no resources.

## Execution and recovery

Run from the committed harness root with its existing Python environment:

```bash
campaign_sut=DataflowSystemLunaNativeCampaignCombined20260915Rep1
NATIVE_CAMPAIGN_ROUND=first python kb.py run --sut "$campaign_sut" \
    --parallel --isolate --watchdog-min 35

NATIVE_CAMPAIGN_ROUND=recovery1 python kb.py rerun-failed --sut "$campaign_sut" \
    --include-missing --parallel --isolate --watchdog-min 35
NATIVE_CAMPAIGN_ROUND=recovery2 python kb.py rerun-failed --sut "$campaign_sut" \
    --include-missing --parallel --isolate --watchdog-min 35
```

The recovery commands select score-zero/unscored and missing tasks. If the
protocol instead selects every score below 1, add `--all-failed` consistently
for both rounds and all eight classes. `kb.py tasks --ids ...` accepts the same
round environment. Each task sends exactly one model turn with
`empty_turn_retries=0` and the pilot's 1,800-second turn deadline. The longer
watchdog leaves time to persist artifacts after that deadline.

Reusing a completed round is rejected. A later explicit round atomically
moves the entire previous task directory into
`system_scratch/<SUT>/_attempts/<task>/<old-round>/` before reserving the new
attempt. It preserves scorer output, raw events, partial captures, and resource
journals. First-pass scores remain in the archived `evaluation.json`; use
official latest measures for the recovered result and state the pass threshold.

`stats.json` retains current-attempt accounting plus separate `first_pass` and
`all_attempts` summaries. Unknown or interrupted whole-attempt cost stays null;
observed partial usage/cost remains available and is included in the all-attempt
lower bound. Never report only the latest attempt as total campaign cost.

Frozen rates per million tokens are Luna `$0.20 / $0.02 / $1.20` and Terra
`$2.00 / $0.20 / $12.00` for input/cached-input/output. Reasoning is included
in output. A request above 272,000 input tokens is flagged in
`long_context_step_ids`: published long-context uplift is outside this base
schedule, so `cost_usd` and `observed_cost_usd` become null and the base-rate
estimate is retained separately. Reprice those requests before cost comparison.

## Task resource cleanup

After complete capture and a completed turn, the campaign checks journal-owned
agent/workflow identities, an AVAILABLE agent, the local CU route, and terminal
execution history. It deletes only that agent and workflow, verifies their
absence, and retains the shared CU. `cleanup.json` and `attempt.json` record
the result. Interrupted, incomplete, nonterminal, or uncertain cases retain
their IDs for the supervisor's cleanup pass. No destructor cancels executions.
Review pending cleanup before launching recovery rounds.

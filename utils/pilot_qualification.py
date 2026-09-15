# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""Read-only production admission for the owned pilot and model smoke.

No pass-flag override: recheck the actual engine, gateway route, worker reports,
effective settings, default Python environment and journal-owned resources.
Benchmark admission additionally re-audits both real-model smoke traces.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

from dataflow_agent import TEXERA_API_ENDPOINT, TEXERA_COMPUTING_UNIT_ENDPOINT, login
from utils.execution_journal import execution_report
from utils.native_model_smoke import audit_model_smoke_turn
from utils.pilot_cleanup import PilotResourceAPI, _owned_entry
from utils.pilot_inputs import file_identity
from utils.pilot_runtime import verify_engine_runtime
from utils.resource_journal import read_resource_journal


def _require(value, code):
    if not value:
        raise ValueError(code)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_effective_settings(requested, actual):
    _require(isinstance(actual, dict), "effective_settings_missing")
    optional_absent = {
        "enableRecallTool",
        "enableResumeTool",
        "enableAnswerGrounding",
        "sessionTurns",
        "versionedMode",
    }
    for key, value in requested.items():
        if key in optional_absent and value is False and key not in actual:
            continue
        _require(type(actual.get(key)) is type(value) and actual[key] == value, "effective_setting_mismatch_" + key)
    for key in (
        "columnStats",
        "dataHints",
        "valueFormat",
        "enableCodeInSnapshot",
        "thoughtReplay",
        "enableInspectTool",
    ):
        if requested.get(key) is not True:
            _require(actual.get(key, False) is False, "unexpected_evidence_" + key)


def check_default_python(workflow):
    _require(isinstance(workflow, dict) and isinstance(workflow.get("operators"), list), "workflow_unavailable")
    for operator in workflow["operators"]:
        properties = operator.get("operatorProperties", {})
        _require(
            properties.get("defaultEnv", True) is True and not properties.get("envName"),
            "named_python_environment_not_qualified",
        )


def check_worker_proof(proof, binding, source):
    counts = {"python": 2, "core": 2, "evidence": 8, "profile": 4, "shapes": 2, "diagnostics": 2}
    _require(proof.get("status") == "passed" and proof.get("source") == source, "worker_gates_not_passed")
    reports = proof.get("reports", [])
    _require(
        len(reports) == len(counts) and {row.get("suite") for row in reports} == set(counts), "worker_gates_incomplete"
    )
    for report in reports:
        name = report["suite"]
        service = report.get("service_before", {})
        rows = (report.get("summary") or {}).get("checks" if name == "python" else "reports", [])
        _require(
            report.get("command") == ["bun", "run", f"benchmark/e2e/native-{name}-run.ts"]
            and type(report.get("exit_code")) is int
            and report["exit_code"] == 0
            and report.get("engine_before") == binding
            and report.get("engine_after") == binding
            and service == report.get("service_after")
            and service.get("git_sha") == source
            and service.get("loaded_revision_verified") is True
            and len(rows) == counts[name]
            and {row.get("contextMode") for row in rows} == {"delta", "latest"},
            "worker_gate_evidence_mismatch",
        )


def gateway_route(model):
    """Read live LiteLLM routing; credentials stay inside its existing container."""
    _require(model in {"gpt-5.6-luna", "gpt-5.6-terra"}, "unregistered_model")
    script = """import json,os,urllib.request
h={'Authorization':'Bearer '+os.environ.get('LITELLM_MASTER_KEY','')}
r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:4000/model/info',headers=h),timeout=10)
data=json.load(r)
print(json.dumps([{'name':m.get('model_name'),'model':m.get('litellm_params',{}).get('model'),
'reasoning_effort':m.get('litellm_params',{}).get('reasoning_effort')}
for m in data.get('data',[]) if m.get('model_name') in ['gpt-5.6-luna','gpt-5.6-terra']]))
"""
    result = subprocess.run(
        ["docker", "exec", "litellm", "python", "-c", script], capture_output=True, text=True, timeout=20
    )
    _require(result.returncode == 0, "gateway_route_unavailable")
    matches = [row for row in json.loads(result.stdout) if row.get("name") == model]
    _require(
        len(matches) == 1 and matches[0] == {"name": model, "model": "openai/" + model, "reasoning_effort": "medium"},
        "gateway_model_or_reasoning_mismatch",
    )
    # A process restart or container replacement cannot reuse the admission.
    identity = subprocess.check_output(
        ["docker", "inspect", "litellm", "--format", "{{.Id}} {{.State.StartedAt}}"], text=True, timeout=5
    ).strip()
    return {**matches[0], "container": identity}


class LivePilotQualification:
    def __init__(self, *, build_manifest, worker_proof, candidate_sha, model_proofs=()):
        self.build = json.loads(Path(build_manifest).read_text())
        self.binding = verify_engine_runtime(self.build, "http://127.0.0.1:8085")
        self.worker_path = Path(worker_proof)
        self.worker_identity = file_identity(self.worker_path)
        self.candidate_sha = candidate_sha
        self.model_proofs = [Path(path) for path in model_proofs]
        self.routes = {model: gateway_route(model) for model in ("gpt-5.6-luna", "gpt-5.6-terra")}
        self._check_workers()

    def _check_workers(self):
        _require(file_identity(self.worker_path) == self.worker_identity, "worker_proof_changed")
        check_worker_proof(json.loads(self.worker_path.read_text()), self.binding, self.candidate_sha)

    def _check_models(self):
        _require(len(self.model_proofs) == 2, "both_model_smoke_modes_required")
        modes, identities = set(), []
        for directory in self.model_proofs:
            report = json.loads((directory / "model_smoke.json").read_text())
            _require(report.get("status") == "passed", "real_model_smoke_failed")
            admissions = report.get("admissions", [])
            _require(
                len(admissions) == 9
                and all(
                    row.get("evidence", {}).get("runtime_sha256") == _fingerprint(self.binding) for row in admissions
                ),
                "model_smoke_runtime_unbound",
            )
            modes.add(report.get("context_mode"))
            previous_ids, before, chain = set(), None, None
            for case in ("chain", "cached", "failure", "repair"):
                turn = directory / case
                capture = json.loads((turn / "capture.json").read_text())
                _require(capture.get("complete") is True, "model_smoke_capture_incomplete")
                trace = json.loads((turn / "react_steps.json").read_text())["steps"]
                steps = [row for row in trace if row["id"] not in previous_ids]
                snapshots = {
                    row["stepId"]: row for row in json.loads((turn / "snapshots.json").read_text())["snapshots"]
                }
                head = snapshots[steps[-1]["id"]]
                audit_model_smoke_turn(case, steps, snapshots, head, before=chain if case == "repair" else before)
                if case == "chain":
                    chain = head
                before, previous_ids = head, {row["id"] for row in trace}
            identities.append(file_identity(directory / "model_smoke.json"))
        _require(modes == {"delta", "latest"}, "both_model_smoke_modes_required")
        return identities

    def for_smoke(self, context, system, stage, info):
        return self._check(context, system, stage, info, benchmark=False)

    def __call__(self, context, system, stage, info):
        return self._check(context, system, stage, info, benchmark=True)

    def _check(self, context, system, stage, info, *, benchmark):
        verify_engine_runtime(self.build, context["backend"], expected=self.binding)
        self._check_workers()
        model = context["model_type"]
        _require(gateway_route(model) == self.routes[model], "gateway_route_changed")
        references = {
            "V1Reference": "bdd6b84037bd32818684e291dcdf5933b7a23466",
            "BatchParent": "41cd8ae97e30edfec90c847886ed4068dd0fa9b0",
        }
        _require(
            context["agent_source"]["git_sha"] == references.get(context["arm"], self.candidate_sha),
            "agent_reference_revision_mismatch",
        )
        evidence = {
            "qualified": True,
            "scope": "benchmark" if benchmark else "model-smoke",
            "runtime_sha256": _fingerprint(self.binding),
            "worker_proof": self.worker_identity,
            "gateway": self.routes[model],
        }
        if benchmark:
            evidence["model_proofs"] = self._check_models()
        if stage == "before_resources":
            return evidence
        token = getattr(getattr(system, "agent", None), "_token", None) or login(timeout=(3, 15), allow_redirects=False)
        api = PilotResourceAPI(
            token=token,
            agent_endpoint=system.agent_service_endpoint,
            texera_endpoint=TEXERA_API_ENDPOINT,
            computing_unit_endpoint=TEXERA_COMPUTING_UNIT_ENDPOINT,
        )
        cu_record = read_resource_journal(context["computing_unit_journal"])
        cu = cu_record["owned"].get("computing_unit")
        _require(cu_record["allocations_complete"] and isinstance(cu, dict), "computing_unit_receipt_missing")
        unit = _owned_entry(
            api.request("computing_unit", "GET", "/api/computing-unit"), "computing_unit", cu["id"], cu["name"]
        )
        _require(
            unit.get("type") == "local"
            and unit.get("uri") == context["backend"]
            and cu["id"] == system.computing_unit_id,
            "computing_unit_route_mismatch",
        )
        if stage == "before_setup":
            return evidence
        resources = read_resource_journal(system.pilot_bundle.path / "resource_allocations.jsonl")
        _require(
            resources["allocations_complete"] and resources["references"].get("computing_unit") == cu["id"],
            "owned_setup_incomplete",
        )
        workflow, agent = resources["owned"]["workflow"], resources["owned"]["agent"]
        _owned_entry(api.request("texera", "GET", "/api/workflow/list"), "workflow", workflow["id"], workflow["name"])
        _require(
            isinstance(info, dict)
            and info.get("id") == agent["id"]
            and info.get("name") == agent["name"]
            and info.get("state") == "AVAILABLE"
            and info.get("modelType") == model
            and info.get("driver") == "vercel-tool-use",
            "agent_not_idle_or_owned",
        )
        delegate = info.get("delegate", {})
        _require(
            delegate.get("workflowId") == workflow["id"] and delegate.get("computingUnitId") == cu["id"],
            "agent_delegate_mismatch",
        )
        check_effective_settings(system.agent.settings.to_api_dict(), info.get("settings"))
        route = "/api/agents/" + quote(agent["id"], safe="")
        current_workflow = api.request("agent", "GET", route + "/workflow").get("workflow")
        check_default_python(current_workflow)
        history = api.request("texera", "GET", "/api/executions/" + str(workflow["id"]))
        _require(
            isinstance(history, list)
            and all(row.get("status") in {3, 4, 5, 6} and row.get("cuId") == cu["id"] for row in history),
            "backend_not_terminal",
        )
        report = execution_report(
            Path(context["session_directory"]) / "execution_requests.jsonl",
            workflow_id=workflow["id"],
            computing_unit_id=cu["id"],
            expected_recorder_id=context["recorder"]["instanceId"],
        )
        _require(
            report["journal_status"] in {"open", "closed"}
            and report["other_requests"] == 0
            and report["compilation"]["unknown"] == 0
            and report["runtime"]["unknown"] == 0
            and len(history) == report["requests"]
            and {str(row["eId"]) for row in history} == set(report["engine_execution_ids"]),
            "execution_history_not_accounted",
        )
        if stage in {"before_dispatch", "before_smoke"}:
            steps = api.request("agent", "GET", route + "/react-steps").get("steps")
            _require(steps == [] and current_workflow["operators"] == [] and history == [], "session_not_fresh")
        evidence["resources"] = {"workflow_id": workflow["id"], "agent_id": agent["id"], "computing_unit_id": cu["id"]}
        return evidence

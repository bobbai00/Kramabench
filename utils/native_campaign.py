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

"""Read-only admission of a supervisor-frozen full campaign manifest.

The supervisor qualifies the engine and launches services before freezing this
manifest. Admission rechecks their identities, source, gateway, local input
bindings and requested/effective settings; it never starts or repairs services.
"""

import glob
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

from utils.pilot_inputs import file_identity
from utils.pilot_launch import verify_service_launch
from utils.pilot_qualification import check_effective_settings, gateway_route


STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
HARNESS_ROOT = Path(__file__).resolve().parents[1]


def _require(value, message):
    if not value:
        raise ValueError(message)


def file_binding(path):
    """Hash once at manifest creation, bind device/inode/mtime/ctime thereafter."""
    path = Path(path).absolute()
    before = path.stat()
    identity = file_identity(path)
    after = path.stat()
    signature = {key: getattr(after, key) for key in STAT_FIELDS}
    _require(signature == {key: getattr(before, key) for key in STAT_FIELDS}, "binding_changed_while_hashing")
    return {"path": str(path), **identity, "stat": signature}


def runtime_identity(runtime):
    """Verify the already-qualified local master without rehashing its jars."""
    from utils.pilot_launch import _local_origin

    endpoint = _local_origin(runtime["endpoint"])
    pid = runtime["pid"]
    _require(type(pid) is int and pid > 0, "runtime_pid_invalid")
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rpartition(")")[2].split()
    listeners = subprocess.check_output(
        ["ss", "-H", "-ltnp", "sport", "=", f":{urlsplit(endpoint).port}"], text=True, timeout=5
    )
    _require(
        fields[0] != "Z" and int(fields[19]) == runtime["start_ticks"]
        and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == runtime["boot_id"]
        and {int(value) for value in re.findall(r"\bpid=(\d+)", listeners)} == {pid}
        and b"org.apache.texera.web.ComputingUnitMaster" in (process / "cmdline").read_bytes().split(b"\0"),
        "campaign_runtime_changed",
    )
    return str((process / "cwd").resolve(strict=True))


class CampaignGuard:
    def __init__(self, *, campaign=None):
        from systems.native_campaign_system import CAMPAIGN_ID, OBSERVE_ONLY_CAMPAIGN_ID

        self.campaign = CAMPAIGN_ID if campaign is None else campaign
        _require(self.campaign in {CAMPAIGN_ID, OBSERVE_ONLY_CAMPAIGN_ID}, "campaign_manifest_invalid")
        self.observe_only = self.campaign == OBSERVE_ONLY_CAMPAIGN_ID
        path = os.environ.get("NATIVE_CAMPAIGN_MANIFEST")
        _require(path, "NATIVE_CAMPAIGN_MANIFEST is required before campaign dispatch")
        self.path = Path(path).resolve(strict=True)
        self.binding = file_binding(self.path)
        self.manifest = json.loads(self.path.read_text())
        _require(self.manifest.get("version") == 1
                 and self.manifest.get("campaign") == self.campaign, "campaign_manifest_invalid")
        _require(isinstance(self.manifest.get("bindings"), list) and self.manifest["bindings"],
                 "campaign_input_bindings_missing")

    @staticmethod
    def check_binding(binding, *, hash_content=True):
        path = Path(binding["path"])
        stat = path.stat()
        _require(
            str(path.resolve(strict=True)) == binding["resolved_path"]
            and {key: getattr(stat, key) for key in STAT_FIELDS} == binding["stat"]
            and stat.st_size == binding["bytes"]
            and re.fullmatch(r"[a-f0-9]{64}", binding["sha256"]),
            "campaign_file_binding_changed",
        )
        if hash_content:
            _require(file_identity(path) == {key: binding[key] for key in ("resolved_path", "bytes", "sha256")},
                     "campaign_file_binding_changed")

    def __call__(self, system, stage, info):
        _require(getattr(system, "campaign_id", "NativeCampaign20260915Rep1") == self.campaign,
                 "campaign_namespace_mismatch")
        self.check_binding(self.binding)
        manifest = self.manifest
        git = lambda *args: subprocess.check_output(
            ["git", "-C", str(HARNESS_ROOT), *args], text=True, timeout=10
        ).strip()
        _require(git("rev-parse", "HEAD") == manifest["harness_sha"], "campaign_harness_revision_changed")
        # Official scoring updates this cache during a run. Exclude only its
        # exact path; all benchmark code and other fixture changes still fail.
        _require(not git("status", "--porcelain", "--untracked-files=all", "--", "systems", "utils",
                         "dataflow_agent.py", "kb.py", "evaluate.py", "benchmark",
                         ":(top,exclude)benchmark/fixtures/paraphrase_cache.json"), "campaign_harness_source_dirty")
        _require(system.computing_unit_id == manifest["computing_unit_id"], "campaign_computing_unit_mismatch")
        key = "BatchParent" if system.pilot_spec.key == "BatchParent" else "V2"
        service = manifest["services"][key]
        _require(system.agent_service_endpoint.rstrip("/") == service["endpoint"].rstrip("/"),
                 "campaign_endpoint_mismatch")
        observed = verify_service_launch(service["launch_record"])
        _require(observed["service_endpoint"] == service["endpoint"].rstrip("/")
                 and observed["git_sha"] == service["git_sha"]
                 and observed.get("recorder_url", "").rstrip("/") == manifest["runtime"]["endpoint"].rstrip("/"),
                 "campaign_service_changed")
        if key == "BatchParent" and not self.observe_only:
            _require(observed["git_sha"] == "41cd8ae97e30edfec90c847886ed4068dd0fa9b0",
                     "campaign_batch_parent_revision_mismatch")
        route = gateway_route(system.model_type)
        _require(route == manifest["gateway"][system.model_type], "campaign_gateway_changed")
        execution_directory = Path(runtime_identity(manifest["runtime"]))
        bound = {}
        for binding in manifest["bindings"]:
            self.check_binding(binding, hash_content=False)
            bound[binding["resolved_path"]] = binding
        task_id = system.campaign_query_id
        expected_paths = manifest.get("task_inputs", {}).get(task_id)
        if expected_paths is not None:
            _require(system.campaign_prompt_paths == expected_paths, "campaign_task_input_map_changed")
        domain = task_id.rsplit("-", 2)[0]
        workload = HARNESS_ROOT / "workload" / f"{domain}.json"
        required = [workload, HARNESS_ROOT / "format_hint" / f"{domain}.json"]
        for pattern in system.campaign_prompt_paths:
            _require(not Path(pattern).is_absolute(), "campaign_prompt_path_not_relative")
            local_matches = sorted(path for path in glob.glob(pattern, recursive=True) if Path(path).is_file())
            worker_matches = sorted(os.path.relpath(path, execution_directory) for path in
                                    glob.glob(str(execution_directory / pattern), recursive=True) if Path(path).is_file())
            _require(local_matches and local_matches == worker_matches, "campaign_data_glob_mismatch")
            for relative in local_matches:
                local, worker = Path(relative).resolve(strict=True), (execution_directory / relative).resolve(strict=True)
                _require(str(local) in bound and str(worker) in bound
                         and bound[str(local)]["sha256"] == bound[str(worker)]["sha256"], "campaign_data_binding_missing")
                required.extend((local, worker))
        _require(all(str(path.resolve(strict=True)) in bound for path in required), "campaign_task_binding_missing")
        for resolved in {str(path.resolve(strict=True)) for path in required}:
            self.check_binding(bound[resolved])
        task = [row for row in json.loads(workload.read_text()) if row.get("id") == task_id]
        _require(len(task) == 1 and task[0] == system.workload_data[task_id], "campaign_task_changed")
        tool_surface = None
        if stage != "before_setup":
            _require(isinstance(info, dict) and info.get("id") == system.agent.agent_id
                     and info.get("modelType") == system.model_type and info.get("driver") == "vercel-tool-use",
                     "campaign_agent_identity_mismatch")
            if stage == "before_dispatch":
                _require(info.get("state") == "AVAILABLE", "campaign_agent_not_available")
            delegate = info.get("delegate", {})
            _require(delegate.get("computingUnitId") == system.computing_unit_id
                     and delegate.get("workflowId") == system.agent._workflow_id, "campaign_agent_route_mismatch")
            check_effective_settings(system.agent.settings.to_api_dict(), info.get("settings"))
            if self.observe_only:
                tool_surface = self.check_tool_surface(system.agent)
        evidence = {"qualified": True, "manifest_sha256": self.binding["sha256"],
                    "service": observed, "gateway": route, "harness_sha": manifest["harness_sha"]}
        if tool_surface is not None:
            evidence["tool_surface"] = tool_surface
        return evidence

    @staticmethod
    def check_tool_surface(agent):
        """Qualify the actual registry and prompt, including disabled entries."""
        from systems.native_pilot_system import agent_json

        observed = agent_json(agent, "/system-info")
        _require(isinstance(observed, dict) and isinstance(observed.get("systemPrompt"), str)
                 and observed["systemPrompt"].strip()
                 and isinstance(observed.get("tools"), list) and observed["tools"], "campaign_tool_surface_invalid")
        surface = {key: observed[key] for key in ("systemPrompt", "tools")}
        encoded = json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
        _require(b"inspectresult" not in encoded.lower(), "campaign_inspect_result_forbidden")
        _require(all(isinstance(tool, dict) and isinstance(tool.get("name"), str) and tool["name"]
                     and isinstance(tool.get("description"), str) and isinstance(tool.get("inputSchema"), dict)
                     and type(tool.get("enabled")) is bool for tool in surface["tools"]),
                 "campaign_tool_surface_invalid")
        names = [tool["name"] for tool in surface["tools"]]
        enabled = [tool["name"] for tool in surface["tools"] if tool["enabled"]]
        _require(len(names) == len(set(names)) and "dataflow" in enabled, "campaign_tool_surface_invalid")
        return {**surface, "tool_names": names, "enabled_tool_names": enabled,
                "sha256": hashlib.sha256(encoded).hexdigest()}


def cleanup_campaign_resources(system, bundle, attempt):
    """Delete only captured, idle, owned task resources; retain the campaign CU."""
    from dataflow_agent import TEXERA_API_ENDPOINT, TEXERA_COMPUTING_UNIT_ENDPOINT
    from utils.pilot_cleanup import PilotResourceAPI, _absent, _owned_entry
    from utils.resource_journal import read_resource_journal

    result = {"status": "pending", "completed": [], "unresolved": attempt["resources"],
              "backend_terminal_verified": False}
    bundle.write("cleanup.json", result)
    if not attempt["completed_event"] or not attempt["capture_complete"]:
        result["reason"] = "completed_turn_and_full_capture_required"
        bundle.write("cleanup.json", result)
        return result
    try:
        manifest = system._campaign_guard.manifest
        key = "BatchParent" if system.pilot_spec.key == "BatchParent" else "V2"
        launch = manifest["services"][key]["launch_record"]
        source = verify_service_launch(launch)
        _require(source["service_endpoint"] == system.agent_service_endpoint.rstrip("/"), "cleanup_service_mismatch")
        journal = read_resource_journal(bundle.path / "resource_allocations.jsonl")
        _require(journal["allocations_complete"], "cleanup_incomplete_allocations")
        owned = journal["owned"]
        agent, workflow = owned["agent"], owned["workflow"]
        cuid = system.computing_unit_id
        _require(journal["references"].get("computing_unit") == cuid and attempt["resources"] == {
            "agent_id": agent["id"], "workflow_id": workflow["id"], "computing_unit_id": cuid,
        }, "cleanup_resource_identity_mismatch")
        api = PilotResourceAPI(token=system.agent._token, agent_endpoint=system.agent_service_endpoint,
                               texera_endpoint=TEXERA_API_ENDPOINT, computing_unit_endpoint=TEXERA_COMPUTING_UNIT_ENDPOINT)
        path = "/api/agents/" + quote(agent["id"], safe="")

        def check_terminal():
            current = api.request("agent", "GET", path)
            delegate = current.get("delegate", {})
            _require(current.get("id") == agent["id"] and current.get("name") == agent["name"]
                     and current.get("state") == "AVAILABLE" and delegate.get("workflowId") == workflow["id"]
                     and delegate.get("computingUnitId") == cuid, "cleanup_agent_not_idle")
            history = api.request("texera", "GET", "/api/executions/" + str(workflow["id"]))
            _require(isinstance(history, list) and all(isinstance(row, dict)
                     and type(row.get("status")) is int and row["status"] in {3, 4, 5, 6}
                     and type(row.get("eId")) is int and row["eId"] > 0 and row.get("cuId") == cuid
                     for row in history), "cleanup_execution_not_terminal")
            return history

        _owned_entry(api.request("texera", "GET", "/api/workflow/list"), "workflow", workflow["id"], workflow["name"])
        units = api.request("computing_unit", "GET", "/api/computing-unit")
        matching = [item for item in units if isinstance(item, dict) and item.get("isOwner") is True
                    and isinstance(item.get("computingUnit"), dict) and item["computingUnit"].get("cuid") == cuid]
        _require(len(matching) == 1 and matching[0]["computingUnit"].get("type") == "local"
                 and matching[0]["computingUnit"].get("uri", "").rstrip("/") == manifest["runtime"]["endpoint"].rstrip("/"),
                 "cleanup_computing_unit_mismatch")
        history = check_terminal()
        _require(verify_service_launch(launch) == source, "cleanup_service_changed")
        _require(check_terminal() == history, "cleanup_execution_history_changed")
        result.update(status="cleaning", backend_terminal_verified=True,
                      execution_ids=[row["eId"] for row in history])
        bundle.write("cleanup.json", result)
        api.request("agent", "DELETE", path)
        remaining = api.request("agent", "GET", "/api/agents/")
        _require(isinstance(remaining, dict) and _absent(remaining.get("agents"), None, "id", agent["id"]),
                 "cleanup_agent_deletion_unverified")
        result["completed"].append("agent")
        result["unresolved"] = {"workflow_id": workflow["id"], "computing_unit_id": cuid}
        bundle.write("cleanup.json", result)
        _owned_entry(api.request("texera", "GET", "/api/workflow/list"), "workflow", workflow["id"], workflow["name"])
        api.request("texera", "POST", "/api/workflow/delete", {"wids": [workflow["id"]]})
        _require(_absent(api.request("texera", "GET", "/api/workflow/list"), "workflow", "wid", workflow["id"]),
                 "cleanup_workflow_deletion_unverified")
        result["completed"].append("workflow")
        result.update(status="complete", unresolved={}, retained_computing_unit_id=cuid)
    except Exception as error:
        result.update(status="partial" if result["completed"] else "pending", error_type=type(error).__name__)
        if isinstance(error, ValueError) and str(error).startswith("cleanup_"):
            result["reason"] = str(error)
        elif isinstance(getattr(error, "code", None), str):
            result["reason"] = error.code
    bundle.write("cleanup.json", result)
    return result

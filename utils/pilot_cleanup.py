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

"""Scoped pilot allocation/cleanup; never stop a shared backend to clean up.

Allocation receipts alone are insufficient. Cleanup requires captured artifacts,
a stable source-pinned agent service, a drained recorder with known outcomes,
matching terminal database executions, current ownership and a LOCAL CU. Any
ambiguity preserves the resources and records their IDs. This module does not
retry model turns, guess missing allocations, or kill unknown executions.
"""

import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

from utils.execution_journal import execution_report
from utils.pilot_artifacts import AttemptBundle
from utils.pilot_launch import _local_origin, verify_service_launch
from utils.resource_journal import read_resource_journal


class CleanupBlocked(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise CleanupBlocked(code)


class PilotResourceAPI:
    def __init__(self, *, token, agent_endpoint, texera_endpoint, computing_unit_endpoint):
        if not isinstance(token, str) or not token:
            raise ValueError("authenticated resource API requires a token")
        self._token = token
        self.agent_endpoint = _local_origin(agent_endpoint)
        self.texera_endpoint = _local_origin(texera_endpoint)
        self.computing_unit_endpoint = _local_origin(computing_unit_endpoint)

    def request(self, service, method, path, payload=None):
        origin = {
            "agent": self.agent_endpoint,
            "texera": self.texera_endpoint,
            "computing_unit": self.computing_unit_endpoint,
        }[service]
        _require(
            isinstance(path, str) and path.startswith("/api/") and "?" not in path and "#" not in path,
            "invalid_resource_route",
        )
        response = requests.request(
            method,
            origin + path,
            json=payload,
            headers={"Authorization": "Bearer " + self._token},
            timeout=(3, 15),
            allow_redirects=False,
        )
        _require(200 <= response.status_code < 300, "resource_http_" + str(response.status_code))
        if method == "DELETE" or path == "/api/workflow/delete":
            return None
        return response.json()


def create_owned_computing_unit(api, journal, *, backend):
    backend = _local_origin(backend)
    name = "native-python-" + journal.run_id + "-cu"
    journal.record({"event": "create_intent", "kind": "computing_unit", "name": name})
    data = api.request(
        "computing_unit",
        "POST",
        "/api/computing-unit/create",
        {
            "name": name,
            "unitType": "local",
            "uri": backend,
            "cpuLimit": "NaN",
            "memoryLimit": "NaN",
            "gpuLimit": "NaN",
            "jvmMemorySize": "NaN",
            "shmSize": "NaN",
        },
    )
    unit = data.get("computingUnit") if isinstance(data, dict) else None
    _require(
        isinstance(unit, dict)
        and data.get("isOwner") is True
        and unit.get("type") == "local"
        and unit.get("name") == name
        and _local_origin(unit.get("uri")) == backend,
        "created_computing_unit_did_not_match_request",
    )
    journal.record({"event": "created", "kind": "computing_unit", "name": name, "id": unit.get("cuid")})
    return unit["cuid"]


def _owned_entry(entries, kind, identifier, name):
    field, id_field = ("workflow", "wid") if kind == "workflow" else ("computingUnit", "cuid")
    _require(isinstance(entries, list), "invalid_resource_list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get(field), dict) and entry[field].get(id_field) == identifier
    ]
    _require(
        len(matches) == 1 and matches[0].get("isOwner") is True and matches[0][field].get("name") == name,
        "resource_ownership_or_name_mismatch",
    )
    return matches[0][field]


def _absent(entries, field, id_field, identifier):
    _require(isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries), "invalid_resource_list")
    if field is not None:
        _require(all(isinstance(entry.get(field), dict) for entry in entries), "invalid_resource_list")
    identities = [(entry[field] if field else entry).get(id_field) for entry in entries]
    _require(
        all(
            (isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)))
            if id_field == "id"
            else (type(value) is int and 0 < value <= 2147483647)
            for value in identities
        ),
        "invalid_resource_list",
    )
    return identifier not in identities


def _assess(
    api,
    *,
    task_directory,
    computing_unit_journal,
    workflow_journal,
    execution_journal,
    recorder_id,
    service_launch_record,
):
    cu_record, workflow_record = read_resource_journal(computing_unit_journal), read_resource_journal(workflow_journal)
    _require(
        cu_record["allocations_complete"] and workflow_record["allocations_complete"], "allocation_receipts_incomplete"
    )
    cu, workflow, agent = (
        cu_record["owned"].get("computing_unit"),
        workflow_record["owned"].get("workflow"),
        workflow_record["owned"].get("agent"),
    )
    _require(all(isinstance(item, dict) for item in (cu, workflow, agent)), "created_resource_receipts_missing")
    _require(workflow_record["references"].get("computing_unit") == cu["id"], "computing_unit_reference_mismatch")
    _require(
        cu["name"] == "native-python-" + cu_record["run_id"] + "-cu"
        and workflow["name"] == "native-python-" + workflow_record["run_id"] + "-workflow"
        and agent["name"] == "native-python-" + workflow_record["run_id"] + "-agent",
        "resource_tag_mismatch",
    )
    task_directory = Path(task_directory)
    capture = json.loads((task_directory / "capture.json").read_text())
    attempt = json.loads((task_directory / "attempt.json").read_text())
    expected = {"agent_id": agent["id"], "workflow_id": workflow["id"], "computing_unit_id": cu["id"]}
    _require(isinstance(capture, dict) and capture.get("complete") is True, "artifact_capture_incomplete")
    _require(isinstance(attempt, dict) and attempt.get("resources") == expected, "attempt_resource_identity_mismatch")
    source = verify_service_launch(service_launch_record)
    _require(
        source.get("loaded_revision_verified") is True and source.get("service_endpoint") == api.agent_endpoint,
        "agent_service_identity_mismatch",
    )
    report = execution_report(
        execution_journal, workflow_id=workflow["id"], computing_unit_id=cu["id"], expected_recorder_id=recorder_id
    )
    _require(
        report["journal_status"] == "closed"
        and report["other_requests"] == 0
        and report["compilation"]["unknown"] == 0
        and report["runtime"]["unknown"] == 0
        and len(report["engine_execution_ids"]) == report["requests"],
        "backend_requests_not_fully_accounted",
    )
    with Path(execution_journal).open() as stream:
        header = json.loads(stream.readline())
    _require(
        header.get("computingUnitIds") == [cu["id"]] and _local_origin(header.get("url")) == source.get("recorder_url"),
        "recorder_route_or_scope_mismatch",
    )
    recorder_source = header.get("source")
    _require(
        isinstance(recorder_source, dict)
        and recorder_source.get("dirty") is False
        and isinstance(recorder_source.get("gitSha"), str)
        and re.fullmatch(r"[a-f0-9]{40}", recorder_source["gitSha"]),
        "recorder_source_unqualified",
    )
    current = api.request("agent", "GET", "/api/agents/" + quote(agent["id"], safe=""))
    _require(
        isinstance(current, dict)
        and current.get("id") == agent["id"]
        and current.get("name") == agent["name"]
        and current.get("state") == "AVAILABLE",
        "agent_not_idle_or_identity_changed",
    )
    delegate = current.get("delegate")
    _require(
        isinstance(delegate, dict)
        and delegate.get("workflowId") == workflow["id"]
        and delegate.get("computingUnitId") == cu["id"],
        "agent_delegate_mismatch",
    )
    _owned_entry(api.request("texera", "GET", "/api/workflow/list"), "workflow", workflow["id"], workflow["name"])
    unit = _owned_entry(
        api.request("computing_unit", "GET", "/api/computing-unit"), "computing_unit", cu["id"], cu["name"]
    )
    _require(
        unit.get("type") == "local" and _local_origin(unit.get("uri")) == _local_origin(header.get("backend")),
        "computing_unit_not_the_owned_local_registration",
    )
    history = api.request("texera", "GET", "/api/executions/" + str(workflow["id"]))
    _require(isinstance(history, list) and all(isinstance(row, dict) for row in history), "invalid_execution_history")
    # Utils.maptoStatusCode: Completed=3, Failed=4, Killed=5, CompletedFromCache=6.
    # Ready/Uninitialized=0, Running=1, Paused=2 and unknown=-1 are not terminal.
    _require(
        all(
            type(row.get("status")) is int
            and row["status"] in {3, 4, 5, 6}
            and type(row.get("eId")) is int
            and row["eId"] > 0
            and row.get("cuId") == cu["id"]
            for row in history
        ),
        "backend_execution_not_terminal_or_wrong_unit",
    )
    _require(
        len(history) == len(report["engine_execution_ids"])
        and {str(row["eId"]) for row in history} == set(report["engine_execution_ids"]),
        "execution_history_mismatch",
    )
    current_source = verify_service_launch(service_launch_record)
    _require(current_source == source, "agent_service_changed_during_cleanup_preflight")
    return {
        "agent": agent,
        "workflow": workflow,
        "computing_unit": cu,
        "execution_ids": report["engine_execution_ids"],
        "source": source,
    }


def cleanup_owned_resources(
    api,
    *,
    task_directory,
    computing_unit_journal,
    workflow_journal,
    execution_journal,
    recorder_id,
    service_launch_record,
    output_directory,
):
    """One verified cleanup attempt, with partial IDs retained on any failure.

    This is intentionally not a broad recovery loop. Incomplete allocations or
    unknown executions require investigation, not a guessed DELETE or restart.
    Each cleanup attempt gets a new directory; it never replaces task evidence.
    """
    bundle = AttemptBundle(output_directory)
    result = {
        "version": 1,
        "status": "preflight",
        "stage": "verification",
        "completed": [],
        "unresolved": {},
        "backend_terminal_verified": False,
    }
    bundle.write("cleanup.json", result)
    mutation_attempted = False
    try:
        # Keep known receipt IDs available even if the stronger preflight fails.
        for path in (computing_unit_journal, workflow_journal):
            for kind, receipt in read_resource_journal(path)["owned"].items():
                _require(
                    kind not in result["unresolved"] or result["unresolved"][kind] == receipt["id"],
                    "duplicate_resource_identity",
                )
                result["unresolved"][kind] = receipt["id"]
        bundle.write("cleanup.json", result)
        plan = _assess(
            api,
            task_directory=task_directory,
            computing_unit_journal=computing_unit_journal,
            workflow_journal=workflow_journal,
            execution_journal=execution_journal,
            recorder_id=recorder_id,
            service_launch_record=service_launch_record,
        )
        result.update(backend_terminal_verified=True, execution_ids=plan["execution_ids"], status="cleaning")
        for kind in ("agent", "workflow", "computing_unit"):
            result["stage"] = kind
            bundle.write("cleanup.json", result)  # intent before each destructive request
            identifier = plan[kind]["id"]
            mutation_attempted = True
            if kind == "agent":
                api.request("agent", "DELETE", "/api/agents/" + quote(identifier, safe=""))
                remaining = api.request("agent", "GET", "/api/agents/")
                _require(
                    isinstance(remaining, dict) and _absent(remaining.get("agents"), None, "id", identifier),
                    "agent_deletion_unverified",
                )
            elif kind == "workflow":
                _owned_entry(api.request("texera", "GET", "/api/workflow/list"), kind, identifier, plan[kind]["name"])
                api.request("texera", "POST", "/api/workflow/delete", {"wids": [identifier]})
                _require(
                    _absent(api.request("texera", "GET", "/api/workflow/list"), "workflow", "wid", identifier),
                    "workflow_deletion_unverified",
                )
            else:
                unit = _owned_entry(
                    api.request("computing_unit", "GET", "/api/computing-unit"), kind, identifier, plan[kind]["name"]
                )
                _require(unit.get("type") == "local", "computing_unit_type_changed")
                api.request("computing_unit", "DELETE", "/api/computing-unit/" + str(identifier) + "/terminate")
                _require(
                    _absent(
                        api.request("computing_unit", "GET", "/api/computing-unit"), "computingUnit", "cuid", identifier
                    ),
                    "computing_unit_termination_unverified",
                )
            result["completed"].append(kind)
            del result["unresolved"][kind]
            bundle.write("cleanup.json", result)
        result.update(status="complete", stage="complete")
    except Exception as error:
        result.update(status="partial" if mutation_attempted else "blocked", error_type=type(error).__name__)
        if isinstance(error, CleanupBlocked):
            result["reason"] = error.code
    bundle.write("cleanup.json", result)
    return result

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

"""Own one pilot attempt's isolated services, artifacts and verified cleanup.

This composes the tested runtime pieces, not an automatic qualification bypass.
The mandatory qualification callback must check matching engine/worker builds,
prior correctness/model gates, reference revisions, effective settings and live
resource ownership. An arbitrary {qualified: True} is not production evidence.
There is no permissive CLI, shared-backend restart, campaign or retry loop.
"""

import hashlib
import re
import sys
from pathlib import Path

import requests

from dataflow_agent import TEXERA_API_ENDPOINT, TEXERA_COMPUTING_UNIT_ENDPOINT, login
from native_python_pilot import finalize_pilot_measurements, prepare_pilot_task, run_pilot_task
from systems import native_python_system
from systems.native_pilot_system import NativePilotSystem, PILOT_TASK
from utils.pilot_artifacts import AttemptBundle
from utils.pilot_cleanup import PilotResourceAPI, cleanup_owned_resources, create_owned_computing_unit
from utils.pilot_inputs import file_identity, freeze_pilot_inputs, verify_pilot_inputs
from utils.pilot_launch import (
    _command,
    _listener_free,
    _local_origin,
    launch_agent_service,
    source_snapshot,
    stop_agent_service,
    verify_service_launch,
)
from utils.pilot_resource_smoke import RecorderProcess
from utils.resource_journal import ResourceJournal


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _harness_source():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "benchmark",
        "systems",
        "utils",
        "native_python_pilot.py",
        "kb.py",
        "dataflow_agent.py",
        "pyproject.toml",
        "uv.lock",
    ]
    git = lambda args: _command(["git", "-C", str(root), *args])
    sha = git(["rev-parse", "HEAD"]).strip()
    _require(re.fullmatch(r"[a-f0-9]{40}", sha), "harness_revision_unresolved")
    _require(not git(["status", "--porcelain", "--untracked-files=all", "--", *paths]).strip(), "harness_source_dirty")
    names = sorted(name for name in git(["ls-files", "-z", "--", *paths]).split("\0") if name)
    _require(names, "harness_source_empty")
    digest = hashlib.sha256()
    for name in names:
        identity = file_identity(root / name)
        _require(identity["resolved_path"] == str(root / name), "harness_source_link")
        digest.update(name.encode() + b"\0" + identity["sha256"].encode())
    return {
        "root": str(root),
        "git_sha": sha,
        "source_sha256": digest.hexdigest(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
    }


def _recorder_source(worktree, expected_sha):
    root = Path(worktree).resolve(strict=True)
    paths = ["agent-service/benchmark/e2e/" + name for name in ("execution-recorder.ts", "execution-recorder-run.ts")]
    _require(
        _command(["git", "-C", str(root), "rev-parse", "HEAD"]).strip() == expected_sha, "recorder_revision_changed"
    )
    _require(
        not _command(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", *paths]).strip(),
        "recorder_source_dirty",
    )
    return {
        "root": str(root),
        "gitSha": expected_sha,
        "files": [{"path": name, "sha256": file_identity(root / name)["sha256"]} for name in paths],
    }


def _verify_recorder(recorder, source, *, require_empty):
    _require(recorder.process.poll() is None, "recorder_process_not_live")
    loaded = recorder.metadata.get("source", {})
    _require(
        loaded.get("dirty") is False and all(loaded.get(key) == value for key, value in source.items()),
        "recorder_loaded_source_mismatch",
    )
    response = requests.get(recorder.metadata["url"] + "/__recorder/status", timeout=(2, 3), allow_redirects=False)
    _require(response.status_code == 200, "recorder_health_unavailable")
    status = response.json()
    _require(
        isinstance(status, dict)
        and status.get("version") == 1
        and status.get("kind") == "execution-recorder"
        and all(
            status.get(key) == recorder.metadata.get(key)
            for key in ("instanceId", "backend", "computingUnitIds", "maxRequests")
        )
        and status.get("accepting") is True
        and status.get("sinkConfigured") is True
        and status.get("persistenceFailed") is False
        and type(status.get("pending")) is int
        and status["pending"] == 0
        and type(status.get("completed")) is int
        and status["completed"] >= 0
        and (not require_empty or status["completed"] == 0),
        "recorder_not_idle_or_bound",
    )
    _require(recorder.process.poll() is None, "recorder_process_not_live")
    return status


def run_owned_pilot(
    *,
    arm_key,
    agent_worktree,
    agent_sha,
    recorder_worktree,
    recorder_sha,
    workload_path,
    dataset_directory,
    execution_directory,
    output_directory,
    qualification,
    model_type="gpt-5.6-luna",
    task_output_directory=None,
    backend="http://127.0.0.1:8085",
):
    """Keep this caller alive through launch, one attempt, drain and shutdown.

    qualification(context, system, stage, info) runs before resource allocation
    (system/info are None), then at each normal attempt guard stage. It must
    return a non-secret evidence dict with qualified exactly True or raise.
    The system's local inputs and pinned service/recorder are also rechecked
    independently. A wrong answer or failed qualification stops this one-arm
    session; cleanup is allowed only after its independent terminal/ownership
    checks. Unresolved failures retain resources and private IDs, not retries.
    """
    bundle = AttemptBundle(Path(output_directory).resolve())
    result = {"version": 1, "status": "running", "stage": "preflight", "arm": arm_key, "resources": {}}
    recorder, system = None, None
    drained = False

    def stage(name):
        result["stage"] = name
        bundle.write("session.json", result)

    def qualify(context, current, name, info):
        evidence = qualification(context, current, name, info)
        _require(isinstance(evidence, dict) and evidence.get("qualified") is True, "runtime_qualification_rejected")
        return evidence

    try:
        stage("preflight")
        _require(callable(qualification), "runtime_qualification_required")
        selected = [
            arm for arm in native_python_system.ALL_PILOT_ARMS if arm.key == arm_key and arm.model_type == model_type
        ]
        _require(len(selected) == 1, "unknown_pilot_arm")
        spec = selected[0]
        backend = _local_origin(backend)
        source = source_snapshot(agent_worktree)
        _require(
            isinstance(agent_sha, str)
            and re.fullmatch(r"[a-f0-9]{40}", agent_sha)
            and source["git_sha"] == agent_sha
            and source["source_dirty"] is False
            and isinstance(recorder_sha, str)
            and re.fullmatch(r"[a-f0-9]{40}", recorder_sha),
            "source_preflight_failed",
        )
        recorder_source = _recorder_source(recorder_worktree, recorder_sha)
        harness_source = _harness_source()
        _require(_listener_free(spec.port), "isolated_agent_port_occupied")
        task_output = Path(task_output_directory or Path.cwd() / "system_scratch" / spec.system_name).resolve()
        task_path = task_output / PILOT_TASK
        _require(not task_path.exists() and not task_path.is_symlink(), "first_attempt_already_reserved")
        # Local preparation only, before resource allocation; never invent a CU
        # ID to instantiate a registered arm. The actual arm re-verifies this.
        inputs_system = NativePilotSystem(
            name=spec.system_name, model_type=spec.model_type, output_dir=str(bundle.path / "input-preflight")
        )
        prepare_pilot_task(inputs_system, workload_path=workload_path, dataset_directory=dataset_directory)
        inputs = freeze_pilot_inputs(
            inputs_system, workload_path=workload_path, execution_directory=execution_directory
        )
        context = {
            "arm": spec.key,
            "system_name": spec.system_name,
            "model_type": spec.model_type,
            "reasoning_effort": spec.reasoning_effort,
            "backend": backend,
            "agent_source": source,
            "recorder_source": recorder_source,
            "harness_source": harness_source,
            "inputs": inputs,
            "task_directory": str(task_path),
            "session_directory": str(bundle.path),
        }
        result["preflight"] = qualify(context, None, "before_resources", None)
        bundle.write("inputs.json", inputs)
        bundle.write("manifest.json", context)
        stage("authentication")
        token = login(timeout=(3, 15), allow_redirects=False)
        endpoint = f"http://127.0.0.1:{spec.port}"
        api = PilotResourceAPI(
            token=token,
            agent_endpoint=endpoint,
            texera_endpoint=TEXERA_API_ENDPOINT,
            computing_unit_endpoint=TEXERA_COMPUTING_UNIT_ENDPOINT,
        )
        stage("computing_unit")
        cu_path = bundle.path / "cu_allocations.jsonl"
        with ResourceJournal(cu_path) as journal:
            cuid = create_owned_computing_unit(api, journal, backend=backend)
        result["resources"]["computing_unit_id"] = cuid
        context["computing_unit_journal"] = str(cu_path)
        stage("recorder")
        # Historical agent trees do not contain the new recorder. All arms
        # use the same separately pinned candidate recorder source.
        recorder = RecorderProcess(recorder_worktree, backend=backend, computing_unit_id=cuid, directory=bundle.path)
        bundle.write("recorder.json", recorder.metadata)
        context["recorder"] = recorder.metadata
        _verify_recorder(recorder, recorder_source, require_empty=True)
        stage("agent_service")
        launch = launch_agent_service(
            agent_worktree,
            port=spec.port,
            expected_sha=agent_sha,
            recorder_url=recorder.metadata["url"],
            run_directory=bundle.path / "agent-launch",
        )
        result["agent_service_pid"] = launch["listener"]["pid"]
        launch_record = bundle.path / "agent-launch/launch.json"
        context["service_launch_record"] = str(launch_record)
        bundle.write("manifest.json", context)
        system = getattr(native_python_system, spec.system_name)(
            computing_unit_id=cuid, agent_service_endpoint=endpoint, output_dir=str(task_output)
        )

        def guard(current, name, info):
            _require(_recorder_source(recorder_worktree, recorder_sha) == recorder_source, "recorder_source_changed")
            _require(_harness_source() == harness_source, "harness_source_changed")
            return {
                "inputs": verify_pilot_inputs(current, inputs),
                "service": verify_service_launch(launch_record),
                "recorder": _verify_recorder(recorder, recorder_source, require_empty=name != "after_attempt"),
                "runtime": qualify(context, current, name, info),
                "qualified": True,
            }

        stage("attempt")
        result["verdict"] = run_pilot_task(
            system,
            guard=guard,
            workload_path=workload_path,
            dataset_directory=dataset_directory,
            execution_journal_path=bundle.path / "execution_requests.jsonl",
            recorder_id=recorder.metadata["instanceId"],
        )
        result["resources"] = system._resources()
        stage("recorder_drain")
        recorder.drain()
        drained = True
        stage("measurement_finalization")
        finalize_pilot_measurements(system)
        stage("resource_cleanup")
        result["cleanup"] = cleanup_owned_resources(
            api,
            task_directory=system.pilot_bundle.path,
            computing_unit_journal=cu_path,
            workflow_journal=system.pilot_bundle.path / "resource_allocations.jsonl",
            execution_journal=bundle.path / "execution_requests.jsonl",
            recorder_id=recorder.metadata["instanceId"],
            service_launch_record=launch_record,
            output_directory=bundle.path / "cleanup",
        )
        _require(result["cleanup"]["status"] == "complete", "owned_cleanup_unresolved")
        stage("agent_service_stop")
        result["service_stop"] = stop_agent_service(launch_record)
        verdict = result["verdict"]
        result.update(
            stage="complete",
            status="passed"
            if verdict.get("passed") is True and verdict.get("comparison_eligible") is True
            else "failed",
        )
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
    finally:
        if system is not None:
            result["resources"] = system._resources()
        if recorder is not None and not drained:
            try:
                recorder.drain()
                result["recorder_drained_after_failure"] = True
            except BaseException as error:
                result["recorder_drain_error_type"] = type(error).__name__
        bundle.write("session.json", result)
    return result

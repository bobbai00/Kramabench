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

"""Owned live model-smoke lifecycle; not a benchmark attempt or retry loop."""

import re
from pathlib import Path

from dataflow_agent import TEXERA_API_ENDPOINT, TEXERA_COMPUTING_UNIT_ENDPOINT, login
from systems.native_pilot_system import NativePilotSystem, public_info
from systems.native_python_system import PILOT_ARMS
from utils.native_model_smoke import run_model_smoke
from utils.pilot_artifacts import AttemptBundle
from utils.pilot_cleanup import PilotResourceAPI, cleanup_owned_resources, create_owned_computing_unit
from utils.pilot_launch import launch_agent_service, source_snapshot, stop_agent_service, verify_service_launch
from utils.pilot_resource_smoke import RecorderProcess
from utils.pilot_session import _recorder_source, _verify_recorder
from utils.resource_journal import ResourceJournal


def run_owned_model_smoke(*, worktree, source_sha, context_mode, output_directory, qualification):
    bundle = AttemptBundle(Path(output_directory))
    result = {"version": 1, "status": "running", "context_mode": context_mode, "benchmark_attempt": False}
    recorder, system = None, None
    drained = False
    context = {
        "backend": "http://127.0.0.1:8085",
        "model_type": "gpt-5.6-luna",
        "arm": "Combined",
        "agent_source": source_snapshot(worktree),
        "session_directory": str(bundle.path),
    }
    try:
        if context_mode not in {"delta", "latest"}:
            raise ValueError("invalid_smoke_mode")
        qualification.for_smoke(context, None, "before_resources", None)
        token = login(timeout=(3, 15), allow_redirects=False)
        api = PilotResourceAPI(
            token=token,
            agent_endpoint="http://127.0.0.1:3011",
            texera_endpoint=TEXERA_API_ENDPOINT,
            computing_unit_endpoint=TEXERA_COMPUTING_UNIT_ENDPOINT,
        )
        cu_path = bundle.path / "cu_allocations.jsonl"
        with ResourceJournal(cu_path) as journal:
            cuid = create_owned_computing_unit(api, journal, backend=context["backend"])
        context["computing_unit_journal"] = str(cu_path)
        recorder_source = _recorder_source(worktree, source_sha)
        recorder = RecorderProcess(worktree, backend=context["backend"], computing_unit_id=cuid, directory=bundle.path)
        context["recorder"] = recorder.metadata
        _verify_recorder(recorder, recorder_source, require_empty=True)
        launch_agent_service(
            worktree,
            port=3011,
            expected_sha=source_sha,
            recorder_url=recorder.metadata["url"],
            run_directory=bundle.path / "service",
        )
        launch_path = bundle.path / "service/launch.json"
        spec = next(arm for arm in PILOT_ARMS if arm.key == "Combined")
        settings = {**spec.settings(), "context_mode": context_mode, "message_layout": "native"}
        system = NativePilotSystem(
            name="NativePythonModelSmoke" + context_mode.title(),
            computing_unit_id=cuid,
            agent_service_endpoint="http://127.0.0.1:3011",
            output_dir=str(bundle.path),
            **settings,
        )
        task = AttemptBundle(bundle.path / "owned")
        system.pilot_bundle = task
        qualification.for_smoke(context, system, "before_setup", None)
        system._setup_agent()

        def guard(current, stage, info):
            service = verify_service_launch(launch_path)
            _verify_recorder(recorder, recorder_source, require_empty=stage in {"before_smoke", "before_chain"})
            evidence = qualification.for_smoke(context, current, stage, info)
            return {**evidence, "agent_service": service}

        result["model_smoke"] = run_model_smoke(
            system, output_directory=bundle.path / "model", guard=guard, context_mode=context_mode
        )
        capture, trace, snapshots, info, workflow = system._capture(task)
        for name, value in (
            ("capture.json", capture),
            ("react_steps.json", trace),
            ("snapshots.json", snapshots),
            ("agent_info.json", public_info(info)),
            ("workflow.json", workflow),
            ("attempt.json", {"resources": system._resources()}),
        ):
            task.write(name, value)
        recorder.drain()
        drained = True
        result["cleanup"] = cleanup_owned_resources(
            api,
            task_directory=task.path,
            computing_unit_journal=cu_path,
            workflow_journal=task.path / "resource_allocations.jsonl",
            execution_journal=bundle.path / "execution_requests.jsonl",
            recorder_id=recorder.metadata["instanceId"],
            service_launch_record=launch_path,
            output_directory=bundle.path / "cleanup",
        )
        if result["cleanup"]["status"] != "complete":
            raise ValueError("model_smoke_cleanup_unresolved")
        result["service_stop"] = stop_agent_service(launch_path)
        result["status"] = result["model_smoke"]["status"]
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, ValueError) and re.fullmatch(r"[A-Za-z_]{1,120}", str(error)):
            result["reason"] = str(error)
    finally:
        if system is not None:
            result["resources"] = system._resources()
        if recorder is not None and not drained:
            try:
                recorder.drain()
            except BaseException as error:
                result["recorder_drain_error"] = type(error).__name__
        bundle.write("model-session.json", result)
    return result

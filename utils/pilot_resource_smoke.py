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

"""Live setup/capture/cleanup only: no task, model request or engine execution.

Keep the orchestrator alive until its recorder and isolated agent have stopped.
An external tool runner may reap background children when their caller exits.
Failures preserve private evidence and resource IDs; this is not a recovery
loop and never restarts the shared backend or adopts another process by port.
"""

import argparse
import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path

from dataflow_agent import TEXERA_API_ENDPOINT, TEXERA_COMPUTING_UNIT_ENDPOINT, login
from systems.native_pilot_system import public_info
from systems.native_python_system import DataflowSystemLunaPythonPilotV2Control20260915Rep1 as PilotSystem
from utils.execution_journal import execution_report
from utils.pilot_artifacts import AttemptBundle
from utils.pilot_cleanup import PilotResourceAPI, cleanup_owned_resources, create_owned_computing_unit
from utils.pilot_launch import _listener_free, _local_origin, launch_agent_service, source_snapshot, stop_agent_service
from utils.resource_journal import ResourceJournal


class RecorderProcess:
    """Own a fresh recorder child, with bounded startup and graceful drain."""

    def __init__(self, worktree, *, backend, computing_unit_id, directory):
        self.metadata = None
        self.process = None
        directory = Path(directory)
        descriptor = os.open(directory / "recorder.stderr.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            self.process = subprocess.Popen(
                [
                    "bun",
                    "benchmark/e2e/execution-recorder-run.ts",
                    "--backend",
                    backend,
                    "--computing-units",
                    str(computing_unit_id),
                    "--journal",
                    str(directory / "execution_requests.jsonl"),
                    "--port",
                    "0",
                ],
                cwd=Path(worktree) / "agent-service",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stream,
            )
        try:
            line = bytearray()
            deadline = time.monotonic() + 15
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b"\n" not in line:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise RuntimeError("recorder startup timed out")
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk or len(line) + len(chunk) > 65536:
                        raise RuntimeError("recorder startup metadata missing or excessive")
                    line.extend(chunk)
            value = json.loads(line.split(b"\n", 1)[0])
            if (
                not isinstance(value, dict)
                or value.get("pid") != self.process.pid
                or value.get("computingUnitIds") != [computing_unit_id]
                or _local_origin(value.get("backend")) != backend
                or not isinstance(value.get("instanceId"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value["instanceId"])
                or value.get("pending") != 0
                or value.get("completed") != 0
                or value.get("accepting") is not True
                or self.process.poll() is not None
            ):
                raise RuntimeError("recorder startup identity mismatch")
            _local_origin(value.get("url"))
            self.metadata = value
        except BaseException:
            # The exact child handle is known; do not kill a port or cancel an
            # admitted request. A timeout here still leaves its state unresolved.
            self.drain()
            raise

    def drain(self):
        if self.process.poll() is None:
            self.process.terminate()
        status = self.process.wait(timeout=25)
        if self.process.stdout:
            self.process.stdout.close()
        if status != 0:
            raise RuntimeError("recorder did not exit cleanly")


def capture_setup(system, bundle):
    """Save actual service artifacts before checking this was a no-run setup."""
    bundle.write("attempt.json", {"setup_only": True, "dispatched": False, "resources": system._resources()})
    capture, trace, snapshots, info, workflow = system._capture(bundle)
    for name, value in (
        ("capture.json", capture),
        ("react_steps.json", trace),
        ("snapshots.json", snapshots),
        ("agent_info.json", public_info(info)),
        ("workflow.json", workflow),
    ):
        bundle.write(name, value)
    content = workflow.get("workflow") if isinstance(workflow, dict) else None
    steps = trace.get("steps") if isinstance(trace, dict) else None
    if (
        capture.get("complete") is not True
        or not isinstance(info, dict)
        or info.get("state") != "AVAILABLE"
        or not isinstance(steps, list)
        or any(step.get("role") in {"agent", "tool", "user"} for step in steps)
        or not isinstance(content, dict)
        or content.get("operators") != []
        or content.get("links") != []
    ):
        raise RuntimeError("setup-only service capture was incomplete or nonempty")
    return system._resources()


def run_resource_smoke(worktree, *, expected_sha, output_directory, port=3011, backend="http://127.0.0.1:8085"):
    bundle = AttemptBundle(Path(output_directory).resolve())
    result = {"version": 1, "status": "running", "stage": "preflight", "setup_only": True, "resources": {}}
    recorder, system = None, None
    recorder_drained = False

    def stage(name):
        result["stage"] = name
        bundle.write("smoke.json", result)

    try:
        stage("preflight")
        backend = _local_origin(backend)
        source = source_snapshot(worktree)
        result["source"] = source
        if (
            not isinstance(expected_sha, str)
            or not re.fullmatch(r"[a-f0-9]{40}", expected_sha)
            or source["git_sha"] != expected_sha
            or source["source_dirty"] is not False
            or type(port) is not int
            or not 1024 <= port <= 65535
            or port == 3001
            or not _listener_free(port)
        ):
            raise RuntimeError("source or isolated port preflight failed")
        stage("authentication")
        token = login(timeout=(3, 15), allow_redirects=False)
        endpoint = f"http://127.0.0.1:{port}"
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
        stage("recorder")
        recorder = RecorderProcess(worktree, backend=backend, computing_unit_id=cuid, directory=bundle.path)
        bundle.write("recorder.json", recorder.metadata)
        stage("agent_service")
        launch = launch_agent_service(
            worktree,
            port=port,
            expected_sha=expected_sha,
            recorder_url=recorder.metadata["url"],
            run_directory=bundle.path / "agent-launch",
        )
        result["agent_service_pid"] = launch["listener"]["pid"]
        record_path = bundle.path / "agent-launch/launch.json"
        stage("setup")
        system = PilotSystem(computing_unit_id=cuid, agent_service_endpoint=endpoint, output_dir=str(bundle.path))
        task_bundle = AttemptBundle(bundle.path / "setup")
        system.pilot_bundle = task_bundle
        system._setup_agent()
        result["resources"] = system._resources()
        stage("capture")
        capture_setup(system, task_bundle)
        stage("recorder_drain")
        recorder.drain()
        recorder_drained = True
        report = execution_report(
            bundle.path / "execution_requests.jsonl",
            workflow_id=result["resources"]["workflow_id"],
            computing_unit_id=cuid,
            expected_recorder_id=recorder.metadata["instanceId"],
        )
        bundle.write("execution_report.json", report)
        if report["journal_status"] != "closed" or report["requests"] != 0:
            raise RuntimeError("setup-only recorder was not closed and empty")
        stage("resource_cleanup")
        cleanup = cleanup_owned_resources(
            api,
            task_directory=task_bundle.path,
            computing_unit_journal=cu_path,
            workflow_journal=task_bundle.path / "resource_allocations.jsonl",
            execution_journal=bundle.path / "execution_requests.jsonl",
            recorder_id=recorder.metadata["instanceId"],
            service_launch_record=record_path,
            output_directory=bundle.path / "cleanup",
        )
        result["cleanup"] = cleanup
        if cleanup["status"] != "complete":
            raise RuntimeError("owned resource cleanup is unresolved")
        stage("agent_service_stop")
        result["service_stop"] = stop_agent_service(record_path)
        result.update(status="passed", stage="complete")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
    finally:
        if system is not None:
            result["resources"] = system._resources()
        # Only drain our exact recorder child. Never delete resources or stop
        # the agent after an unknown setup/capture/cleanup failure.
        if recorder is not None and not recorder_drained:
            try:
                recorder.drain()
                result["recorder_drained_after_failure"] = True
            except BaseException as error:
                result["recorder_drain_error_type"] = type(error).__name__
        bundle.write("smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--output", required=True, help="Fresh private smoke directory; never an official task attempt")
    parser.add_argument("--port", type=int, default=3011)
    parser.add_argument("--backend", default="http://127.0.0.1:8085")
    args = parser.parse_args()
    result = run_resource_smoke(
        args.worktree,
        expected_sha=args.sha,
        output_directory=args.output,
        port=args.port,
        backend=args.backend,
    )
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

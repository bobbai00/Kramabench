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

"""Source-pinned, agent-only launches through the supported local-dev launcher.

Never adopts an occupied port, resets a deployment pointer, installs packages,
or starts/stops a shared JVM. A launch record is not backend/CU/task/LLM
qualification: the outer pilot guard must bind those independent checks too.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import requests

from utils.pilot_artifacts import AttemptBundle
from utils.pilot_provenance import inspect_service_listener


_SOURCE_PATHS = [
    "agent-service/src",
    "agent-service/package.json",
    "agent-service/yarn.lock",
    "agent-service/bun.lock",
    "agent-service/bun.lockb",
]
_LAUNCHER_PATHS = ["bin/local-dev.sh", "bin/local-dev"]


def _command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("pilot provenance command failed")
    return result.stdout


def _local_origin(value):
    try:
        url = urlsplit(value)
        if (
            url.scheme in {"http", "https"}
            and url.hostname in {"127.0.0.1", "localhost", "::1"}
            and url.port
            and not url.username
            and not url.password
            and not url.query
            and not url.fragment
            and url.path in {"", "/"}
        ):
            return value.rstrip("/")
    except (TypeError, ValueError):
        pass
    raise ValueError("expected a credential-free loopback origin with a port")


def source_snapshot(worktree):
    root = Path(worktree).resolve(strict=True)
    git = lambda args: _command(["git", "-C", str(root), *args])
    if Path(git(["rev-parse", "--show-toplevel"]).strip()).resolve() != root:
        raise RuntimeError("expected a repository root")
    sha = git(["rev-parse", "HEAD"]).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RuntimeError("unresolved source revision")

    def tree_digest(paths):
        names = sorted(name for name in git(["ls-files", "-z", "--", *paths]).split("\0") if name)
        if not names:
            raise RuntimeError("source tree is empty")
        digest = hashlib.sha256()
        for name in names:
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise RuntimeError("runtime source links cannot establish a pinned build")
            digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    return {
        "source_root": str(root),
        "git_sha": sha,
        "source_dirty": bool(
            git(
                [
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    *_SOURCE_PATHS,
                    *_LAUNCHER_PATHS,
                ]
            ).strip()
        ),
        "source_sha256": tree_digest(_SOURCE_PATHS),
        "launcher_sha256": tree_digest(_LAUNCHER_PATHS),
        "dependency_directory": str((root / "agent-service/node_modules").resolve()),
    }


def _listener_free(port):
    return not _command(["ss", "-H", "-ltn", "sport", "=", f":{port}"]).strip()


def _boot_clock():
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "ticks": int(float(Path("/proc/uptime").read_text().split()[0]) * os.sysconf("SC_CLK_TCK")),
    }


def _execution_route_matches(pid, expected):
    # Never export the full process environment. Compare only routing fields;
    # even mismatched values (possibly credential-bearing URLs) are not logged.
    try:
        selected = {}
        for entry in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
            key, _, value = entry.partition(b"=")
            if key in {
                b"WORKFLOW_EXECUTION_SERVICE_ENDPOINT",
                b"EXECUTION_ENDPOINT_TEMPLATE",
            }:
                selected[key] = value.decode()
        return selected.get(b"WORKFLOW_EXECUTION_SERVICE_ENDPOINT", "").rstrip("/") == expected and not selected.get(
            b"EXECUTION_ENDPOINT_TEMPLATE"
        )
    except (OSError, UnicodeError):
        return False


def _healthy(endpoint):
    try:
        return requests.get(endpoint + "/api/healthcheck", timeout=(2, 3), allow_redirects=False).status_code == 200
    except requests.RequestException:
        return False


def _agents_empty(endpoint):
    try:
        response = requests.get(endpoint + "/api/agents/", timeout=(2, 3), allow_redirects=False)
        if response.status_code != 200:
            return False
        payload = response.json()
        return isinstance(payload, dict) and payload.get("agents") == []
    except (requests.RequestException, ValueError):
        return False


def _same_process(pid, start_ticks):
    try:
        suffix = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
        return int(suffix[19]) == start_ticks and suffix[0] != "Z"
    except FileNotFoundError:
        return False
    # Other errors are unknown, not proof of termination.


def _launch_command(arguments, *, environment, worktree, log_path):
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        return subprocess.run(
            arguments,
            cwd=worktree,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=90,
        )


def _validate_listener(listener, source):
    if (
        listener.get("status") != "resolved"
        or listener.get("source_dirty") is not False
        or listener.get("source_root") != source["source_root"]
        or listener.get("git_sha") != source["git_sha"]
    ):
        raise RuntimeError("service source does not match the pinned launch")


def launch_agent_service(worktree, *, port, expected_sha, recorder_url, run_directory):
    if type(port) is not int or not 1024 <= port <= 65535 or port == 3001:
        raise ValueError("use an explicit non-default unprivileged agent port")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", expected_sha):
        raise ValueError("a full pinned source SHA is required")
    recorder_url = _local_origin(recorder_url)
    source = source_snapshot(worktree)
    if source["source_dirty"] is not False or source["git_sha"] != expected_sha:
        raise RuntimeError("pilot source is dirty or does not match the expected revision")
    if not _listener_free(port):
        raise RuntimeError("agent port is occupied; never adopt or restart its listener")
    bundle = AttemptBundle(run_directory)
    state_directory = bundle.path.resolve() / "launcher-state"
    # Fresh state prevents an old deploy-source pointer from choosing a different
    # worktree. No --worktree/--branch/full-stack operation is sent to the launcher.
    state_directory.mkdir(mode=0o700)
    record = {
        "version": 1,
        "status": "launching",
        "loaded_revision_verified": False,
        "source_before": source,
        "service_endpoint": f"http://127.0.0.1:{port}",
        "state_directory": str(state_directory),
        "recorder_url": recorder_url,
        "before": _boot_clock(),
        "listener": None,
        "stage": "launcher",
    }
    bundle.write("launch.json", record)
    environment = dict(os.environ)
    environment.update(
        TEXERA_AGENT_SERVICE_PORT=str(port),
        TEXERA_LOCAL_DEV_DIR=str(state_directory),
        WORKFLOW_EXECUTION_SERVICE_ENDPOINT=recorder_url,
        EXECUTION_ENDPOINT_TEMPLATE="",
    )
    arguments = [
        str(Path(source["source_root"]) / "bin/local-dev.sh"),
        "up",
        "agent-service",
        "--skip-build",
    ]
    try:
        result = _launch_command(
            arguments,
            environment=environment,
            worktree=source["source_root"],
            log_path=bundle.path / "launcher.log",
        )
        record["launcher_exit_code"] = result.returncode
        listener = inspect_service_listener(record["service_endpoint"])
        record["listener"] = listener
        record["source_after"] = source_snapshot(source["source_root"])
        if result.returncode:
            raise RuntimeError("agent launcher failed; inspect its private log")
        record["stage"] = "source_verification"
        _validate_listener(listener, source)
        if record["source_after"] != source:
            raise RuntimeError("source changed during launch")
        if listener.get("process_start_ticks", -1) < record["before"]["ticks"]:
            raise RuntimeError("listener process predates this launch")
        if _boot_clock()["boot_id"] != record["before"]["boot_id"]:
            raise RuntimeError("process boot identity changed")
        record["stage"] = "execution_route"
        if not _execution_route_matches(listener["pid"], recorder_url):
            raise RuntimeError("execution route mismatch or template bypass")
        record["stage"] = "health"
        if not _healthy(record["service_endpoint"]):
            raise RuntimeError("agent health check failed")
        record.update(status="qualified", stage="qualified", loaded_revision_verified=True)
        bundle.write("launch.json", record)
        return record
    except BaseException as error:
        record.update(
            status="failed",
            loaded_revision_verified=False,
            error_type=type(error).__name__,
        )
        if record["listener"] is None:
            record["listener"] = inspect_service_listener(record["service_endpoint"])
        bundle.write("launch.json", record)
        # Leave the observed process identity available for verified cleanup;
        # never run a broad stop/restart after an ambiguous launcher failure.
        raise


def verify_service_launch(record_path):
    with Path(record_path).open() as stream:
        record = json.load(stream)
    if (
        record.get("version") != 1
        or record.get("status") != "qualified"
        or record.get("loaded_revision_verified") is not True
    ):
        raise RuntimeError("service has no qualified source-pinned launch record")
    endpoint = _local_origin(record["service_endpoint"])
    recorder_url = _local_origin(record["recorder_url"])
    source = source_snapshot(record["source_before"]["source_root"])
    if source != record["source_before"] or record["source_after"] != source:
        raise RuntimeError("service source changed since launch")
    if _boot_clock()["boot_id"] != record["before"]["boot_id"]:
        raise RuntimeError("process boot identity changed")
    listener = inspect_service_listener(endpoint)
    _validate_listener(listener, source)
    expected = record["listener"]
    if any(listener.get(key) != expected.get(key) for key in ("pid", "process_start_ticks", "entry_sha256")):
        raise RuntimeError("service process changed since launch")
    if not _execution_route_matches(listener["pid"], recorder_url):
        raise RuntimeError("service execution route changed")
    if not _healthy(endpoint):
        raise RuntimeError("agent health check failed")
    return {
        **listener,
        "loaded_revision_verified": True,
        "service_endpoint": endpoint,
        "recorder_url": recorder_url,
    }


def stop_agent_service(record_path):
    """Stop only the still-matching, empty service from a qualified launch.

    This does not cancel executions or dispose of workflows/computing units.
    The pilot runner must first finish and clean up those owned resources.
    A failed/ambiguous launch is deliberately not an automatic stop target.
    """
    record_path = Path(record_path).resolve(strict=True)
    record = json.loads(record_path.read_text())
    listener = verify_service_launch(record_path)
    endpoint = listener["service_endpoint"]
    port = urlsplit(endpoint).port
    state = record_path.parent / "launcher-state"
    if state.is_symlink() or str(state.resolve(strict=True)) != record["state_directory"] or not state.is_dir():
        raise RuntimeError("launcher state is not the owned launch directory")
    pointer = state / "deploy-source"
    if pointer.exists() and (pointer.is_symlink() or pointer.read_text().strip() != listener["source_root"]):
        raise RuntimeError("launcher state points at another source tree")
    if not _agents_empty(endpoint):
        raise RuntimeError("service still has agents or their absence could not be verified")
    bundle = AttemptBundle(record_path.parent / ("stop-" + uuid.uuid4().hex))
    result = {
        "version": 1,
        "status": "stopping",
        "service_endpoint": endpoint,
        "pid": listener["pid"],
        "process_start_ticks": listener["process_start_ticks"],
        "started_at": time.time(),
        "artifact_directory": str(bundle.path),
    }
    bundle.write("stop.json", result)
    environment = dict(os.environ)
    environment.update(TEXERA_AGENT_SERVICE_PORT=str(port), TEXERA_LOCAL_DEV_DIR=str(state))
    try:
        # Recheck immediately before the supported single-service command.
        verify_service_launch(record_path)
        if not _agents_empty(endpoint):
            raise RuntimeError("service acquired agents during cleanup preflight")
        command = _launch_command(
            [
                str(Path(listener["source_root"]) / "bin/local-dev.sh"),
                "down",
                "agent-service",
            ],
            environment=environment,
            worktree=listener["source_root"],
            log_path=bundle.path / "launcher.log",
        )
        result["launcher_exit_code"] = command.returncode
        deadline = time.monotonic() + 10
        while _same_process(listener["pid"], listener["process_start_ticks"]):
            if time.monotonic() >= deadline:
                raise RuntimeError("owned service termination is unresolved")
            time.sleep(0.1)
        if not _listener_free(port):
            raise RuntimeError("agent port is occupied after shutdown; do not stop its new listener")
        # Actual termination and the command's exit code are separate evidence.
        result.update(status="stopped", finished_at=time.time())
        bundle.write("stop.json", result)
        return result
    except BaseException as error:
        result.update(
            status="unresolved",
            error_type=type(error).__name__,
            finished_at=time.time(),
        )
        bundle.write("stop.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description="Launch, verify, or stop only a source-pinned isolated agent service.")
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--worktree", required=True)
    launch.add_argument("--port", required=True, type=int)
    launch.add_argument("--sha", required=True)
    launch.add_argument("--recorder", required=True)
    launch.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--record", required=True)
    stop = commands.add_parser("stop")
    stop.add_argument("--record", required=True)
    args = parser.parse_args()
    try:
        if args.action == "launch":
            result = launch_agent_service(
                args.worktree,
                port=args.port,
                expected_sha=args.sha,
                recorder_url=args.recorder,
                run_directory=args.output,
            )
        elif args.action == "verify":
            result = verify_service_launch(args.record)
        else:
            result = stop_agent_service(args.record)
        print(json.dumps(result))
        return 0
    except Exception as error:
        # Do not print raw process errors, environment or argument values.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

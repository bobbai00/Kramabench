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

"""Read-only local listener/source resolution. Never guess a worktree by port.

This resolves on-disk source associated with a live Linux listener. It does
NOT prove that a previously started/non-watching process loaded today's HEAD;
the runner also needs a source-pinned launch record and before/after checks.
No process environment, credentials or full command lines enter the result.
"""

import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


def inspect_service_listener(endpoint, *, proc_root=Path("/proc"), run=subprocess.run):
    unknown = {
        "status": "unavailable",
        "reason": "unsupported_endpoint",
        "git_sha": None,
        "source_dirty": None,
        "loaded_revision_verified": False,
    }
    try:
        url = urlsplit(endpoint)
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or not url.port
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            return unknown

        def command(args):
            result = run(args, capture_output=True, text=True, timeout=5)
            if result.returncode:
                raise RuntimeError("command_failed")
            return result.stdout.strip()

        listeners = command(["ss", "-H", "-ltnp", "sport", "=", f":{url.port}"])
        pids = {int(pid) for pid in re.findall(r"\bpid=(\d+)", listeners)}
        if len(pids) != 1:
            return {**unknown, "reason": "no_unique_listener"}
        pid = pids.pop()
        process = Path(proc_root) / str(pid)

        def start_ticks():
            # comm may contain spaces/parentheses; fields after its last ')' are
            # stable Linux stat fields (state=3, starttime=22).
            suffix = (process / "stat").read_text().rpartition(")")[2].split()
            return int(suffix[19])

        before = start_ticks()
        cwd = (process / "cwd").resolve(strict=True)
        args = (process / "cmdline").read_bytes().split(b"\0")
        candidates = []
        for raw in args:
            arg = raw.decode("utf-8", errors="replace")
            if arg == "src/server.ts" or arg.endswith("/src/server.ts"):
                path = Path(arg)
                candidates.append((path if path.is_absolute() else cwd / path).resolve(strict=True))
        if len(candidates) != 1:
            return {**unknown, "reason": "not_an_agent_entry_point"}
        entry = candidates[0]
        root = Path(command(["git", "-C", str(entry.parent), "rev-parse", "--show-toplevel"])).resolve(strict=True)
        if entry != root / "agent-service" / "src" / "server.ts":
            return {**unknown, "reason": "entry_point_outside_agent_source"}
        sha = command(["git", "-C", str(root), "rev-parse", "HEAD"])
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            return {**unknown, "reason": "invalid_source_revision"}
        # Include runtime templates and dependency declarations, not local
        # node_modules/venv links or unrelated user edits elsewhere in the tree.
        dirty = bool(
            command(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                    "--",
                    "agent-service/src",
                    "agent-service/package.json",
                    "agent-service/yarn.lock",
                    "agent-service/bun.lock",
                    "agent-service/bun.lockb",
                ]
            )
        )
        digest = hashlib.sha256(entry.read_bytes()).hexdigest()
        if start_ticks() != before:
            return {**unknown, "reason": "process_changed"}
        return {
            "status": "resolved",
            "reason": None,
            "pid": pid,
            "process_start_ticks": before,
            "source_root": str(root),
            "entry_point": str(entry),
            "entry_sha256": digest,
            "git_sha": sha,
            "source_dirty": dirty,
            "loaded_revision_verified": False,
        }
    except Exception:
        return {**unknown, "reason": "resolution_failed"}

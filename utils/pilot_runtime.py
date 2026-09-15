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

"""Build and bind the local pilot engine; never launch/restart a shared JVM.

Build identity and functional qualification are separate. A matching fresh
master, classpath, worker source and interpreter do not replace actual worker
and model tests. No complete argv, environment or worker startup payload is
exported: those can contain database, storage and model credentials.
"""

import hashlib
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from utils.pilot_artifacts import AttemptBundle
from utils.pilot_inputs import file_identity
from utils.pilot_launch import _boot_clock, _command, _local_origin


BUILD_COMMAND = ["sbt", "-no-colors", "WorkflowExecutionService/dist"]
MASTER_CLASS = "org.apache.texera.web.ComputingUnitMaster"


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _digest_files(root, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        _require(not path.is_symlink() and path.resolve().is_relative_to(root), "engine_source_link")
        identity = file_identity(path)
        digest.update(name.encode() + b"\0" + identity["sha256"].encode())
    return digest.hexdigest()


def engine_source(worktree, expected_sha=None):
    root = Path(worktree).resolve(strict=True)
    git = lambda args: _command(["git", "-C", str(root), *args])
    _require(Path(git(["rev-parse", "--show-toplevel"]).strip()).resolve() == root, "engine_source_not_root")
    sha = git(["rev-parse", "HEAD"]).strip()
    _require(
        re.fullmatch(r"[a-f0-9]{40}", sha) and (expected_sha is None or sha == expected_sha),
        "engine_source_revision_mismatch",
    )
    paths = [
        "amber/src/main",
        "amber/build.sbt",
        "common",
        "build.sbt",
        "project",
        ".jvmopts",
        "pyproject.toml",
        "uv.lock",
        "bin/python-proto-gen.sh",
    ]
    _require(not git(["status", "--porcelain", "--untracked-files=all", "--", *paths]).strip(), "engine_source_dirty")
    names = [name for name in git(["ls-files", "-z", "--", *paths]).split("\0") if name]
    _require(names, "engine_source_empty")
    proto_root = root / "amber/src/main/python/proto"
    protos = [str(path.relative_to(root)) for path in proto_root.rglob("*.py") if path.is_file()]
    _require(len(protos) > 1, "engine_python_protos_missing")
    return {
        "source_root": str(root),
        "git_sha": sha,
        "source_sha256": _digest_files(root, names),
        "proto_sha256": _digest_files(root, protos),
    }


def python_runtime(path):
    """Read interpreter/prefix and installed distribution versions, not a worker.

    No package installation or network access. Distribution versions are not
    immutable hashes of every dependency file; matching worker E2E remains a
    separate gate. A named per-operator PVE must be rejected independently.
    """
    path = Path(path).absolute()
    _require(path.is_file() and os.access(path, os.X_OK), "engine_python_unavailable")
    script = """import hashlib, importlib.metadata, json, sys
packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
print(json.dumps({'version': sys.version, 'prefix': sys.prefix,
 'packages_sha256': hashlib.sha256(json.dumps(packages).encode()).hexdigest()}))
"""
    completed = subprocess.run([str(path), "-I", "-c", script], capture_output=True, text=True, timeout=30)
    _require(completed.returncode == 0, "engine_python_probe_failed")
    value = json.loads(completed.stdout)
    return {"path": str(path), "executable": file_identity(path), **value}


def archive_jars(path):
    """Fingerprint the actual packaged classpath without extracting/overwriting."""
    entries, roots = {}, set()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if entry.is_dir() or not entry.filename.endswith(".jar"):
                continue
            _require(len(parts) == 3 and parts[1] == "lib" and ".." not in parts, "invalid_engine_archive_layout")
            name = parts[-1]
            _require(name not in entries and 0 < entry.file_size <= 256 * 1024 * 1024, "invalid_engine_archive_jar")
            roots.add(parts[0])
            digest = hashlib.sha256()
            with archive.open(entry) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries[name] = {"name": name, "bytes": entry.file_size, "sha256": digest.hexdigest()}
    _require(len(roots) == 1 and entries and len(entries) <= 2000, "invalid_engine_archive_classpath")
    _require(any(name.startswith("org.apache.texera.amber-") for name in entries), "engine_archive_missing_master")
    return [entries[name] for name in sorted(entries)]


def _process_ticks(process):
    fields = (process / "stat").read_text().rpartition(")")[2].split()
    _require(fields[0] != "Z", "engine_process_exited")
    return int(fields[19])


def inspect_engine_process(endpoint, *, proc_root=Path("/proc")):
    endpoint = _local_origin(endpoint)
    listeners = _command(["ss", "-H", "-ltnp", "sport", "=", f":{urlsplit(endpoint).port}"])
    pids = {int(value) for value in re.findall(r"\bpid=(\d+)", listeners)}
    _require(len(pids) == 1, "engine_listener_not_unique")
    pid = pids.pop()
    proc_root = Path(proc_root)
    process = proc_root / str(pid)
    ticks = _process_ticks(process)
    cwd = (process / "cwd").resolve(strict=True)
    argv_bytes = (process / "cmdline").read_bytes()
    args = [value.decode() for value in argv_bytes.split(b"\0") if value]
    _require(args.count(MASTER_CLASS) == 1, "engine_listener_not_master")
    main_index = args.index(MASTER_CLASS)
    _require(args[main_index + 1 :] in ([], ["--cluster", "false"]), "engine_cluster_mode_not_local")
    indexes = [index for index, value in enumerate(args[:main_index]) if value in {"-cp", "-classpath"}]
    _require(len(indexes) == 1 and indexes[0] + 1 < main_index, "engine_classpath_missing")
    classpath = args[indexes[0] + 1].split(os.pathsep)
    _require(classpath and len(classpath) <= 2000 and len(set(classpath)) == len(classpath), "engine_classpath_invalid")
    _require(
        not any(
            value.startswith(("-javaagent", "-agentlib", "-agentpath", "-Xbootclasspath", "--module-path"))
            for value in args[:main_index]
        ),
        "engine_code_injection_unsupported",
    )
    properties = dict(value[2:].partition("=")[::2] for value in args[:main_index] if value.startswith("-D"))
    _require(
        not any(
            key in properties
            for key in ("config.file", "config.resource", "config.url", "user.dir", "java.system.class.loader")
        ),
        "engine_config_override_unsupported",
    )
    environment_bytes = (process / "environ").read_bytes()
    environment = dict(value.partition(b"=")[::2] for value in environment_bytes.split(b"\0") if b"=" in value)
    _require(
        not any(
            environment.get(key)
            for key in (b"JAVA_TOOL_OPTIONS", b"JDK_JAVA_OPTIONS", b"_JAVA_OPTIONS", b"PYTHONPATH", b"PYTHONHOME")
        ),
        "engine_injected_environment_unsupported",
    )
    python = properties.get("UDF_PYTHON_PATH", environment.get(b"UDF_PYTHON_PATH", b"").decode())
    _require(python and Path(python).is_absolute(), "engine_python_path_not_explicit")
    jars = []
    for value in classpath:
        path = Path(value)
        if not path.is_absolute():
            path = cwd / path
        _require(path.is_file() and path.suffix == ".jar", "engine_classpath_not_jar")
        identity = file_identity(path)
        jars.append({"name": path.name, **identity})
    _require(
        _process_ticks(process) == ticks
        and (process / "cwd").resolve() == cwd
        and (process / "cmdline").read_bytes() == argv_bytes
        and (process / "environ").read_bytes() == environment_bytes,
        "engine_process_changed",
    )
    return {
        "version": 1,
        "endpoint": endpoint,
        "pid": pid,
        "start_ticks": ticks,
        "boot_id": (proc_root / "sys/kernel/random/boot_id").read_text().strip(),
        "cwd": str(cwd),
        "java": file_identity(process / "exe"),
        "python_path": python,
        "classpath": jars,
        "argv_sha256": hashlib.sha256(argv_bytes).hexdigest(),
        "environment_sha256": hashlib.sha256(environment_bytes).hexdigest(),
    }


def verify_engine_runtime(build, endpoint, *, expected=None, proc_root=Path("/proc")):
    _require(
        isinstance(build, dict)
        and build.get("version") == 1
        and build.get("status") == "built"
        and build.get("command") == BUILD_COMMAND
        and type(build.get("exit_code")) is int
        and build["exit_code"] == 0
        and isinstance(build.get("source_before"), dict)
        and build.get("source_before") == build.get("source_after")
        and isinstance(build.get("completed_clock"), dict)
        and isinstance(build.get("archive"), dict)
        and isinstance(build.get("jars"), list)
        and isinstance(build.get("python"), dict),
        "engine_build_evidence_incomplete",
    )
    source = build["source_before"]
    current_source = engine_source(source["source_root"])
    # A later docs/checkpoint commit does not change the built engine. Preserve
    # the actual build SHA and compare every runtime-source/proto field; the
    # separately pinned agent/harness manifest records their current heads.
    _require(
        {key: value for key, value in current_source.items() if key != "git_sha"}
        == {key: value for key, value in source.items() if key != "git_sha"},
        "engine_source_changed",
    )
    _require(file_identity(build["archive"]["resolved_path"]) == build["archive"], "engine_archive_changed")
    _require(archive_jars(build["archive"]["resolved_path"]) == build["jars"], "engine_archive_classpath_changed")
    observed = inspect_engine_process(endpoint, proc_root=proc_root)
    _require(observed["cwd"] == str(Path(source["source_root"]) / "amber"), "engine_working_directory_mismatch")
    _require(
        observed["boot_id"] == build["completed_clock"]["boot_id"]
        and observed["start_ticks"] >= build["completed_clock"]["ticks"],
        "engine_process_predates_build",
    )
    _require(
        observed["python_path"] == build["python"]["path"]
        and python_runtime(observed["python_path"]) == build["python"],
        "engine_python_changed",
    )
    jars = [{key: item[key] for key in ("name", "bytes", "sha256")} for item in observed["classpath"]]
    _require(sorted(jars, key=lambda item: item["name"]) == build["jars"], "engine_classpath_build_mismatch")
    _require(
        all(
            Path(item["resolved_path"]).is_relative_to(Path(observed["cwd"]) / "target")
            for item in observed["classpath"]
        ),
        "engine_classpath_outside_candidate",
    )
    bound = {
        **observed,
        "source": source,
        "archive": build["archive"],
        "python": build["python"],
        "build_bound": True,
        "live_correctness_qualified": False,
    }
    _require(expected is None or expected == bound, "bound_engine_runtime_changed")
    return bound


def build_engine(worktree, *, expected_sha, output_directory):
    """Compile/package only into an idle candidate worktree, never deploy it.

    Caller must use a fresh process and keep it alive until the build returns.
    Source and generated Python protos are bound before/after sbt. Regenerate
    protos using the supported script separately if absent or changed. This
    function neither unpacks the distribution nor modifies a live master.
    """
    bundle = AttemptBundle(Path(output_directory).resolve())
    source = engine_source(worktree, expected_sha)
    root = Path(source["source_root"])
    # No worker-source regeneration here. Still refuse any local master using
    # this worktree; it prevents accidentally qualifying an in-place rebuild.
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if MASTER_CLASS.encode() in args:
                _require((process / "cwd").resolve() != root / "amber", "candidate_engine_already_live")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    version = re.search(r'ThisBuild\s*/\s*version\s*:=\s*"([A-Za-z0-9.-]+)"', (root / "build.sbt").read_text())
    _require(version, "engine_distribution_version_unresolved")
    archive_path = root / "amber/target/universal" / ("amber-" + version[1] + ".zip")
    record = {
        "version": 1,
        "status": "building",
        "command": BUILD_COMMAND,
        "source_before": source,
        "started_clock": _boot_clock(),
        "exit_code": None,
    }
    bundle.write("build.json", record)
    try:
        record["python"] = python_runtime(root / ".venv/bin/python")
        descriptor = os.open(bundle.path / "build.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            result = subprocess.run(
                BUILD_COMMAND, cwd=root, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=1800
            )
        record["exit_code"] = result.returncode
        _require(result.returncode == 0, "engine_build_failed")
        record["source_after"] = engine_source(root, expected_sha)
        _require(record["source_after"] == source, "engine_source_changed_during_build")
        _require(python_runtime(root / ".venv/bin/python") == record["python"], "engine_python_changed_during_build")
        record["archive"] = file_identity(archive_path)
        record["jars"] = archive_jars(archive_path)
        _require(file_identity(archive_path) == record["archive"], "engine_archive_changed_during_read")
        record.update(status="built", completed_clock=_boot_clock())
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        bundle.write("build.json", record)
        raise
    bundle.write("build.json", record)
    return record

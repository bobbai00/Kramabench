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

"""Read-only frozen pilot inputs, not service/engine/model qualification.

Bind the actual executor workload, in-memory task, ordered prompt paths, file
bytes at both harness and worker locations, prompt and frozen harness pricing.
Gold answers, raw rows and prompt text are not copied into the manifest. A
successful check here cannot authorize a model dispatch without the remaining
source/runtime/resource and prior-correctness qualification gates.
"""

import hashlib
import json
import math
from pathlib import Path

from systems.cost_utils import price_schedule
from systems.native_pilot_system import PILOT_TASK, build_pilot_prompt


PILOT_DATA_FILES = ("water-body-testing-2012.csv", "water-body-testing-2013.csv")


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_identity(path):
    """Hash actual bytes; detect path/stat changes across a streaming read."""
    path = Path(path)
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), "input_not_regular_file")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    _require(
        path.resolve(strict=True) == resolved
        and all(
            getattr(before, field) == getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "input_changed_while_hashing",
    )
    return {"resolved_path": str(resolved), "bytes": after.st_size, "sha256": digest.hexdigest()}


def _pricing(model):
    pricing = price_schedule(model)
    _require(isinstance(pricing, dict), "pilot_price_unavailable")
    rates = pricing.get("rates")
    _require(
        pricing.get("version") == 1
        and pricing.get("kind") == "token_estimate"
        and pricing.get("model") == model
        and pricing.get("currency") == "USD"
        and pricing.get("unit") == "per_token"
        and pricing.get("input_includes_cached") is True
        and pricing.get("output_includes_reasoning") is True
        and isinstance(pricing.get("source"), dict)
        and isinstance(rates, dict)
        and set(rates) == {"input", "output", "cache_read", "cache_creation"}
        and all(type(value) in {int, float} and math.isfinite(value) and value >= 0 for value in rates.values())
        and rates["input"] > 0
        and rates["output"] > 0,
        "pilot_price_unavailable",
    )
    return pricing


def freeze_pilot_inputs(system, *, workload_path, execution_directory):
    """Call after read-only dataset preparation, from the actual harness CWD.

    execution_directory must be the worker process's actual working directory,
    checked separately by runtime qualification. Multiple copies of a file are
    allowed only when their bytes match; every resolved location is frozen.
    """
    _require(system.model_type in {"gpt-5.6-luna", "gpt-5.6-terra"}, "pilot_model_changed")
    cwd = Path.cwd().resolve(strict=True)
    execution_directory = Path(execution_directory).resolve(strict=True)
    dataset = Path(system.dataset_directory).resolve(strict=True)
    workload_path = Path(workload_path).absolute()
    workload = file_identity(workload_path)
    _require(workload["bytes"] <= 8 * 1024 * 1024, "pilot_workload_too_large")
    tasks = json.loads(workload_path.read_text(encoding="utf-8"))
    _require(isinstance(tasks, list), "pilot_workload_not_list")
    selected = [task for task in tasks if isinstance(task, dict) and task.get("id") == PILOT_TASK]
    _require(len(selected) == 1 and selected[0].get("answer_type") == "numeric_exact", "pilot_task_mismatch")
    task = selected[0]
    _require(task == system.workload_data.get(PILOT_TASK), "pilot_task_mismatch")
    _require(task.get("data_sources") == list(PILOT_DATA_FILES), "pilot_source_policy_changed")
    _require(isinstance(task.get("query"), str) and task["query"].strip(), "pilot_task_mismatch")
    paths, prompt = build_pilot_prompt(system)
    _require(len(paths) == 2 and len(set(paths)) == 2 and isinstance(prompt, str), "pilot_prompt_files_mismatch")
    files = []
    for prompt_path, name in zip(paths, PILOT_DATA_FILES):
        relative = Path(prompt_path)
        _require(
            not relative.is_absolute() and ".." not in relative.parts and relative.name == name,
            "pilot_prompt_path_invalid",
        )
        harness = file_identity(cwd / relative)
        _require(harness == file_identity(dataset / name), "pilot_prompt_files_mismatch")
        worker = file_identity(execution_directory / relative)
        _require(worker["sha256"] == harness["sha256"] and worker["bytes"] == harness["bytes"], "worker_data_mismatch")
        files.append({"prompt_path": prompt_path, "harness": harness, "worker": worker})
    _require(workload == file_identity(workload_path), "input_changed_while_hashing")
    return {
        "version": 1,
        "kind": "native-python-pilot-inputs",
        "task_id": PILOT_TASK,
        "model": system.model_type,
        "harness_directory": str(cwd),
        "dataset_directory": str(dataset),
        "execution_directory": str(execution_directory),
        "workload_path": str(workload_path),
        "workload": workload,
        "task_sha256": _digest(task),
        "files": files,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_utf8_bytes": len(prompt.encode()),
        "pricing": _pricing(system.model_type),
        "runtime_qualified": False,
    }


def verify_pilot_inputs(system, manifest):
    _require(
        isinstance(manifest, dict)
        and manifest.get("version") == 1
        and manifest.get("kind") == "native-python-pilot-inputs",
        "invalid_pilot_inputs_manifest",
    )
    _require(str(Path.cwd().resolve()) == manifest.get("harness_directory"), "pilot_working_directory_changed")
    observed = freeze_pilot_inputs(
        system,
        workload_path=manifest["workload_path"],
        execution_directory=manifest["execution_directory"],
    )
    _require(observed == manifest, "pilot_inputs_changed")
    return {"inputs_verified": True, "manifest_sha256": _digest(manifest), "runtime_qualified": False}

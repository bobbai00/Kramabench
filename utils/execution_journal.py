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

"""Read-only, scoped denominators from the parent's durable execution journal.

One admission/start is one possible backend request, not one operator, batch,
or proven execution. A crash between the fsynced start and forwarding cannot
establish whether the backend received it. Missing finishes stay unknown.
The caller must independently qualify the recorder/source and bind its ID;
matching an ID in a file is not service provenance or termination evidence.
"""

import json
import math
import re
from pathlib import Path


_OUTCOMES = ("passed", "failed", "not_attempted", "unknown")
_BASE_KEYS = (
    "version",
    "requestId",
    "workflowId",
    "computingUnitId",
    "startedAt",
    "operatorCount",
    "targetOperatorIds",
    "structuredResults",
    "requestSummaryTruncated",
)


def _positive_int(value):
    return type(value) is int and 0 < value <= 9007199254740991


def _nonnegative(value):
    try:
        return type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _identity(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", value))


def _rates(values, total):
    counts = {outcome: values.count(outcome) for outcome in _OUTCOMES}
    denominator = counts["passed"] + counts["failed"]
    return {
        **counts,
        "known_attempted": denominator,
        "pass_rate": counts["passed"] / denominator if denominator else None,
        "known_coverage": (total - counts["unknown"]) / total if total else None,
    }


def _measured_phases(record):
    """Same measured-phase policy as the recorder; never trust display labels.

    Recheck the version/consistency before accepting persisted values. Python
    callback preflight and runtime-start RPC success are not compiler passes.
    """
    unknown = ("unknown", "unknown", None)
    if (
        record.get("transport") != "response"
        or type(record.get("httpStatus")) is not int
        or not 200 <= record["httpStatus"] < 300
    ):
        return unknown
    phase = record.get("phaseMetrics")
    if not isinstance(phase, dict) or record.get("phaseEvidence") != "recorded":
        return unknown
    compilation, runtime = phase.get("compilation"), phase.get("runtimeStartAttempted")
    duration = phase.get("compilationDurationMs")
    terminal = compilation in {"passed", "failed"} if isinstance(compilation, str) else False
    if (
        type(phase.get("version")) is not int
        or phase["version"] != 1
        or compilation not in ("passed", "failed", "not_attempted", "running")
        or type(runtime) is not bool
        or (terminal and not _nonnegative(duration))
        or (not terminal and duration is not None)
        or (runtime and compilation != "passed")
        or (record.get("success") is True and not runtime)
    ):
        return unknown
    execution = "unknown"
    if not runtime:
        execution = "not_attempted"
    elif record.get("success") is True and record.get("state") in (
        "Completed",
        "CompletedFromCache",
    ):
        execution = "passed"
    elif record.get("success") is False and record.get("state") in (
        "Failed",
        "Killed",
        "Terminated",
        "Completed",
    ):
        execution = "failed"
    return "unknown" if compilation == "running" else compilation, execution, duration


def execution_report(journal_path, *, workflow_id, computing_unit_id, expected_recorder_id):
    if not _positive_int(workflow_id) or not _positive_int(computing_unit_id) or not _identity(expected_recorder_id):
        raise ValueError("explicit workflow, computing unit and recorder identities are required")
    report = {
        "version": 1,
        "journal_status": "missing",
        "requests": None,
        "workflow_id": workflow_id,
        "computing_unit_id": computing_unit_id,
        "recorder_id": expected_recorder_id,
        "other_requests": 0,
        "finished_requests": 0,
        "unfinished_requests": 0,
        "duplicate_events": 0,
        "compilation": _rates([], 0),
        "runtime": _rates([], 0),
        "request_duration_ms_observed": None,
        "compilation_duration_ms_observed": None,
        "engine_execution_ids": [],
        "issues": [],
        "backend_termination_verified": False,
    }
    if journal_path is None or not Path(journal_path).exists():
        return report
    path = Path(journal_path)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        return {
            **report,
            "journal_status": "invalid",
            "issues": ["invalid_journal_size_or_type"],
        }
    header, footer = None, None
    starts, finishes, conflicts = {}, {}, set()
    issues = set()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except (ValueError, RecursionError):
                issues.add("malformed_event")
                continue
            if not isinstance(event, dict) or type(event.get("version")) is not int or event["version"] != 1:
                issues.add("invalid_event_version")
                continue
            if event.get("instanceId") != expected_recorder_id:
                # Never adopt another journal and relabel it as this attempt.
                return {
                    **report,
                    "journal_status": "invalid",
                    "issues": ["recorder_identity_mismatch"],
                }
            kind = event.get("kind")
            if kind == "recorder_started":
                units = event.get("computingUnitIds")
                if (
                    header is not None
                    or starts
                    or finishes
                    or not isinstance(units, list)
                    or not units
                    or any(not _positive_int(unit) for unit in units)
                ):
                    issues.add("invalid_header")
                else:
                    header = event
                continue
            if header is None:
                issues.add("missing_header")
                continue
            if footer is not None:
                issues.add("event_after_close")
            if kind == "recorder_closed":
                footer = event
                continue
            if kind not in ("request_started", "request_finished"):
                issues.add("unknown_event_kind")
                continue
            record = event.get("record")
            if (
                not isinstance(record, dict)
                or type(record.get("version")) is not int
                or record["version"] != 1
                or not _identity(record.get("requestId"))
                or not _positive_int(record.get("workflowId"))
                or not _positive_int(record.get("computingUnitId"))
                or record["computingUnitId"] not in header["computingUnitIds"]
            ):
                issues.add("invalid_request_identity")
                continue
            identifier = record["requestId"]
            destination = starts if kind == "request_started" else finishes
            if identifier in destination:
                if destination[identifier] == record:
                    report["duplicate_events"] += 1
                else:
                    conflicts.add(identifier)
                    issues.add("conflicting_event")
            else:
                destination[identifier] = record
            if kind == "request_finished" and identifier not in starts:
                conflicts.add(identifier)
                issues.add("finish_without_start")
    if header is None:
        return {
            **report,
            "journal_status": "invalid",
            "issues": sorted(issues | {"missing_header"}),
        }
    if computing_unit_id not in header["computingUnitIds"]:
        return {
            **report,
            "journal_status": "invalid",
            "issues": ["unregistered_computing_unit"],
        }
    if footer is not None and (
        type(footer.get("completed")) is not int
        or footer["completed"] != len(finishes)
        or type(footer.get("pending")) is not int
        or footer["pending"] != 0
        or footer.get("persistenceFailed") is not False
        or set(starts) != set(finishes)
    ):
        issues.add("inconsistent_footer")
    compilation, runtime, elapsed, compilation_ms, execution_ids = [], [], [], [], set()
    for identifier in starts.keys() | finishes.keys():
        base = starts.get(identifier, finishes.get(identifier))
        finish = finishes.get(identifier)
        if finish is not None and any(base.get(key) != finish.get(key) for key in _BASE_KEYS):
            conflicts.add(identifier)
            issues.add("request_identity_changed")
        if not any(
            record and record["workflowId"] == workflow_id and record["computingUnitId"] == computing_unit_id
            for record in (base, finish)
        ):
            report["other_requests"] += 1
            continue
        if finish is None:
            report["unfinished_requests"] += 1
        else:
            report["finished_requests"] += 1
        if identifier in conflicts or finish is None:
            compilation.append("unknown")
            runtime.append("unknown")
            continue
        compile_outcome, runtime_outcome, measured_ms = _measured_phases(finish)
        compilation.append(compile_outcome)
        runtime.append(runtime_outcome)
        if measured_ms is not None:
            compilation_ms.append(measured_ms)
        if _nonnegative(finish.get("elapsedMs")):
            elapsed.append(finish["elapsedMs"])
        execution_id = finish.get("executionId")
        if isinstance(execution_id, str) and re.fullmatch(r"\d{1,30}", execution_id):
            execution_ids.add(execution_id)
    total = len(compilation)
    report.update(
        journal_status="invalid" if issues else "closed" if footer is not None else "open",
        requests=total,
        issues=sorted(issues),
        compilation=_rates(compilation, total),
        runtime=_rates(runtime, total),
        request_duration_ms_observed=sum(elapsed) if elapsed or total == 0 else None,
        compilation_duration_ms_observed=sum(compilation_ms) if compilation_ms or total == 0 else None,
        engine_execution_ids=sorted(execution_ids),
    )
    return report

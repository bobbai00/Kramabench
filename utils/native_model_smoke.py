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

"""Real-model smoke protocol and input-timed audit, not a benchmark task.

The caller supplies an already-owned fresh V2 session and a mandatory live
qualification callback. This module neither launches services nor allocates,
deletes, retries, or stops anything. Run separately for DELTA and LATEST after
matching worker/engine qualification. Synthetic audit tests are not live proof.
"""

import json
import re
from pathlib import Path

from systems.native_pilot_system import public_info
from utils.pilot_artifacts import AttemptBundle


LOAD_CODE = """def load():
    import pandas as pd
    return pd.DataFrame({'x': [1, 2, 3], 'note': ['HIDDEN_' + str(1000000 + i) for i in (1, 2, 3)]})"""

SMOKE_CASES = (
    (
        "chain",
        "This is a Python-native integration test. In ONE dataflow batch, create source (udf), filtered (filter "
        "of source, declaring columns ['x'], keeping x >= 2), and answer (project of filtered, keeping only x). "
        "Put answer before filtered before source in the operators array to exercise forward references. "
        "Use the following source code exactly, without simplifying its expressions:\n" + LOAD_CODE + "\n"
        "Observe only answer, never source or filtered in this turn. Use no other operators or tools. "
        'After checking its actual result, end with just JSON {"values": [the observed x values]}.',
    ),
    (
        "cached",
        "Now retrieve source's cached content explicitly with dataflow operators=[] and observe=['source']. "
        "Make no edits and use no other tools. Summarize the observed note values.",
    ),
    (
        "failure",
        "Now deliberately test hidden-error reporting. In one batch update only filtered, retaining input source "
        "and columns ['x'], to this function:\ndef apply(row):\n    raise ValueError('SMOKE_CALLBACK_FAILURE')\n"
        "Set observe=[] so no successful table content is requested. Do not repair it yet. After the failure, "
        "use inspectError for filtered, then explain the reported user-function line. No other tools or edits.",
    ),
    (
        "repair",
        "Repair filtered to keep x >= 3, and add extended (project of the existing answer, keeping only x) "
        "in the SAME dataflow batch. Put extended before filtered to test a new consumer alongside an upstream "
        "edit. Do not replace source or answer. Observe only extended; verify the changed downstream output. "
        'Use no other tools. End with just JSON {"values": [the observed x values]}.',
    ),
)


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _text(value, depth=0):
    """Read actual input content, including JSON/SDK parts, without rebuilding context."""
    if depth > 16:
        return ""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return value
        return _text(decoded, depth + 1) if isinstance(decoded, (dict, list)) else value
    if isinstance(value, dict):
        return "\n".join(_text(item, depth + 1) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_text(item, depth + 1) for item in value)
    return ""


def _input(step):
    return "\n".join(
        _text(message.get("content"))
        for message in step.get("inputMessages", [])
        if isinstance(message, dict) and message.get("role") in {"user", "tool"}
    )


def _normalized(value):
    return " ".join(value.split())


def _rows(head, operator_id, expected):
    info = head.get("results", {}).get(operator_id, {})
    _require(info.get("sampleRecords") == expected and not info.get("nativeError"), "executed_rows_mismatch")
    version = info.get("resultVersion")
    _require(
        isinstance(version, dict)
        and version.get("complete") is True
        and isinstance(version.get("id"), str)
        and re.fullmatch(r"[a-f0-9]{64}", version["id"]),
        "result_binding_missing",
    )
    return version["id"]


def _seen_observation(operator_id, calls, steps, snapshots, head):
    for index, call in calls:
        if call["toolName"] != "dataflow" or operator_id not in call["input"].get("observe", []):
            continue
        snapshot = snapshots.get(steps[index]["id"], {})
        frame = snapshot.get("nativeObservations", {}).get(operator_id, {})
        text = frame.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        version = head.get("results", {}).get(operator_id, {}).get("resultVersion", {}).get("id")
        if not version or frame.get("resultVersion") != version:
            continue
        for after in steps[index + 1 :]:
            if _normalized(text) in _normalized(_input(after)):
                return {
                    "operator_id": operator_id,
                    "tool_step_id": steps[index]["id"],
                    "input_step_id": after["id"],
                    "result_version": version,
                }
    raise ValueError("not_seen_after_observe")


def audit_model_smoke_turn(case, trace, snapshots, head, *, before=None):
    """Require executed results AND later real inputs, not a snapshot-only pass.

    trace contains only this turn's new steps in service order. snapshots map
    those IDs to actual REST snapshots. Caller must separately establish live
    source/engine provenance and completion; arbitrary supplied JSON is not it.
    """
    _require(case in {name for name, _ in SMOKE_CASES}, "unknown_smoke_case")
    _require(isinstance(trace, list) and all(isinstance(step, dict) for step in trace), "invalid_trace")
    steps = [step for step in trace if step.get("role") == "agent"]
    _require(steps and all(isinstance(step.get("id"), str) for step in steps), "missing_model_steps")
    _require(len({step["id"] for step in steps}) == len(steps), "duplicate_step_identity")
    _require(
        all(isinstance(step.get("inputMessages"), list) and step["inputMessages"] for step in steps),
        "missing_model_input",
    )
    _require(steps[-1].get("isEnd") is True and not steps[-1].get("toolCalls"), "no_model_final_step")
    calls = [(index, call) for index, step in enumerate(steps) for call in step.get("toolCalls", [])]
    _require(
        all(isinstance(call, dict) and isinstance(call.get("input"), dict) for _, call in calls), "invalid_tool_call"
    )
    allowed = {"dataflow", "inspectError"} if case == "failure" else {"dataflow"}
    _require(calls and all(call.get("toolName") in allowed for _, call in calls), "unexpected_tool")
    batches = [(i, call["input"]) for i, call in calls if call["toolName"] == "dataflow"]
    _require(len(batches) == 1, "expected_one_batch")
    index, batch = batches[0]
    operators, observed = batch.get("operators"), batch.get("observe", [])
    _require(isinstance(operators, list) and all(isinstance(op, dict) for op in operators), "invalid_batch")
    _require(isinstance(observed, list), "invalid_observe")
    report = {"case": case, "model_steps": len(steps), "observations": []}

    if case == "chain":
        _require([op.get("id") for op in operators] == ["answer", "filtered", "source"], "dependent_batch_missing")
        answer, filtered, source = operators
        _require(
            answer.get("op") == "project"
            and answer.get("input") == "filtered"
            and answer.get("keep") == ["x"]
            and filtered.get("op") == "filter"
            and filtered.get("input") == "source"
            and filtered.get("columns") == ["x"]
            and source.get("op") == "udf"
            and source.get("code", "").strip() == LOAD_CODE
            and observed == ["answer"],
            "dependent_batch_contract_mismatch",
        )
        _rows(head, "answer", [{"x": 2}, {"x": 3}])
        _require(not any(re.search(r"HIDDEN_100000[123]", _input(step)) for step in steps), "hidden_source_row")
        report["observations"].append(_seen_observation("answer", calls, steps, snapshots, head))
    elif case == "cached":
        _require(operators == [] and observed == ["source"], "cached_observe_contract_mismatch")
        _require(
            not any(re.search(r"HIDDEN_100000[123]", _input(step)) for step in steps[: index + 1]),
            "source_seen_before_cached_observe",
        )
        old = (before or {}).get("results", {}).get("source", {})
        current = head.get("results", {}).get("source", {})
        _require(
            old.get("resultVersion") is not None
            and old.get("materialization") is not None
            and current.get("resultVersion") == old["resultVersion"]
            and current.get("materialization") == old["materialization"]
            and current.get("sampleRecords") == old.get("sampleRecords"),
            "cached_binding_changed",
        )
        proof = _seen_observation("source", calls, steps, snapshots, head)
        witnessed = next(step for step in steps if step["id"] == proof["input_step_id"])
        _require(all("HIDDEN_" + str(1000000 + i) in _input(witnessed) for i in (1, 2, 3)), "cached_rows_not_seen")
        report["observations"].append(proof)
    elif case == "failure":
        _require(
            len(operators) == 1 and operators[0].get("id") == "filtered" and observed == [],
            "hidden_failure_contract_mismatch",
        )
        error = head.get("results", {}).get("filtered", {}).get("nativeError", {})
        _require(
            head.get("results", {}).get("answer", {}).get("sampleRecords") is None,
            "stale_downstream_rows_after_failure",
        )
        diagnostic = error.get("diagnostic", {})
        revision = diagnostic.get("operatorRevision")
        _require(
            diagnostic.get("code") == "CALLBACK_RUNTIME"
            and diagnostic.get("stage") == "execute"
            and diagnostic.get("source", {}).get("line") == 2
            and isinstance(revision, str)
            and re.fullmatch(r"[a-f0-9]{16}", revision)
            and isinstance(error.get("executionId"), str)
            and error["executionId"].isdigit(),
            "source_mapped_runtime_error_missing",
        )
        inspections = [(i, call) for i, call in calls if call["toolName"] == "inspectError"]
        _require(len(inspections) == 1 and inspections[0][0] > index, "inspection_missing_or_early")
        inspected_at, inspection = inspections[0]
        _require(inspection["input"].get("operatorId") == "filtered", "wrong_error_inspected")
        header = f"[CALLBACK_RUNTIME; execute; revision {revision}]"
        _require(header in _input(steps[inspected_at]), "error_not_seen_before_inspection")
        outputs = [
            item.get("output")
            for item in steps[inspected_at].get("toolResults", [])
            if item.get("toolCallId") == inspection.get("toolCallId")
        ]
        _require(len(outputs) == 1 and isinstance(outputs[0], str) and outputs[0].strip(), "inspection_result_missing")
        _require(
            any(_normalized(outputs[0]) in _normalized(_input(step)) for step in steps[inspected_at + 1 :]),
            "inspection_not_seen_after_pull",
        )
        report.update(
            error_execution_id=error["executionId"],
            error_revision=revision,
            inspection_step_id=steps[inspected_at]["id"],
        )
    else:
        _require(
            [op.get("id") for op in operators] == ["extended", "filtered"] and observed == ["extended"],
            "repair_batch_missing",
        )
        extended, filtered = operators
        _require(
            extended.get("op") == "project"
            and extended.get("input") == "answer"
            and extended.get("keep") == ["x"]
            and filtered.get("op") == "filter"
            and filtered.get("input") == "source",
            "repair_dependency_mismatch",
        )
        version = _rows(head, "answer", [{"x": 3}])
        _rows(head, "extended", [{"x": 3}])
        old = (before or {}).get("results", {}).get("answer", {}).get("resultVersion", {}).get("id")
        _require(old and version != old, "downstream_version_not_invalidated")
        _require(not head.get("results", {}).get("filtered", {}).get("nativeError"), "repair_error_not_cleared")
        report["observations"].append(_seen_observation("extended", calls, steps, snapshots, head))

    if case in {"chain", "repair"}:
        try:
            answer = json.loads(steps[-1].get("content", ""))
        except (ValueError, TypeError):
            answer = None
        _require(answer == {"values": [2, 3] if case == "chain" else [3]}, "final_answer_mismatch")
    return report


def _capture(system, bundle):
    capture, trace, snapshots, info, head_workflow = system._capture(bundle)
    for name, value in (
        ("capture.json", capture),
        ("react_steps.json", trace),
        ("snapshots.json", snapshots),
        ("agent_info.json", public_info(info)),
        ("workflow.json", head_workflow),
    ):
        bundle.write(name, value)
    _require(capture.get("complete") is True, "incomplete_service_capture")
    return trace, snapshots, info, head_workflow


def run_model_smoke(system, *, output_directory, guard, context_mode, model_type="gpt-5.6-luna"):
    """Run on an already-owned session; require live admission for every turn.

    guard(system, stage, info) must validate source/engine/worker/ownership and
    return {qualified: True}. Its implementation/provenance belongs to the
    outer runner, not to this callback-shaped interface. Failure retains all
    artifacts and stops this fixture without canceling unknown backend work.
    """
    _require(callable(guard) and context_mode in {"delta", "latest"}, "live_qualification_guard_required")
    _require(model_type in {"gpt-5.6-luna", "gpt-5.6-terra"}, "smoke_model_or_driver_mismatch")
    bundle = AttemptBundle(Path(output_directory))
    report = {
        "version": 1,
        "status": "running",
        "context_mode": context_mode,
        "model_type": model_type,
        "cases": [],
        "resources": system._resources(),
        "admissions": [],
    }
    bundle.write("model_smoke.json", report)

    def qualify(stage, info):
        evidence = guard(system, stage, info)
        _require(isinstance(evidence, dict) and evidence.get("qualified") is True, "live_qualification_failed")
        report["admissions"].append({"stage": stage, "evidence": evidence})
        bundle.write("model_smoke.json", report)
        required = {
            "agentMode": "native",
            "nativeToolMode": "batch",
            "nativeCatalogVersion": "v2",
            "parallelToolCalls": False,
            "contextMode": context_mode,
            "messageLayout": "native",
            "nativeProfileCollection": True,
            "columnStats": False,
            "dataLevel": 1,
            "flowLevel": 1,
            "maxOperatorResultCharLimit": 2000,
            "maxOperatorResultCellCharLimit": 3000,
            "maxSteps": 25,
            "maxResultRows": 0,
            "enableCodeInSnapshot": False,
            "thoughtReplay": False,
            "contextWindowTokens": 0,
        }
        _require(isinstance(info, dict) and info.get("state") == "AVAILABLE", "agent_not_idle")
        _require(
            info.get("modelType") == model_type and info.get("driver") == "vercel-tool-use",
            "smoke_model_or_driver_mismatch",
        )
        settings = info.get("settings", {})
        _require(
            all(type(settings.get(key)) is type(value) and settings[key] == value for key, value in required.items()),
            "smoke_settings_mismatch",
        )

    try:
        trace, _, info, workflow = _capture(system, AttemptBundle(bundle.path / "initial"))
        qualify("before_smoke", info)
        _require(trace.get("steps") == [] and workflow.get("workflow", {}).get("operators") == [], "session_not_fresh")
        previous_ids, before, chain_head = set(), None, None
        for case, prompt in SMOKE_CASES:
            turn = AttemptBundle(bundle.path / case)
            turn.write("prompt.txt", prompt, text=True)
            turn.write("attempt.json", {"case": case, "dispatched": False, "resources": system._resources()})
            qualify("before_" + case, info)
            outcome, failure = None, None
            try:
                turn.write("attempt.json", {"case": case, "dispatched": True, "resources": system._resources()})
                system.agent.max_turn_seconds = 1800
                outcome = system.agent.run(prompt, empty_turn_retries=0, on_event=turn.event)
            except BaseException as error:
                failure = type(error).__name__
            finally:
                turn.write(
                    "transport.json", {"completed": bool(outcome and outcome.completed is True), "error_type": failure}
                )
                trace, captured, info, _ = _capture(system, turn)
            qualify("after_" + case, info)
            _require(
                outcome and outcome.completed is True and not outcome.error and not outcome.stopped and failure is None,
                "model_turn_incomplete",
            )
            steps = [step for step in trace["steps"] if step["id"] not in previous_ids]
            snapshots = {snapshot["stepId"]: snapshot for snapshot in captured["snapshots"]}
            _require(steps and steps[-1]["id"] in snapshots, "final_snapshot_missing")
            head = snapshots[steps[-1]["id"]]
            audit = audit_model_smoke_turn(
                case, steps, snapshots, head, before=chain_head if case == "repair" else before
            )
            turn.write("audit.json", audit)
            report["cases"].append(audit)
            if case == "chain":
                chain_head = head
            before, previous_ids = head, {step["id"] for step in trace["steps"]}
            bundle.write("model_smoke.json", report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        # Audit errors are static codes; do not serialize arbitrary API/guard errors.
        if type(error) is ValueError and re.fullmatch(r"[a-z_]+", str(error)):
            report["reason"] = str(error)
    bundle.write("model_smoke.json", report)
    return report

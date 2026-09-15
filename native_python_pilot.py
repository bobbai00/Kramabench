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

"""One native pilot attempt through KramaBench's actual Executor/Evaluator.

This is the orchestration entry point, not a service launcher. The caller must
supply the qualification/ownership/provenance guard; there is intentionally no
default no-op guard or CLI bypass. See docs/native-python-pilot.md for the live
gates, recorder binding and resource cleanup still required before running models.
"""

import json
from pathlib import Path

import kb
from benchmark.benchmark import Evaluator, Executor
from systems.native_pilot_system import NativePilotSystem, PILOT_TASK
from utils.collector_measurements import collector_report
from utils.execution_journal import execution_report


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def prepare_pilot_task(system, *, workload_path, dataset_directory):
    """Prepare only local inputs; use the executor's exact workload, not -tiny."""
    with Path(workload_path).open() as stream:
        tasks = [task for task in json.load(stream) if task.get("id") == PILOT_TASK]
    if len(tasks) != 1 or tasks[0].get("answer_type") != "numeric_exact":
        raise ValueError("expected exactly one frozen numeric-exact pilot task")
    system.process_dataset(dataset_directory)
    system.workload_data[PILOT_TASK] = tasks[0]


def run_pilot_task(
    system,
    *,
    guard,
    workload_path,
    dataset_directory,
    fixture_directory=None,
    execution_journal_path=None,
    recorder_id=None,
):
    """Exactly one task, no cache reuse, replacement attempt, or recovery round.

    Raw answer scores survive failed qualification but are labeled ineligible
    for the treatment comparison. A rejected preflight makes no model call and
    is unscored. Unknown/incomplete cost never becomes zero. Model completion,
    official answer correctness and execution-phase pass rates remain separate.
    """
    if not isinstance(system, NativePilotSystem):
        raise TypeError("run_pilot_task requires a registered native pilot SUT")
    if (execution_journal_path is None) != (recorder_id is None):
        raise ValueError("execution journal and bound recorder ID must be provided together")
    system.bind_pilot_guard(guard)
    prepare_pilot_task(system, workload_path=workload_path, dataset_directory=dataset_directory)
    executor = Executor(
        system,
        system.name,
        str(workload_path),
        system.output_dir,
        run_subtasks=False,
        use_truth_subset=True,
        task_id_filter=[PILOT_TASK],
    )
    response = executor.run_next_task()
    bundle = system.pilot_bundle
    bundle.write("response.json", response)
    with (bundle.path / "attempt.json").open() as stream:
        attempt = json.load(stream)
    with (bundle.path / "stats.json").open() as stream:
        stats = json.load(stream)
    verdict = {
        "version": 1,
        "task_id": PILOT_TASK,
        "answer_type": "numeric_exact",
        "expected_metric": "success",
        "pass_threshold": 0.9,
        "score": None,
        "passed": None,
        "comparison_eligible": attempt["dispatched"] and attempt["qualification"] == "passed",
        "attempt_status": attempt["status"],
        "reason": "not_dispatched" if not attempt["dispatched"] else None,
        "cost_usd": stats["cost_usd"],
        "cost_status": stats["cost_status"],
        "observed_cost_usd": stats["observed_cost_usd"],
    }
    if attempt["dispatched"]:
        try:
            evaluator = Evaluator(
                str(workload_path),
                str(fixture_directory or Path(__file__).parent / "benchmark/fixtures"),
                system.output_dir,
                run_subtasks=False,
                evaluate_pipeline=False,
                task_id_filter=[PILOT_TASK],
            )
            evaluated = evaluator.evaluate_results([response])
            if len(evaluated) != 1:
                raise ValueError("expected one official task evaluation")
            official = evaluated[0]
            bundle.write("evaluation.json", official)
            score = official.get("success")
            # This task's official metric is binary. Missing/wrong metrics are
            # unscored; never use the max of unrelated evaluator fields.
            if type(score) in {int, float} and score in (0, 1):
                verdict.update(score=score, passed=score >= verdict["pass_threshold"])
            else:
                verdict["reason"] = "missing_expected_metric"
        except Exception as error:
            verdict.update(reason="official_evaluation_failed", error_type=type(error).__name__)
    bundle.write("verdict.json", verdict)
    binding = None
    if execution_journal_path is not None:
        binding = {"journal_path": str(Path(execution_journal_path).resolve()), "recorder_id": recorder_id}
    bundle.write(
        "pilot_metrics.json",
        _attempt_measurements(bundle, attempt, stats, system.pilot_spec.collection, binding),
    )
    return verdict


def _attempt_measurements(bundle, attempt, stats, collection_requested, binding):
    execution_measurements = {
        "journal_status": "missing",
        "requests": None,
        "compilation": {"pass_rate": None},
        "runtime": {"pass_rate": None},
        "backend_termination_verified": False,
    }
    if binding is not None:
        try:
            resources = attempt["resources"]
            execution_measurements = execution_report(
                binding["journal_path"],
                workflow_id=resources.get("workflow_id"),
                computing_unit_id=resources.get("computing_unit_id"),
                expected_recorder_id=binding["recorder_id"],
            )
        except Exception as error:
            # Missing setup identity or unreadable evidence cannot erase the
            # official answer. Nor can it become a made-up compile pass rate.
            execution_measurements.update(journal_status="unavailable", error_type=type(error).__name__)
    try:
        snapshots = json.loads((bundle.path / "snapshots.json").read_text())
        capture = json.loads((bundle.path / "capture.json").read_text())
        collector_measurements = collector_report(
            snapshots,
            workflow_id=attempt["resources"].get("workflow_id"),
            execution_ids=execution_measurements.get("engine_execution_ids")
            if execution_measurements.get("journal_status") in {"open", "closed"}
            else None,
            collection_requested=collection_requested,
            snapshots_complete=capture.get("complete") is True,
        )
    except Exception as error:
        collector_measurements = {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "observed_collection_ms": None,
            "whole_attempt_overhead_complete": False,
        }
    return {
        "measurement_stage": "after_attempt",
        "recorder_binding": binding,
        "trace": _json_value(kb.react_metrics(bundle.path)),
        "step_tokens": kb.step_token_rows(bundle.path),
        "usage_status": stats["usage_status"],
        "input_trace_complete": stats["input_trace_complete"],
        "execution_measurements": execution_measurements,
        "collector_measurements": collector_measurements,
    }


def finalize_pilot_measurements(system):
    """Refresh only derived measurements after the originally bound recorder drains.

    No agent, guard, executor, evaluator or cleanup call is made. An open,
    invalid or foreign recorder cannot replace the initial measurement report.
    Closure does not prove backend termination, complete phase coverage or
    whole-attempt collector overhead; their existing unknowns remain explicit.
    Repeating this local refresh on unchanged artifacts is idempotent.
    """
    if not isinstance(system, NativePilotSystem):
        raise TypeError("finalization requires the original native pilot SUT")
    bundle = system.pilot_bundle
    previous = json.loads((bundle.path / "pilot_metrics.json").read_text())
    binding = previous.get("recorder_binding")
    if (
        not isinstance(binding, dict)
        or not isinstance(binding.get("journal_path"), str)
        or not Path(binding["journal_path"]).is_absolute()
        or not isinstance(binding.get("recorder_id"), str)
        or not binding["recorder_id"]
    ):
        raise ValueError("original recorder binding is required")
    attempt = json.loads((bundle.path / "attempt.json").read_text())
    config = json.loads((bundle.path / "config.json").read_text())
    stats = json.loads((bundle.path / "stats.json").read_text())
    if config.get("system_name") != system.name or config.get("agent_settings") != system.pilot_spec.settings():
        raise ValueError("pilot settings changed before measurement finalization")
    metrics = _attempt_measurements(bundle, attempt, stats, system.pilot_spec.collection, binding)
    if metrics["execution_measurements"]["journal_status"] != "closed":
        raise ValueError("original recorder journal must be valid and closed")
    metrics["measurement_stage"] = "after_recorder_drain"
    bundle.write("pilot_metrics.json", metrics)
    return metrics

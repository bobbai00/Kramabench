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
gates, recorder persistence and cleanup still required before running models.
"""

import json
from pathlib import Path

import kb
from benchmark.benchmark import Evaluator, Executor
from systems.native_pilot_system import NativePilotSystem, PILOT_TASK


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def run_pilot_task(system, *, guard, workload_path, dataset_directory, fixture_directory=None):
    """Exactly one task, no cache reuse, replacement attempt, or recovery round.

    Raw answer scores survive failed qualification but are labeled ineligible
    for the treatment comparison. A rejected preflight makes no model call and
    is unscored. Unknown/incomplete cost never becomes zero. Model completion,
    official answer correctness and execution-phase pass rates remain separate.
    """
    if not isinstance(system, NativePilotSystem):
        raise TypeError("run_pilot_task requires a registered native pilot SUT")
    with Path(workload_path).open() as stream:
        tasks = [task for task in json.load(stream) if task.get("id") == PILOT_TASK]
    if len(tasks) != 1 or tasks[0].get("answer_type") != "numeric_exact":
        raise ValueError("expected exactly one frozen numeric-exact pilot task")
    system.bind_pilot_guard(guard)
    system.process_dataset(dataset_directory)
    # Use the same exact workload as the official executor/evaluator, not a
    # best-effort domain inference or a -tiny workload silently overriding it.
    system.workload_data[PILOT_TASK] = tasks[0]
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
    bundle.write(
        "pilot_metrics.json",
        {
            "trace": _json_value(kb.react_metrics(bundle.path)),
            "step_tokens": kb.step_token_rows(bundle.path),
            "usage_status": stats["usage_status"],
            "input_trace_complete": stats["input_trace_complete"],
            "execution_measurements": "attach recorder journal and report measured denominators separately",
            "collector_measurements": "deduplicate producing result versions before aggregation",
        },
    )
    return verdict

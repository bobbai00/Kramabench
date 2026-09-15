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

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from utils.native_model_smoke import LOAD_CODE, audit_model_smoke_turn, run_model_smoke


def model_step(identifier, calls=(), observation=""):
    return {
        "id": identifier,
        "role": "agent",
        "isEnd": not calls,
        "inputMessages": [{"role": "system", "content": "prompt"}, {"role": "user", "content": observation}],
        "toolCalls": [
            {"toolName": name, "toolCallId": identifier + str(i), "input": args} for i, (name, args) in enumerate(calls)
        ],
        "content": '{"values":[2,3]}',
    }


def result(rows, version="a" * 64):
    return {
        "sampleRecords": rows,
        "resultVersion": {"id": version, "complete": True},
        "materialization": {"executionId": "100"},
    }


class ModelSmokeAuditTest(unittest.TestCase):
    def setUp(self):
        self.batch = {
            "operators": [
                {"op": "project", "id": "answer", "input": "filtered", "keep": ["x"]},
                {
                    "op": "filter",
                    "id": "filtered",
                    "input": "source",
                    "columns": ["x"],
                    "func": "def apply(row):\n    return row['x'] >= 2",
                },
                {"op": "udf", "id": "source", "code": LOAD_CODE},
            ],
            "observe": ["answer"],
        }
        self.frame = {"text": "Complete table\nx\n2\n3", "resultVersion": "a" * 64}
        self.head = {
            "results": {
                "source": result([{"x": i, "note": "HIDDEN_" + str(1000000 + i)} for i in (1, 2, 3)]),
                "answer": result([{"x": 2}, {"x": 3}]),
            },
            "nativeObservations": {"answer": self.frame},
        }
        self.steps = [
            model_step("batch", [("dataflow", self.batch)]),
            model_step("final", observation=self.frame["text"]),
        ]
        self.snapshots = {"batch": copy.deepcopy(self.head), "final": copy.deepcopy(self.head)}

    def audit(self, case="chain", before=None):
        return audit_model_smoke_turn(case, self.steps, self.snapshots, self.head, before=before)

    def test_dependent_batch_needs_a_later_actual_input_observation(self):
        report = self.audit()
        self.assertEqual(report["observations"][0]["input_step_id"], "final")
        self.assertEqual(report["observations"][0]["tool_step_id"], "batch")

    def test_system_prompt_echo_is_not_observation_evidence(self):
        self.steps[-1]["inputMessages"] = [{"role": "system", "content": self.frame["text"]}]
        with self.assertRaisesRegex(ValueError, "not_seen_after_observe"):
            self.audit()

    def test_later_snapshot_is_not_retroactive_model_input(self):
        self.steps[0]["inputMessages"][1]["content"] = self.frame["text"]
        self.steps[-1]["inputMessages"][1]["content"] = "not observed"
        with self.assertRaisesRegex(ValueError, "not_seen_after_observe"):
            self.audit()

    def test_debug_steps_or_missing_inputs_cannot_qualify_as_real_model(self):
        del self.steps[0]["inputMessages"]
        with self.assertRaisesRegex(ValueError, "missing_model_input"):
            self.audit()

    def test_hidden_source_row_leak_is_rejected(self):
        self.steps[-1]["inputMessages"][1]["content"] += "\nHIDDEN_1000001"
        with self.assertRaisesRegex(ValueError, "hidden_source_row"):
            self.audit()

    def test_explicit_observe_and_actual_answer_are_both_required(self):
        self.batch["observe"] = []
        with self.assertRaises(ValueError):
            self.audit()
        self.batch["observe"] = ["answer"]
        self.steps[-1]["content"] = '{"values":[1,2,3]}'
        with self.assertRaisesRegex(ValueError, "final_answer_mismatch"):
            self.audit()

    def test_cached_observe_keeps_producing_identity(self):
        before = copy.deepcopy(self.head)
        text = "Complete table\nx note\n1 HIDDEN_1000001\n2 HIDDEN_1000002\n3 HIDDEN_1000003"
        self.head["nativeObservations"] = {"source": {"text": text, "resultVersion": "a" * 64}}
        self.steps = [
            model_step("observe", [("dataflow", {"operators": [], "observe": ["source"]})]),
            model_step("final", observation=text),
        ]
        self.snapshots = {"observe": self.head, "final": self.head}
        self.assertEqual(self.audit("cached", before)["case"], "cached")
        self.steps[0]["inputMessages"][1]["content"] = "HIDDEN_1000001"
        with self.assertRaisesRegex(ValueError, "source_seen_before_cached_observe"):
            self.audit("cached", before)
        self.steps[0]["inputMessages"][1]["content"] = ""
        self.head["results"]["source"]["resultVersion"]["id"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "cached_binding_changed"):
            self.audit("cached", before)

    def test_failure_needs_source_mapped_error_seen_before_inspection(self):
        diagnostic = {
            "code": "CALLBACK_RUNTIME",
            "stage": "execute",
            "operatorRevision": "1" * 16,
            "source": {"line": 2},
        }
        self.head = {
            "results": {"filtered": {"nativeError": {"executionId": "100", "diagnostic": diagnostic}}},
            "nativeObservations": {},
        }
        error = "[CALLBACK_RUNTIME; execute; revision " + "1" * 16 + "] apply line 2"
        self.steps = [
            model_step("error", [("dataflow", {"operators": [{"id": "filtered", "op": "filter"}], "observe": []})]),
            model_step("inspect", [("inspectError", {"operatorId": "filtered"})], observation=error),
            model_step("final", observation=error + " Inspection from worker"),
        ]
        # The actual model-facing inspectError tool returns a string. The
        # authenticated REST inspection endpoint instead returns {text, ...}.
        self.steps[1]["toolResults"] = [{"toolCallId": "inspect0", "output": "Inspection from worker"}]
        self.snapshots = {step["id"]: self.head for step in self.steps}
        self.assertEqual(self.audit("failure")["case"], "failure")
        self.steps[1]["inputMessages"] = [{"role": "user", "content": "no error here"}]
        with self.assertRaisesRegex(ValueError, "error_not_seen_before_inspection"):
            self.audit("failure")
        self.steps[1]["inputMessages"] = [{"role": "user", "content": error}]
        self.head["results"]["answer"] = {"sampleRecords": [{"x": 2}]}
        with self.assertRaisesRegex(ValueError, "stale_downstream_rows_after_failure"):
            self.audit("failure")

    def test_repair_extends_existing_graph_and_invalidates_downstream(self):
        before = copy.deepcopy(self.head)
        frame = {"text": "Complete table\nx\n3", "resultVersion": "c" * 64}
        self.head = {
            "results": {"answer": result([{"x": 3}], "b" * 64), "extended": result([{"x": 3}], "c" * 64)},
            "nativeObservations": {"extended": frame},
        }
        batch = {
            "operators": [
                {"id": "extended", "op": "project", "input": "answer", "keep": ["x"]},
                {"id": "filtered", "op": "filter", "input": "source"},
            ],
            "observe": ["extended"],
        }
        self.steps = [model_step("repair", [("dataflow", batch)]), model_step("final", observation=frame["text"])]
        self.steps[-1]["content"] = '{"values":[3]}'
        self.snapshots = {step["id"]: self.head for step in self.steps}
        self.assertEqual(self.audit("repair", before)["case"], "repair")
        self.head["results"]["answer"]["resultVersion"] = before["results"]["answer"]["resultVersion"]
        with self.assertRaisesRegex(ValueError, "downstream_version_not_invalidated"):
            self.audit("repair", before)

    def test_unqualified_runner_fails_before_any_model_call(self):
        system = Mock()
        with self.assertRaises(ValueError):
            run_model_smoke(system, output_directory="unused", guard=None, context_mode="delta")
        system.agent.run.assert_not_called()


class ModelSmokeRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "model"
        self.system = Mock()
        self.system._resources.return_value = {"agent_id": "owned", "workflow_id": 456, "computing_unit_id": 321}
        settings = {
            "agentMode": "native",
            "nativeToolMode": "batch",
            "nativeCatalogVersion": "v2",
            "contextMode": "delta",
            "messageLayout": "native",
            "parallelToolCalls": False,
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
        self.info = {
            "state": "AVAILABLE",
            "settings": settings,
            "delegate": {"token": "private-token"},
            "modelType": "gpt-5.6-luna",
            "driver": "vercel-tool-use",
        }
        self.system._capture.return_value = (
            {"complete": True},
            {"steps": []},
            {"snapshots": []},
            self.info,
            {"workflow": {"operators": [], "links": []}},
        )

    def test_guard_failure_never_dispatches_or_cleans_resources(self):
        report = run_model_smoke(
            self.system, output_directory=self.path, guard=lambda *args: {"qualified": False}, context_mode="delta"
        )
        self.assertEqual(report["status"], "failed")
        self.system.agent.run.assert_not_called()
        self.system.cleanup.assert_not_called()
        self.assertNotIn("private-token", (self.path / "initial/agent_info.json").read_text())

    def test_incomplete_turn_is_captured_not_retried_or_canceled(self):
        self.system.agent.run.return_value = SimpleNamespace(completed=False, error=None, stopped=False)
        report = run_model_smoke(
            self.system, output_directory=self.path, guard=lambda *args: {"qualified": True}, context_mode="delta"
        )
        self.assertEqual(report["reason"], "model_turn_incomplete")
        self.assertEqual(self.system.agent.run.call_count, 1)
        self.assertEqual(self.system.agent.run.call_args.kwargs["empty_turn_retries"], 0)
        self.assertEqual(self.system.agent.max_turn_seconds, 1800)
        self.assertTrue((self.path / "chain/snapshots.json").exists())
        self.system.agent.stop.assert_not_called()
        self.system.cleanup.assert_not_called()

    def test_effective_mode_mismatch_is_rejected_before_model_use(self):
        report = run_model_smoke(
            self.system, output_directory=self.path, guard=lambda *args: {"qualified": True}, context_mode="latest"
        )
        self.assertEqual(report["reason"], "smoke_settings_mismatch")
        self.system.agent.run.assert_not_called()
        self.assertEqual(json.loads((self.path / "model_smoke.json").read_text())["status"], "failed")

    def test_wrong_model_is_not_a_pilot_qualification(self):
        self.info["modelType"] = "another-model"
        report = run_model_smoke(
            self.system, output_directory=self.path, guard=lambda *args: {"qualified": True}, context_mode="delta"
        )
        self.assertEqual(report["reason"], "smoke_model_or_driver_mismatch")
        self.system.agent.run.assert_not_called()

    def test_all_four_turns_use_new_steps_and_stop_after_the_final_audit(self):
        fixture = ModelSmokeAuditTest()
        fixture.setUp()
        chain = copy.deepcopy(fixture.head)
        cached = copy.deepcopy(chain)
        cached["nativeObservations"] = {
            "source": {
                "text": "Complete table\nHIDDEN_1000001 HIDDEN_1000002 HIDDEN_1000003",
                "resultVersion": "a" * 64,
            }
        }
        error_text = "[CALLBACK_RUNTIME; execute; revision " + "1" * 16 + "] apply line 2"
        failed = {
            "results": {
                "source": chain["results"]["source"],
                "filtered": {
                    "nativeError": {
                        "executionId": "100",
                        "diagnostic": {
                            "code": "CALLBACK_RUNTIME",
                            "stage": "execute",
                            "operatorRevision": "1" * 16,
                            "source": {"line": 2},
                        },
                    }
                },
            }
        }
        repaired = {
            "results": {"answer": result([{"x": 3}], "b" * 64), "extended": result([{"x": 3}], "c" * 64)},
            "nativeObservations": {"extended": {"text": "Complete table\nx\n3", "resultVersion": "c" * 64}},
        }
        repair_args = {
            "operators": [
                {"id": "extended", "op": "project", "input": "answer", "keep": ["x"]},
                {"id": "filtered", "op": "filter", "input": "source"},
            ],
            "observe": ["extended"],
        }
        phases = [
            (
                chain,
                [
                    model_step("chain-batch", [("dataflow", fixture.batch)]),
                    model_step("chain-final", observation=fixture.frame["text"]),
                ],
            ),
            (
                cached,
                [
                    model_step("cached-batch", [("dataflow", {"operators": [], "observe": ["source"]})]),
                    model_step("cached-final", observation=cached["nativeObservations"]["source"]["text"]),
                ],
            ),
            (
                failed,
                [
                    model_step("failure-batch", [("dataflow", {"operators": [{"id": "filtered"}], "observe": []})]),
                    model_step(
                        "failure-inspect", [("inspectError", {"operatorId": "filtered"})], observation=error_text
                    ),
                    model_step("failure-final", observation=error_text + "\ninspection"),
                ],
            ),
            (
                repaired,
                [
                    model_step("repair-batch", [("dataflow", repair_args)]),
                    model_step("repair-final", observation=repaired["nativeObservations"]["extended"]["text"]),
                ],
            ),
        ]
        phases[2][1][1]["toolResults"] = [{"toolCallId": "failure-inspect0", "output": "inspection"}]
        phases[3][1][-1]["content"] = '{"values":[3]}'
        captures = [self.system._capture.return_value]
        cumulative_steps, cumulative_snapshots = [], []
        for head, steps in phases:
            cumulative_steps.extend(steps)
            cumulative_snapshots.extend({**copy.deepcopy(head), "stepId": step["id"]} for step in steps)
            captures.append(
                (
                    {"complete": True},
                    {"steps": list(cumulative_steps)},
                    {"snapshots": list(cumulative_snapshots)},
                    self.info,
                    {"workflow": {"operators": []}},
                )
            )
        self.system._capture.side_effect = captures
        self.system.agent.run.return_value = SimpleNamespace(completed=True, error=None, stopped=False)
        stages = []

        def guard(system, stage, info):
            stages.append(stage)
            return {"qualified": True}

        report = run_model_smoke(self.system, output_directory=self.path, guard=guard, context_mode="delta")
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual([entry["stage"] for entry in report["admissions"]], stages)
        self.assertTrue(all(entry["evidence"]["qualified"] is True for entry in report["admissions"]))
        self.assertEqual([case["case"] for case in report["cases"]], ["chain", "cached", "failure", "repair"])
        self.assertEqual(self.system.agent.run.call_count, 4)
        self.assertEqual(
            stages,
            [
                "before_smoke",
                "before_chain",
                "after_chain",
                "before_cached",
                "after_cached",
                "before_failure",
                "after_failure",
                "before_repair",
                "after_repair",
            ],
        )
        self.system.cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()

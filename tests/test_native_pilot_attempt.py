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

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dataflow_agent import AgentSettings, MessageResult
from systems.native_python_system import DataflowSystemLunaPythonPilotV2Control20260915Rep1 as Arm


TASK = {
    "id": "environment-easy-3",
    "query": "Fixture query, not a benchmark answer.",
    "answer_type": "numeric_exact",
    "answer": 17,
    "data_sources": ["fixture.csv"],
}
STEP = {
    "id": "s1",
    "role": "agent",
    "isEnd": True,
    "content": "17",
    "toolCalls": [],
    "inputMessages": [{"role": "user", "content": TASK["query"]}],
    "usage": {"inputTokens": 100, "outputTokens": 20, "cachedInputTokens": 80, "totalTokens": 120},
}


class NativePilotAttemptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.arm = Arm(output_dir=self.temporary.name, computing_unit_id=321)
        self.arm.dataset_directory = "data/environment/input"
        self.arm.dataset = {"fixture.csv": None}
        self.arm.workload_data = {TASK["id"]: TASK}
        self.arm._expand_data_sources = Mock(return_value=["data/environment/input/fixture.csv"])
        self.agent = SimpleNamespace(
            agent_id="fixture-agent",
            _workflow_id=456,
            _computing_unit_id=321,
            _token="private-token",
            agent_service_endpoint="http://localhost:3011",
            settings=AgentSettings(),
            max_turn_seconds=None,
            run=Mock(),
            cleanup=Mock(),
        )

        def setup():
            self.arm.agent = self.agent

        self.arm._setup_agent = Mock(side_effect=setup)
        self.guard = Mock(return_value={"fixture_guard": True})
        self.arm.bind_pilot_guard(self.guard)
        self.info = {
            "id": "fixture-agent",
            "modelType": "gpt-5.6-luna",
            "driver": "vercel-tool-use",
            "state": "AVAILABLE",
            "settings": self.agent.settings.to_api_dict(),
            "delegate": {"workflowId": 456, "computingUnitId": 321, "userToken": "private-token"},
        }
        self.react = {"steps": [STEP], "state": "AVAILABLE"}
        self.api = patch("systems.native_pilot_system.agent_json", side_effect=self.request).start()
        self.addCleanup(patch.stopall)

    def request(self, agent, resource):
        if resource == "":
            return self.info
        if resource == "/react-steps":
            return self.react
        if resource == "/workflow":
            return {"workflow": {"operators": [], "links": []}}
        if resource.startswith("/snapshots/"):
            return {"stepId": resource.split("/")[-1], "results": {"op": {"profile": {"version": "fixture"}}}}
        raise AssertionError(resource)

    def serve(self):
        return self.arm.serve_query(TASK["query"], TASK["id"], TASK["data_sources"])

    def artifact(self, name):
        return json.loads((Path(self.temporary.name) / TASK["id"] / name).read_text())

    def test_complete_attempt_saves_inputs_settings_snapshots_and_one_dispatch(self):
        self.agent.run.return_value = MessageResult("17", [], {}, {}, False, completed=True)
        result = self.serve()
        self.agent.run.assert_called_once()
        self.assertEqual(self.agent.run.call_args.kwargs["empty_turn_retries"], 0)
        self.assertEqual(result["explanation"]["answer"], "17")
        self.assertGreater(result["cost_usd"], 0)
        self.assertEqual(self.artifact("attempt.json")["status"], "completed")
        self.assertEqual(self.artifact("react_steps.json")["steps"][0]["inputMessages"], STEP["inputMessages"])
        self.assertEqual(self.artifact("snapshots.json")["snapshots"][0]["stepId"], "s1")
        self.assertNotIn("private-token", json.dumps(self.artifact("config.json")))
        self.assertEqual(
            [call.args[1] for call in self.guard.call_args_list], ["before_setup", "before_dispatch", "after_attempt"]
        )
        with self.assertRaises(FileExistsError):
            self.serve()
        self.agent.run.assert_called_once()

    def test_transport_failure_preserves_streamed_progress_if_rest_is_down(self):
        def run(prompt, *, on_event, **kwargs):
            on_event({"type": "step", "step": STEP})
            raise ConnectionError("must not log raw credentials: private-token")

        self.agent.run.side_effect = run

        def request(agent, resource):
            if resource == "" and self.agent.run.call_count == 0:
                return self.info
            raise ConnectionError("private-token")

        self.api.side_effect = request
        result = self.serve()
        self.agent.run.assert_called_once()
        self.assertIsNone(result["cost_usd"])
        self.assertGreater(self.artifact("stats.json")["observed_cost_usd"], 0)
        self.assertEqual(self.artifact("react_steps.json")["steps"], [STEP])
        self.assertEqual(self.artifact("attempt.json")["status"], "interrupted")
        self.assertFalse(self.artifact("attempt.json")["backend_termination_verified"])
        self.arm.cleanup()
        self.agent.cleanup.assert_not_called()
        self.assertEqual(self.artifact("attempt.json")["resources"]["workflow_id"], 456)
        self.assertNotIn("private-token", json.dumps(self.artifact("attempt.json")))

    def test_preflight_failure_never_dispatches_and_keeps_resources(self):
        def guard(system, stage, info):
            if stage == "before_dispatch":
                raise ValueError("wrong build")
            return {"fixture_guard": True}

        self.guard.side_effect = guard
        result = self.serve()
        self.agent.run.assert_not_called()
        self.assertEqual(self.artifact("attempt.json")["status"], "not_started")
        self.assertFalse(result["pilot_evaluation_eligible"])
        self.assertIsNone(result["cost_usd"])

    def test_setup_failure_has_a_bundle_without_reset_or_retry(self):
        def setup():
            self.arm.agent = self.agent
            self.agent.agent_id = None
            raise ConnectionError("private-token")

        self.arm._setup_agent.side_effect = setup
        self.serve()
        self.arm._setup_agent.assert_called_once()
        self.agent.run.assert_not_called()
        self.assertEqual(self.artifact("attempt.json")["status"], "not_started")
        self.assertEqual(self.artifact("attempt.json")["resources"]["workflow_id"], 456)

    def test_postflight_drift_is_not_a_qualified_success(self):
        self.agent.run.return_value = MessageResult("17", [], {}, {}, False, completed=True)

        def guard(system, stage, info):
            if stage == "after_attempt":
                raise ValueError("source drift")
            return {}

        self.guard.side_effect = guard
        result = self.serve()
        self.assertEqual(self.artifact("attempt.json")["status"], "completed")
        self.assertFalse(result["pilot_evaluation_eligible"])
        self.assertEqual(self.artifact("attempt.json")["qualification"], "failed")

    def test_synthetic_error_is_not_misparsed_as_an_answer(self):
        self.agent.run.return_value = MessageResult("Error: row 17", [], {}, {}, False, completed=True)
        self.react = {
            "steps": [STEP, {"id": "err", "role": "agent", "isEnd": True, "content": "Error: row 17"}],
            "state": "AVAILABLE",
        }
        result = self.serve()
        self.assertEqual(result["explanation"]["answer"], "")
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(self.artifact("attempt.json")["status"], "agent_error")

    def test_observed_final_answer_survives_missing_completion_frame(self):
        self.agent.run.return_value = MessageResult("17", [], {}, {}, False, completed=False)
        result = self.serve()
        self.assertEqual(result["explanation"]["answer"], "17")
        self.assertEqual(self.artifact("attempt.json")["status"], "interrupted")
        self.assertIsNone(result["cost_usd"])
        self.assertFalse(self.artifact("stats.json")["input_trace_complete"])

    def test_unbound_arm_and_nonpilot_task_fail_before_any_network_call(self):
        self.arm._pilot_guard = None
        with self.assertRaisesRegex(RuntimeError, "guard"):
            self.serve()
        self.arm._setup_agent.assert_not_called()
        self.arm.bind_pilot_guard(self.guard)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.arm.serve_query("different", "different-id", [])
        self.arm._setup_agent.assert_not_called()

    def test_dataset_preparation_makes_no_live_agent(self):
        self.arm.process_dataset(str(Path(self.temporary.name) / "data"))
        self.arm._setup_agent.assert_not_called()

    def test_official_executor_and_evaluator_use_the_same_single_attempt(self):
        from native_python_pilot import run_pilot_task

        workload = Path(self.temporary.name) / "workload.json"
        workload.write_text(json.dumps([TASK]))
        self.agent.run.return_value = MessageResult("17", [], {}, {}, False, completed=True)
        with patch("benchmark.benchmark.GPTInterface", side_effect=AssertionError("no LLM judge for numeric exact")):
            verdict = run_pilot_task(
                self.arm, guard=self.guard, workload_path=workload, dataset_directory="data/environment/input"
            )
        self.agent.run.assert_called_once()
        self.assertEqual(verdict["score"], 1)
        self.assertIs(verdict["passed"], True)
        self.assertEqual(verdict["pass_threshold"], 0.9)
        self.assertEqual(verdict["expected_metric"], "success")
        self.assertEqual(self.artifact("evaluation.json")["success"], 1)
        self.assertEqual(self.artifact("response.json")["task_id"], TASK["id"])
        self.assertEqual(self.artifact("pilot_metrics.json")["trace"]["agent_steps"], 1)

    def test_preflight_failure_is_unscored_not_an_official_wrong_answer(self):
        from native_python_pilot import run_pilot_task

        workload = Path(self.temporary.name) / "workload.json"
        workload.write_text(json.dumps([TASK]))
        self.guard.side_effect = ValueError("missing qualification")
        verdict = run_pilot_task(
            self.arm, guard=self.guard, workload_path=workload, dataset_directory="data/environment/input"
        )
        self.agent.run.assert_not_called()
        self.assertIsNone(verdict["score"])
        self.assertIsNone(verdict["passed"])
        self.assertEqual(verdict["reason"], "not_dispatched")

    def test_malformed_service_trace_falls_back_to_preserved_stream(self):
        def run(prompt, *, on_event, **kwargs):
            on_event({"type": "step", "step": STEP})
            return MessageResult("17", [], {}, {}, False, completed=True)

        self.agent.run.side_effect = run
        self.react = {"steps": ["not a step"], "state": "AVAILABLE"}
        result = self.serve()
        self.assertEqual(self.artifact("react_steps.json")["steps"], [STEP])
        self.assertEqual(self.artifact("capture.json")["trace_source"], "stream_partial")
        self.assertFalse(self.artifact("capture.json")["complete"])
        self.assertIsNone(result["cost_usd"])

    def test_snapshot_identity_mismatch_does_not_pass_capture(self):
        self.agent.run.return_value = MessageResult("17", [], {}, {}, False, completed=True)

        def request(agent, resource):
            if resource.startswith("/snapshots/"):
                return {"stepId": "different-step", "results": {}}
            return self.request(agent, resource)

        self.api.side_effect = request
        self.serve()
        self.assertFalse(self.artifact("capture.json")["complete"])

    def test_missing_input_capture_does_not_erase_a_real_model_answer(self):
        self.agent.run.return_value = MessageResult("17", [], {}, {}, False, completed=True)
        self.react = {
            "steps": [{key: value for key, value in STEP.items() if key != "inputMessages"}],
            "state": "AVAILABLE",
        }
        result = self.serve()
        self.assertEqual(result["explanation"]["answer"], "17")
        self.assertFalse(self.artifact("stats.json")["input_trace_complete"])


if __name__ == "__main__":
    unittest.main()

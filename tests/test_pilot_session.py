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
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from systems.native_python_system import PILOT_ARMS
from utils.pilot_session import _verify_recorder, run_owned_pilot


class PilotSessionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calls = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.source = {"source_root": str(self.root), "git_sha": "a" * 40, "source_dirty": False}
        self.contexts = []

        def qualify(context, system, stage, info):
            self.contexts.append((context.copy(), stage))
            self.calls.append(stage)
            return {"qualified": True, "fixture_only": True}

        self.qualification = Mock(side_effect=qualify)
        self.patches = {}

        def mock(name, **kwargs):
            value = self.stack.enter_context(patch("utils.pilot_session." + name, **kwargs))
            self.patches[name] = value
            return value

        mock("source_snapshot", return_value=self.source)
        mock("_listener_free", return_value=True)
        mock("_harness_source", return_value={"git_sha": "b" * 40})
        mock("_recorder_source", return_value={"root": str(self.root), "gitSha": "a" * 40, "files": []})
        mock("_verify_recorder", return_value={"pending": 0})
        mock("freeze_pilot_inputs", return_value={"fixture": True})
        mock("verify_pilot_inputs", return_value={"inputs_verified": True})
        mock("prepare_pilot_task")
        mock("login", return_value="private-token")
        mock("PilotResourceAPI")
        mock("create_owned_computing_unit", side_effect=lambda *a, **kw: self.calls.append("create_cu") or 321)
        recorder = mock("RecorderProcess").return_value
        recorder.metadata = {"url": "http://127.0.0.1:39193", "instanceId": "fixture-recorder"}
        recorder.drain.side_effect = lambda: self.calls.append("drain")
        mock(
            "launch_agent_service",
            side_effect=lambda *a, **kw: self.calls.append("launch") or {"listener": {"pid": 99}},
        )
        mock("verify_service_launch", return_value={"loaded_revision_verified": True})
        mock("finalize_pilot_measurements", side_effect=lambda *a: self.calls.append("finalize") or {"fixture": True})
        mock(
            "cleanup_owned_resources",
            side_effect=lambda *a, **kw: self.calls.append("cleanup") or {"status": "complete"},
        )
        mock("stop_agent_service", side_effect=lambda *a: self.calls.append("stop") or {"status": "stopped"})

        def attempt(system, *, guard, **kwargs):
            self.calls.append("attempt")
            guard(system, "before_setup", None)
            system.pilot_bundle = Mock(path=Path(system.output_dir) / "environment-easy-3")
            guard(system, "before_dispatch", {"fixture": True})
            guard(system, "after_attempt", {"fixture": True})
            return {"passed": True, "comparison_eligible": True}

        mock("run_pilot_task", side_effect=attempt)
        self.attempt = attempt

    def run_session(self, **overrides):
        arguments = {
            "arm_key": "V2Control",
            "agent_worktree": self.root,
            "agent_sha": "a" * 40,
            "recorder_worktree": self.root,
            "recorder_sha": "a" * 40,
            "workload_path": self.root / "workload.json",
            "dataset_directory": self.root / "data",
            "execution_directory": self.root,
            "output_directory": self.root / "session",
            "task_output_directory": self.root / "tasks",
            "qualification": self.qualification,
        }
        arguments.update(overrides)
        return run_owned_pilot(**arguments)

    def test_owned_one_attempt_orders_drain_finalization_cleanup_and_stop(self):
        result = self.run_session()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            self.calls,
            [
                "before_resources",
                "create_cu",
                "launch",
                "attempt",
                "before_setup",
                "before_dispatch",
                "after_attempt",
                "drain",
                "finalize",
                "cleanup",
                "stop",
            ],
        )
        self.patches["run_pilot_task"].assert_called_once()
        self.assertEqual(self.patches["verify_pilot_inputs"].call_count, 3)
        self.assertEqual(self.patches["verify_service_launch"].call_count, 3)
        self.assertNotIn("private-token", (self.root / "session/session.json").read_text())
        self.assertEqual(self.contexts[0][1], "before_resources")

    def test_missing_or_negative_runtime_qualification_never_allocates(self):
        for index, qualifier in enumerate((None, Mock(return_value={"qualified": False}), Mock(return_value={}))):
            with self.subTest(index=index):
                result = self.run_session(qualification=qualifier, output_directory=self.root / str(index))
                self.assertEqual(result["status"], "failed")
                self.patches["create_owned_computing_unit"].assert_not_called()
                self.patches["run_pilot_task"].assert_not_called()

    def test_existing_first_attempt_is_not_replaced(self):
        reserved = self.root / "tasks/environment-easy-3"
        reserved.mkdir(parents=True)
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.patches["login"].assert_not_called()
        self.assertTrue(reserved.exists())

    def test_foreign_or_dirty_agent_revision_is_not_launched(self):
        self.patches["source_snapshot"].return_value = {**self.source, "source_dirty": True}
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.patches["create_owned_computing_unit"].assert_not_called()

    def test_harness_drift_cannot_admit_the_attempt(self):
        self.patches["_harness_source"].side_effect = [{"git_sha": "b" * 40}, {"git_sha": "c" * 40}]
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.patches["cleanup_owned_resources"].assert_not_called()
        self.patches["stop_agent_service"].assert_not_called()
        self.assertNotIn("before_dispatch", self.calls)

    def test_unresolved_attempt_only_drains_owned_recorder_and_preserves_ids(self):
        self.patches["run_pilot_task"].side_effect = ConnectionError("private-token")
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["resources"]["computing_unit_id"], 321)
        self.assertTrue(result["recorder_drained_after_failure"])
        self.patches["cleanup_owned_resources"].assert_not_called()
        self.patches["stop_agent_service"].assert_not_called()
        self.assertNotIn("private-token", json.dumps(result))

    def test_failed_cleanup_does_not_stop_service_or_retry(self):
        self.patches["cleanup_owned_resources"].return_value = {"status": "blocked"}
        self.patches["cleanup_owned_resources"].side_effect = None
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "resource_cleanup")
        self.patches["stop_agent_service"].assert_not_called()
        self.patches["run_pilot_task"].assert_called_once()

    def test_wrong_or_ineligible_answer_is_not_reported_as_success(self):
        def incorrect(*args, **kwargs):
            self.attempt(*args, **kwargs)
            return {"passed": False, "comparison_eligible": True}

        self.patches["run_pilot_task"].side_effect = incorrect
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "complete")
        self.patches["run_pilot_task"].assert_called_once()

    def test_all_arms_use_their_own_frozen_port_and_one_candidate_recorder(self):
        for spec in PILOT_ARMS:
            with self.subTest(arm=spec.key):
                result = self.run_session(arm_key=spec.key, output_directory=self.root / spec.key)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(self.patches["launch_agent_service"].call_args.kwargs["port"], spec.port)
                self.assertEqual(self.patches["RecorderProcess"].call_args.args[0], self.root)


class RecorderAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.source = {"root": "/fixture", "gitSha": "a" * 40, "files": []}
        self.status = {
            "version": 1,
            "kind": "execution-recorder",
            "instanceId": "fixture",
            "backend": "http://127.0.0.1:8085",
            "computingUnitIds": [321],
            "maxRequests": 1000,
            "pending": 0,
            "completed": 0,
            "accepting": True,
            "sinkConfigured": True,
            "persistenceFailed": False,
        }
        self.recorder = Mock()
        self.recorder.process.poll.return_value = None
        self.recorder.metadata = {
            **self.status,
            "url": "http://127.0.0.1:39193",
            "source": {**self.source, "dirty": False},
        }
        self.request = patch("utils.pilot_session.requests.get").start()
        self.addCleanup(patch.stopall)
        self.request.return_value.status_code = 200
        self.request.return_value.json.side_effect = lambda: self.status

    def test_live_bound_recorder_is_empty_before_dispatch_but_can_have_completed_work_after(self):
        self.assertEqual(_verify_recorder(self.recorder, self.source, require_empty=True), self.status)
        self.status["completed"] = 2
        self.assertEqual(_verify_recorder(self.recorder, self.source, require_empty=False), self.status)
        with self.assertRaisesRegex(ValueError, "idle_or_bound"):
            _verify_recorder(self.recorder, self.source, require_empty=True)
        self.assertIs(self.request.call_args.kwargs["allow_redirects"], False)

    def test_foreign_scope_unpersisted_pending_or_exited_recorder_is_refused(self):
        for key, value in (
            ("instanceId", "foreign"),
            ("computingUnitIds", [999]),
            ("pending", 1),
            ("sinkConfigured", False),
            ("persistenceFailed", True),
            ("accepting", False),
            ("completed", True),
        ):
            with self.subTest(key=key):
                old = self.status[key]
                self.status[key] = value
                with self.assertRaises(ValueError):
                    _verify_recorder(self.recorder, self.source, require_empty=False)
                self.status[key] = old
        self.recorder.process.poll.return_value = 0
        with self.assertRaisesRegex(ValueError, "not_live"):
            _verify_recorder(self.recorder, self.source, require_empty=False)

    def test_source_mismatch_is_rejected_before_status_request(self):
        self.recorder.metadata["source"]["gitSha"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "source_mismatch"):
            _verify_recorder(self.recorder, self.source, require_empty=True)
        self.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

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
from unittest.mock import Mock, patch

from utils.pilot_resource_smoke import capture_setup, run_resource_smoke
from utils.pilot_artifacts import AttemptBundle


class ResourceSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.system = Mock()
        self.system._resources.return_value = {"agent_id": "owned", "workflow_id": 456, "computing_unit_id": 321}
        self.capture = (
            {"complete": True},
            {"steps": []},
            {"snapshots": []},
            {"id": "owned", "state": "AVAILABLE", "delegate": {"token": "private-token"}},
            {"workflow": {"operators": [], "links": []}},
        )
        self.system._capture.return_value = self.capture

    def test_setup_capture_persists_identities_but_never_raw_credentials(self):
        bundle = AttemptBundle(self.directory / "capture")
        capture_setup(self.system, bundle)
        attempt = json.loads((bundle.path / "attempt.json").read_text())
        self.assertFalse(attempt["dispatched"])
        self.assertTrue(attempt["setup_only"])
        self.assertEqual(attempt["resources"]["workflow_id"], 456)
        self.system.serve_query.assert_not_called()
        self.assertNotIn("private-token", (bundle.path / "agent_info.json").read_text())

    def test_nonempty_or_incomplete_capture_is_not_setup_only_evidence(self):
        for index, value in (
            (0, {"complete": False}),
            (1, {"steps": [{"id": "unexpected", "role": "agent"}]}),
            (3, {"state": "RUNNING"}),
            (4, {"workflow": {"operators": [{"operatorID": "unexpected"}], "links": []}}),
        ):
            with self.subTest(index=index):
                candidate = list(self.capture)
                candidate[index] = value
                self.system._capture.return_value = tuple(candidate)
                bundle = AttemptBundle(self.directory / str(index))
                with self.assertRaises(RuntimeError):
                    capture_setup(self.system, bundle)
                self.assertTrue((bundle.path / "attempt.json").exists())
                self.assertTrue((bundle.path / "capture.json").exists())

    def test_source_or_port_preflight_fails_before_allocations(self):
        with (
            patch(
                "utils.pilot_resource_smoke.source_snapshot", return_value={"source_dirty": True, "git_sha": "a" * 40}
            ),
            patch("utils.pilot_resource_smoke.login") as login,
            patch("utils.pilot_resource_smoke.create_owned_computing_unit") as create,
        ):
            result = run_resource_smoke(self.directory, expected_sha="a" * 40, output_directory=self.directory / "run")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "preflight")
        login.assert_not_called()
        create.assert_not_called()

    def test_orchestration_captures_then_drains_then_cleans_then_stops(self):
        order = []
        with (
            patch(
                "utils.pilot_resource_smoke.source_snapshot", return_value={"source_dirty": False, "git_sha": "a" * 40}
            ),
            patch("utils.pilot_resource_smoke._listener_free", return_value=True),
            patch("utils.pilot_resource_smoke.login", return_value="private-token"),
            patch("utils.pilot_resource_smoke.create_owned_computing_unit", return_value=321),
            patch("utils.pilot_resource_smoke.RecorderProcess") as Recorder,
            patch("utils.pilot_resource_smoke.launch_agent_service", return_value={"listener": {"pid": 99}}),
            patch("utils.pilot_resource_smoke.PilotSystem", return_value=self.system),
            patch("utils.pilot_resource_smoke.capture_setup", side_effect=lambda *args: order.append("capture")),
            patch(
                "utils.pilot_resource_smoke.execution_report", return_value={"journal_status": "closed", "requests": 0}
            ),
            patch(
                "utils.pilot_resource_smoke.cleanup_owned_resources",
                side_effect=lambda *a, **kw: order.append("cleanup") or {"status": "complete"},
            ),
            patch(
                "utils.pilot_resource_smoke.stop_agent_service",
                side_effect=lambda *a: order.append("stop") or {"status": "stopped"},
            ),
        ):
            recorder = Recorder.return_value
            recorder.metadata = {"url": "http://127.0.0.1:39193", "instanceId": "fixture-recorder"}
            recorder.drain.side_effect = lambda: order.append("drain")
            result = run_resource_smoke(self.directory, expected_sha="a" * 40, output_directory=self.directory / "run")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(order, ["capture", "drain", "cleanup", "stop"])
        self.system.serve_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()

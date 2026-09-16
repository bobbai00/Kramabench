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

import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from utils.pilot_model_session import run_owned_model_smoke


class PilotModelSessionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.mocks = {}
        for name in ("source_snapshot", "login", "PilotResourceAPI", "create_owned_computing_unit",
                     "_recorder_source", "RecorderProcess", "_verify_recorder", "launch_agent_service",
                     "verify_service_launch", "NativePilotSystem", "run_model_smoke", "cleanup_owned_resources",
                     "stop_agent_service"):
            self.mocks[name] = stack.enter_context(patch("utils.pilot_model_session." + name))
        self.mocks["source_snapshot"].return_value = {"git_sha": "a" * 40}
        self.mocks["create_owned_computing_unit"].return_value = 321
        self.mocks["RecorderProcess"].return_value.metadata = {
            "url": "http://127.0.0.1:39000", "instanceId": "fixture"}
        self.mocks["NativePilotSystem"].return_value._capture.return_value = ({}, {"steps": []}, {}, {}, {})
        self.mocks["NativePilotSystem"].return_value._resources.return_value = {"computing_unit_id": 321}
        self.mocks["run_model_smoke"].return_value = {"status": "passed"}
        self.mocks["cleanup_owned_resources"].return_value = {"status": "complete"}
        self.mocks["stop_agent_service"].return_value = {"status": "stopped"}
        self.qualification = Mock()

    def run_smoke(self, **kwargs):
        return run_owned_model_smoke(
            worktree=self.root, source_sha="a" * 40, context_mode="delta",
            output_directory=self.root / ("smoke-" + str(kwargs.get("agent_port", "default"))),
            qualification=self.qualification, **kwargs,
        )

    def test_default_and_explicit_ports_route_api_launch_and_agent_consistently(self):
        for port in (None, 3013):
            with self.subTest(port=port):
                result = self.run_smoke(**({} if port is None else {"agent_port": port}))
                selected = 3011 if port is None else port
                endpoint = f"http://127.0.0.1:{selected}"
                self.assertEqual(result["status"], "passed")
                self.assertEqual(self.mocks["PilotResourceAPI"].call_args.kwargs["agent_endpoint"], endpoint)
                self.assertEqual(self.mocks["launch_agent_service"].call_args.kwargs["port"], selected)
                settings = self.mocks["NativePilotSystem"].call_args.kwargs
                self.assertEqual(settings["agent_service_endpoint"], endpoint)
                self.assertEqual(settings["max_steps"], 25)
                self.assertEqual(settings["max_operator_result_char_limit"], 2000)
                self.assertEqual(settings["context_mode"], "delta")

    def test_invalid_or_shared_default_service_ports_fail_before_resources(self):
        for port in (0, 1023, 3001, 65536, True, "3013", "http://remote.invalid:3013", 3013.0):
            with self.subTest(port=port):
                with self.assertRaisesRegex(ValueError, "invalid_smoke_agent_port"):
                    self.run_smoke(agent_port=port)
        self.qualification.for_smoke.assert_not_called()
        for mock in self.mocks.values():
            mock.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

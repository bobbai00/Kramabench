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
from unittest.mock import patch

from utils.pilot_launch import (
    launch_agent_service,
    stop_agent_service,
    verify_service_launch,
)


class PilotLaunchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "worktree"
        self.root.mkdir()
        self.directory = Path(self.temporary.name) / "launch"
        self.sha = "a" * 40
        self.source = {
            "source_root": str(self.root),
            "git_sha": self.sha,
            "source_dirty": False,
            "source_sha256": "b" * 64,
            "launcher_sha256": "c" * 64,
        }
        self.listener = {
            "status": "resolved",
            "pid": 12345,
            "process_start_ticks": 1100,
            "source_root": str(self.root),
            "git_sha": self.sha,
            "source_dirty": False,
            "entry_sha256": "d" * 64,
            "loaded_revision_verified": False,
        }
        patches = {
            "source_snapshot": dict(return_value=self.source),
            "_listener_free": dict(return_value=True),
            "_boot_clock": dict(return_value={"boot_id": "fixture-boot", "ticks": 1000}),
            "inspect_service_listener": dict(return_value=self.listener),
            "_execution_route_matches": dict(return_value=True),
            "_healthy": dict(return_value=True),
            "_launch_command": dict(return_value=SimpleNamespace(returncode=0)),
        }
        self.mocks = {
            name: patch("utils.pilot_launch." + name, **arguments).start() for name, arguments in patches.items()
        }
        self.addCleanup(patch.stopall)

    def launch(self):
        return launch_agent_service(
            self.root,
            port=3011,
            expected_sha=self.sha,
            recorder_url="http://127.0.0.1:39193",
            run_directory=self.directory,
        )

    def test_only_the_named_agent_service_is_launched_and_source_is_bound(self):
        record = self.launch()
        self.assertEqual(record["status"], "qualified")
        self.assertTrue(record["loaded_revision_verified"])
        command = self.mocks["_launch_command"].call_args
        self.assertEqual(
            command.args[0],
            [
                str(self.root / "bin/local-dev.sh"),
                "up",
                "agent-service",
                "--skip-build",
            ],
        )
        self.assertEqual(command.kwargs["environment"]["TEXERA_AGENT_SERVICE_PORT"], "3011")
        self.assertEqual(
            command.kwargs["environment"]["WORKFLOW_EXECUTION_SERVICE_ENDPOINT"],
            "http://127.0.0.1:39193",
        )
        self.assertEqual(command.kwargs["environment"]["EXECUTION_ENDPOINT_TEMPLATE"], "")
        self.assertEqual(
            Path(command.kwargs["environment"]["TEXERA_LOCAL_DEV_DIR"]),
            self.directory / "launcher-state",
        )
        self.assertEqual(
            json.loads((self.directory / "launch.json").read_text())["listener"]["pid"],
            12345,
        )
        verified = verify_service_launch(self.directory / "launch.json")
        self.assertEqual(verified["pid"], 12345)
        with self.assertRaises(FileExistsError):
            self.launch()
        self.mocks["_launch_command"].assert_called_once()

    def test_an_occupied_port_is_never_adopted_or_restarted(self):
        self.mocks["_listener_free"].return_value = False
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            self.launch()
        self.mocks["_launch_command"].assert_not_called()

    def test_dirty_or_wrong_source_fails_before_launch(self):
        for change in ({"source_dirty": True}, {"git_sha": "e" * 40}):
            self.mocks["source_snapshot"].return_value = {**self.source, **change}
            with self.assertRaises(RuntimeError):
                self.launch()
        self.mocks["_launch_command"].assert_not_called()

    def test_source_change_during_launch_is_not_provenance_success(self):
        self.mocks["source_snapshot"].side_effect = [
            self.source,
            {**self.source, "source_sha256": "e" * 64},
        ]
        with self.assertRaisesRegex(RuntimeError, "source"):
            self.launch()
        record = json.loads((self.directory / "launch.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["loaded_revision_verified"])
        self.assertEqual(record["listener"]["pid"], 12345)

    def test_old_process_or_execution_template_bypass_is_not_qualified(self):
        self.mocks["inspect_service_listener"].return_value = {
            **self.listener,
            "process_start_ticks": 999,
        }
        with self.assertRaisesRegex(RuntimeError, "process"):
            self.launch()
        self.assertFalse(json.loads((self.directory / "launch.json").read_text())["loaded_revision_verified"])

    def test_a_restarted_listener_cannot_reuse_a_launch_record(self):
        self.launch()
        self.mocks["inspect_service_listener"].return_value = {
            **self.listener,
            "process_start_ticks": 1200,
        }
        with self.assertRaisesRegex(RuntimeError, "process"):
            verify_service_launch(self.directory / "launch.json")

    def test_post_launch_execution_route_change_fails_verification(self):
        self.launch()
        self.mocks["_execution_route_matches"].return_value = False
        with self.assertRaisesRegex(RuntimeError, "route"):
            verify_service_launch(self.directory / "launch.json")

    def test_unsafe_ports_and_credential_urls_fail_before_launch(self):
        for port in (0, 80, 3001, True, "3011", 70000):
            with self.subTest(port=port), self.assertRaises(ValueError):
                launch_agent_service(
                    self.root,
                    port=port,
                    expected_sha=self.sha,
                    recorder_url="http://127.0.0.1:39193",
                    run_directory=self.directory,
                )
        with self.assertRaises(ValueError):
            launch_agent_service(
                self.root,
                port=3011,
                expected_sha=self.sha,
                recorder_url="http://user:secret@127.0.0.1:39193",
                run_directory=self.directory,
            )
        self.mocks["_launch_command"].assert_not_called()

    def test_stop_targets_only_the_owned_empty_agent_service(self):
        self.launch()
        with (
            patch("utils.pilot_launch._agents_empty", return_value=True),
            patch("utils.pilot_launch._same_process", return_value=False),
        ):
            result = stop_agent_service(self.directory / "launch.json")
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(
            self.mocks["_launch_command"].call_args.args[0],
            [str(self.root / "bin/local-dev.sh"), "down", "agent-service"],
        )
        self.assertEqual(
            self.mocks["_launch_command"].call_args.kwargs["environment"]["TEXERA_AGENT_SERVICE_PORT"],
            "3011",
        )

    def test_an_agent_service_with_resources_is_not_stopped(self):
        self.launch()
        with patch("utils.pilot_launch._agents_empty", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "agents"):
                stop_agent_service(self.directory / "launch.json")
        self.mocks["_launch_command"].assert_called_once()  # launch, not down


if __name__ == "__main__":
    unittest.main()

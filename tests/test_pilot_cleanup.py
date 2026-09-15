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
from unittest.mock import patch

from utils.pilot_cleanup import PilotResourceAPI, cleanup_owned_resources, create_owned_computing_unit
from utils.resource_journal import ResourceJournal
from test_execution_journal import HEADER, request, footer


class PilotCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cu_journal = self.directory / "cu.jsonl"
        self.workflow_journal = self.directory / "resources.jsonl"
        self.execution_journal = self.directory / "execution_requests.jsonl"
        self.api = PilotResourceAPI(
            token="private-token",
            agent_endpoint="http://127.0.0.1:3011",
            texera_endpoint="http://127.0.0.1:8080",
            computing_unit_endpoint="http://127.0.0.1:8888",
        )
        with ResourceJournal(self.cu_journal) as journal:
            self.cu_name = "native-python-" + journal.run_id + "-cu"
            journal.record({"event": "create_intent", "kind": "computing_unit", "name": self.cu_name})
            journal.record({"event": "created", "kind": "computing_unit", "name": self.cu_name, "id": 321})
        with ResourceJournal(self.workflow_journal) as journal:
            self.workflow_name = "native-python-" + journal.run_id + "-workflow"
            self.agent_name = "native-python-" + journal.run_id + "-agent"
            journal.record({"event": "reference", "kind": "computing_unit", "id": 321})
            for kind, name, identifier in (
                ("workflow", self.workflow_name, 456),
                ("agent", self.agent_name, "fixture-agent"),
            ):
                journal.record({"event": "create_intent", "kind": kind, "name": name})
                journal.record({"event": "created", "kind": kind, "name": name, "id": identifier})
        self.header = {
            **HEADER,
            "computingUnitIds": [321],
            "url": "http://127.0.0.1:39193",
            "backend": "http://127.0.0.1:8085",
            "source": {"gitSha": "a" * 40, "dirty": False},
        }
        self.events = [self.header, *request("r"), self.footer(1)]
        self.execution_journal.write_text("".join(json.dumps(event) + "\n" for event in self.events))
        (self.directory / "capture.json").write_text(json.dumps({"complete": True}))
        (self.directory / "attempt.json").write_text(
            json.dumps(
                {
                    "resources": {
                        "agent_id": "fixture-agent",
                        "workflow_id": 456,
                        "computing_unit_id": 321,
                    }
                }
            )
        )
        self.agent = {
            "id": "fixture-agent",
            "name": self.agent_name,
            "state": "AVAILABLE",
            "delegate": {"workflowId": 456, "computingUnitId": 321},
        }
        self.workflow = {"isOwner": True, "workflow": {"wid": 456, "name": self.workflow_name}}
        self.unit = {
            "isOwner": True,
            "computingUnit": {"cuid": 321, "name": self.cu_name, "type": "local", "uri": self.header["backend"]},
        }
        self.history = [{"eId": 101, "cuId": 321, "status": 3}]
        self.mutations = []
        self.request = patch.object(self.api, "request", side_effect=self.serve).start()
        self.source = patch(
            "utils.pilot_cleanup.verify_service_launch",
            return_value={
                "loaded_revision_verified": True,
                "service_endpoint": self.api.agent_endpoint,
                "recorder_url": self.header["url"],
                "pid": 100,
                "process_start_ticks": 1000,
            },
        ).start()
        self.addCleanup(patch.stopall)

    def serve(self, service, method, path, payload=None):
        if method != "GET":
            self.mutations.append((service, method, path, payload))
            if path == "/api/agents/fixture-agent":
                self.agent = None
            elif path == "/api/workflow/delete":
                self.workflow = None
            elif path == "/api/computing-unit/321/terminate":
                self.unit = None
            else:
                raise AssertionError(path)
            return None
        if path == "/api/agents/fixture-agent":
            return self.agent
        if path == "/api/agents/":
            return {"agents": [self.agent] if self.agent else []}
        if path == "/api/workflow/list":
            return [self.workflow] if self.workflow else []
        if path == "/api/computing-unit":
            return [self.unit] if self.unit else []
        if path == "/api/executions/456":
            return self.history
        raise AssertionError(path)

    def cleanup(self):
        return cleanup_owned_resources(
            self.api,
            task_directory=self.directory,
            computing_unit_journal=self.cu_journal,
            workflow_journal=self.workflow_journal,
            execution_journal=self.execution_journal,
            recorder_id=HEADER["instanceId"],
            service_launch_record=self.directory / "launch.json",
            output_directory=self.directory / "cleanup",
        )

    def footer(self, completed):
        return {**footer(completed), "computingUnitIds": [321], "backend": self.header["backend"]}

    def test_malformed_absence_response_stops_cleanup(self):
        def malformed(service, method, path, payload=None):
            if path == "/api/agents/":
                return {"agents": [{"name": "missing-identity"}]}
            return self.serve(service, method, path, payload)

        self.request.side_effect = malformed
        report = self.cleanup()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["completed"], [])
        self.assertIn("agent", report["unresolved"])
        self.assertEqual(len(self.mutations), 1)

    def test_only_owned_resources_are_deleted_after_terminal_verification(self):
        report = self.cleanup()
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["backend_terminal_verified"])
        self.assertEqual(
            self.mutations,
            [
                ("agent", "DELETE", "/api/agents/fixture-agent", None),
                ("texera", "POST", "/api/workflow/delete", {"wids": [456]}),
                ("computing_unit", "DELETE", "/api/computing-unit/321/terminate", None),
            ],
        )
        self.assertNotIn("private-token", (self.directory / "cleanup/cleanup.json").read_text())

    def test_active_backend_or_agent_never_reaches_delete(self):
        self.history[0]["status"] = 1
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_active_agent_is_not_stopped_as_a_cleanup_shortcut(self):
        self.agent["state"] = "RUNNING"
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_owned_unit_must_be_local_and_match_the_recorded_backend(self):
        self.unit["computingUnit"]["type"] = "kubernetes"
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_reference_or_name_is_not_sufficient_ownership(self):
        self.workflow["isOwner"] = False
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_incomplete_capture_or_execution_journal_preserves_resources(self):
        self.execution_journal.write_text("".join(json.dumps(event) + "\n" for event in self.events[:-1]))
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_unknown_or_unmatched_execution_outcomes_do_not_become_terminal(self):
        self.history[0]["eId"] = 999
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_another_workflow_using_the_unit_blocks_automatic_cleanup(self):
        other = request("other-workflow")
        for event in other:
            event["record"]["workflowId"] = 999
        events = [*self.events[:-1], *other, self.footer(2)]
        self.execution_journal.write_text("".join(json.dumps(event) + "\n" for event in events))
        self.assertEqual(self.cleanup()["status"], "blocked")
        self.assertEqual(self.mutations, [])

    def test_partial_cleanup_keeps_the_unresolved_ids(self):
        def fail(service, method, path, payload=None):
            if path == "/api/workflow/delete":
                raise ConnectionError("private-token")
            return self.serve(service, method, path, payload)

        self.request.side_effect = fail
        report = self.cleanup()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["completed"], ["agent"])
        self.assertEqual(report["unresolved"], {"workflow": 456, "computing_unit": 321})
        self.assertNotIn("private-token", str(report))

    def test_zero_execution_setup_can_be_cleaned_without_inventing_a_pass_rate(self):
        self.execution_journal.write_text(json.dumps(self.header) + "\n" + json.dumps(self.footer(0)) + "\n")
        self.history = []
        report = self.cleanup()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["execution_ids"], [])

    def test_computing_unit_creation_is_explicit_journaled_and_not_discovery(self):
        path = self.directory / "new-cu.jsonl"

        def create(service, method, route, payload=None):
            self.assertEqual((service, method, route), ("computing_unit", "POST", "/api/computing-unit/create"))
            self.assertEqual(payload["unitType"], "local")
            self.assertIn('"create_intent"', path.read_text())
            return {
                "isOwner": True,
                "computingUnit": {"cuid": 999, "type": "local", "uri": payload["uri"], "name": payload["name"]},
            }

        self.request.side_effect = create
        with ResourceJournal(path) as journal:
            self.assertEqual(create_owned_computing_unit(self.api, journal, backend=self.header["backend"]), 999)
        self.assertIn('"created"', path.read_text())


if __name__ == "__main__":
    unittest.main()

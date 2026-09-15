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
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dataflow_agent import DataflowAgent
from utils.resource_journal import ResourceJournal, read_resource_journal


class ResourceJournalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "resources.jsonl"

    def test_creation_receipts_are_private_and_references_are_not_ownership(self):
        with ResourceJournal(self.path) as journal:
            journal.record({"event": "reference", "kind": "computing_unit", "id": 321})
            journal.record({"event": "create_intent", "kind": "workflow", "name": "owned-fixture"})
            pending = read_resource_journal(self.path)
            self.assertEqual(pending["pending"], ["workflow"])
            self.assertNotIn("computing_unit", pending["owned"])
            journal.record({"event": "created", "kind": "workflow", "name": "owned-fixture", "id": 456})
            self.assertEqual(read_resource_journal(self.path)["owned"]["workflow"]["id"], 456)
            self.assertFalse(read_resource_journal(self.path)["closed"])
        self.assertTrue(read_resource_journal(self.path)["closed"])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            ResourceJournal(self.path)

    def test_unknown_receipts_retries_and_secret_fields_are_rejected(self):
        with ResourceJournal(self.path) as journal:
            for invalid in (
                {"event": "created", "kind": "agent", "id": "unknown", "name": "fixture"},
                {"event": "reference", "kind": "computing_unit", "id": 321, "token": "private-token"},
                {"event": "create_intent", "kind": "workflow", "name": "fixture", "delegate": {}},
            ):
                with self.assertRaises(ValueError):
                    journal.record(invalid)
            journal.record({"event": "create_intent", "kind": "workflow", "name": "fixture"})
            with self.assertRaises(ValueError):
                journal.record({"event": "create_intent", "kind": "workflow", "name": "fixture"})
        self.assertNotIn("private-token", self.path.read_text())
        self.assertEqual(read_resource_journal(self.path)["pending"], ["workflow"])

    def test_a_dangling_intent_is_not_automatically_recovered_or_retried(self):
        with ResourceJournal(self.path) as journal:
            journal.record({"event": "create_intent", "kind": "workflow", "name": "fixture"})
        report = read_resource_journal(self.path)
        self.assertEqual(report["owned"], {})
        self.assertEqual(report["pending"], ["workflow"])
        self.assertFalse(report["allocations_complete"])

    def test_changed_names_and_truncated_journals_fail_validation(self):
        with ResourceJournal(self.path) as journal:
            journal.record({"event": "create_intent", "kind": "agent", "name": "fixture"})
            with self.assertRaises(ValueError):
                journal.record({"event": "created", "kind": "agent", "name": "another", "id": "new-agent"})
        with self.path.open("a") as stream:
            stream.write('{"event":')
        with self.assertRaises(ValueError):
            read_resource_journal(self.path)


class SetupJournalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "resources.jsonl"
        self.agent = DataflowAgent(
            computing_unit_id=321, workflow_name="workflow-fixture", agent_name="agent-fixture", verbosity_level=0
        )
        self.login = patch("dataflow_agent.login", return_value="private-token").start()
        self.discovery = patch("dataflow_agent.get_or_create_computing_unit", return_value=321).start()
        self.workflow = patch("dataflow_agent.create_workflow", return_value=456).start()
        self.create_agent = patch(
            "dataflow_agent.create_agent", return_value=SimpleNamespace(id="new-agent", name="agent-fixture")
        ).start()
        self.addCleanup(patch.stopall)

    def test_receipts_are_written_before_the_next_allocation(self):
        def create_agent(**kwargs):
            recorded = read_resource_journal(self.path)
            self.assertEqual(recorded["owned"]["workflow"]["id"], 456)
            self.assertEqual(recorded["pending"], ["agent"])
            self.assertEqual(kwargs["timeout"], (3, 15))
            self.assertIs(kwargs["allow_redirects"], False)
            return SimpleNamespace(id="new-agent", name="agent-fixture")

        self.create_agent.side_effect = create_agent
        with ResourceJournal(self.path) as journal:
            self.agent.setup(on_resource=journal.record)
        self.assertEqual(read_resource_journal(self.path)["owned"]["agent"]["id"], "new-agent")
        self.assertNotIn("private-token", self.path.read_text())
        self.discovery.assert_not_called()

    def test_an_intent_write_failure_prevents_resource_creation(self):
        def failure(event):
            if event["event"] == "create_intent":
                raise OSError("simulated journal failure")

        with self.assertRaises(OSError):
            self.agent.setup(on_resource=failure)
        self.workflow.assert_not_called()
        self.create_agent.assert_not_called()

    def test_failed_receipt_retains_the_created_id_and_stops_before_agent_creation(self):
        def failure(event):
            if event["event"] == "created":
                raise OSError("simulated journal failure")

        with self.assertRaises(OSError):
            self.agent.setup(on_resource=failure)
        self.assertEqual(self.agent._workflow_id, 456)
        self.create_agent.assert_not_called()

    def test_journaled_setup_requires_an_explicit_cu_and_synchronous_callback(self):
        self.agent.computing_unit_id = None
        with self.assertRaises(ValueError):
            self.agent.setup(on_resource=Mock())
        self.login.assert_not_called()
        self.discovery.assert_not_called()
        self.agent.computing_unit_id = 321
        with self.assertRaises(TypeError):
            self.agent.setup(on_resource=lambda event: True)
        self.workflow.assert_not_called()

    def test_legacy_setup_keeps_the_old_optional_transport_policy(self):
        self.agent.setup()
        self.assertNotIn("timeout", self.workflow.call_args.kwargs)
        self.assertNotIn("allow_redirects", self.create_agent.call_args.kwargs)

    def test_actual_native_setup_wires_the_journal_without_changing_settings(self):
        from systems.native_python_system import DataflowSystemLunaPythonPilotDataOnly20260915Rep1 as Arm
        from utils.pilot_artifacts import AttemptBundle

        system = Arm(output_dir=self.temporary.name, computing_unit_id=321)
        system.pilot_bundle = AttemptBundle(Path(self.temporary.name) / "attempt")
        self.create_agent.side_effect = lambda **kwargs: SimpleNamespace(id="new-agent", name=kwargs["name"])
        system._setup_agent()
        recorded = read_resource_journal(system.pilot_bundle.path / "resource_allocations.jsonl")
        self.assertEqual(recorded["owned"]["agent"]["id"], "new-agent")
        self.assertEqual(recorded["owned"]["workflow"]["id"], 456)
        self.assertNotIn("computing_unit", recorded["owned"])
        self.assertTrue(recorded["allocations_complete"])
        settings = self.create_agent.call_args.kwargs["settings"].to_api_dict()
        self.assertEqual(settings["nativeCatalogVersion"], "v2")
        self.assertIs(settings["nativeProfileCollection"], True)

    def test_native_setup_without_a_reserved_attempt_fails_before_api_calls(self):
        from systems.native_python_system import DataflowSystemLunaPythonPilotDataOnly20260915Rep1 as Arm

        system = Arm(output_dir=self.temporary.name, computing_unit_id=321)
        with self.assertRaisesRegex(RuntimeError, "reserved attempt"):
            system._setup_agent()
        self.login.assert_not_called()
        self.workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()

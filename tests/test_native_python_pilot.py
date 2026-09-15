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
from inspect import Parameter, signature
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from dataflow_agent import AgentSettings, DataflowAgent
from utils.pilot_artifacts import AttemptBundle


class NativePythonClientTest(unittest.TestCase):
    def test_new_options_do_not_shift_legacy_positional_parameters(self):
        from systems.dataflow_system import DataflowSystem

        for constructor in (AgentSettings, DataflowAgent, DataflowSystem):
            params = signature(constructor).parameters
            for name in ("native_catalog_version", "native_profile_collection"):
                self.assertIs(params[name].kind, Parameter.KEYWORD_ONLY)
            if constructor is not AgentSettings:
                self.assertIs(params["computing_unit_id"].kind, Parameter.KEYWORD_ONLY)

    def test_legacy_payload_does_not_opt_into_v2_or_collection(self):
        payload = AgentSettings().to_api_dict()
        self.assertNotIn("nativeCatalogVersion", payload)
        self.assertNotIn("nativeProfileCollection", payload)

    def test_explicit_v2_settings_survive_client_and_wire(self):
        for collection in (False, True):
            with self.subTest(collection=collection):
                agent = DataflowAgent(
                    native_tool_mode="batch",
                    native_catalog_version="v2",
                    native_profile_collection=collection,
                    parallel_tool_calls=False,
                )
                payload = agent.settings.to_api_dict()
                self.assertEqual(payload["nativeCatalogVersion"], "v2")
                self.assertEqual(payload["nativeToolMode"], "batch")
                self.assertIs(payload["nativeProfileCollection"], collection)
                self.assertIs(payload["parallelToolCalls"], False)
                self.assertNotIn("nativeFlowEvidence", payload)

    def test_pinned_computing_unit_survives_fresh_setups(self):
        with (
            patch("dataflow_agent.login", return_value="test-token"),
            patch("dataflow_agent.get_or_create_computing_unit") as discover,
            patch("dataflow_agent.create_workflow", return_value=123),
            patch("dataflow_agent.create_agent", return_value=SimpleNamespace(id="test-agent", name="test")) as create,
            patch("dataflow_agent.delete_agent"),
            patch("dataflow_agent.delete_workflow"),
        ):
            agent = DataflowAgent(computing_unit_id=321, verbosity_level=0)
            for _ in range(2):
                agent.setup()
                self.assertEqual(create.call_args.kwargs["computing_unit_id"], 321)
                agent.cleanup()
            discover.assert_not_called()

    def test_invalid_computing_unit_fails_before_network_calls(self):
        for value in (0, -1, True, "321", 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                DataflowAgent(computing_unit_id=value)


class NativePythonArmsTest(unittest.TestCase):
    def test_registered_arms_isolate_framework_collection_and_exposure(self):
        import systems
        from systems.native_python_system import PILOT_ARMS

        expected = {
            "V1Reference": (3013, None, None, True, None, 1, 1),
            "BatchParent": (3012, None, "batch", False, None, 1, 1),
            "V2Control": (3011, "v2", "batch", False, False, 0, 0),
            "CollectorOnly": (3011, "v2", "batch", False, True, 0, 0),
            "DataOnly": (3011, "v2", "batch", False, True, 1, 0),
            "FlowOnly": (3011, "v2", "batch", False, True, 0, 1),
            "Combined": (3011, "v2", "batch", False, True, 1, 1),
        }
        self.assertEqual({arm.key for arm in PILOT_ARMS}, set(expected))
        with TemporaryDirectory() as directory:
            for spec in PILOT_ARMS:
                with self.subTest(arm=spec.key):
                    arm = getattr(systems, spec.system_name)(output_dir=directory, computing_unit_id=321)
                    port, version, mode, parallel, collect, data, flow = expected[spec.key]
                    self.assertEqual(arm.agent_service_endpoint, f"http://localhost:{port}")
                    self.assertEqual(arm.native_catalog_version, version)
                    self.assertEqual(arm.native_tool_mode, mode)
                    self.assertIs(arm.parallel_tool_calls, parallel)
                    self.assertIs(arm.native_profile_collection, collect)
                    self.assertEqual((arm.data_level, arm.flow_level), (data, flow))
                    self.assertIsNone(arm.native_flow_evidence)
                    self.assertEqual(arm.computing_unit_id, 321)
                    self.assertEqual(arm.model_type, "gpt-5.6-luna")
                    self.assertEqual(arm.agent_mode, "native")
                    self.assertEqual(arm.context_mode, "delta")
                    self.assertEqual(arm.result_selection, "all")
                    self.assertFalse(arm.column_stats)
                    self.assertFalse(arm.thought_replay)
                    self.assertFalse(arm.enable_code_in_snapshot)
                    self.assertTrue(arm.attempt_reflection)
                    self.assertEqual(arm.max_steps, 25)
                    self.assertEqual(arm.execution_timeout_minutes, 10)
                    self.assertEqual(arm.max_operator_result_char_limit, 2000)
                    self.assertEqual(arm.max_operator_result_cell_char_limit, 3000)
                    # Exercise the production system-to-client boundary without
                    # authenticating, creating resources, or calling an LLM.
                    arm.pilot_bundle = AttemptBundle(Path(directory) / spec.key)
                    with patch.object(DataflowAgent, "setup"):
                        arm._setup_agent()
                    payload = arm.agent.settings.to_api_dict()
                    self.assertEqual(arm.agent.computing_unit_id, 321)
                    self.assertEqual(payload["dataLevel"], data)
                    self.assertEqual(payload["flowLevel"], flow)
                    self.assertEqual(payload.get("nativeCatalogVersion"), version)
                    self.assertIs(payload.get("nativeProfileCollection"), collect)
                    self.assertNotIn("nativeFlowEvidence", payload)

    def test_pilot_requires_an_explicit_isolated_computing_unit(self):
        from systems.native_python_system import DataflowSystemLunaPythonPilotV2Control20260915Rep1 as Arm

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "NATIVE_PYTHON_PILOT_CUID"):
                Arm()
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"NATIVE_PYTHON_PILOT_CUID": "456"}):
            self.assertEqual(Arm(output_dir=directory).computing_unit_id, 456)
            self.assertEqual(Arm(output_dir=directory, computing_unit_id=321).computing_unit_id, 321)
            for value in (0, -1, True, "321"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    Arm(output_dir=directory, computing_unit_id=value)

    def test_treatment_overrides_cannot_reuse_a_frozen_sut_name(self):
        from systems.native_python_system import DataflowSystemLunaPythonPilotV2Control20260915Rep1 as Arm

        for key, value in (
            ("model_type", "other"),
            ("native_profile_collection", True),
            ("data_level", 1),
            ("flow_level", 1),
            ("max_steps", 26),
            ("native_flow_evidence", True),
            ("column_stats", True),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frozen"):
                Arm(computing_unit_id=321, **{key: value})
        with TemporaryDirectory() as directory:
            arm = Arm(output_dir=directory, computing_unit_id=321, max_steps=25)
            self.assertEqual(arm.max_steps, 25)


if __name__ == "__main__":
    unittest.main()

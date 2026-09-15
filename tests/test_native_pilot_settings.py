# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import unittest
from tempfile import TemporaryDirectory

from dataflow_agent import AgentSettings, DataflowAgent


class NativePilotSettingsTest(unittest.TestCase):
    def test_defaults_leave_experimental_settings_absent(self):
        payload = AgentSettings().to_api_dict()
        self.assertNotIn("nativeToolMode", payload)
        self.assertNotIn("nativeFlowEvidence", payload)

    def test_client_transmits_explicit_native_settings(self):
        agent = DataflowAgent(native_tool_mode="batch", native_flow_evidence=True)
        payload = agent.settings.to_api_dict()
        self.assertEqual(payload["nativeToolMode"], "batch")
        self.assertIs(payload["nativeFlowEvidence"], True)

    def test_explicit_false_reaches_service(self):
        payload = AgentSettings(native_flow_evidence=False).to_api_dict()
        self.assertIs(payload["nativeFlowEvidence"], False)

    def test_pilot_arms_share_anchor_and_use_distinct_ports(self):
        import systems

        variants = [
            ("Baseline", 3010, None, None, True),
            ("Batch", 3008, "batch", None, False),
            ("Evidence", 3009, None, True, True),
        ]
        with TemporaryDirectory() as directory:
            for variant, port, mode, evidence, parallel in variants:
                with self.subTest(variant=variant):
                    arm = getattr(systems, f"DataflowSystemLunaNativePilot{variant}20260914Rep1")(
                        output_dir=directory,
                    )
                    self.assertEqual(arm.agent_service_endpoint, f"http://localhost:{port}")
                    self.assertEqual(arm.native_tool_mode, mode)
                    self.assertEqual(arm.native_flow_evidence, evidence)
                    self.assertEqual(arm.parallel_tool_calls, parallel)
                    self.assertEqual(arm.model_type, "gpt-5.6-luna")
                    self.assertEqual(arm.agent_mode, "native")
                    self.assertEqual(arm.context_mode, "delta")
                    self.assertEqual(arm.result_selection, "all")
                    self.assertFalse(arm.column_stats)
                    self.assertEqual(arm.data_level, 1)
                    self.assertEqual(arm.max_steps, 25)
                    self.assertEqual(arm.max_operator_result_char_limit, 2000)


if __name__ == "__main__":
    unittest.main()

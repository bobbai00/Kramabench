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

"""Evidence-only wire contract; no services, LLM calls or benchmark writes."""

import unittest
from tempfile import TemporaryDirectory

from dataflow_agent import AgentSettings, DataflowAgent
from systems.dataflow_system import DataflowSystem


class EvidenceOnlySettingsTest(unittest.TestCase):
    def test_all_four_arms_preserve_explicit_booleans(self):
        for data in (False, True):
            for flow in (False, True):
                with self.subTest(data=data, flow=flow):
                    settings = AgentSettings(data_evidence=data, flow_evidence=flow)
                    wire = settings.to_api_dict()
                    self.assertIs(wire["dataEvidence"], data)
                    self.assertIs(wire["flowEvidence"], flow)
                    for name in ("dataLevel", "flowLevel", "columnStats", "statScopes", "nativeFlowEvidence"):
                        self.assertNotIn(name, wire)

    def test_client_carries_new_fields(self):
        agent = DataflowAgent(data_evidence=True, flow_evidence=False)
        self.assertIs(agent.settings.data_evidence, True)
        self.assertIs(agent.settings.flow_evidence, False)

    def test_removed_constructor_options_are_not_aliases(self):
        for key in ("data_level", "flow_level", "column_stats", "stat_scopes", "native_flow_evidence"):
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    AgentSettings(**{key: False})
                with self.assertRaises(TypeError):
                    DataflowAgent(**{key: False})
                with self.assertRaisesRegex(ValueError, "removed"):
                    DataflowSystem(**{key: False})

    def test_client_rejects_truthy_non_booleans(self):
        for key in ("data_evidence", "flow_evidence"):
            for value in (1, "true", None):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    AgentSettings(**{key: value}).to_api_dict()

    def test_sut_keeps_exposure_separate_from_collection(self):
        with TemporaryDirectory() as output_dir:
            sut = DataflowSystem(
                name="EvidenceOnlyWireTest", output_dir=output_dir,
                native_catalog_version="v2", native_tool_mode="batch",
                native_profile_collection=True, data_evidence=False, flow_evidence=True,
            )
            self.assertIs(sut.data_evidence, False)
            self.assertIs(sut.flow_evidence, True)
            self.assertIs(sut.native_profile_collection, True)


if __name__ == "__main__":
    unittest.main()

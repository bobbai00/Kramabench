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
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import systems
from systems.evidence_only_system import EVIDENCE_ONLY_ARMS
from systems.native_campaign_system import NativeCampaignSystem, EVIDENCE_ONLY_CAMPAIGN_ID
from utils.native_campaign import CampaignGuard


class EvidenceOnlyPilotTest(unittest.TestCase):
    def test_new_campaign_keeps_no_inspector_admission(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text(json.dumps({'version': 1, 'campaign': EVIDENCE_ONLY_CAMPAIGN_ID,
                                            'bindings': [{'fixture': True}]}))
            with patch.dict(os.environ, {'NATIVE_CAMPAIGN_MANIFEST': str(manifest)}):
                guard = CampaignGuard(campaign=EVIDENCE_ONLY_CAMPAIGN_ID)
                self.assertTrue(guard.observe_only)
                with self.assertRaisesRegex(ValueError, 'campaign_manifest_invalid'):
                    CampaignGuard(campaign='unregistered')

    def test_matched_terra_medium_matrix(self):
        self.assertEqual(len(EVIDENCE_ONLY_ARMS), 4)
        self.assertEqual({(a.data, a.flow) for a in EVIDENCE_ONLY_ARMS},
                         {(False, False), (True, False), (False, True), (True, True)})
        common = []
        with TemporaryDirectory() as output:
            for spec in EVIDENCE_ONLY_ARMS:
                self.assertEqual(spec.model_type, 'gpt-5.6-terra')
                self.assertEqual(spec.reasoning_effort, 'medium')
                settings = spec.settings()
                for key in ('data_level', 'flow_level', 'column_stats', 'native_flow_evidence', 'stats_enabled'):
                    self.assertNotIn(key, settings)
                self.assertTrue(settings['native_profile_collection'])
                self.assertFalse(settings['enable_inspect_tool'])
                self.assertEqual(settings['max_steps'], 25)
                self.assertEqual(settings['max_operator_result_char_limit'], 2000)
                arm = getattr(systems, spec.system_name)(output_dir=output, computing_unit_id=123)
                self.assertIsInstance(arm, NativeCampaignSystem)
                self.assertIs(arm.data_evidence, spec.data)
                self.assertIs(arm.flow_evidence, spec.flow)
                self.assertEqual(arm.campaign_id, spec.campaign_id)
                self.assertIn('EvidenceOnlyPilot', spec.system_name)
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    getattr(systems, spec.system_name)(output_dir=output, computing_unit_id=123,
                                                      data_evidence=not spec.data)
                common.append({k: v for k, v in settings.items() if k not in ('data_evidence', 'flow_evidence')})
        self.assertTrue(all(row == common[0] for row in common))


if __name__ == '__main__':
    unittest.main()

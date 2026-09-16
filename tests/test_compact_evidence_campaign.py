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
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import systems
from systems.compact_evidence_system import COMPACT_EVIDENCE_ARMS, COMPACT_SMOKE_ARMS
from systems.evidence_only_system import EvidenceOnlyArm
from systems.native_campaign_system import COMPACT_EVIDENCE_CAMPAIGN_ID
from utils.native_campaign import CampaignGuard


class CompactEvidenceCampaignTest(unittest.TestCase):
    def test_matched_full_arms_and_disjoint_smoke(self):
        self.assertEqual(len(COMPACT_EVIDENCE_ARMS), 3)
        self.assertEqual({(a.data, a.flow) for a in COMPACT_EVIDENCE_ARMS},
                         {(True, False), (False, True), (True, True)})
        self.assertEqual({a.port for a in COMPACT_EVIDENCE_ARMS}, {3017, 3018, 3019})
        all_arms = COMPACT_EVIDENCE_ARMS + COMPACT_SMOKE_ARMS
        self.assertEqual(len({a.system_name for a in all_arms}), 6)
        with TemporaryDirectory() as output:
            for spec in all_arms:
                self.assertEqual(spec.settings(), EvidenceOnlyArm(spec.key, spec.data, spec.flow).settings())
                self.assertEqual(spec.model_type, 'gpt-5.6-terra')
                self.assertEqual(spec.reasoning_effort, 'medium')
                self.assertEqual(spec.campaign_id, COMPACT_EVIDENCE_CAMPAIGN_ID)
                arm = getattr(systems, spec.system_name)(output_dir=output, computing_unit_id=123)
                self.assertEqual(arm.agent_service_endpoint, f'http://localhost:{spec.port}')
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    getattr(systems, spec.system_name)(output_dir=output, computing_unit_id=123,
                                                      data_evidence=not spec.data)

    def test_registered_campaign_requires_no_inspector(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text(json.dumps({'version': 1, 'campaign': COMPACT_EVIDENCE_CAMPAIGN_ID,
                                            'bindings': [{'fixture': True}]}))
            with patch.dict(os.environ, {'NATIVE_CAMPAIGN_MANIFEST': str(manifest)}):
                self.assertTrue(CampaignGuard(campaign=COMPACT_EVIDENCE_CAMPAIGN_ID).observe_only)
                with self.assertRaisesRegex(ValueError, 'campaign_manifest_invalid'):
                    CampaignGuard(campaign='unregistered')


if __name__ == '__main__':
    unittest.main()

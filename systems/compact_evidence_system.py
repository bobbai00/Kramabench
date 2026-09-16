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

"""Matched compact-evidence campaign, with separate smoke artifact namespaces."""

from dataclasses import dataclass, replace

from .evidence_only_system import EvidenceOnlyArm
from .native_campaign_system import COMPACT_EVIDENCE_CAMPAIGN_ID, _make_arm


@dataclass(frozen=True)
class CompactEvidenceArm(EvidenceOnlyArm):
    campaign_id: str = COMPACT_EVIDENCE_CAMPAIGN_ID
    phase: str = 'Full'

    @property
    def system_name(self):
        return f'DataflowSystemTerraCompactEvidence{self.phase}{self.key}20260916Rep1'


COMPACT_EVIDENCE_ARMS = (
    CompactEvidenceArm('DataOnly', True, False, port=3017),
    CompactEvidenceArm('FlowOnly', False, True, port=3018),
    CompactEvidenceArm('Combined', True, True, port=3019),
)
COMPACT_SMOKE_ARMS = tuple(replace(spec, phase='Smoke') for spec in COMPACT_EVIDENCE_ARMS)
__all__ = [spec.system_name for spec in COMPACT_EVIDENCE_ARMS + COMPACT_SMOKE_ARMS]
for _spec in COMPACT_EVIDENCE_ARMS + COMPACT_SMOKE_ARMS:
    globals()[_spec.system_name] = _make_arm(_spec)

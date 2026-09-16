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

"""Matched V2 evidence-only pilot; fresh namespaces, no legacy-setting aliases."""

from dataclasses import dataclass

from .native_campaign_system import EVIDENCE_ONLY_CAMPAIGN_ID, _make_arm


@dataclass(frozen=True)
class EvidenceOnlyArm:
    key: str
    data: bool
    flow: bool
    port: int = 3015
    model_type: str = 'gpt-5.6-terra'
    reasoning_effort: str = 'medium'
    campaign_id: str = EVIDENCE_ONLY_CAMPAIGN_ID

    @property
    def system_name(self):
        return f'DataflowSystemTerraEvidenceOnlyPilot{self.key}20260916Rep1'

    def settings(self):
        return {
            'model_type': self.model_type, 'driver': None,
            'agent_mode': 'native', 'context_mode': 'delta',
            'native_tool_mode': 'batch', 'native_catalog_version': 'v2',
            'native_profile_collection': True,
            'data_evidence': self.data, 'flow_evidence': self.flow,
            'parallel_tool_calls': False, 'max_steps': 25, 'max_operator_edits': 0,
            'max_operator_result_char_limit': 2000,
            'max_operator_result_cell_char_limit': 3000,
            'operator_result_serialization_mode': 'tsv',
            'tool_timeout_seconds': 240, 'execution_timeout_minutes': 10,
            'result_selection': 'all', 'attempt_reflection': True,
            'enable_code_in_snapshot': False, 'thought_replay': False,
            'context_window_tokens': 0, 'static_compaction': False,
            'enable_inspect_tool': False, 'enable_render_prefs': False,
            'enable_recall_tool': False, 'enable_resume_tool': False,
            'enable_answer_grounding': False, 'error_reflection': False,
            'few_shot_prompt': False, 'agent_turns': False,
            'session_turns': False, 'versioned_mode': False, 'message_layout': None,
        }


EVIDENCE_ONLY_ARMS = (
    EvidenceOnlyArm('NoEvidence', False, False),
    EvidenceOnlyArm('DataOnly', True, False),
    EvidenceOnlyArm('FlowOnly', False, True),
    EvidenceOnlyArm('Combined', True, True),
)
__all__ = [spec.system_name for spec in EVIDENCE_ONLY_ARMS]
for _spec in EVIDENCE_ONLY_ARMS:
    globals()[_spec.system_name] = _make_arm(_spec)

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

"""2x2 probe: DELTA vs the opBlock LATEST layout, with and without evidence.

Every knob below is copied verbatim from the 2026-09-16 compact-evidence
campaign (`evidence_only_system.EvidenceOnlyArm.settings`) so the DELTA arms
reproduce that protocol. Only two things vary:

    context_mode     delta -> latest
    message_layout   None  -> "opBlock"

`opBlock` is the retained-observation layout: one user message per operator in
creation order, each carrying that operator's inputs, parameters and the result
ITS CURRENT REVISION produced. Today's LATEST layout drops an operator's result
to `result not requested` the moment the agent observes something else, which
rewrites an early message for a reason unrelated to that operator and voids the
provider's prefix cache from there down. opBlock removes that rewrite; the
retained body is bound to its result version, so a redefinition or a re-execution
discards it rather than presenting a stale measurement as current.

This is a 5-task probe, NOT a campaign: it measures cost, step count and cache
split. It is far too small for an accuracy claim.
"""

import os

from .dataflow_system import DataflowSystem

# Frozen protocol shared by all four arms — the compact-evidence campaign's.
_FROZEN = dict(
    model_type="gpt-5.6-terra",
    driver=None,
    agent_mode="native",
    native_tool_mode="batch",
    native_catalog_version="v2",
    native_profile_collection=True,
    parallel_tool_calls=False,
    max_steps=25,
    max_operator_edits=0,
    max_operator_result_char_limit=2000,
    max_operator_result_cell_char_limit=3000,
    operator_result_serialization_mode="tsv",
    tool_timeout_seconds=240,
    execution_timeout_minutes=10,
    result_selection="all",
    attempt_reflection=True,
    enable_code_in_snapshot=False,
    # LATEST-only reinjection. It renders BEFORE `# Current Dataflow`, so under a
    # per-operator layout it lands in the one message that has to stay
    # byte-stable and restarts the cache every step. Off in every arm so the
    # two context modes stay comparable.
    thought_replay=False,
    context_window_tokens=0,
    static_compaction=False,
    enable_inspect_tool=False,
    enable_render_prefs=False,
    enable_recall_tool=False,
    enable_resume_tool=False,
    enable_answer_grounding=False,
    error_reflection=False,
    few_shot_prompt=False,
    agent_turns=False,
    session_turns=False,
    versioned_mode=False,
)


class _LayoutProbeBase(DataflowSystem):
    _CONTEXT_MODE = "delta"
    _MESSAGE_LAYOUT = None
    _DATA = False
    _FLOW = False
    _NAME = "_LayoutProbeBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", os.environ.get("LAYOUT_PROBE_AGENT_ENDPOINT", "http://localhost:3001"))
        super().__init__(
            context_mode=self._CONTEXT_MODE,
            message_layout=self._MESSAGE_LAYOUT,
            data_evidence=self._DATA,
            flow_evidence=self._FLOW,
            name=self._NAME,
            verbose=verbose,
            *args,
            **_FROZEN,
            **kwargs,
        )


class DataflowSystemTerraLayoutDeltaPlain20260919(_LayoutProbeBase):
    """Control: the campaign's DELTA protocol, evidence off."""

    _CONTEXT_MODE = "delta"
    _MESSAGE_LAYOUT = None
    _DATA = False
    _FLOW = False
    _NAME = "DataflowSystemTerraLayoutDeltaPlain20260919"


class DataflowSystemTerraLayoutDeltaEvidence20260919(_LayoutProbeBase):
    """Control + statistics: the campaign's Combined arm."""

    _CONTEXT_MODE = "delta"
    _MESSAGE_LAYOUT = None
    _DATA = True
    _FLOW = True
    _NAME = "DataflowSystemTerraLayoutDeltaEvidence20260919"


class DataflowSystemTerraLayoutOpBlockPlain20260919(_LayoutProbeBase):
    """LATEST with one retained-observation message per operator, evidence off."""

    _CONTEXT_MODE = "latest"
    _MESSAGE_LAYOUT = "opBlock"
    _DATA = False
    _FLOW = False
    _NAME = "DataflowSystemTerraLayoutOpBlockPlain20260919"


class DataflowSystemTerraLayoutOpBlockEvidence20260919(_LayoutProbeBase):
    """opBlock + statistics."""

    _CONTEXT_MODE = "latest"
    _MESSAGE_LAYOUT = "opBlock"
    _DATA = True
    _FLOW = True
    _NAME = "DataflowSystemTerraLayoutOpBlockEvidence20260919"


LAYOUT_PROBE_ARMS = (
    DataflowSystemTerraLayoutDeltaPlain20260919,
    DataflowSystemTerraLayoutDeltaEvidence20260919,
    DataflowSystemTerraLayoutOpBlockPlain20260919,
    DataflowSystemTerraLayoutOpBlockEvidence20260919,
)
__all__ = [cls.__name__ for cls in LAYOUT_PROBE_ARMS]

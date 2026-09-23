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

"""Native mode as an expression program, 2026-09-22.

The agent writes the dataflow as `name = operator(...)` statements over the
single operator catalog, with three custom operators that carry Python
(filter, map, process). Everything else is the 2026-09-16 compact-evidence
campaign's frozen protocol: Terra-medium, DELTA, profiling on, 25 steps,
2,000-character results; the arms differ only in which evidence is on.

The retired `native_tool_mode` / `native_catalog_version` knobs are not sent:
the service rejects them, which is how the historical arms stay historical.
"""

import os

from .dataflow_system import DataflowSystem

_PROTOCOL = dict(
    model_type="gpt-5.6-terra",
    driver=None,
    agent_mode="native",
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


class _NativeExpressionBase(DataflowSystem):
    _CONTEXT_MODE = "delta"
    _DATA = True
    _FLOW = False
    _SEED = False
    _NAME = "_NativeExpressionBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault(
            "agent_service_endpoint", os.environ.get("NATIVE_EXPR_AGENT_ENDPOINT", "http://localhost:3061")
        )
        super().__init__(
            context_mode=self._CONTEXT_MODE,
            data_evidence=self._DATA,
            flow_evidence=self._FLOW,
            seed_sources=self._SEED,
            name=self._NAME,
            verbose=verbose,
            *args,
            **_PROTOCOL,
            **kwargs,
        )


class DataflowSystemTerraNativeExprDataOnly20260922(_NativeExpressionBase):
    """Expression program, DELTA, Data-only evidence: the campaign protocol on the new surface."""

    _NAME = "DataflowSystemTerraNativeExprDataOnly20260922"


class DataflowSystemTerraNativeExprSeededDataOnly20260922(_NativeExpressionBase):
    """The same with seeded source roots (the service scans the listed files first)."""

    _SEED = True
    _NAME = "DataflowSystemTerraNativeExprSeededDataOnly20260922"


class DataflowSystemTerraNativeExprFlowOnly20260922(_NativeExpressionBase):
    """Expression program, DELTA, Flow-only evidence (operator counters, no data measurements)."""

    _DATA = False
    _FLOW = True
    _NAME = "DataflowSystemTerraNativeExprFlowOnly20260922"


class DataflowSystemTerraNativeExprCombined20260922(_NativeExpressionBase):
    """Expression program, DELTA, Data + Flow evidence."""

    _DATA = True
    _FLOW = True
    _NAME = "DataflowSystemTerraNativeExprCombined20260922"


NATIVE_EXPRESSION_ARMS = (
    DataflowSystemTerraNativeExprDataOnly20260922,
    DataflowSystemTerraNativeExprSeededDataOnly20260922,
    DataflowSystemTerraNativeExprFlowOnly20260922,
    DataflowSystemTerraNativeExprCombined20260922,
)
__all__ = [cls.__name__ for cls in NATIVE_EXPRESSION_ARMS]

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

"""Pre-registered one-task/one-attempt Python-native pilot, not measured results.

Reference services must run their recorded historical revisions. A port alone
does not prove provenance. All V2 arms use one candidate build and differ only
in collection/exposure. New settings require a new result namespace, not an
override under an existing SUT name. See docs/native-python-pilot.md.
"""

import os
from dataclasses import dataclass

from .native_pilot_system import NativePilotSystem


@dataclass(frozen=True)
class PilotArm:
    key: str
    port: int
    catalog: str | None
    tool_mode: str | None
    parallel: bool
    collection: bool | None
    data: int
    flow: int

    @property
    def system_name(self) -> str:
        return f"DataflowSystemLunaPythonPilot{self.key}20260915Rep1"

    def settings(self) -> dict:
        return {
            "model_type": "gpt-5.6-luna",
            "driver": None,
            "agent_mode": "native",
            "context_mode": "delta",
            "native_tool_mode": self.tool_mode,
            "native_catalog_version": self.catalog,
            "native_profile_collection": self.collection,
            "native_flow_evidence": None,  # unrelated legacy prototype flag
            "parallel_tool_calls": self.parallel,
            "max_steps": 25,
            "max_operator_edits": 0,
            "max_operator_result_char_limit": 2000,
            "max_operator_result_cell_char_limit": 3000,
            "operator_result_serialization_mode": "tsv",
            "tool_timeout_seconds": 240,
            "execution_timeout_minutes": 10,
            "result_selection": "all",
            "flow_level": self.flow,
            "data_level": self.data,
            "stats_enabled": False,
            "column_stats": False,
            "attempt_reflection": True,
            "enable_code_in_snapshot": False,
            "thought_replay": False,
            "context_window_tokens": 0,
            "static_compaction": False,
            "enable_inspect_tool": False,
            "enable_render_prefs": False,
            "enable_recall_tool": False,
            "enable_resume_tool": False,
            "enable_answer_grounding": False,
            "error_reflection": False,
            "few_shot_prompt": False,
            "agent_turns": False,
            "session_turns": False,
            "versioned_mode": False,
            "message_layout": None,
        }


# In legacy V1 the anchor's L1/L1 carries schema/source structure, NOT the V2
# evidence treatment. V2 always renders its mandatory typed structural schema;
# its dataLevel/flowLevel switches control the new selected-evidence channels.
PILOT_ARMS = (
    PilotArm("V1Reference", 3013, None, None, True, None, 1, 1),
    PilotArm("BatchParent", 3012, None, "batch", False, None, 1, 1),
    PilotArm("V2Control", 3011, "v2", "batch", False, False, 0, 0),
    PilotArm("CollectorOnly", 3011, "v2", "batch", False, True, 0, 0),
    PilotArm("DataOnly", 3011, "v2", "batch", False, True, 1, 0),
    PilotArm("FlowOnly", 3011, "v2", "batch", False, True, 0, 1),
    PilotArm("Combined", 3011, "v2", "batch", False, True, 1, 1),
)


def _make_arm(spec: PilotArm):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        self.agent = None  # Base destructor also runs after failed validation.
        frozen = spec.settings()
        for key, value in frozen.items():
            if key in kwargs and (type(kwargs[key]) is not type(value) or kwargs[key] != value):
                raise ValueError(f"{spec.system_name}: {key} is frozen; use a new SUT name")
        # Only operational arguments may vary. Otherwise a legacy alias such
        # as schema_in_result=True could silently turn the V2 data channel on.
        operational = {"output_dir", "agent_service_endpoint", "computing_unit_id"}
        unknown = kwargs.keys() - frozen.keys() - operational
        if unknown:
            raise ValueError(f"{spec.system_name}: frozen settings reject overrides {sorted(unknown)}")
        if "computing_unit_id" not in kwargs:
            raw = os.environ.get("NATIVE_PYTHON_PILOT_CUID", "")
            if not raw.isdecimal() or int(raw) <= 0:
                raise ValueError("NATIVE_PYTHON_PILOT_CUID must identify the isolated pilot computing unit")
            kwargs["computing_unit_id"] = int(raw)
        cuid = kwargs["computing_unit_id"]
        if type(cuid) is not int or cuid <= 0:
            raise ValueError("computing_unit_id must be a positive integer")
        kwargs.setdefault("agent_service_endpoint", f"http://localhost:{spec.port}")
        kwargs.update(frozen)
        NativePilotSystem.__init__(self, *args, name=spec.system_name, verbose=verbose, **kwargs)
        self.pilot_spec = spec

    return type(
        spec.system_name,
        (NativePilotSystem,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__doc__": f"One-attempt Python-native pilot: {spec.key}; no results claimed.",
        },
    )


__all__ = [spec.system_name for spec in PILOT_ARMS]
for _spec in PILOT_ARMS:
    globals()[_spec.system_name] = _make_arm(_spec)

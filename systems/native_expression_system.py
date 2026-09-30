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
campaign's frozen protocol: DELTA, profiling on, 25 steps,
2,000-character results; the arms differ only in which evidence is on.

The retired `native_tool_mode` / `native_catalog_version` knobs are not sent:
the service rejects them, which is how the historical arms stay historical.
"""

import os

from .dataflow_system import DataflowSystem

_PROTOCOL = dict(
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
    _MODEL = "gpt-5.6-terra"
    _CONTEXT_MODE = "delta"
    _DATA = True
    _FLOW = False
    _SEED = False
    _REASONING = None  # None = server default (off); True = dataflow(reasoning, ...)
    _CHECKS = None  # None = server default (off); True = `check:` lines
    _NAME = "_NativeExpressionBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault(
            "agent_service_endpoint",
            os.environ.get("NATIVE_EXPR_AGENT_ENDPOINT", "http://localhost:3061"),
        )
        super().__init__(
            model_type=self._MODEL,
            context_mode=self._CONTEXT_MODE,
            data_evidence=self._DATA,
            flow_evidence=self._FLOW,
            seed_sources=self._SEED,
            dataflow_reasoning=self._REASONING,
            native_checks=self._CHECKS,
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


class DataflowSystemLunaNativeExprDataOnly20260922(_NativeExpressionBase):
    """The Terra Data-only protocol on gpt-5.6-luna: same knobs, cheaper model."""

    _MODEL = "gpt-5.6-luna"
    _NAME = "DataflowSystemLunaNativeExprDataOnly20260922"


# 2026-09-23 evidence factorial on surface v4 (doc on filter/map, 15 operators,
# orthogonal evidence budgets: data lines share the table's budget, flow lines
# have their own). One rep per model, four arms each; new names so the 09-22
# results stay untouched.
_EVIDENCE_ARMS = {
    "DataOnly": (True, False, "Data evidence only."),
    "FlowOnly": (
        False,
        True,
        "Flow evidence only (operator counters, contracts, upstream summaries).",
    ),
    "Combined": (True, True, "Data and flow evidence, each within its own budget."),
    "NoEvidence": (
        False,
        False,
        "Neither family: shape, schema and sampled rows only.",
    ),
}
_FACTORIAL_ARMS = []
for _model_tag, _model in (("Luna", "gpt-5.6-luna"), ("Terra", "gpt-5.6-terra")):
    for _arm, (_data, _flow, _doc) in _EVIDENCE_ARMS.items():
        _name = f"DataflowSystem{_model_tag}NativeExpr{_arm}20260923"
        _cls = type(
            _name,
            (_NativeExpressionBase,),
            {
                "__doc__": f"{_doc} Surface v4, {_model}, DELTA, 2,000 chars, 25 steps.",
                "__module__": __name__,
                "_MODEL": _model,
                "_DATA": _data,
                "_FLOW": _flow,
                "_NAME": _name,
            },
        )
        globals()[_name] = _cls
        _FACTORIAL_ARMS.append(_cls)

# 2026-09-24: reasoning argument A/B, and evidence made additive (data lines get
# their own budget; sample rows are identical in every arm). The *20260923 arms
# ran before the data budget change; the *20260924 arms run on the new code.
for _model_tag, _model in (("Luna", "gpt-5.6-luna"), ("Terra", "gpt-5.6-terra")):
    for _arm, (_data, _flow, _doc) in _EVIDENCE_ARMS.items():
        for _reasoning in (False, True):
            _name = f"DataflowSystem{_model_tag}NativeExpr{_arm}{'Reasoning' if _reasoning else ''}20260924"
            _cls = type(
                _name,
                (_NativeExpressionBase,),
                {
                    "__doc__": f"{_doc} Additive evidence budgets{', reasoning argument on' if _reasoning else ''}; {_model}, DELTA, 2,000 chars, 25 steps.",
                    "__module__": __name__,
                    "_MODEL": _model,
                    "_DATA": _data,
                    "_FLOW": _flow,
                    "_REASONING": True if _reasoning else None,
                    "_NAME": _name,
                },
            )
            globals()[_name] = _cls
            _FACTORIAL_ARMS.append(_cls)

# Controls for the 2026-09-24 smoke test: the *20260923 settings under new names,
# run against the agent that still serves the 2026-09-23 code, on the same tasks.
for _arm in ("DataOnly", "Combined"):
    _data, _flow, _doc = _EVIDENCE_ARMS[_arm]
    _name = f"DataflowSystemLunaNativeExpr{_arm}Control20260924"
    _cls = type(
        _name,
        (_NativeExpressionBase,),
        {
            "__doc__": f"{_doc} Control: 2026-09-23 code and prompt; gpt-5.6-luna.",
            "__module__": __name__,
            "_MODEL": "gpt-5.6-luna",
            "_DATA": _data,
            "_FLOW": _flow,
            "_NAME": _name,
        },
    )
    globals()[_name] = _cls
    _FACTORIAL_ARMS.append(_cls)

# 2026-09-27: the 20260924 settings, used for the stuck-task runs: first on the
# code with the state-registry picks (model request deadline + turn retry, step
# lifecycle logs, whole-Arrow-batch engine I/O, pandas-named blank headers, CRLF
# kept in quoted fields), then with the hang fixes that followed (execution
# requests past Bun's 240 s sweep, dead-worker detection, input readers that
# fail loudly, 64 MB result commits, a lighter scan).
for _model_tag, _model in (("Luna", "gpt-5.6-luna"), ("Terra", "gpt-5.6-terra")):
    for _arm, (_data, _flow, _doc) in _EVIDENCE_ARMS.items():
        _name = f"DataflowSystem{_model_tag}NativeExpr{_arm}20260927"
        _cls = type(
            _name,
            (_NativeExpressionBase,),
            {
                "__doc__": f"{_doc} 2026-09-27 code (timeouts, engine batches); {_model}, DELTA, 2,000 chars, 25 steps.",
                "__module__": __name__,
                "_MODEL": _model,
                "_DATA": _data,
                "_FLOW": _flow,
                "_NAME": _name,
            },
        )
        globals()[_name] = _cls
        _FACTORIAL_ARMS.append(_cls)

# 2026-09-28: operator checks. Each operator's own assumptions (a scan's first
# row is a header, a sum adds distinct records, a lookup join adds columns, the
# steps inside a process function keep rows) tested by the engine on the full
# tables; only violations become `check:` lines. A/B on the Data-only protocol,
# one knob; the Control arm is the same code with checks off.
for _model_tag, _model in (("Luna", "gpt-5.6-luna"), ("Terra", "gpt-5.6-terra")):
    for _checks in (False, True):
        for _rep in (1, 2, 3):
            _name = f"DataflowSystem{_model_tag}NativeExprDataOnly{'Checks' if _checks else 'Control'}20260928Rep{_rep}"
            _cls = type(
                _name,
                (_NativeExpressionBase,),
                {
                    "__doc__": f"Data evidence{', operator checks on' if _checks else ', checks off (control)'}; {_model}, DELTA, 2,000 chars, 25 steps; rep {_rep}.",
                    "__module__": __name__,
                    "_MODEL": _model,
                    "_DATA": True,
                    "_FLOW": False,
                    "_CHECKS": True if _checks else None,
                    "_NAME": _name,
                },
            )
            globals()[_name] = _cls
            _FACTORIAL_ARMS.append(_cls)

# Ablation (agent :3077, a frozen copy whose prompt omits the `check:` paragraph):
# check lines without the prompt's instruction about them, and more Control reps.
for _rep in (1, 2, 3):
    for _tag, _checks in (("ChecksNoPrompt", True), ("Control", None)):
        _name = f"DataflowSystemLunaNativeExprDataOnly{_tag}20260928Rep{_rep + (3 if _tag == 'Control' else 0)}"
        _cls = type(
            _name,
            (_NativeExpressionBase,),
            {
                "__doc__": f"Data evidence, {'check lines without the prompt paragraph' if _checks else 'checks off (control)'}; gpt-5.6-luna; rep {_rep}.",
                "__module__": __name__,
                "_MODEL": "gpt-5.6-luna",
                "_CHECKS": _checks,
                "_NAME": _name,
            },
        )
        globals()[_name] = _cls
        _FACTORIAL_ARMS.append(_cls)

# v2 (agent :3078): repeated records only when each source lists the record
# once and several sources list it; no join or merge row checks; a paragraph
# that says unchecked operators are unverified. Paired with Control reps 7-9.
for _rep in (1, 2, 3):
    for _tag, _checks, _r in (("ChecksV2", True, _rep), ("Control", None, _rep + 6)):
        _name = f"DataflowSystemLunaNativeExprDataOnly{_tag}20260928Rep{_r}"
        _cls = type(
            _name,
            (_NativeExpressionBase,),
            {
                "__doc__": f"Data evidence, {'v2 operator checks' if _checks else 'checks off (control)'}; gpt-5.6-luna; rep {_r}.",
                "__module__": __name__,
                "_MODEL": "gpt-5.6-luna",
                "_CHECKS": _checks,
                "_NAME": _name,
            },
        )
        globals()[_name] = _cls
        _FACTORIAL_ARMS.append(_cls)

for _checks in (False, True):
    _name = f"DataflowSystemLunaNativeExprDataOnly{'Checks' if _checks else 'Control'}20260928Smoke"
    _cls = type(
        _name,
        (_NativeExpressionBase,),
        {
            "__doc__": "Smoke test of the 2026-09-28 checks arms (not a result).",
            "__module__": __name__,
            "_MODEL": "gpt-5.6-luna",
            "_CHECKS": True if _checks else None,
            "_NAME": _name,
        },
    )
    globals()[_name] = _cls
    _FACTORIAL_ARMS.append(_cls)

# 2026-09-29 stats campaign: dataflow with no stats (plain records: shape,
# schema, sampled rows) vs dataflow with stats (data evidence + flow evidence,
# including the facts derived from operator semantics: header rows that read as
# data, what a distinct count counts, group keys that pool entities, unread
# flag columns), on Luna and Terra, 3 reps each. Same code for all arms; the
# script-agent peers are CodeAgentSystem{Luna,Terra}Chars5kGuidedRep{0,1,2}.
for _model_tag, _model in (("Luna", "gpt-5.6-luna"), ("Terra", "gpt-5.6-terra")):
    for _arm, _stats in (("NoStats", False), ("Stats", True)):
        for _rep in (1, 2, 3):
            _name = f"DataflowSystem{_model_tag}NativeExpr{_arm}20260929Rep{_rep}"
            _cls = type(
                _name,
                (_NativeExpressionBase,),
                {
                    "__doc__": f"{'Data + flow evidence' if _stats else 'No evidence: plain records'}; {_model}, DELTA, 2,000 chars, 25 steps; rep {_rep}.",
                    "__module__": __name__,
                    "_MODEL": _model,
                    "_DATA": _stats,
                    "_FLOW": _stats,
                    "_NAME": _name,
                },
            )
            globals()[_name] = _cls
            _FACTORIAL_ARMS.append(_cls)

NATIVE_EXPRESSION_ARMS = (
    DataflowSystemLunaNativeExprDataOnly20260922,
    DataflowSystemTerraNativeExprDataOnly20260922,
    DataflowSystemTerraNativeExprSeededDataOnly20260922,
    DataflowSystemTerraNativeExprFlowOnly20260922,
    DataflowSystemTerraNativeExprCombined20260922,
    *_FACTORIAL_ARMS,
)
__all__ = [cls.__name__ for cls in NATIVE_EXPRESSION_ARMS]

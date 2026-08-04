"""LongDS experiment arms.

Deliberately NOT registered in `systems/__init__.py`: these are instantiated
directly by `run_longds.py`, and KramaBench's own harness must never pick them up
(they answer a multi-turn protocol it cannot drive). Keeping them here also means
adding a LongDS arm cannot disturb a KramaBench arm.

Baseline is the same knob set as the campaign's `_LunaAnchor` — DELTA, 1k result
chars, no stats, no code-in-snapshot — so LongDS numbers sit next to the gpt-5.6
KramaBench Anchor without a knob translation step. Two deliberate differences:

  * `max_steps=40` (LongDS's own per-turn cap) instead of 25. LakeQA showed the
    step cap binds first on harder benchmarks, and LongDS turn 1 has to build the
    whole analytical state from raw files.
  * `agent_service_endpoint` is :3001, the main checkout. `_LunaBase` defaults to
    :3004 (the code-lean worktree), which is not running.

`context_window_tokens=0` is the append-only condition: DELTA replays every
turn's events and nothing is trimmed by a token budget. It is already the default
— stated explicitly because for this benchmark it is a design choice, not an
inherited default.
"""
from systems.dataflow_system import DataflowSystem

#: The main-checkout agent-service. Verified serving from
#: dataflow-agent/agent-service (not a worktree) at the time these arms landed.
AGENT_SERVICE_ENDPOINT = "http://localhost:3001"


class LongDSBase(DataflowSystem):
    _MODEL = "gpt-5.6-luna"
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _STATS = False
    _CODE = False
    _MAX_STEPS = 40
    _NAME = "LongDSBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", AGENT_SERVICE_ENDPOINT)
        super().__init__(
            model_type=self._MODEL,
            context_mode=self._CONTEXT_MODE,
            max_steps=self._MAX_STEPS,
            flow_level=1,
            data_level=2 if self._STATS else 1,
            column_stats=self._STATS,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            enable_code_in_snapshot=self._CODE,
            context_window_tokens=0,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class LongDSLunaDelta1k(LongDSBase):
    """The baseline: gpt-5.6-luna, DELTA, 1k, no stats, append-only history."""

    _NAME = "LongDSLunaDelta1k"


class LongDSLunaDelta1kTaskLast(LongDSBase):
    """Baseline + the request moved under the history.

    Pure cache/attention change: same bytes, same order of everything else. The
    baseline put the newest request FIRST, so each new turn changed byte 13 of the
    prompt and invalidated the identical history below it (measured: consecutive
    turns shared 13 leading characters of a 233 KB prompt). Isolating it here
    means any accuracy difference is attributable to placement alone.
    """

    _NAME = "LongDSLunaDelta1kTaskLast"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("user_task_placement", "bottom")
        super().__init__(verbose=verbose, *args, **kwargs)


class LongDSLunaRecall(LongDSBase):
    """The experiment: catalogue prior turns, let the model pull what it needs.

    Prior turns collapse into `# Prior Turns` (request / operators touched /
    answer) and the model recalls specifics through `recallState`. The request
    sits last so the catalogue and the current turn's events stay a stable prefix.

    The bet: LongDS failures are state failures, not retrieval failures, and a
    dataflow's state is addressable (named operators, per-turn revisions,
    materialized results), so the model can fetch the exact earlier definition
    rather than re-deriving it — which is where the numeric drift comes from.
    """

    _NAME = "LongDSLunaRecall"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("user_task_placement", "bottom")
        kwargs.setdefault("turn_history", "index")
        kwargs.setdefault("enable_recall_tool", True)
        super().__init__(verbose=verbose, *args, **kwargs)


class LongDSLunaRecallPushFull(LongDSBase):
    """Ablation: the recall tool WITHOUT trimming history.

    Separates "the model can pull state" from "the push side got leaner". If the
    tool only helps once history is trimmed, the mechanism is attention/burial,
    not retrieval ability.
    """

    _NAME = "LongDSLunaRecallPushFull"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("user_task_placement", "bottom")
        kwargs.setdefault("enable_recall_tool", True)
        super().__init__(verbose=verbose, *args, **kwargs)


ARMS = {
    "luna-delta-1k": LongDSLunaDelta1k,
    "luna-task-last": LongDSLunaDelta1kTaskLast,
    "luna-recall": LongDSLunaRecall,
    "luna-recall-pushfull": LongDSLunaRecallPushFull,
}

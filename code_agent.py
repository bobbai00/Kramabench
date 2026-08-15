# -*- coding: utf-8 -*-
"""
Code Agent - Wrapper for smolagents CodeAgent for KramaBench benchmarking.
"""

import os
import re
import time
from typing import Optional, Any
from dataclasses import dataclass

from smolagents import CodeAgent
from smolagents.models import ChatMessage, MessageRole, OpenAIServerModel
from smolagents.memory import ActionStep

from code_agent_custom_prompt import CUSTOM_INSTRUCTIONS, FINE_GRAINED_INSTRUCTIONS, DATA_PITFALLS_INSTRUCTIONS

# Default settings (CODE_AGENT_MAX_STEPS env var overrides default)
DEFAULT_MODEL_TYPE = "claude-haiku-4.5"
DEFAULT_MAX_STEPS = int(os.environ.get("CODE_AGENT_MAX_STEPS", 50))
DEFAULT_API_BASE = os.environ.get("CODE_AGENT_API_BASE", "http://localhost:4000")
# "dummy" only works against a proxy that enforces no key. The local-dev LiteLLM
# sets a master_key, and any other key sends it looking for a virtual key in a
# database it does not have — the request fails with 400 `no_db_connection`.
# Same env-var convention code_agent_session.py already uses.
DEFAULT_API_KEY = os.environ.get("CODE_AGENT_API_KEY") or os.environ.get("LITELLM_MASTER_KEY") or "dummy"

# Customized prompt setting (set to "true" to enable)
CUSTOMIZED_PROMPT_ENABLED = os.environ.get("CUSTOMIZED_PROMPT_ENABLED", "false").lower() == "true"

# Fine-grained prompt setting (set to "true" to use one-line-per-action prompt)
FINE_GRAINED_PROMPT_ENABLED = os.environ.get("FINE_GRAINED_PROMPT_ENABLED", "false").lower() == "true"
# Data-pitfalls guidance WITHOUT the one-op-per-step granularity mandate.
PITFALLS_PROMPT_ENABLED = os.environ.get("PITFALLS_PROMPT_ENABLED", "false").lower() == "true"

# Max print outputs length (set to limit characters shown to agent per code execution, empty/0 = no limit)
_max_print_env = os.environ.get("CODE_AGENT_MAX_PRINT_OUTPUTS_LENGTH", "")
DEFAULT_MAX_PRINT_OUTPUTS_LENGTH = int(_max_print_env) if _max_print_env.isdigit() and int(_max_print_env) > 0 else None

REASONING_EFFORT_SUFFIXES = ("-high", "-medium", "-low")


# OpenAI reasoning models reject a non-default `temperature` (only 1 is allowed)
# and bill reasoning tokens inside `completion_tokens`.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_id: str) -> bool:
    m = (model_id or "").split("/")[-1]
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _parse_model_and_reasoning_effort(model_type: str) -> tuple[str, Optional[str]]:
    """Parse model name like 'gpt-5-mini-medium' into ('gpt-5-mini', 'medium')."""
    for suffix in REASONING_EFFORT_SUFFIXES:
        if model_type.endswith(suffix):
            return model_type[: -len(suffix)], suffix.lstrip("-")
    return model_type, None


class UsageTrackingOpenAIModel(OpenAIServerModel):
    """OpenAIServerModel that accumulates the per-call token-class breakdown that
    smolagents' RunResult.token_usage drops — cache-read and reasoning tokens —
    by reading the raw OpenAI response on every generate() call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_usage()

    @property
    def supports_stop_parameter(self) -> bool:
        # smolagents' built-in regex misses dotted reasoning ids (e.g. "gpt-5.4"),
        # so it would send `stop` and get a 400. All gpt-5*/o* reasoning models
        # reject `stop`; force it off and let smolagents trim stop seqs client-side.
        if _is_reasoning_model(self.model_id or ""):
            return False
        from smolagents.models import supports_stop_parameter as _ssp
        return _ssp(self.model_id or "")

    def reset_usage(self):
        self.usage_acc = {
            "input": 0, "output": 0, "cached": 0, "cache_creation": 0,
            "reasoning": 0, "calls": 0,
        }

    def _track(self, msg):
        try:
            usage = getattr(getattr(msg, "raw", None), "usage", None)
            if usage is None:
                return
            self.usage_acc["calls"] += 1
            self.usage_acc["input"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage_acc["output"] += int(getattr(usage, "completion_tokens", 0) or 0)
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                self.usage_acc["cached"] += int(getattr(ptd, "cached_tokens", 0) or 0)
                # litellm extension; Anthropic bills cache WRITES at 1.25x input,
                # so dropping this undercounts every cache-enabled claude run
                self.usage_acc["cache_creation"] += int(
                    getattr(ptd, "cache_creation_tokens", 0) or 0)
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                self.usage_acc["reasoning"] += int(getattr(ctd, "reasoning_tokens", 0) or 0)
        except Exception:
            pass

    def generate(self, *args, **kwargs):
        msg = super().generate(*args, **kwargs)
        self._track(msg)
        return msg


def _strip_code_blocks(text: str) -> str:
    """Remove code blocks from model output, keeping the reasoning.

    Handles both smolagents default tags (<code>...</code>) and markdown (```...```)."""
    # smolagents default: <code>...</code>
    result = re.sub(r"<code>.*?</code>", "[code omitted]", text, flags=re.DOTALL)
    # markdown style: ```...```
    result = re.sub(r"```.*?```", "[code omitted]", result, flags=re.DOTALL)
    return result.strip()


class NoActionDetailCodeAgent(CodeAgent):
    """CodeAgent subclass that strips code from historical steps to reduce context size."""

    def write_memory_to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)

        # Find the last ActionStep index
        action_steps = [
            (i, step) for i, step in enumerate(self.memory.steps)
            if isinstance(step, ActionStep)
        ]
        last_action_idx = action_steps[-1][0] if action_steps else -1

        for i, memory_step in enumerate(self.memory.steps):
            if isinstance(memory_step, ActionStep) and i != last_action_idx:
                # For historical ActionSteps: emit messages without code detail
                # Keep the reasoning (stripped of code blocks) as assistant message
                if memory_step.model_output is not None and not summary_mode:
                    stripped = _strip_code_blocks(str(memory_step.model_output))
                    messages.append(
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=[{"type": "text", "text": stripped}],
                        )
                    )
                # Skip tool_calls — omit the code call message entirely
                # Keep observations/errors so the agent knows what happened
                if memory_step.observations is not None:
                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL_RESPONSE,
                            content=[{"type": "text", "text": f"Observation:\n{memory_step.observations}"}],
                        )
                    )
                if memory_step.error is not None:
                    error_msg = (
                        "Error:\n" + str(memory_step.error)
                        + "\nNow let's retry: take care not to repeat previous errors! "
                        "If you have retried several times, try a completely different approach.\n"
                    )
                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL_RESPONSE,
                            content=[{"type": "text", "text": error_msg}],
                        )
                    )
            else:
                # Latest ActionStep or non-ActionStep: emit full messages
                messages.extend(memory_step.to_messages(summary_mode=summary_mode))

        return messages


AUTHORIZED_IMPORTS = [
    # Data science essentials (with submodules)
    "numpy.*",      # numpy.linalg, numpy.random, numpy.fft, etc.
    "pandas.*",     # pandas.api, pandas.io, etc.
    "scipy.*",      # scipy.stats, scipy.optimize, scipy.interpolate, etc.
    "sklearn.*",    # sklearn.model_selection, sklearn.preprocessing, sklearn.metrics, etc.
    "matplotlib.*", # matplotlib.pyplot, matplotlib.figure, etc.
    "openpyxl.*",   # for reading .xlsx Excel files via pandas

    # HTML parsing (for pd.read_html)
    "lxml.*",       # lxml.html, lxml.etree - fast HTML/XML parser
    "bs4.*",        # BeautifulSoup for HTML parsing

    # Geospatial data (for .gpkg GeoPackage files)
    "geopandas.*",  # extends pandas for geospatial data
    "fiona.*",      # reads/writes geospatial data formats
    "shapely.*",    # geometric operations
    "pyproj.*",     # coordinate transformations

    # Scientific data formats
    "cdflib.*",     # NASA Common Data Format (.cdf files)

    # Standard library - common
    "json", "csv", "os", "glob", "math", "statistics", "random", "re",
    "datetime", "itertools", "time", "unicodedata", "queue", "stat",
    "textwrap", "string", "io", "pathlib", "functools", "operator",

    # Standard library - with submodules
    "collections.*",  # collections.abc
    "xml.*",          # xml.etree, xml.dom, xml.sax
    "urllib.*",       # urllib.parse, urllib.request
    "html.*",         # html.parser

    # Path handling (used internally by os.path, pandas, etc.)
    "posixpath", "ntpath", "genericpath",

    # Additional useful libraries
    "typing", "copy", "decimal", "fractions", "struct",
    "hashlib", "base64", "logging", "warnings", "bisect", "heapq",
]



@dataclass
class CodeAgentResult:
    """Result from running the code agent."""
    response: str
    reasoning_trace: list[dict]
    elapsed_seconds: float
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    num_steps: int = 0


def extract_reasoning_trace(agent: CodeAgent) -> list[dict]:
    """Extract reasoning trace from agent memory."""
    trace = []
    if not hasattr(agent, "memory") or not agent.memory:
        return trace

    try:
        steps = agent.memory.get_full_steps()
        for idx, step in enumerate(steps):
            # Skip task entries
            if isinstance(step, dict) and "task" in step and "step_number" not in step:
                continue

            entry = {"step": step.get("step_number", idx + 1) if isinstance(step, dict) else idx + 1}

            # Extract relevant fields
            def get(key):
                return step.get(key) if isinstance(step, dict) else getattr(step, key, None)

            if get("model_output"):
                entry["model_output"] = str(get("model_output"))[:1000]
            if get("code_action"):
                entry["code"] = str(get("code_action"))[:2000]
            if get("action_output"):
                entry["output"] = str(get("action_output"))[:500]
            if get("observations"):
                entry["observations"] = str(get("observations"))[:500]
            if get("error"):
                entry["error"] = str(get("error"))
            if get("is_final_answer"):
                entry["is_final_answer"] = True

            if len(entry) > 1:
                trace.append(entry)
    except Exception:
        pass

    return trace


class CodeAgentWrapper:
    """Wrapper for smolagents CodeAgent."""

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        max_steps: int = DEFAULT_MAX_STEPS,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        authorized_imports: list[str] = None,
        verbosity_level: int = 1,
        use_fine_grained_prompt: bool = None,
        use_custom_prompt: bool = None,
        use_pitfalls_prompt: bool = None,
        max_print_outputs_length: int = None,
        no_action_detail: bool = False,
    ):
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.authorized_imports = authorized_imports or AUTHORIZED_IMPORTS
        self.verbosity_level = verbosity_level
        # If not explicitly set, fall back to environment variable
        self.use_fine_grained_prompt = use_fine_grained_prompt if use_fine_grained_prompt is not None else FINE_GRAINED_PROMPT_ENABLED
        # If not explicitly set, fall back to environment variable
        self.use_custom_prompt = use_custom_prompt if use_custom_prompt is not None else CUSTOMIZED_PROMPT_ENABLED
        # Data-pitfalls-only guidance (no granularity mandate)
        self.use_pitfalls_prompt = use_pitfalls_prompt if use_pitfalls_prompt is not None else PITFALLS_PROMPT_ENABLED
        # If not explicitly set, fall back to environment variable (None = no limit)
        self.max_print_outputs_length = max_print_outputs_length if max_print_outputs_length is not None else DEFAULT_MAX_PRINT_OUTPUTS_LENGTH
        self.no_action_detail = no_action_detail
        self._agent: Optional[CodeAgent] = None
        self._model: Optional[OpenAIServerModel] = None

    def setup(self) -> "CodeAgentWrapper":
        """Setup the agent."""
        model_id, reasoning_effort = _parse_model_and_reasoning_effort(self.model_type)
        model_kwargs = {}
        if reasoning_effort:
            model_kwargs["reasoning_effort"] = reasoning_effort
        # Reasoning models (gpt-5*/o*) only accept the default temperature (1);
        # sending temperature=0.2 returns a 400. Omit it for those.
        if not _is_reasoning_model(model_id):
            model_kwargs["temperature"] = 0.2
        self._model = UsageTrackingOpenAIModel(
            model_id=model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            **model_kwargs,
        )

        # Build agent kwargs
        agent_kwargs = {
            "tools": [],
            "model": self._model,
            "additional_authorized_imports": self.authorized_imports,
            "max_steps": self.max_steps,
            "verbosity_level": self.verbosity_level,
            "executor_kwargs": {"additional_functions": {"open": open}},
        }

        # Add max print outputs length if set
        if self.max_print_outputs_length is not None:
            agent_kwargs["max_print_outputs_length"] = self.max_print_outputs_length

        # Add custom instructions if enabled
        if self.use_fine_grained_prompt:
            agent_kwargs["instructions"] = FINE_GRAINED_INSTRUCTIONS
        elif self.use_custom_prompt:
            agent_kwargs["instructions"] = CUSTOM_INSTRUCTIONS
        elif self.use_pitfalls_prompt:
            agent_kwargs["instructions"] = DATA_PITFALLS_INSTRUCTIONS

        agent_cls = NoActionDetailCodeAgent if self.no_action_detail else CodeAgent
        self._agent = agent_cls(**agent_kwargs)
        return self

    def run(self, prompt: str) -> CodeAgentResult:
        """Run the agent with a prompt."""
        if not self._agent:
            raise RuntimeError("Agent not set up. Call setup() first.")

        error = None
        response = ""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        num_steps = 0
        elapsed = 0.0

        try:
            # Use return_full_result=True to get RunResult with token usage and timing
            result = self._agent.run(prompt, return_full_result=True)

            # Extract response (output)
            response = str(result.output) if result.output else ""

            # Extract token usage
            if result.token_usage:
                input_tokens = result.token_usage.input_tokens or 0
                output_tokens = result.token_usage.output_tokens or 0
                total_tokens = result.token_usage.total_tokens or 0

            # Extract timing
            if result.timing and result.timing.end_time:
                elapsed = result.timing.end_time - result.timing.start_time

            # Count steps
            num_steps = len(result.steps) if result.steps else 0

        except (Exception, SystemExit) as e:
            # Agent-generated code sometimes calls `raise SystemExit(...)` or
            # sys.exit() as its own error handling. SystemExit is a
            # BaseException, so smolagents' `except Exception` lets it escape and
            # kill the whole workload process. Catch it here so one bad task
            # fails cleanly (scored 0) instead of aborting every remaining task.
            error = f"SystemExit: {e}" if isinstance(e, SystemExit) else str(e)

        # Get reasoning trace from agent memory (more detailed than result.steps)
        trace = extract_reasoning_trace(self._agent)

        # Use trace length if we didn't get steps from result
        if num_steps == 0 and trace:
            num_steps = len(trace)

        # Per-class breakdown (cache-read + reasoning) captured by the tracking
        # model, which smolagents' aggregate token_usage omits. Fall back to the
        # accumulator's own input/output if smolagents didn't report them.
        acc = getattr(self._model, "usage_acc", {}) or {}
        reasoning_tokens = int(acc.get("reasoning", 0) or 0)
        cached_tokens = int(acc.get("cached", 0) or 0)
        cache_creation_tokens = int(acc.get("cache_creation", 0) or 0)
        if not input_tokens and acc.get("input"):
            input_tokens = int(acc["input"])
        if not output_tokens and acc.get("output"):
            output_tokens = int(acc["output"])
        if not total_tokens:
            total_tokens = input_tokens + output_tokens

        cost_usd = 0.0
        try:
            from systems.cost_utils import compute_cost
            c = compute_cost(
                self.model_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )
            cost_usd = c if c is not None else 0.0
        except Exception:
            pass

        return CodeAgentResult(
            response=response,
            reasoning_trace=trace,
            elapsed_seconds=elapsed,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cost_usd=cost_usd,
            num_steps=num_steps,
        )

    def reset(self):
        """Reset the agent state."""
        if self._model is not None and hasattr(self._model, "reset_usage"):
            self._model.reset_usage()
        if self._agent and self._model:
            # Build agent kwargs
            agent_kwargs = {
                "tools": [],
                "model": self._model,
                "additional_authorized_imports": self.authorized_imports,
                "max_steps": self.max_steps,
                "verbosity_level": self.verbosity_level,
                "executor_kwargs": {"additional_functions": {"open": open}},
            }

            # Add max print outputs length if set
            if self.max_print_outputs_length is not None:
                agent_kwargs["max_print_outputs_length"] = self.max_print_outputs_length

            # Add custom instructions if enabled
            if self.use_fine_grained_prompt:
                agent_kwargs["instructions"] = FINE_GRAINED_INSTRUCTIONS
            elif self.use_custom_prompt:
                agent_kwargs["instructions"] = CUSTOM_INSTRUCTIONS
            elif self.use_pitfalls_prompt:
                agent_kwargs["instructions"] = DATA_PITFALLS_INSTRUCTIONS

            agent_cls = NoActionDetailCodeAgent if self.no_action_detail else CodeAgent
            self._agent = agent_cls(**agent_kwargs)

    def cleanup(self):
        """Cleanup resources."""
        self._agent = None
        self._model = None

    def __enter__(self) -> "CodeAgentWrapper":
        return self.setup()

    def __exit__(self, *args):
        self.cleanup()
        return False

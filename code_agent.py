# -*- coding: utf-8 -*-
"""
Code Agent - Wrapper for smolagents CodeAgent for KramaBench benchmarking.
"""

import os
import time
from typing import Optional, Any
from dataclasses import dataclass

from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel

from code_agent_custom_prompt import CUSTOM_INSTRUCTIONS, FINE_GRAINED_INSTRUCTIONS

# Default settings (CODE_AGENT_MAX_STEPS env var overrides default)
DEFAULT_MODEL_TYPE = "claude-haiku-4.5"
DEFAULT_MAX_STEPS = int(os.environ.get("CODE_AGENT_MAX_STEPS", 50))
DEFAULT_API_BASE = "http://localhost:9096/api"
DEFAULT_API_KEY = "dummy"

# Customized prompt setting (set to "true" to enable)
CUSTOMIZED_PROMPT_ENABLED = os.environ.get("CUSTOMIZED_PROMPT_ENABLED", "false").lower() == "true"

# Fine-grained prompt setting (set to "true" to use one-line-per-action prompt)
FINE_GRAINED_PROMPT_ENABLED = os.environ.get("FINE_GRAINED_PROMPT_ENABLED", "false").lower() == "true"

# Max print outputs length (set to limit characters shown to agent per code execution, empty/0 = no limit)
_max_print_env = os.environ.get("CODE_AGENT_MAX_PRINT_OUTPUTS_LENGTH", "")
DEFAULT_MAX_PRINT_OUTPUTS_LENGTH = int(_max_print_env) if _max_print_env.isdigit() and int(_max_print_env) > 0 else None

# Imports the agent is allowed to use
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
        max_print_outputs_length: int = None,
    ):
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.authorized_imports = authorized_imports or AUTHORIZED_IMPORTS
        self.verbosity_level = verbosity_level
        # If not explicitly set, fall back to environment variable
        self.use_fine_grained_prompt = use_fine_grained_prompt if use_fine_grained_prompt is not None else FINE_GRAINED_PROMPT_ENABLED
        # If not explicitly set, fall back to environment variable (None = no limit)
        self.max_print_outputs_length = max_print_outputs_length if max_print_outputs_length is not None else DEFAULT_MAX_PRINT_OUTPUTS_LENGTH
        self._agent: Optional[CodeAgent] = None
        self._model: Optional[OpenAIServerModel] = None

    def setup(self) -> "CodeAgentWrapper":
        """Setup the agent."""
        self._model = OpenAIServerModel(
            model_id=self.model_type,
            api_base=self.api_base,
            api_key=self.api_key,
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
        elif CUSTOMIZED_PROMPT_ENABLED:
            agent_kwargs["instructions"] = CUSTOM_INSTRUCTIONS

        self._agent = CodeAgent(**agent_kwargs)
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

        except Exception as e:
            error = str(e)

        # Get reasoning trace from agent memory (more detailed than result.steps)
        trace = extract_reasoning_trace(self._agent)

        # Use trace length if we didn't get steps from result
        if num_steps == 0 and trace:
            num_steps = len(trace)

        return CodeAgentResult(
            response=response,
            reasoning_trace=trace,
            elapsed_seconds=elapsed,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            num_steps=num_steps,
        )

    def reset(self):
        """Reset the agent state."""
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
            elif CUSTOMIZED_PROMPT_ENABLED:
                agent_kwargs["instructions"] = CUSTOM_INSTRUCTIONS

            self._agent = CodeAgent(**agent_kwargs)

    def cleanup(self):
        """Cleanup resources."""
        self._agent = None
        self._model = None

    def __enter__(self) -> "CodeAgentWrapper":
        return self.setup()

    def __exit__(self, *args):
        self.cleanup()
        return False

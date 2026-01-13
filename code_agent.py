# -*- coding: utf-8 -*-
"""
Code Agent - Wrapper for smolagents CodeAgent for KramaBench benchmarking.
"""

import time
from typing import Optional, Any
from dataclasses import dataclass

from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel

# Default settings
DEFAULT_MODEL_TYPE = "claude-haiku-4.5"
DEFAULT_MAX_STEPS = 50
DEFAULT_API_BASE = "http://localhost:9096/api"
DEFAULT_API_KEY = "dummy"

# Imports the agent is allowed to use
AUTHORIZED_IMPORTS = ["numpy", "pandas", "json", "csv", "os", "glob"]

# Instructions for the agent about file access
FILE_INSTRUCTIONS = """You have access to Python's built-in open() function to read files.
Use open(filepath, 'r') to read text files and open(filepath, 'rb') for binary files.
You can also use pandas.read_csv(), pandas.read_json(), pandas.read_html() etc. for structured data.
Always use the absolute file paths provided in the task."""


@dataclass
class CodeAgentResult:
    """Result from running the code agent."""
    response: str
    reasoning_trace: list[dict]
    elapsed_seconds: float
    error: Optional[str] = None


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
    ):
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.authorized_imports = authorized_imports or AUTHORIZED_IMPORTS
        self.verbosity_level = verbosity_level
        self._agent: Optional[CodeAgent] = None
        self._model: Optional[OpenAIServerModel] = None

    def setup(self) -> "CodeAgentWrapper":
        """Setup the agent."""
        self._model = OpenAIServerModel(
            model_id=self.model_type,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        self._agent = CodeAgent(
            tools=[],
            model=self._model,
            additional_authorized_imports=self.authorized_imports,
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            instructions=FILE_INSTRUCTIONS,
            executor_kwargs={"additional_functions": {"open": open}},
        )
        return self

    def run(self, prompt: str) -> CodeAgentResult:
        """Run the agent with a prompt."""
        if not self._agent:
            raise RuntimeError("Agent not set up. Call setup() first.")

        start_time = time.time()
        error = None
        response = ""

        try:
            response = self._agent.run(prompt)
            response = str(response) if response else ""
        except Exception as e:
            error = str(e)

        elapsed = time.time() - start_time
        trace = extract_reasoning_trace(self._agent)

        return CodeAgentResult(
            response=response,
            reasoning_trace=trace,
            elapsed_seconds=elapsed,
            error=error,
        )

    def reset(self):
        """Reset the agent state."""
        if self._agent and self._model:
            self._agent = CodeAgent(
                tools=[],
                model=self._model,
                additional_authorized_imports=self.authorized_imports,
                max_steps=self.max_steps,
                verbosity_level=self.verbosity_level,
                instructions=FILE_INSTRUCTIONS,
                executor_kwargs={"additional_functions": {"open": open}},
            )

    def cleanup(self):
        """Cleanup resources."""
        self._agent = None
        self._model = None

    def __enter__(self) -> "CodeAgentWrapper":
        return self.setup()

    def __exit__(self, *args):
        self.cleanup()
        return False

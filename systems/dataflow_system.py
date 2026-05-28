# -*- coding: utf-8 -*-
"""
DataflowSystem - KramaBench System wrapper for Texera DataflowAgent.

This module provides a System implementation that uses the Texera Agent Service
to solve benchmark tasks via dataflow-based agents.
"""

import os
import json
import time
from typing import Dict, List, Optional

from benchmark.benchmark_api import System
from dataflow_agent import DataflowAgent, MessageResult, get_agent_workflow, get_agent_react_steps
from systems.data_source_utils import expand_data_sources
from utils.answer_parser import parse_answer


class DataflowSystem(System):
    """
    KramaBench System wrapper for Texera DataflowAgent.

    This system sends queries to the Texera Agent Service and returns
    the agent's responses in the format expected by KramaBench.
    """

    def __init__(
        self,
        model_type: str = None,
        driver: str = None,
        max_steps: int = None,
        max_operator_result_char_limit: int = None,
        max_operator_result_cell_char_limit: int = None,
        operator_result_serialization_mode: str = None,
        tool_timeout_seconds: int = None,
        execution_timeout_minutes: int = None,
        agent_mode: str = None,
        context_mode: str = None,
        parallel_tool_calls: bool = None,
        allowed_operator_types: Optional[List[str]] = None,
        disabled_tools: Optional[List[str]] = None,
        stats_enabled: bool = False,
        include_operator_properties: bool = None,
        max_operator_edits: Optional[int] = None,
        lineage_hint_on_stall: Optional[bool] = None,
        verbose: bool = False,
        name: str = "DataflowSystem",
        *args,
        **kwargs
    ):
        """
        Initialize the DataflowSystem.

        Configuration flows from constructor arguments only — subclasses
        (e.g. DataflowSystemHaiku45) pin model-specific values; any caller
        can override per-instance by passing kwargs. Class-level defaults
        live in this method body.

        Args:
            model_type: LLM model type (default: claude-haiku-4.5)
            max_steps: Maximum steps per query (default: 50)
            max_operator_result_char_limit: Max chars for operator results (default: 1000)
            max_operator_result_cell_char_limit: Max chars per cell (default: 2000)
            operator_result_serialization_mode: Result format (default: tsv)
            tool_timeout_seconds: Tool timeout (default: 240)
            execution_timeout_minutes: Execution timeout (default: 4)
            agent_mode: Agent mode (default: code)
            context_mode: Context selection policy (default: latest)
            parallel_tool_calls: Allow parallel tool calls (default: True)
            allowed_operator_types: Optional whitelist of operator types; None uses server default
            disabled_tools: Optional list of tool names to disable
            max_operator_edits: Optional convergence guard cap; None uses server default
            lineage_hint_on_stall: Optional lineage hint toggle for convergence guard stalls
            verbose: Enable verbose logging
            name: System name for benchmark identification
        """
        super().__init__(name, verbose=verbose, *args, **kwargs)

        self.model_type = model_type or "claude-haiku-4.5"
        # None -> let agent-service auto-derive the driver from model_type.
        self.driver = driver
        self.max_steps = max_steps or 50
        self.max_operator_result_char_limit = max_operator_result_char_limit or 1000
        self.max_operator_result_cell_char_limit = max_operator_result_cell_char_limit or 2000
        self.operator_result_serialization_mode = operator_result_serialization_mode or "tsv"
        self.tool_timeout_seconds = tool_timeout_seconds or 240
        self.execution_timeout_minutes = execution_timeout_minutes or 4
        self.agent_mode = agent_mode or "code"
        self.context_mode = context_mode or "latest"
        self.parallel_tool_calls = True if parallel_tool_calls is None else parallel_tool_calls
        self.allowed_operator_types = allowed_operator_types
        self.disabled_tools = disabled_tools
        self.stats_enabled = stats_enabled
        # None -> server default (true).
        self.include_operator_properties = include_operator_properties
        self.max_operator_edits = max_operator_edits
        self.lineage_hint_on_stall = lineage_hint_on_stall

        self.agent: Optional[DataflowAgent] = None
        self.output_dir = kwargs.get("output_dir", f"./system_scratch/{name}")
        self.workload_data: Dict[str, dict] = {}  # Map task_id -> task dict (for ground truth)
        self.format_hints: Dict[str, str] = {}  # Map task_id -> format_hint string

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """
        Process the dataset by storing the directory path.

        The DataflowAgent accesses files directly via paths in the prompt,
        so we only need to store the directory and collect the file list.

        Args:
            dataset_directory: Path to the dataset directory
        """
        self.dataset_directory = dataset_directory
        self.dataset = {}

        # Collect file list for reference
        for dirpath, _, filenames in os.walk(dataset_directory):
            for fname in filenames:
                rel_path = os.path.relpath(
                    os.path.join(dirpath, fname), dataset_directory
                )
                self.dataset[rel_path] = None  # Placeholder - agent reads files directly

        if self.verbose:
            print(f"[DataflowSystem] Found {len(self.dataset)} files in {dataset_directory}")

        # Try to load workload for ground truth lookup
        self._load_workload(dataset_directory)

        # Try to load format hints
        self._load_format_hints(dataset_directory)

        # Initialize the agent. Best-effort: when `evaluate.py --use_system_cache`
        # short-circuits to the cached response file, `serve_query` is never
        # called and the agent isn't actually needed. Don't make scoring-only
        # runs fall over just because the JVM stack happens to be down.
        try:
            self._setup_agent()
        except Exception as e:
            if self.verbose:
                print(f"[DataflowSystem] Agent setup failed ({e}); proceeding "
                      f"without live agent. Live serve_query calls will fail; "
                      f"this is safe when --use_system_cache is in effect.")
            self.agent = None

    def _load_workload(self, dataset_directory: str) -> None:
        """Load workload files to enable ground truth saving."""
        # Infer workload directory from dataset directory
        # dataset_directory is like: .../data/{domain}/input
        # workload is at: .../workload/{domain}.json
        try:
            parts = dataset_directory.rstrip('/').split('/')
            if 'data' in parts:
                data_idx = parts.index('data')
                domain = parts[data_idx + 1]  # e.g., "legal"
                project_root = '/'.join(parts[:data_idx])

                # Try loading both regular and tiny workloads
                for suffix in ['', '-tiny']:
                    workload_path = os.path.join(project_root, 'workload', f'{domain}{suffix}.json')
                    if os.path.exists(workload_path):
                        with open(workload_path, 'r') as f:
                            tasks = json.load(f)
                            for task in tasks:
                                self.workload_data[task['id']] = task
                        if self.verbose:
                            print(f"[DataflowSystem] Loaded {len(tasks)} tasks from {workload_path}")
        except Exception as e:
            if self.verbose:
                print(f"[DataflowSystem] Could not load workload for ground truth: {e}")

    def _load_format_hints(self, dataset_directory: str) -> None:
        """Load format hints for the domain."""
        try:
            parts = dataset_directory.rstrip('/').split('/')
            if 'data' in parts:
                data_idx = parts.index('data')
                domain = parts[data_idx + 1]
                project_root = '/'.join(parts[:data_idx])
                hint_path = os.path.join(project_root, 'format_hint', f'{domain}.json')
                if os.path.exists(hint_path):
                    with open(hint_path, 'r') as f:
                        hints = json.load(f)
                        for hint in hints:
                            self.format_hints[hint['id']] = hint.get('format_hint', '')
                    if self.verbose:
                        print(f"[DataflowSystem] Loaded {len(hints)} format hints from {hint_path}")
        except Exception as e:
            if self.verbose:
                print(f"[DataflowSystem] Could not load format hints: {e}")

    def _setup_agent(self) -> None:
        """Initialize and setup the DataflowAgent."""
        if self.verbose:
            print(f"[DataflowSystem] Setting up agent with model: {self.model_type}")
            print(f"[DataflowSystem] Agent settings: max_steps={self.max_steps}, mode={self.agent_mode}, context_mode={self.context_mode}")

        self.agent = DataflowAgent(
            model_type=self.model_type,
            driver=self.driver,
            max_steps=self.max_steps,
            max_operator_result_char_limit=self.max_operator_result_char_limit,
            max_operator_result_cell_char_limit=self.max_operator_result_cell_char_limit,
            operator_result_serialization_mode=self.operator_result_serialization_mode,
            tool_timeout_seconds=self.tool_timeout_seconds,
            execution_timeout_minutes=self.execution_timeout_minutes,
            agent_mode=self.agent_mode,
            context_mode=self.context_mode,
            parallel_tool_calls=self.parallel_tool_calls,
            allowed_operator_types=self.allowed_operator_types,
            disabled_tools=self.disabled_tools,
            stats_enabled=self.stats_enabled,
            include_operator_properties=self.include_operator_properties,
            max_operator_edits=self.max_operator_edits,
            lineage_hint_on_stall=self.lineage_hint_on_stall,
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()

    def _build_prompt(self, query: str, file_paths: List[str], format_hint: str = "") -> str:
        """
        Build the prompt for the agent.

        Args:
            query: The natural language query
            file_paths: List of file paths available for the query
            format_hint: Optional format hint for the expected answer format

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Note: All paths are relative. Some paths may contain wildcards (e.g., "folder/*" or "file-*.csv"). Use glob patterns to match and read those files.

Question: {query}

Answer format: {format_hint}

Your last line MUST BE: **Final Answer: <value>**"""

        return prompt

    def _expand_data_sources(self, data_sources: List[str]) -> List[str]:
        """
        Expand wildcard patterns in data_sources to actual file paths.

        Args:
            data_sources: List of file patterns (may contain wildcards)

        Returns:
            List of actual file paths (relative to current working directory)
        """
        return expand_data_sources(
            data_sources=data_sources,
            dataset_directory=self.dataset_directory,
            all_files=list(self.dataset.keys()),
            verbose=self.verbose
        )

    def serve_query(
        self,
        query: str,
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None
    ) -> Dict:
        """
        Serve a query using the DataflowAgent.

        Args:
            query: Natural language query
            query_id: Unique identifier for the query
            subset_files: Optional list of specific files to use

        Returns:
            Dictionary with explanation, pipeline_code, and token usage
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call process_dataset() first.")

        # Expand wildcards and build file paths
        if subset_files:
            file_paths = self._expand_data_sources(subset_files)
        else:
            # Use a recursive wildcard instead of listing every file
            file_paths = [os.path.relpath(self.dataset_directory) + "/**/*"]

        if self.verbose:
            print(f"[DataflowSystem] Processing query: {query_id}")
            print(f"[DataflowSystem] Using {len(file_paths)} files")

        # Build prompt with file paths and format hint
        format_hint = self.format_hints.get(query_id, "")
        prompt = self._build_prompt(query, file_paths, format_hint=format_hint)

        # Save prompt for debugging
        query_output_dir = os.path.join(self.output_dir, query_id)
        os.makedirs(query_output_dir, exist_ok=True)
        prompt_path = os.path.join(query_output_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        # Save config.json with run parameters
        config = {
            "system_name": self.name,
            "model_type": self.model_type,
            "driver": self.driver,
            "query_id": query_id,
            "dataset_directory": str(self.dataset_directory),
            "num_files": len(file_paths),
            "subset_files": subset_files,
            "agent_settings": {
                "max_steps": self.max_steps,
                "max_operator_result_char_limit": self.max_operator_result_char_limit,
                "max_operator_result_cell_char_limit": self.max_operator_result_cell_char_limit,
                "operator_result_serialization_mode": self.operator_result_serialization_mode,
                "tool_timeout_seconds": self.tool_timeout_seconds,
                "execution_timeout_minutes": self.execution_timeout_minutes,
                "agent_mode": self.agent_mode,
                "context_mode": self.context_mode,
                "parallel_tool_calls": self.parallel_tool_calls,
                "allowed_operator_types": self.allowed_operator_types,
                "disabled_tools": self.disabled_tools,
                "stats_enabled": self.stats_enabled,
                "include_operator_properties": self.include_operator_properties,
                "max_operator_edits": self.max_operator_edits,
                "lineage_hint_on_stall": self.lineage_hint_on_stall,
            }
        }
        config_path = os.path.join(query_output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # WORKAROUND: Create a fresh agent for each query to avoid workflow state pollution
        # The reset() API doesn't properly clear workflow operators, causing stale operators
        # to accumulate and be reused incorrectly across tasks.
        try:
            if self.verbose:
                print(f"[DataflowSystem] Creating fresh agent for clean workflow state...")
            self.agent.cleanup()  # Delete old agent and workflow
            self._setup_agent()   # Create new agent with fresh workflow
        except Exception as e:
            if self.verbose:
                print(f"[DataflowSystem] Fresh agent creation warning: {e}")
            # Fallback to reset if cleanup/setup fails
            try:
                self.agent.reset()
            except Exception as e2:
                if self.verbose:
                    print(f"[DataflowSystem] Reset fallback warning: {e2}")

        # Run the agent with timing
        start_time = time.time()
        try:
            result: MessageResult = self.agent.run(prompt)
        except Exception as e:
            print(f"[DataflowSystem] Error running agent: {e}")
            return {
                "explanation": {
                    "id": "main-task",
                    "answer": f"Error: {str(e)}",
                },
                "pipeline_code": "",
                "token_usage": 0,
                "token_usage_input": 0,
                "token_usage_output": 0,
            }
        elapsed_seconds = time.time() - start_time

        # Save response for debugging
        response_path = os.path.join(query_output_dir, "response.txt")
        with open(response_path, "w") as f:
            f.write(result.response or "(empty response)")

        # Fetch the full ReAct trace from the agent — this replaces the old
        # `messages.json` export. We always save it so every trace is auditable.
        react_data: Dict = {}
        react_steps: List[Dict] = []
        try:
            react_data = get_agent_react_steps(
                agent_id=self.agent.agent_id,
                agent_endpoint=self.agent.agent_service_endpoint,
            )
            react_steps = react_data.get("steps", []) or []
            react_path = os.path.join(query_output_dir, "react_steps.json")
            with open(react_path, "w") as f:
                json.dump(react_data, f, indent=2, default=str)
            if self.verbose:
                print(f"[DataflowSystem] React steps saved ({len(react_steps)} steps) to {react_path}")
        except Exception as e:
            if self.verbose:
                print(f"[DataflowSystem] Could not save react steps: {e}")

        if self.verbose:
            print(f"[DataflowSystem] Raw response length: {len(result.response) if result.response else 0}")
            print(f"[DataflowSystem] React steps count: {len(react_steps)}")
            print(f"[DataflowSystem] Stopped: {result.stopped}, Error: {result.error}")

        # Extract token usage
        usage = result.usage or {}
        token_usage = usage.get("total_tokens", 0) or usage.get("totalTokens", 0)
        token_usage_input = usage.get("input_tokens", 0) or usage.get("inputTokens", 0)
        token_usage_output = usage.get("output_tokens", 0) or usage.get("outputTokens", 0)

        # Step count: prefer the WS-derived count from MessageResult.stats; fall
        # back to counting agent steps with tool calls in the ReAct trace.
        stats_from_service = result.stats or {}
        num_steps = int(stats_from_service.get("steps") or 0)
        if num_steps == 0 and react_steps:
            num_steps = sum(
                1
                for s in react_steps
                if s.get("role") == "agent" and (s.get("toolCalls") or [])
            )

        # Save stats.json
        stats = {
            "input_tokens": token_usage_input,
            "output_tokens": token_usage_output,
            "total_tokens": token_usage,
            "num_steps": num_steps,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
        stats_path = os.path.join(query_output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Parse answer from response (or from messages if response is empty)
        answer = parse_answer(result.response, result.messages)

        if self.verbose:
            print(f"[DataflowSystem] Answer: {answer[:200]}..." if len(str(answer)) > 200 else f"[DataflowSystem] Answer: {answer}")
            print(f"[DataflowSystem] Token usage: {token_usage}")

        # Build the explanation dict
        explanation = {
            "id": "main-task",
            "answer": answer,
        }

        # Save answer.json
        answer_path = os.path.join(query_output_dir, "answer.json")
        with open(answer_path, "w") as f:
            json.dump(explanation, f, indent=2)

        # Save ground_truth.json if available
        if query_id in self.workload_data:
            task = self.workload_data[query_id]
            ground_truth = {
                "id": task.get("id"),
                "query": task.get("query"),
                "answer": task.get("answer"),
                "answer_type": task.get("answer_type"),
                "data_sources": task.get("data_sources", []),
            }
            ground_truth_path = os.path.join(query_output_dir, "ground_truth.json")
            with open(ground_truth_path, "w") as f:
                json.dump(ground_truth, f, indent=2)
            if self.verbose:
                print(f"[DataflowSystem] Ground truth saved: {task.get('answer')}")

        # Save workflow.json from the agent
        try:
            workflow = get_agent_workflow(
                agent_id=self.agent.agent_id,
                agent_endpoint=self.agent.agent_service_endpoint
            )
            workflow_path = os.path.join(query_output_dir, "workflow.json")
            with open(workflow_path, "w") as f:
                json.dump(workflow, f, indent=2)
            if self.verbose:
                print(f"[DataflowSystem] Workflow saved to {workflow_path}")
        except Exception as e:
            if self.verbose:
                print(f"[DataflowSystem] Could not save workflow: {e}")

        return {
            "explanation": explanation,
            "pipeline_code": "",  # Skip pipeline eval
            "token_usage": token_usage,
            "token_usage_input": token_usage_input,
            "token_usage_output": token_usage_output,
        }

    def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.agent:
            try:
                self.agent.cleanup()
            except Exception as e:
                if self.verbose:
                    print(f"[DataflowSystem] Cleanup warning: {e}")
            self.agent = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class DataflowSystemHaiku45(DataflowSystem):
    """DataflowSystem using Claude Haiku 4.5 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGPT5Mini(DataflowSystem):
    """DataflowSystem using GPT-5 Mini model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5Mini",
            verbose=verbose,
            *args,
            **kwargs
        )


# ────────────────────────────────────────────────────────────────────────
# Haiku-4.5 × {context-mode, stats} matrix — 3 × 2 = 6 variants.
# Each subclass pins one combination so the benchmark can sweep both
# dimensions by name (no env vars required). All other agent parameters
# follow `DataflowSystemHaiku45`'s defaults.
# ────────────────────────────────────────────────────────────────────────


class _Haiku45Variant(DataflowSystem):
    """Base for the Haiku 4.5 sweep — subclasses only override name + the
    `context_mode` / `stats_enabled` pair so the matrix stays declarative."""

    _CONTEXT_MODE: str = "latest"
    _STATS_ENABLED: bool = False
    _NAME: str = "DataflowSystemHaiku45"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            context_mode=self._CONTEXT_MODE,
            stats_enabled=self._STATS_ENABLED,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemHaiku45LatestStatsOff(_Haiku45Variant):
    _CONTEXT_MODE = "latest"
    _STATS_ENABLED = False
    _NAME = "DataflowSystemHaiku45LatestStatsOff"


class DataflowSystemHaiku45LatestStatsOn(_Haiku45Variant):
    _CONTEXT_MODE = "latest"
    _STATS_ENABLED = True
    _NAME = "DataflowSystemHaiku45LatestStatsOn"


class DataflowSystemHaiku45DeltaStatsOff(_Haiku45Variant):
    _CONTEXT_MODE = "delta"
    _STATS_ENABLED = False
    _NAME = "DataflowSystemHaiku45DeltaStatsOff"


class DataflowSystemHaiku45DeltaStatsOn(_Haiku45Variant):
    _CONTEXT_MODE = "delta"
    _STATS_ENABLED = True
    _NAME = "DataflowSystemHaiku45DeltaStatsOn"


class DataflowSystemHaiku45FullStatsOff(_Haiku45Variant):
    _CONTEXT_MODE = "full"
    _STATS_ENABLED = False
    _NAME = "DataflowSystemHaiku45FullStatsOff"


class DataflowSystemHaiku45FullStatsOn(_Haiku45Variant):
    _CONTEXT_MODE = "full"
    _STATS_ENABLED = True
    _NAME = "DataflowSystemHaiku45FullStatsOn"


# ────────────────────────────────────────────────────────────────────────
# GPT family × {latest, delta} with stats=on. Pins gpt-5-mini and gpt-5.2
# from the LiteLLM catalogue so the benchmark can sweep models alongside
# the Haiku 4.5 matrix above. Stats are on for all four — the comparison
# is purely about model × context-mode at the same render configuration.
# ────────────────────────────────────────────────────────────────────────


class _GPTStatsOnVariant(DataflowSystem):
    """Base for the GPT-family stats-on sweep. Subclasses pin model + context."""

    _MODEL_TYPE: str = "gpt-5-mini"
    _CONTEXT_MODE: str = "latest"
    _NAME: str = "DataflowSystem"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type=self._MODEL_TYPE,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            context_mode=self._CONTEXT_MODE,
            stats_enabled=True,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestStatsOn(_GPTStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _CONTEXT_MODE = "latest"
    _NAME = "DataflowSystemGPT5MiniLatestStatsOn"


class DataflowSystemGPT5MiniDeltaStatsOn(_GPTStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _CONTEXT_MODE = "delta"
    _NAME = "DataflowSystemGPT5MiniDeltaStatsOn"


class DataflowSystemGPT5MiniFullStatsOn(_GPTStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _CONTEXT_MODE = "full"
    _NAME = "DataflowSystemGPT5MiniFullStatsOn"


class DataflowSystemGPT52LatestStatsOn(_GPTStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _CONTEXT_MODE = "latest"
    _NAME = "DataflowSystemGPT52LatestStatsOn"


class DataflowSystemGPT52DeltaStatsOn(_GPTStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _CONTEXT_MODE = "delta"
    _NAME = "DataflowSystemGPT52DeltaStatsOn"


class DataflowSystemGPT52FullStatsOn(_GPTStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _CONTEXT_MODE = "full"
    _NAME = "DataflowSystemGPT52FullStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan1: latest-mode convergence guard + lineage stall hint. These variants
# isolate the guard against the existing LatestStatsOn systems and keep the
# near-term focus on LATEST context rather than FULL/DELTA.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestGuardStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _MAX_OPERATOR_EDITS = 2
    _LINEAGE_HINT_ON_STALL = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            max_operator_edits=self._MAX_OPERATOR_EDITS,
            lineage_hint_on_stall=self._LINEAGE_HINT_ON_STALL,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestGuardStatsOn(_GPTLatestGuardStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestGuardStatsOn"


class DataflowSystemGPT52LatestGuardStatsOn(_GPTLatestGuardStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestGuardStatsOn"


# ────────────────────────────────────────────────────────────────────────
# In-house local model under the text-mode `local-react` driver. Forces
# the local driver explicitly (model_type="local-llm" would auto-derive it
# server-side, but we pin it for clarity). Stats off; operator properties
# off — note the local-react context assembler ignores
# includeOperatorProperties regardless (server.ts:62-66), so this only
# documents intent. max_steps capped at 20.
# ────────────────────────────────────────────────────────────────────────


class DataflowSystemLocalLlm(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=20,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm",
            verbose=verbose,
            *args,
            **kwargs,
        )

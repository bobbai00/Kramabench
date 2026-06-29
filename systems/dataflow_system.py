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
        max_operator_edits: int = 0,
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
        schema_in_result: bool = False,
        loader_hint: bool = False,
        value_format_flags: bool = False,
        lineage_stats: bool = False,
        lineage_error_context: bool = False,
        join_telemetry: bool = False,
        graph_audit: bool = False,
        coercion_telemetry: bool = False,
        compact_stats: bool = False,
        thought_replay: bool = False,
        thought_replay_k: int = 10,
        error_reflection: bool = False,
        error_reflection_threshold: int = 3,
        few_shot_prompt: bool = False,
        table_structure_hint: bool = False,
        loader_escalation: bool = False,
        frontier_depth: int = 0,
        flow_level: int = 0,
        data_level: int = 0,
        max_result_rows: int = 0,
        max_loaders_per_source: int = 0,
        attempt_reflection: bool = False,
        tool_dialect: str = None,
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
            verbose: Enable verbose logging
            name: System name for benchmark identification
        """
        super().__init__(name, verbose=verbose, *args, **kwargs)

        self.model_type = model_type or "claude-haiku-4.5"
        # None -> let agent-service auto-derive the driver from model_type.
        self.driver = driver
        self.max_steps = max_steps or 50
        # Convergence guard: max consecutive same-operator edits before reject.
        # 0 = disabled (no-op default; baseline reproduces). Kept as-is (not
        # `or`) so an explicit 0 stays 0.
        self.max_operator_edits = max_operator_edits
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
        # None -> server default (true).
        self.include_operator_properties = include_operator_properties
        # ---- DECORATE control: the two ordinal facet levels (CONTEXT-DESIGN §5) ----
        # Context decoration is controlled purely by flow_level / data_level (each
        # expands, agent-service side, into the per-rung render flags via the rung
        # catalog). The individual decoration booleans are no longer sent on the
        # wire; for back-compat the legacy per-lever kwargs (schema_in_result,
        # loader_hint, table_structure_hint, …) are TRANSLATED here to the minimum
        # level that enables them, and OR'd with any explicit level. This collapses
        # the old non-cumulative one-off configs onto the cumulative ladder.
        #   flow L1 loader_hint | L2 lineage_stats/lineage_error_context |
        #          L3 graph_audit/join_telemetry
        #   data L1 schema_in_result | L2 table_structure_hint |
        #          L3 value_format_flags/coercion_telemetry
        flow_from_flags = 0
        if loader_hint:
            flow_from_flags = max(flow_from_flags, 1)
        if lineage_stats or lineage_error_context:
            flow_from_flags = max(flow_from_flags, 2)
        if graph_audit or join_telemetry:
            flow_from_flags = max(flow_from_flags, 3)
        data_from_flags = 0
        if schema_in_result or stats_enabled or compact_stats:
            data_from_flags = max(data_from_flags, 1)
        if table_structure_hint:
            data_from_flags = max(data_from_flags, 2)
        if value_format_flags or coercion_telemetry:
            data_from_flags = max(data_from_flags, 3)
        self.flow_level = max(flow_level, flow_from_flags)
        self.data_level = max(data_level, data_from_flags)
        # SELECT reinjection + static prior (kept fine-grained knobs).
        self.thought_replay = thought_replay
        self.thought_replay_k = thought_replay_k
        self.error_reflection = error_reflection
        self.error_reflection_threshold = error_reflection_threshold
        self.few_shot_prompt = few_shot_prompt
        self.loader_escalation = loader_escalation
        self.max_result_rows = max_result_rows
        # Global loader-proliferation budget (plan5); 0 = disabled (no-op).
        self.max_loaders_per_source = max_loaders_per_source
        # Attempt-reflection block on heavily-edited operators (plan3); no-op default.
        self.attempt_reflection = attempt_reflection
        # Tool-call dialect for the local-react driver. Default to the new Qwen
        # XML format ("qwen-xml"), matching the agent-service default; the
        # react-text variants opt in to the previous ReAct text format. Ignored
        # by the vercel-tool-use driver.
        self.tool_dialect = tool_dialect or "qwen-xml"

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
            max_operator_edits=self.max_operator_edits,
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
            include_operator_properties=self.include_operator_properties,
            thought_replay=self.thought_replay,
            thought_replay_k=self.thought_replay_k,
            error_reflection=self.error_reflection,
            error_reflection_threshold=self.error_reflection_threshold,
            few_shot_prompt=self.few_shot_prompt,
            loader_escalation=self.loader_escalation,
            flow_level=self.flow_level,
            data_level=self.data_level,
            max_result_rows=self.max_result_rows,
            max_loaders_per_source=self.max_loaders_per_source,
            attempt_reflection=self.attempt_reflection,
            tool_dialect=self.tool_dialect,
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
                "max_operator_edits": self.max_operator_edits,
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
                "include_operator_properties": self.include_operator_properties,
                "thought_replay": self.thought_replay,
                "thought_replay_k": self.thought_replay_k,
                "error_reflection": self.error_reflection,
                "error_reflection_threshold": self.error_reflection_threshold,
                "few_shot_prompt": self.few_shot_prompt,
                "loader_escalation": self.loader_escalation,
                "flow_level": self.flow_level,
                "data_level": self.data_level,
                "max_result_rows": self.max_result_rows,
                "max_loaders_per_source": self.max_loaders_per_source,
                "attempt_reflection": self.attempt_reflection,
                "tool_dialect": self.tool_dialect,
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

        # Extract token usage (with per-class breakdown: reasoning + cache-read)
        usage = result.usage or {}
        token_usage = usage.get("total_tokens", 0) or usage.get("totalTokens", 0)
        token_usage_input = usage.get("input_tokens", 0) or usage.get("inputTokens", 0)
        token_usage_output = usage.get("output_tokens", 0) or usage.get("outputTokens", 0)
        token_usage_reasoning = usage.get("reasoning_tokens", 0) or usage.get("reasoningTokens", 0)
        token_usage_cached = (
            usage.get("cached_input_tokens", 0) or usage.get("cachedInputTokens", 0)
            or usage.get("cache_read_input_tokens", 0)
        )
        cost_usd = 0.0
        try:
            from systems.cost_utils import compute_cost
            c = compute_cost(
                self.model_type,
                input_tokens=token_usage_input,
                output_tokens=token_usage_output,
                cached_tokens=token_usage_cached,
            )
            cost_usd = c if c is not None else 0.0
        except Exception:
            pass

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
            "reasoning_tokens": token_usage_reasoning,
            "cached_tokens": token_usage_cached,
            "cost_usd": cost_usd,
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
            "token_usage_reasoning": token_usage_reasoning,
            "token_usage_cached": token_usage_cached,
            "cost_usd": cost_usd,
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


# ─────────────────────────────────────────────────────────────────────────
# The "converge" stack = the cost-minimized DataFlow configuration:
# LATEST context (aggregated history) + flow_level=1 (loader-remediation) +
# data_level=1 (compact typed Schema line) + a loader-proliferation budget and
# attempt-reflection (ACT-side convergence guards). Expressed via the two
# ordinal DECORATE levels (see claude/CONTEXT-DESIGN.md §5/§8b).
# ─────────────────────────────────────────────────────────────────────────


class DataflowSystemGPT52LatestSchemaConverge(DataflowSystem):
    """gpt-5.2 converge stack (flow_level=1, data_level=1). Thesis arm."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="latest",
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52LatestSchemaConverge",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestSchemaConverge(DataflowSystem):
    """gpt-5.4 converge stack, LATEST context (flow_level=1, data_level=1).
    gpt-5.4 peer of DataflowSystemGPT52LatestSchemaConverge; pairs with the
    DELTA variant below for a clean context-mode A/B (only context_mode differs;
    both capped at max_steps=25)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestSchemaConverge",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54DeltaSchemaConverge(DataflowSystem):
    """gpt-5.4 converge stack, DELTA context (flow_level=1, data_level=1).
    DELTA-context counterpart of DataflowSystemGPT54LatestSchemaConverge — same
    knobs, context_mode=delta (per-step incremental context instead of the
    aggregated LATEST snapshot)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="delta",
            max_steps=25,
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54DeltaSchemaConverge",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaConverge(DataflowSystem):
    """gpt-5-mini converge stack (flow_level=1, data_level=1). Thesis arm
    (the gpt-5-mini peer for the symmetric DataFlow-vs-Script comparison)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniLatestSchemaConverge",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaConvergeTableStruct(DataflowSystem):
    """Converge + data_level=2 — the structural-profile rung (#25/26/27), the
    campaign's one accuracy win (+8 single / +4 best-of-2)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=1,
            data_level=2,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniLatestSchemaConvergeTableStruct",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaConvergeFewShot(DataflowSystem):
    """Converge + few-shot worked-examples prior (W2, the −5.9% cost win)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            few_shot_prompt=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniLatestSchemaConvergeFewShot",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaConvergeCap20(DataflowSystem):
    """Converge + a hard 20-step budget (#31, ACT-side cost lever)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_steps=20,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniLatestSchemaConvergeCap20",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaConvergeLevels(DataflowSystem):
    """Reference level-config SUT — converge plus flow_level=2 / data_level=2
    set explicitly, demonstrating the ordinal DECORATE knobs."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            max_loaders_per_source=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniLatestSchemaConvergeLevels",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaConvergeThoughtReplay(DataflowSystem):
    """Converge + thought-replay reinjection (SELECT-side). Re-injects the last
    K=10 agent reasoning events as a `# Reasoning` block (thought + compressed
    actions) plus per-operator `Last edited: step N` back-pointers. Only fires
    under LATEST context — built on the converge base so the knob actually takes
    effect rather than silently no-op'ing."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=1,
            data_level=1,
            max_loaders_per_source=2,
            attempt_reflection=True,
            thought_replay=True,
            thought_replay_k=10,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniLatestSchemaConvergeThoughtReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────
# Haiku-4.5 2x2 parameter study. Two axes, four configs; everything else held
# constant (model=claude-haiku-4.5, context_mode=latest, data annotation fixed
# at data_level=2 = mid-level structural-profile stats, char limits 1000/3000 —
# matching DataflowSystemHaiku45 / the converge stacks).
#   Axis 1 — thought_replay (the new SELECT-side reasoning-reinjection rung): off/on
#   Axis 2 — data-lineage flow section (the `# Operators needing attention`
#            block rendered AFTER the dataflow): off (flow_level=0) /
#            on (flow_level=2, which enables the lineage rungs: cardinality +
#            lineage-error-context, on top of loader remediation).
# thought_replay only fires under LATEST, so context_mode=latest is required for
# the axis to be meaningful.
# ─────────────────────────────────────────────────────────────────────────


class DataflowSystemHaiku45Annot2(DataflowSystem):
    """2x2 baseline: data annotation L2 only. lineage OFF, thought_replay OFF."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="latest",
            flow_level=0,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45Annot2",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaiku45Annot2Lineage(DataflowSystem):
    """2x2: lineage ON (flow_level=2), thought_replay OFF. data annotation L2."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45Annot2Lineage",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaiku45Annot2LineageErrorReflect(DataflowSystem):
    """Haiku-4.5: latest + lineage (flow2) + data L2, replay OFF, error-reflection ON
    (fold). Haiku prints often, so this exercises the new print-output surfacing too."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            thought_replay=False,
            error_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45Annot2LineageErrorReflect",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaiku45Annot2ThoughtReplay(DataflowSystem):
    """2x2: lineage OFF, thought_replay ON (K=10). data annotation L2."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="latest",
            flow_level=0,
            data_level=2,
            thought_replay=True,
            thought_replay_k=10,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45Annot2ThoughtReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaiku45Annot2LineageThoughtReplay(DataflowSystem):
    """2x2: lineage ON (flow_level=2) + thought_replay ON (K=10). data annotation L2."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            thought_replay=True,
            thought_replay_k=10,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45Annot2LineageThoughtReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaiku45DeltaLineageReplay(DataflowSystem):
    """DELTA-mode counterpart of the both-on config: context_mode=delta + lineage
    (flow_level=2) + data annotation L2 + thought_replay flag set. NOTE: thought_replay
    is a no-op under DELTA (the `# Reasoning` reinjection is LATEST-only) — DELTA already
    renders each event's thought inline, so this config has the reasoning trajectory
    natively. Pairs with DataflowSystemHaiku45Annot2Lineage (LATEST, lineage on,
    thoughts OFF) for a thoughts-present vs thoughts-absent comparison with lineage held on."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="delta",
            flow_level=2,
            data_level=2,
            thought_replay=True,
            thought_replay_k=10,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemHaiku45DeltaLineageReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniAnnot2LineageThoughtReplay(DataflowSystem):
    """gpt-5-mini peer of the Haiku both-flags system: LATEST + lineage (flow_level=2)
    + data annotation L2 + recent-events (thoughtReplay) ON with K=5."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            thought_replay=True,
            thought_replay_k=5,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniAnnot2LineageThoughtReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniAnnot2Lineage(DataflowSystem):
    """gpt-5-mini peer of the Haiku lineage-only system: LATEST + lineage (flow_level=2)
    + data annotation L2, recent-events OFF. The no-replay control."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniAnnot2Lineage",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniAnnot2LineageErrorReflect(DataflowSystem):
    """gpt-5-mini: LATEST + lineage (flow_level=2) + data annotation L2, recent-events
    OFF, error-reflection ON. A/B treatment over DataflowSystemGPT5MiniAnnot2Lineage
    to test whether surfacing per-operator attempt/error history breaks the churn
    loops (same config otherwise)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="latest",
            flow_level=2,
            data_level=2,
            thought_replay=False,
            error_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniAnnot2LineageErrorReflect",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54Annot2LineageErrorReflect(DataflowSystem):
    """gpt-5.4: LATEST + lineage (flow_level=2) + data annotation L2, recent-events
    OFF, error-reflection ON (folded into the operator block). max_steps=50."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=50,
            flow_level=2,
            data_level=2,
            thought_replay=False,
            error_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54Annot2LineageErrorReflect",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54Annot2LineageThoughtReplay(DataflowSystem):
    """gpt-5.4 latest + lineage (flow_level=2) + data annotation L2 + recent-events
    (thoughtReplay) ON with K=5. max_steps=50 (matches the code-agent default for the
    head-to-head comparison)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=50,
            flow_level=2,
            data_level=2,
            thought_replay=True,
            thought_replay_k=5,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54Annot2LineageThoughtReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54Annot2Lineage(DataflowSystem):
    """gpt-5.4 latest + lineage (flow_level=2) + data annotation L2, recent-events
    (thoughtReplay) OFF — current dataflow + lineage section only. The no-replay
    control peer of DataflowSystemGPT54Annot2LineageThoughtReplay. max_steps=50."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=50,
            flow_level=2,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54Annot2Lineage",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemSonnet46Annot2LineageThoughtReplay(DataflowSystem):
    """Sonnet-4.6 peer of the Haiku/gpt-5-mini both-flags system: LATEST + lineage
    (flow_level=2) + data annotation L2 + recent-events (thoughtReplay) ON with K=5.
    max_steps capped at 20."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-sonnet-4.6",
            context_mode="latest",
            max_steps=20,
            flow_level=2,
            data_level=2,
            thought_replay=True,
            thought_replay_k=5,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemSonnet46Annot2LineageThoughtReplay",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm1(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm1",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm2(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm2",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm3(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm3",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm4(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm4",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm5(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm5",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm6(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm6",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm7(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm7",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm8(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm8",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm9(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=10,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm9",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm10(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=15,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm10",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm11(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=15,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm11",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm12(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=15,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm12",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm13(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=15,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm13",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm14(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=15,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm14",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm15(DataflowSystem):
    """DataflowSystem using the in-house local model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=15,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm15",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV1(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV1",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV2(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV2",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV3(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV3",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV4(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV4",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV5(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV5",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV6(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV6",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV7(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV7",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV8(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV8",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV9(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV9",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BV10(DataflowSystem):
    """DataflowSystem using the in-house local 30B model via the local-react driver."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BV10",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlm30BBaseModel(DataflowSystem):
    """DataflowSystem using the in-house local 30B BASE (non-instruct) model
    via the local-react driver. Config mirrors the 30B sweep variants."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemLocalLlm30BBaseModel",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ────────────────────────────────────────────────────────────────────────
# In-house local model under the PREVIOUS ReAct text tool-call dialect
# (tool_dialect="react-text"). Same local-react driver as the blocks above,
# but the model emits Thought / Action / Action Input text instead of the
# Qwen3-Coder `<tool_call><function=…><parameter=…>` XML — so these pin
# tool_dialect explicitly. The default dialect is "qwen-xml" (the new format),
# so every other local system above uses Qwen. Config otherwise mirrors the
# 30B variants (max_steps=25).
# ────────────────────────────────────────────────────────────────────────


class _LocalLlmReactTextVariant(DataflowSystem):
    """Base for the local model under the react-text tool-call dialect.
    Subclasses only override `name` so the sweep stays declarative."""

    _NAME: str = "DataflowSystemLocalLlmReactText"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="local-llm",
            driver="local-react",
            tool_dialect="react-text",
            max_steps=25,
            stats_enabled=False,
            include_operator_properties=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemLocalLlmReactText1(_LocalLlmReactTextVariant):
    _NAME = "DataflowSystemLocalLlmReactText1"


class DataflowSystemLocalLlmReactText2(_LocalLlmReactTextVariant):
    _NAME = "DataflowSystemLocalLlmReactText2"


class DataflowSystemLocalLlmReactText3(_LocalLlmReactTextVariant):
    _NAME = "DataflowSystemLocalLlmReactText3"


class DataflowSystemLocalLlmReactText4(_LocalLlmReactTextVariant):
    _NAME = "DataflowSystemLocalLlmReactText4"


class DataflowSystemLocalLlmReactText5(_LocalLlmReactTextVariant):
    _NAME = "DataflowSystemLocalLlmReactText5"

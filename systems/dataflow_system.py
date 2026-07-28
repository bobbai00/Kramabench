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
        agent_turns: bool = False,
        context_window_tokens: int = 0,
        static_compaction: bool = False,
        frontier_decay_config: Optional[Dict[str, object]] = None,
        fold_resolved_revisions_config: Optional[Dict[str, object]] = None,
        probe_retirement_config: Optional[Dict[str, object]] = None,
        enable_inspect_tool: bool = False,
        enable_render_prefs: bool = False,
        enable_code_in_snapshot: bool = False,
        compaction_strategy: str = "compress",
        deck_sample_ratio: float = 0.10,
        error_reflection: bool = False,
        error_reflection_threshold: int = 3,
        few_shot_prompt: bool = False,
        table_structure_hint: bool = False,
        frontier_depth: int = 0,
        flow_level: int = 0,
        data_level: int = 0,
        max_result_rows: int = 0,
        attempt_reflection: bool = False,
        column_stats: bool = False,
        value_format: bool = False,
        data_hints: bool = False,
        tool_dialect: str = None,
        summarize_params: Optional[Dict[str, object]] = None,
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
        self.agent_turns = agent_turns
        self.context_window_tokens = context_window_tokens
        self.static_compaction = static_compaction
        self.frontier_decay_config = frontier_decay_config
        self.fold_resolved_revisions_config = fold_resolved_revisions_config
        self.probe_retirement_config = probe_retirement_config
        self.enable_inspect_tool = enable_inspect_tool
        self.enable_render_prefs = enable_render_prefs
        self.enable_code_in_snapshot = enable_code_in_snapshot
        self.compaction_strategy = compaction_strategy
        self.deck_sample_ratio = deck_sample_ratio
        self.error_reflection = error_reflection
        self.error_reflection_threshold = error_reflection_threshold
        self.few_shot_prompt = few_shot_prompt
        self.max_result_rows = max_result_rows
        # Attempt-reflection block on heavily-edited operators (plan3); no-op default.
        self.attempt_reflection = attempt_reflection
        self.column_stats = column_stats
        self.value_format = value_format
        self.data_hints = data_hints
        # Tool-call dialect for the local-react driver. Default to the new Qwen
        # XML format ("qwen-xml"), matching the agent-service default; the
        # react-text variants opt in to the previous ReAct text format. Ignored
        # by the vercel-tool-use driver.
        self.tool_dialect = tool_dialect or "qwen-xml"
        # Deep-partial summarize() params patch (e.g. force detail="shape" =
        # counts+schema+stats, no data rows). None = preset unchanged.
        self.summarize_params = summarize_params

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

    # Generic answer-format sentence per answer_type — used for SUBTASKS, which
    # have no curated format_hint in format_hint/<domain>.json (those cover only
    # main tasks). Without this a subtask prompt gets an empty "Answer format:"
    # line and the agent free-forms prose/variable-names/wrong-id, which the
    # F1/exact scorers zero out. Keep these answer-agnostic (no value leakage).
    _SUBTASK_FORMAT_HINTS = {
        "list_exact": "Return ONLY a flat list of the matching values, as a comma-separated list "
                      "(e.g. value1, value2, value3). No prose, no column names, no explanation — just the values.",
        "list_approximate": "Return ONLY a flat list of the matching values, as a comma-separated list "
                            "(e.g. value1, value2, value3). No prose or explanation — just the values.",
        "numeric_exact": "The result should be a single numeric value only (no units, no prose).",
        "numeric_approximate": "The result should be a single numeric value only (no units, no prose).",
        "string_exact": "The result should be a single value only (no prose or explanation).",
        "string_approximate": "The result should be a single short value/label only (no prose or explanation).",
    }

    def _load_format_hints(self, dataset_directory: str) -> None:
        """Load format hints for the domain (main tasks) and synthesize format
        hints for every subtask from its answer_type (so subtask runs get a
        proper 'Answer format:' line instead of an empty one)."""
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
                # synthesize subtask format hints from answer_type
                wl_path = os.path.join(project_root, 'workload', f'{domain}.json')
                if os.path.exists(wl_path):
                    n = 0
                    with open(wl_path, 'r') as f:
                        for task in json.load(f):
                            for st in (task.get('subtasks', []) if isinstance(task, dict) else []):
                                sid, at = st.get('id'), st.get('answer_type')
                                if sid and sid not in self.format_hints and at in self._SUBTASK_FORMAT_HINTS:
                                    self.format_hints[sid] = self._SUBTASK_FORMAT_HINTS[at]
                                    n += 1
                    if self.verbose and n:
                        print(f"[DataflowSystem] Synthesized {n} subtask format hints from answer_type")
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
            agent_turns=self.agent_turns,
            context_window_tokens=self.context_window_tokens,
            static_compaction=self.static_compaction,
            frontier_decay_config=self.frontier_decay_config,
            fold_resolved_revisions_config=self.fold_resolved_revisions_config,
            probe_retirement_config=self.probe_retirement_config,
            enable_inspect_tool=self.enable_inspect_tool,
            enable_render_prefs=self.enable_render_prefs,
            enable_code_in_snapshot=self.enable_code_in_snapshot,
            compaction_strategy=self.compaction_strategy,
            deck_sample_ratio=self.deck_sample_ratio,
            error_reflection=self.error_reflection,
            error_reflection_threshold=self.error_reflection_threshold,
            few_shot_prompt=self.few_shot_prompt,
            flow_level=self.flow_level,
            data_level=self.data_level,
            max_result_rows=self.max_result_rows,
            attempt_reflection=self.attempt_reflection,
            column_stats=self.column_stats,
            value_format=self.value_format,
            data_hints=self.data_hints,
            tool_dialect=self.tool_dialect,
            summarize_params=self.summarize_params,
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
        subset_note = (
            "\nNote: This is the FULL inventory of the domain's data lake — only a small "
            "subset of these files is relevant to the question. Identify and use only the "
            "relevant file(s); do not try to load everything.\n"
            if getattr(self, "list_all_files", False)
            else ""
        )
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}
{subset_note}
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
        elif getattr(self, "list_all_files", False):
            # Exploration-list mode: enumerate EVERY file in the domain lake in
            # the prompt (same placement as oracle gold files). Removes the
            # enumeration burden while keeping the selection problem open.
            rel = os.path.relpath(self.dataset_directory)
            file_paths = sorted(os.path.join(rel, k) for k in self.dataset.keys())
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
                "agent_turns": self.agent_turns,
                "context_window_tokens": self.context_window_tokens,
                "static_compaction": self.static_compaction,
                "frontier_decay_config": self.frontier_decay_config,
                "fold_resolved_revisions_config": self.fold_resolved_revisions_config,
                "probe_retirement_config": self.probe_retirement_config,
                "enable_inspect_tool": self.enable_inspect_tool,
                "enable_render_prefs": self.enable_render_prefs,
                "enable_code_in_snapshot": self.enable_code_in_snapshot,
                "compaction_strategy": self.compaction_strategy,
                "deck_sample_ratio": self.deck_sample_ratio,
                "error_reflection": self.error_reflection,
                "error_reflection_threshold": self.error_reflection_threshold,
                "few_shot_prompt": self.few_shot_prompt,
                "flow_level": self.flow_level,
                "data_level": self.data_level,
                "max_result_rows": self.max_result_rows,
                "attempt_reflection": self.attempt_reflection,
                "column_stats": self.column_stats,
                "value_format": self.value_format,
                "data_hints": self.data_hints,
                "tool_dialect": self.tool_dialect,
                "summarize_params": self.summarize_params,
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
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestSchemaConverge",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestSchemaConvergeLevels(DataflowSystem):
    """gpt-5.4 LATEST converge with flow_level=2 / data_level=2 (richer DECORATE:
    L2 flow = cardinality alarms + errored-op upstream context; L2 data = the
    structural-profile `Data hints:` block). Identical to
    DataflowSystemGPT54LatestSchemaConverge otherwise (latest, max_steps=25)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=2,
            data_level=2,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestSchemaConvergeLevels",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestColumnStats(DataflowSystem):
    """gpt-5.4 LATEST at the notes' best operating point — flow_level=1
    (pre-emptive loader remediation only; the post-hoc flow rungs are net-negative
    per claude/FINDINGS.md) + data_level=2 (the `tableStructureHint` accuracy win)
    — PLUS the standalone `column_stats` overlay: the full per-column
    `Column Stats:` block (null/mean/min/max/distinct/top-N) that the CODE-mode
    system prompt documents but no data level renders. A/Bs "does explicit
    per-column metrics help?" on top of the best config. Otherwise identical to
    DataflowSystemGPT54LatestSchemaConverge (latest, max_steps=25)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestColumnStats",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestColumnStats(DataflowSystem):
    """gpt-5.2 twin of DataflowSystemGPT54LatestColumnStats — same settings
    (LATEST, flow_level=1, data_level=2, column_stats on, max_steps=25,
    loaders=2, attempt_reflection, char limits 1000/3000), only the model
    differs (gpt-5.2). For the cross-model comparison of the column-stats config."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52LatestColumnStats",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestColumnStatsOnly(DataflowSystem):
    """gpt-5.4 LATEST, the LEANEST column-stats-only view: flow_level=0 and
    data_level=0 (no ladder annotations at all), column_stats ON, value_format &
    data_hints OFF. The model sees only: operator summary + Properties + the
    Inputs/Output shape line + the Result TSV + the per-column `Column Stats:`
    block. Isolates the effect of column stats alone. Other params match
    DataflowSystemGPT54LatestColumnStats (max_steps=25, loaders=2, reflection)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            value_format=False,
            data_hints=False,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestColumnStatsOnly",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestColumnStatsOnly(DataflowSystem):
    """gpt-5.2 twin of DataflowSystemGPT54LatestColumnStatsOnly — the leanest
    column-stats-only view (LATEST, flow_level=0, data_level=0, column_stats on,
    value_format/data_hints off, max_steps=25, loaders=2, attempt_reflection,
    char limits 1000/3000). Only the model differs (gpt-5.2)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="latest",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            value_format=False,
            data_hints=False,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52LatestColumnStatsOnly",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestColumnStatsOldStyle(DataflowSystem):
    """Old-branch replication probe: IDENTICAL config to
    DataflowSystemGPT52LatestColumnStatsOnly (gpt-5.2, LATEST, flow0/data0,
    column_stats on, hints off, steps=25, char limits 1000/3000), but run AFTER
    two old-branch behaviors were restored in the agent service + engine:
      1. the rich worked example (full stats-validation reasoning + the
         explicit-filter self-correction beat) in the system prompt, and
      2. old-style error rendering — the engine error carries only the
         exception + user-code line pointer; the errored operator's code is
         rendered by the agent-service in the context instead.
    Compared against ColumnStatsOnly (same 20-task subset) to measure the lift."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="latest",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            value_format=False,
            data_hints=False,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52LatestColumnStatsOldStyle",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestColumnStatsDataHints(DataflowSystem):
    """Recovery probe: identical to DataflowSystemGPT52LatestColumnStatsOnly
    (gpt-5.2, LATEST, flow_level=0, data_level=0, column_stats on) but with the
    standalone `data_hints` flag ON. Tests whether the worker's data-derived
    `Data hints:` block (esp. the "N columns are Unnamed -> re-read with
    header=/skiprows=" signal) recovers the multi-row-header spreadsheet
    failures. ONLY data_hints changes vs the baseline, isolating its effect."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="latest",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            value_format=False,
            data_hints=True,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52LatestColumnStatsDataHints",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaColumnStatsDataHints(DataflowSystem):
    """DELTA counterpart of DataflowSystemGPT52LatestColumnStatsDataHints — same
    knobs (gpt-5.2, flow_level=0, data_level=0, column_stats on, data_hints on,
    value_format off, max_steps=25, attempt_reflection, char limits 1000/3000),
    only context_mode=delta (per-step Thought/Action/Observation trajectory
    instead of the aggregated LATEST snapshot)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            value_format=False,
            data_hints=True,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaColumnStatsDataHints",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaColumnStatsDataHintsNoParallel(DataflowSystem):
    """Parallel-tool-calls ablation: identical to DataflowSystemGPT52DeltaColumnStatsDataHints
    (gpt-5.2, delta, flow=0/data=0, column_stats on, data_hints on, window OFF,
    max_steps=25, attempt_reflection, char limits 1000/3000) but with
    parallel_tool_calls=False — the model emits ONE tool call per turn instead of
    batching independent actions. Measures the cost of forcing sequential actions
    (more turns) against any accuracy effect, with the context window off."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            value_format=False,
            data_hints=True,
            attempt_reflection=True,
            parallel_tool_calls=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaColumnStatsDataHintsNoParallel",
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
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54DeltaSchemaConverge",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Context-window compaction sweep (gpt-5.2, DELTA). Base config mirrors
# DataflowSystemGPT52DeltaColumnStatsDataHints (flow=0/data=0, steps=25,
# attempt_reflection, char limits 1000/3000, column_stats+data_hints on — which
# now feed the compress DECK, since the raw-delta suffix is forced schema-only).
# Each window is run with both compaction strategies so compress (fold prefix →
# stats deck) can be compared head-to-head against sliding (drop oldest events)
# at a MATCHED budget. Windows bound the assembled trajectory user-message
# (system prompt ≈ 3.3k tokens is separate): 3k triggers compaction from ~step 4,
# 6k from ~step 7 (the hard tail).
# ---------------------------------------------------------------------------


class DataflowSystemGPT52DeltaWin3kCompress(DataflowSystem):
    """gpt-5.2 DELTA, 3k-token trajectory window, compress compaction: the oldest
    events fold into a `# Dataflow (compacted)` stats deck (Column Schema and stats
    + a 10% row peek), then raw delta resumes. Aggressive window (compacts from
    ~step 4) — the compress arm of the 3k pair."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=3000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin3kCompress",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaWin3kSliding(DataflowSystem):
    """gpt-5.2 DELTA, 3k-token window, sliding compaction (the naive baseline): the
    oldest whole agent events are dropped until it fits, leaving an omitted-marker.
    The comparator for DataflowSystemGPT52DeltaWin3kCompress at a matched budget."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=3000,
            compaction_strategy="sliding",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin3kSliding",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaWin6kCompress(DataflowSystem):
    """gpt-5.2 DELTA, 6k-token window, compress compaction (gentler window: compacts
    from ~step 7, so only the hard tail folds). The compress arm of the 6k pair."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=6000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin6kCompress",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaWin6kSliding(DataflowSystem):
    """gpt-5.2 DELTA, 6k-token window, sliding compaction — the comparator for
    DataflowSystemGPT52DeltaWin6kCompress at a matched budget."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=6000,
            compaction_strategy="sliding",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin6kSliding",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaWin3kCompressPromptAware(DataflowSystem):
    """gpt-5.2 DELTA, 3k window, compress — identical config to Win3kCompress, but
    run AFTER the DELTA prompt gained the conditional `## History Compaction`
    section (injected when window>0 AND compress). Measures the lift from telling
    the model its history folds into a trustworthy stats deck, vs the pre-prompt
    baseline (Win3kCompress = 62.7%)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=3000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin3kCompressPromptAware",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Lean-deck iteration (gpt-5.2). Same config as the Win{3k,6k}Compress SUTs, but
# run AFTER the deck was made lean (DECK_STATS_MAX_COLS=12, DECK_MAX_ROWS=5 in
# context-utils.ts) so the compress deck stays smaller than the raw trajectory on
# wide tables. Compared against the recorded heavy-deck compress numbers and the
# sliding baselines to test whether a lean deck recovers compress at 6k.
# ---------------------------------------------------------------------------


class DataflowSystemGPT52DeltaWin3kCompressLean(DataflowSystem):
    """gpt-5.2 DELTA, 3k window, compress with the LEAN deck (capped stats cols +
    fewer rows). Re-run of Win3kCompress under the leaner deck rendering."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=3000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin3kCompressLean",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaWin6kCompressLean(DataflowSystem):
    """gpt-5.2 DELTA, 6k window, compress with the LEAN deck. The key iteration:
    at 6k the heavy deck lost to sliding (59.1% vs 60.5%) because wide-table stats
    ballooned the deck; this tests whether the lean deck recovers compress at 6k."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=6000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT52DeltaWin6kCompressLean",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# gpt-5-mini cross-model replication of the compaction sweep (same config as the
# gpt-5.2 sweep, model_type="gpt-5-mini") — checks whether "compress wins at the
# tight window" generalizes across models.
# ---------------------------------------------------------------------------


class DataflowSystemGPT5MiniDeltaWin3kCompress(DataflowSystem):
    """gpt-5-mini DELTA, 3k window, compress compaction (cross-model replication)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=3000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniDeltaWin3kCompress",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniDeltaWin3kSliding(DataflowSystem):
    """gpt-5-mini DELTA, 3k window, sliding compaction (cross-model comparator)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=3000,
            compaction_strategy="sliding",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniDeltaWin3kSliding",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniDeltaWin6kCompress(DataflowSystem):
    """gpt-5-mini DELTA, 6k window, compress compaction (cross-model replication)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=6000,
            compaction_strategy="compress",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniDeltaWin6kCompress",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniDeltaWin6kSliding(DataflowSystem):
    """gpt-5-mini DELTA, 6k window, sliding compaction (cross-model comparator)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=0,
            column_stats=True,
            data_hints=True,
            attempt_reflection=True,
            context_window_tokens=6000,
            compaction_strategy="sliding",
            deck_sample_ratio=0.10,
            max_result_rows=8,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT5MiniDeltaWin6kSliding",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestSchemaConvergeErrorReflect(DataflowSystem):
    """HYBRID: SchemaConverge LATEST core + selective ERROR-REFLECTION reinjection.
    Identical to DataflowSystemGPT54LatestSchemaConverge (latest, flow=1, data=1,
    steps=25) except error_reflection=True: when an operator errors repeatedly, its
    failed-attempt history is folded back into the context. Targets the latest
    'thrash' failure (e.g. legal-hard-1) without paying for the full delta trajectory."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=1,
            attempt_reflection=True,
            error_reflection=True,
            error_reflection_threshold=2,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestSchemaConvergeErrorReflect",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestSchemaConvergeReinject(DataflowSystem):
    """HYBRID: SchemaConverge LATEST core + selective reinjection of BOTH error
    history (error_reflection) AND recent reasoning (thought_replay K=5). Same knobs
    as DataflowSystemGPT54LatestSchemaConverge otherwise. The fuller 'selective
    add-back': latest's compactness + delta's error-recovery/trajectory memory,
    added back only where it pays off."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=1,
            attempt_reflection=True,
            error_reflection=True,
            error_reflection_threshold=2,
            thought_replay=True,
            thought_replay_k=5,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestSchemaConvergeReinject",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54LatestSchemaConvergeAgentTurns(DataflowSystem):
    """Idea-1 arm: SchemaConverge LATEST core + a `# Agent Turns` section (the full
    Thought/Action/Observation trajectory, rendered delta-style). Same knobs as
    DataflowSystemGPT54LatestSchemaConverge (latest, flow=1, data=1, steps=25)
    plus agent_turns=True — a hybrid of latest's compact current-state and delta's
    trajectory, to test whether appending the turn history to latest helps."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=1,
            attempt_reflection=True,
            agent_turns=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54LatestSchemaConvergeAgentTurns",
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


# ─────────────────────────────────────────────────────────────────────────
# Gate-0 headroom pair (learned-context-selection project). Byte-identical
# arms (gpt-5.4, flow_level=0 → NO "Operators needing attention" flow section,
# data_level=2, thought_replay OFF, max_steps=12) EXCEPT context_mode. The ONLY
# variable is history representation: latest-core (# Current Dataflow + per-op
# result summaries) vs full per-event Thought/Action/Observation trajectory.
# max_steps capped at 12 because gpt-5.4 reasoning is slow (~1-2 min/step), so a
# churning task at 25 steps ran ~50 min; 12 bounds per-task wall-clock.
# ─────────────────────────────────────────────────────────────────────────


class DataflowSystemGPT54Gate0Latest(DataflowSystem):
    """Gate-0 LATEST arm: gpt-5.4 latest-core (flow=0, data=2, no replay), steps=12."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=12,
            flow_level=0,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54Gate0Latest",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54Gate0Delta(DataflowSystem):
    """Gate-0 DELTA arm: gpt-5.4 full Thought/Action/Observation trajectory.
    Identical to DataflowSystemGPT54Gate0Latest except context_mode='delta'."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="delta",
            max_steps=12,
            flow_level=0,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54Gate0Delta",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54AllLatest(DataflowSystem):
    """All-domains comparison, LATEST arm: gpt-5.4 latest-core (flow=0, data=2, no
    replay). Same as Gate-0 latest but max_steps=25 and result char limit=3000."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="latest",
            max_steps=25,
            flow_level=0,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=3000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54AllLatest",
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT54AllDelta(DataflowSystem):
    """All-domains comparison, DELTA arm: gpt-5.4 full Thought/Action/Observation
    trajectory. Identical to DataflowSystemGPT54AllLatest except context_mode='delta'."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            context_mode="delta",
            max_steps=25,
            flow_level=0,
            data_level=2,
            thought_replay=False,
            max_operator_result_char_limit=3000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemGPT54AllDelta",
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


# ---------------------------------------------------------------------------
# gpt-5.2 latest-vs-delta × 2k-vs-5k result-char-limit sweep, column stats ON.
# Held constant across all four: model gpt-5.2, flow_level=1 (loader remediation),
# data_level=1 (Schema line), column_stats=True (the per-column stats block),
# max_steps=25, attempt_reflection, cell char limit 3000, NO compact tool. The
# ONLY variables are context_mode (latest|delta) and max_operator_result_char_limit
# (2000|5000) — a clean 2×2 for the cost/accuracy comparison.
# ---------------------------------------------------------------------------
class _GPT52StatsSweep(DataflowSystem):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 2000
    _NAME = "_GPT52StatsSweep"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=1,
            column_stats=True,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestStats2k(_GPT52StatsSweep):
    """gpt-5.2, LATEST, column stats ON, 2k result char limit."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 2000
    _NAME = "DataflowSystemGPT52LatestStats2k"


class DataflowSystemGPT52LatestStats5k(_GPT52StatsSweep):
    """gpt-5.2, LATEST, column stats ON, 5k result char limit."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52LatestStats5k"


class DataflowSystemGPT52DeltaStats2k(_GPT52StatsSweep):
    """gpt-5.2, DELTA, column stats ON, 2k result char limit."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 2000
    _NAME = "DataflowSystemGPT52DeltaStats2k"


class DataflowSystemGPT52DeltaStats5k(_GPT52StatsSweep):
    """gpt-5.2, DELTA, column stats ON, 5k result char limit."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52DeltaStats5k"


# Schema-only twins of the 5k stats arms: column stats OFF, only the Schema line
# (output column names + types, data_level=1) kept. Isolates "just the schema" vs
# "schema + full per-column stats" at the 5k operating point, for latest & delta.
class _GPT52SchemaOnlySweep(DataflowSystem):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "_GPT52SchemaOnlySweep"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=1,          # Schema line only
            column_stats=False,    # stats block OFF
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52Latest3kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, LATEST, 3k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52Latest3kSchemaOnly"


class DataflowSystemGPT52Delta3kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, DELTA, 3k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52Delta3kSchemaOnly"


class DataflowSystemGPT52Delta1kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, DELTA, 1k, schema line ON, column stats OFF — the starvation
    end of the sampling axis (pairs with DataflowSystemGPT52DeltaStats1kD2)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52Delta1kSchemaOnly"


class DataflowSystemGPT52Latest1kCodeInSnap(_GPT52SchemaOnlySweep):
    """gpt-5.2, LATEST, 1k, schema-only + code shown in snapshot (short summaries).
    C3 (versions) ray of the gpt-5.2 mini-mirror star — delta->latest + code ON,
    vs anchor DataflowSystemGPT52Delta1kSchemaOnly."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52Latest1kCodeInSnap"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


# ---------------------------------------------------------------------------
# Probe-prompt fresh controls. Config-identical to the C1/C2/C3 base arms;
# NEW names so runs land in fresh scratch dirs (the base arms' folders keep
# their pre-probe vintage). The knob is the agent-service prompt itself: the
# raw-probe principles + worked-example beats are PERMANENT in the service
# since dataflow-agent acf87127f (+ 5c10913e6, 57bd2fd0a), so any run of
# these classes carries them. Rerun set: the probing-issue tasks from the
# deep dives (format-blinded loads, dirty headers, key traps) + controls.
# ---------------------------------------------------------------------------
class DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt(_GPT52SchemaOnlySweep):
    """Delta5kSchemaOnly config under the raw-probe prompt vintage."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt"


class DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt(_GPT52SchemaOnlySweep):
    """Delta1kSchemaOnly config under the raw-probe prompt vintage — the
    shared anchor of the probe-star (C1' sampling ray → 5k; C2' profiling
    ray → DeltaStats1kD2ProbePrompt)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt"


class DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt(_GPT52SchemaOnlySweep):
    """Latest5kSchemaOnly config under the raw-probe prompt vintage (C3'
    history pair partner of Delta5kSchemaOnlyProbePrompt)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt"


class DataflowSystemGPT52Latest3kSchemaOnlyProbePrompt(_GPT52SchemaOnlySweep):
    """Latest3kSchemaOnly config under the raw-probe prompt vintage."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52Latest3kSchemaOnlyProbePrompt"


class DataflowSystemGPT52Latest5kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, LATEST, 5k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52Latest5kSchemaOnly"


class DataflowSystemGPT52Delta5kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, DELTA, 5k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52Delta5kSchemaOnly"


class DataflowSystemGPT52Latest7kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, LATEST, 7k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 7000
    _NAME = "DataflowSystemGPT52Latest7kSchemaOnly"


class DataflowSystemGPT52Delta7kSchemaOnly(_GPT52SchemaOnlySweep):
    """gpt-5.2, DELTA, 7k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 7000
    _NAME = "DataflowSystemGPT52Delta7kSchemaOnly"


# ---------------------------------------------------------------------------
# data_level=2 + result-char sweep (gpt-5.2). Same recipe as _GPT52StatsSweep
# (column_stats ON, flow_level=1, max_steps=25, attempt_reflection) but with
# data_level=2 — the `Output Table profile:` block (all-null rows/cols by name,
# duplicate-row count, unnamed-header count) ON. Two result-char points: 5k
# (recovery test vs the data_level=1 5k arms) and 10k (full runs). Optional
# static_compaction demonstrates the DELTA-only auto-fold flag in isolation.
# ---------------------------------------------------------------------------
class _GPT52SweepD2(DataflowSystem):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _STATIC_COMPACTION = False
    _NAME = "_GPT52SweepD2"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            static_compaction=self._STATIC_COMPACTION,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestStats3kD2(_GPT52SweepD2):
    """gpt-5.2, LATEST, 3k, column stats ON + data_level=2."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2"


class DataflowSystemGPT52DeltaStats2kD2(_GPT52SweepD2):
    """gpt-5.2, DELTA, 2k, column stats ON + data_level=2 (C5 — sampling 2k + stats)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 2000
    _NAME = "DataflowSystemGPT52DeltaStats2kD2"


class DataflowSystemGPT52LatestStats1kD2(_GPT52SweepD2):
    """gpt-5.2, LATEST, 1k, column stats ON + data_level=2 (C6 — latest + stats,
    latest-twin of C2 DeltaStats1kD2)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52LatestStats1kD2"


class DataflowSystemGPT52LatestStats3kD2SmallTableControl(_GPT52SweepD2):
    """Current-code control: Latest 3k D2 with frontier decay disabled."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2SmallTableControl"


class DataflowSystemGPT52LatestStats3kD2FrontierDecay(_GPT52SweepD2):
    """Latest 3k D2 with only the conservative settled-result overlay enabled."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2FrontierDecay"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            frontier_decay_config={
                "sampleRows": 3,
                "minStepsSinceEdit": 1,
                "minConsumerStepsSinceEdit": 1,
                "minConsumerStepsSinceHealthy": 1,
            },
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaStats3kD2(_GPT52SweepD2):
    """gpt-5.2, DELTA, 3k, column stats ON + data_level=2."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52DeltaStats3kD2"


class DataflowSystemGPT52DeltaStats3kD2ProbePrompt(_GPT52SweepD2):
    """DeltaStats3kD2 (best arm) config under the raw-probe prompt vintage
    (see the ProbePrompt banner in the schema-only sweep section)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52DeltaStats3kD2ProbePrompt"


class DataflowSystemGPT52DeltaStats1kD2ProbePrompt(_GPT52SweepD2):
    """DeltaStats1kD2 config under the raw-probe prompt vintage (C2'
    profiling ray of the probe-star, vs Delta1kSchemaOnlyProbePrompt)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52DeltaStats1kD2ProbePrompt"


class DataflowSystemGPT52DeltaStats1kD2(_GPT52SweepD2):
    """gpt-5.2, DELTA, 1k, column stats ON + data_level=2 — does the profile
    substitute for sample rows when the render budget is starved? (pairs with
    DataflowSystemGPT52Delta1kSchemaOnly)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52DeltaStats1kD2"


class DataflowSystemGPT52DeltaStatsNoRows1kD2ProbePrompt(_GPT52SweepD2):
    """No-sample-rows stats arm (Bob's hypothesis: the per-column stats block —
    not the sample rows — is the real evidence carrier, and rows can even
    mislead when a wide table renders only one unrepresentative row).

    Same config as DeltaStats1kD2ProbePrompt (delta, 1k, data_level=2,
    column_stats) but the LATEST render detail is forced to "shape" via a
    summarize-params patch: Inputs/Output shape line + Schema + Column Stats +
    structural hints, ZERO data rows. (Delta history already renders as shape,
    so the DAG carries no raw rows anywhere.) One-knob (rows on -> off) vs
    DataflowSystemGPT52DeltaStats1kD2ProbePrompt. Probe-prompt vintage."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT52DeltaStatsNoRows1kD2ProbePrompt"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            summarize_params={
                "operators": {"defaults": {"result": {"latest": {"detail": "shape"}}}}
            },
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaStats3kD2FoldControl(_GPT52SweepD2):
    """Current-code DELTA control: Delta 3k D2 with every experimental overlay off.

    Fresh control for the rank-3 fold-resolved-revisions A/B — the historical
    DataflowSystemGPT52DeltaStats3kD2 run predates the permanent renderer rules
    (small-table stats suppression), so it is not code-matched."""

    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52DeltaStats3kD2FoldControl"


class DataflowSystemGPT52DeltaStats3kD2FoldResolved(_GPT52SweepD2):
    """Delta 3k D2 + rank-3 fold-resolved-revisions rule (audit Rank 3).

    Once a later revision of an operator has executed successfully and been
    consumed healthily (+1 grace event), prior revisions' code/result payloads
    render as one-line resolution facts."""

    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52DeltaStats3kD2FoldResolved"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            fold_resolved_revisions_config={"graceEvents": 1},
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52LatestStats3kD2Explore(_GPT52SweepD2):
    """Exploration-mode twin of the Latest 3k control: identical settings; the
    exploration comes from running WITHOUT --use_truth_subset (kb.py
    --no-oracle), so the prompt carries the domain lake's recursive glob
    instead of the task's gold files. Separate name keeps scratch dirs
    disjoint from the oracle runs."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2Explore"


class DataflowSystemGPT52DeltaStats3kD2Explore(_GPT52SweepD2):
    """Exploration-mode twin of the Delta 3k control (see Latest twin)."""

    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52DeltaStats3kD2Explore"


class DataflowSystemGPT52LatestStats3kD2ExploreList(_GPT52SweepD2):
    """Exploration-LIST mode: run with --no-oracle, but the prompt enumerates
    every file in the domain lake (list_all_files) instead of a glob.
    Decomposes the discovery tax: (oracle -> list) = selection cost,
    (list -> glob) = enumeration cost."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2ExploreList"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(verbose=verbose, *args, **kwargs)
        self.list_all_files = True


class DataflowSystemGPT52DeltaStats3kD2ExploreList(_GPT52SweepD2):
    """Delta twin of the exploration-LIST mode."""

    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52DeltaStats3kD2ExploreList"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(verbose=verbose, *args, **kwargs)
        self.list_all_files = True


class DataflowSystemGPT52LatestStats3kD2Lean3(_GPT52SweepD2):
    """E1 lean-push arm: Latest 3k D2 with the row sample capped at 3.

    Isolates the damage of a lean default render WITHOUT the pull tool.
    Matched to DataflowSystemGPT52LatestStats3kD2SmallTableControl except
    max_result_rows."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2Lean3"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(max_result_rows=3, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT52LatestStats3kD2Lean3Pull(_GPT52SweepD2):
    """E1 demand-paging arm: lean push (3-row samples) + the read-only
    inspectResult pull tool. Pulls persist across steps via the tail
    `# Inspections` section (append-mostly — cache-friendly by construction)."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2Lean3Pull"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(max_result_rows=3, enable_inspect_tool=True, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT52LatestStats3kD2ProbeRetire(_GPT52SweepD2):
    """Latest 3k D2 + rank-4 probe-retirement rule (audit Rank 4).

    A settled orphan probe whose discovery is provably encoded downstream
    (probe output value inside a quoted literal of a later healthy connected
    operator's code) renders as a compact extracted fact instead of its table.
    Control arm: DataflowSystemGPT52LatestStats3kD2SmallTableControl (same
    code, overlay off)."""

    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT52LatestStats3kD2ProbeRetire"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            probe_retirement_config={"minStepsSinceEdit": 2, "minValueLength": 4},
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT52DeltaStats5kD2FreshControl(_GPT52SweepD2):
    """Current-code control for the render-prefs A/B: Delta 5k D2, all flags
    off. The historical DataflowSystemGPT52DeltaStats5kD2 predates the
    permanent small-table renderer rule, so it is not code-matched."""

    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52DeltaStats5kD2FreshControl"


class DataflowSystemGPT52DeltaStats5kD2RenderPrefs(_GPT52SweepD2):
    """DELTA 5k D2 + write-time render prefs: the agent declares per-version
    outputSummary (minimal|standard) and showOutputStatistics on
    createOrModifyOperator; a landed version's prefs govern the observations
    rendered while it was active (append-only — old events never change).
    Comparators: DataflowSystemGPT52DeltaStats5kD2 (stats base) and the
    Delta 5k schema-only arm."""

    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52DeltaStats5kD2RenderPrefs"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(enable_render_prefs=True, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT52LatestStats5kD2(_GPT52SweepD2):
    """gpt-5.2, LATEST, 5k, column stats ON + data_level=2 (Output Table profile)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52LatestStats5kD2"


class DataflowSystemGPT52DeltaStats5kD2(_GPT52SweepD2):
    """gpt-5.2, DELTA, 5k, column stats ON + data_level=2 (Output Table profile)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52DeltaStats5kD2"


class DataflowSystemGPT52LatestStats7kD2(_GPT52SweepD2):
    """gpt-5.2, LATEST, 7k, column stats ON + data_level=2."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 7000
    _NAME = "DataflowSystemGPT52LatestStats7kD2"


class DataflowSystemGPT52DeltaStats7kD2(_GPT52SweepD2):
    """gpt-5.2, DELTA, 7k, column stats ON + data_level=2."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 7000
    _NAME = "DataflowSystemGPT52DeltaStats7kD2"


class DataflowSystemGPT52LatestStats10kD2(_GPT52SweepD2):
    """gpt-5.2, LATEST, 10k result chars, column stats ON + data_level=2."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 10000
    _NAME = "DataflowSystemGPT52LatestStats10kD2"


class DataflowSystemGPT52DeltaStats10kD2(_GPT52SweepD2):
    """gpt-5.2, DELTA, 10k result chars, column stats ON + data_level=2."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 10000
    _NAME = "DataflowSystemGPT52DeltaStats10kD2"


# ---------------------------------------------------------------------------
# gpt-5-mini 3k replica of the gpt-5.2 data-context sweep. These hold the
# DataflowAgent knobs constant with the gpt-5.2 arms and only change model_type.
# ---------------------------------------------------------------------------
class _GPT5MiniSchemaOnlySweep(DataflowSystem):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "_GPT5MiniSchemaOnlySweep"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=1,
            column_stats=False,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatest3kSchemaOnly(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, LATEST, 3k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT5MiniLatest3kSchemaOnly"


class DataflowSystemGPT5MiniDelta3kSchemaOnly(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, DELTA, 3k, schema line ON, column stats OFF."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT5MiniDelta3kSchemaOnly"


class _GPT5MiniSweepD2(DataflowSystem):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "_GPT5MiniSweepD2"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestStats3kD2(_GPT5MiniSweepD2):
    """gpt-5-mini, LATEST, 3k, column stats ON + data_level=2."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT5MiniLatestStats3kD2"


class DataflowSystemGPT5MiniDeltaStats3kD2(_GPT5MiniSweepD2):
    """gpt-5-mini, DELTA, 3k, column stats ON + data_level=2."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 3000
    _NAME = "DataflowSystemGPT5MiniDeltaStats3kD2"


class DataflowSystemGPT5MiniDeltaStats5kD2(_GPT5MiniSweepD2):
    """gpt-5-mini, DELTA, 5k, column stats ON + data_level=2 (C4 — sampling 5k + stats)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT5MiniDeltaStats5kD2"


class DataflowSystemGPT5MiniDeltaStats2kD2(_GPT5MiniSweepD2):
    """gpt-5-mini, DELTA, 2k, column stats ON + data_level=2 (C5 — sampling 2k + stats)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 2000
    _NAME = "DataflowSystemGPT5MiniDeltaStats2kD2"


class DataflowSystemGPT5MiniLatestStats1kD2(_GPT5MiniSweepD2):
    """gpt-5-mini, LATEST, 1k, column stats ON + data_level=2 (C6 — latest + stats,
    latest-twin of C2 DeltaStats1kD2)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT5MiniLatestStats1kD2"


# --- gpt-5-mini C1/C2 knob arms (subtask-eval study) ---
# C1 char cap: Delta1k vs Delta5k (schema-only). C2 profile: Delta1k schema vs
# DeltaStats1kD2 (both 1k). Matched one-knob pairs on the mini substrate.
class DataflowSystemGPT5MiniDelta1kSchemaOnly(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, DELTA, 1k, schema-only (C1 anchor / C2 anchor)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnly"


class DataflowSystemGPT5MiniDelta5kSchemaOnly(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, DELTA, 5k, schema-only (C1 ray — the rows knob)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnly"


class DataflowSystemGPT5MiniDelta2kSchemaOnly(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, DELTA, 2k, schema-only (C7 — mid sampling point of the rows
    knob, schema-only twin of C5 DeltaStats2kD2)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 2000
    _NAME = "DataflowSystemGPT5MiniDelta2kSchemaOnly"


class DataflowSystemGPT5MiniDeltaStats1kD2(_GPT5MiniSweepD2):
    """gpt-5-mini, DELTA, 1k, column stats ON + data_level=2 (C2 ray — the stats knob)."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2"


# --- code-in-snapshot experiment (LATEST, 1k, gpt-5-mini) ---
# Does showing the agent its OWN code in the snapshot (with short summaries) help?
# Baseline = plain LATEST-1k schema-only; ray = same + enableCodeInSnapshot.
class DataflowSystemGPT5MiniLatest1kSchemaOnly(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, LATEST, 1k, schema-only (code-in-snapshot BASELINE, flag off)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT5MiniLatest1kSchemaOnly"


class DataflowSystemGPT5MiniLatest1kCodeInSnap(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, LATEST, 1k, schema-only + code shown in snapshot (short summaries)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnap"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


# --- Variance replicates (gpt-5-mini): config-identical to anchor + C1..C6,
# NEW names so each lands in its own scratch dir = independent single-shot run.
# Two per base arm -> with the original, 3 independent samples per knob to
# estimate the run-to-run randomness floor. Placed AFTER all base arms so the
# subclass references resolve. (No recovery rounds when run — raw single-shot.)
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate1(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate1"
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate2(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate2"

class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate1(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate1"
class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate2(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate2"

class DataflowSystemGPT5MiniDeltaStats1kD2Replicate1(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate1"
class DataflowSystemGPT5MiniDeltaStats1kD2Replicate2(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate2"

class DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate1(DataflowSystemGPT5MiniLatest1kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate1"
class DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate2(DataflowSystemGPT5MiniLatest1kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate2"

class DataflowSystemGPT5MiniDeltaStats5kD2Replicate1(DataflowSystemGPT5MiniDeltaStats5kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats5kD2Replicate1"
class DataflowSystemGPT5MiniDeltaStats5kD2Replicate2(DataflowSystemGPT5MiniDeltaStats5kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats5kD2Replicate2"

class DataflowSystemGPT5MiniDeltaStats2kD2Replicate1(DataflowSystemGPT5MiniDeltaStats2kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats2kD2Replicate1"
class DataflowSystemGPT5MiniDeltaStats2kD2Replicate2(DataflowSystemGPT5MiniDeltaStats2kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats2kD2Replicate2"

class DataflowSystemGPT5MiniLatestStats1kD2Replicate1(DataflowSystemGPT5MiniLatestStats1kD2):
    _NAME = "DataflowSystemGPT5MiniLatestStats1kD2Replicate1"
class DataflowSystemGPT5MiniLatestStats1kD2Replicate2(DataflowSystemGPT5MiniLatestStats1kD2):
    _NAME = "DataflowSystemGPT5MiniLatestStats1kD2Replicate2"


# --- Replicate0: clean single-shot re-run of the 7 base arms (anchor+C1-C6),
# because the base arms' round0 traces were overwritten in-place by their 2
# recovery rounds. Config-identical, new names, NO retries when run -> a 3rd
# clean single-shot trace set per knob (variance triple = Rep0/Rep1/Rep2).
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate0(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate0"
class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate0(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate0"
class DataflowSystemGPT5MiniDeltaStats1kD2Replicate0(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate0"
class DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate0(DataflowSystemGPT5MiniLatest1kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate0"
class DataflowSystemGPT5MiniDeltaStats5kD2Replicate0(DataflowSystemGPT5MiniDeltaStats5kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats5kD2Replicate0"
class DataflowSystemGPT5MiniDeltaStats2kD2Replicate0(DataflowSystemGPT5MiniDeltaStats2kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats2kD2Replicate0"
class DataflowSystemGPT5MiniLatestStats1kD2Replicate0(DataflowSystemGPT5MiniLatestStats1kD2):
    _NAME = "DataflowSystemGPT5MiniLatestStats1kD2Replicate0"


# --- Replicate3/Replicate4: extend every knob to 5 clean single-shot reps.
# --- C7 (Delta2kSchemaOnly) Replicate0-4: new knob, 5 reps from scratch.
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate3(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate3"
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate4(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate4"

class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate3(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate3"
class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate4(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate4"

class DataflowSystemGPT5MiniDeltaStats1kD2Replicate3(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate3"
class DataflowSystemGPT5MiniDeltaStats1kD2Replicate4(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate4"

class DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate3(DataflowSystemGPT5MiniLatest1kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate3"
class DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate4(DataflowSystemGPT5MiniLatest1kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate4"

class DataflowSystemGPT5MiniDeltaStats5kD2Replicate3(DataflowSystemGPT5MiniDeltaStats5kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats5kD2Replicate3"
class DataflowSystemGPT5MiniDeltaStats5kD2Replicate4(DataflowSystemGPT5MiniDeltaStats5kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats5kD2Replicate4"

class DataflowSystemGPT5MiniDeltaStats2kD2Replicate3(DataflowSystemGPT5MiniDeltaStats2kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats2kD2Replicate3"
class DataflowSystemGPT5MiniDeltaStats2kD2Replicate4(DataflowSystemGPT5MiniDeltaStats2kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats2kD2Replicate4"

class DataflowSystemGPT5MiniLatestStats1kD2Replicate3(DataflowSystemGPT5MiniLatestStats1kD2):
    _NAME = "DataflowSystemGPT5MiniLatestStats1kD2Replicate3"
class DataflowSystemGPT5MiniLatestStats1kD2Replicate4(DataflowSystemGPT5MiniLatestStats1kD2):
    _NAME = "DataflowSystemGPT5MiniLatestStats1kD2Replicate4"

class DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate0(DataflowSystemGPT5MiniDelta2kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate0"
class DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate1(DataflowSystemGPT5MiniDelta2kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate1"
class DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate2(DataflowSystemGPT5MiniDelta2kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate2"
class DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate3(DataflowSystemGPT5MiniDelta2kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate3"
class DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate4(DataflowSystemGPT5MiniDelta2kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta2kSchemaOnlyReplicate4"


# --- C8: LATEST 5k + code-in-snapshot (wide-sampling twin of C3), 5 reps.
class DataflowSystemGPT5MiniLatest5kCodeInSnap(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, LATEST, 5k, schema-only + code shown in snapshot (C8 —
    5k twin of C3 Latest1kCodeInSnap)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT5MiniLatest5kCodeInSnap"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)

class DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate0(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate0"
class DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate1(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate1"
class DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate2(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate2"
class DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate3(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate3"
class DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate4(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    _NAME = "DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate4"


# Static-compaction demonstrator: byte-identical to DataflowSystemGPT52DeltaStats5k
# (DELTA, 5k, column_stats ON, flow_level=1, data_level=1) EXCEPT static_compaction
# is ON — so the accuracy/cost delta vs that baseline isolates the compaction flag.
class DataflowSystemGPT52DeltaStats5kCompact(_GPT52StatsSweep):
    """gpt-5.2, DELTA, 5k, column stats ON, data_level=1 + static compaction ON."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52DeltaStats5kCompact"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(verbose=verbose, static_compaction=True, *args, **kwargs)


class DataflowSystemGPT52DeltaStats5kCompactEC(_GPT52StatsSweep):
    """gpt-5.2, DELTA, 5k, column stats ON, data_level=1 + static compaction ON,
    EDIT-CONVERGENCE rule (fold at the frontier operator's active-edit-run boundary;
    monotone/cache-friendly). Identical to DeltaStats5k except the flag; vs the
    token-rule DeltaStats5kCompact it isolates the rule change."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT52DeltaStats5kCompactEC"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        # compaction_rule defaults to editConvergence agent-service side.
        super().__init__(verbose=verbose, static_compaction=True, *args, **kwargs)

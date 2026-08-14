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
        role_policy_config: Optional[Dict[str, object]] = None,
        source_provenance_hint: Optional[bool] = None,
        user_task_placement: Optional[str] = None,
        turn_history: Optional[str] = None,
        enable_recall_tool: bool = False,
        enable_resume_tool: bool = False,
        enable_answer_grounding: bool = False,
        # Standalone telemetry renders. Distinct from the legacy `coercion_telemetry`
        # kwarg above, which rides the cumulative DECORATE data ladder (it forces
        # data_level=3 and thereby the whole stats bundle). These flip ONLY the
        # telemetry lines, so an arm can carry them without inheriting stats.
        coercion_facts: bool = False,
        row_lineage: bool = False,
        versioned_mode: bool = False,
        session_turns: bool = False,
        recall_max_result_chars: Optional[int] = None,
        recall_operator_level: bool = False,
        spec_audit: bool = False,
        # None -> service default (currently False; the heads-table A/B showed no effect).
        versioned_heads: bool | None = None,
        index_rich_tables: Optional[int] = None,
        index_detailed_operators: Optional[int] = None,
        index_thin_observations: Optional[bool] = None,
        agent_service_endpoint: Optional[str] = None,
        fold_resolved_revisions_config: Optional[Dict[str, object]] = None,
        probe_retirement_config: Optional[Dict[str, object]] = None,
        enable_inspect_tool: bool = False,
        enable_render_prefs: bool = False,
        enable_code_in_snapshot: bool = True,  # service default; False now genuinely transmitted
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
        self.role_policy_config = role_policy_config
        self.source_provenance_hint = source_provenance_hint
        self.user_task_placement = user_task_placement
        self.turn_history = turn_history
        self.enable_recall_tool = enable_recall_tool
        self.enable_resume_tool = enable_resume_tool
        self.enable_answer_grounding = enable_answer_grounding
        self.coercion_facts = coercion_facts
        self.row_lineage = row_lineage
        self.versioned_mode = versioned_mode
        self.session_turns = session_turns
        self.recall_max_result_chars = recall_max_result_chars
        self.recall_operator_level = recall_operator_level
        self.spec_audit = spec_audit
        self.versioned_heads = versioned_heads
        self.index_rich_tables = index_rich_tables
        self.index_detailed_operators = index_detailed_operators
        self.index_thin_observations = index_thin_observations
        self.agent_service_endpoint = agent_service_endpoint
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
            role_policy_config=self.role_policy_config,
            source_provenance_hint=self.source_provenance_hint,
            user_task_placement=self.user_task_placement,
            turn_history=self.turn_history,
            enable_recall_tool=self.enable_recall_tool,
            enable_resume_tool=self.enable_resume_tool,
            enable_answer_grounding=self.enable_answer_grounding,
            coercion_facts=self.coercion_facts,
            row_lineage=self.row_lineage,
            versioned_mode=self.versioned_mode,
            session_turns=self.session_turns,
            recall_max_result_chars=self.recall_max_result_chars,
            recall_operator_level=self.recall_operator_level,
            spec_audit=self.spec_audit,
            versioned_heads=self.versioned_heads,
            index_rich_tables=self.index_rich_tables,
            index_detailed_operators=self.index_detailed_operators,
            index_thin_observations=self.index_thin_observations,
            **({"agent_service_endpoint": self.agent_service_endpoint} if self.agent_service_endpoint else {}),
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
        # Provenance: which service + which agent-service commit produced this run.
        # Cross-vintage comparison (same config, different service build) silently
        # invalidated an experiment once; stamp it so it is always auditable.
        _endpoint = getattr(self, "agent_service_endpoint", None) or "http://localhost:3001"
        try:
            import subprocess as _sp, os as _os
            # The service's code lives in the worktree that serves that PORT — not
            # in the main checkout. Stamping main's SHA for a worktree-served port
            # would be a confidently wrong provenance record.
            # VERIFY THIS AGAINST REALITY BEFORE TRUSTING A STAMP:
            #   readlink /proc/$(lsof -tiTCP:PORT -sTCP:LISTEN)/cwd
            # A wrong entry is worse than none — it stamps a confident, false
            # provenance. Both known failures happened:
            #   * :3001 was mapped to the frontier-decay worktree while the
            #     service actually ran from the MAIN checkout, so every gpt-5.2
            #     arm recorded `fee02701d`, a commit on a branch that service
            #     was not running.
            #   * :3005 was absent, so it fell through to the main-repo default
            #     and stamped main's SHA (dirty) for a clean worktree.
            _WORKTREE_BY_PORT = {
                "3002": "~/Desktop/bobflow/dataflow-agent-worktrees/feat-role-policy",
                "3005": "~/Desktop/bobflow/dataflow-agent-worktrees/prompt-fix",
            }
            _port = _endpoint.rsplit(":", 1)[-1].strip("/")
            _svc_dir = _os.path.expanduser(_WORKTREE_BY_PORT.get(_port, "~/Desktop/bobflow/dataflow-agent"))
            _sha = _sp.run(["git", "-C", _svc_dir, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
            _dirty = bool(_sp.run(["git", "-C", _svc_dir, "status", "--porcelain",
                                   "agent-service/src"], capture_output=True, text=True,
                                  timeout=5).stdout.strip())
        except Exception:
            _sha, _dirty = "unknown", None
        config = {
            "agent_service_endpoint": _endpoint,
            "agent_service_git_sha": _sha,
            "agent_service_src_dirty": _dirty,
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
                "role_policy_config": self.role_policy_config,
                "source_provenance_hint": self.source_provenance_hint,
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


# ===========================================================================
#  RULE A — role-keyed render policy ("rich source, lean interior").
#  Needs the agent-service overlay `rolePolicyConfig` (applyRolePolicy), which
#  lives only on the feat-role-policy worktree service. These arms therefore
#  target the ISOLATED service on :3002 so the main :3001 service (and anyone
#  else's runs on it) is never involved.
#  Base = C8 Latest5k+code, identical otherwise -> the only delta is per-op
#  render policy: sources get a wide sample + stats, interior/sink go lean.
# ===========================================================================
ROLE_POLICY_ENDPOINT = "http://localhost:3002"


class _A1RolePolicy(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """Rule A: sources rich (12 rows + stats + structural hints), interior lean (3 rows, no stats)."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(
            role_policy_config={
                "sourceSampleRows": 12,
                "sourceStats": True,
                "sourceStructuralHints": True,
                "interiorSampleRows": 3,
                "interiorStats": False,
                "leanTerminal": True,
            },
            verbose=verbose, *args, **kwargs)


class _A0Control(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """Rule A control: same :3002 service, policy OFF — isolates the policy from
    any service/vintage difference between :3001 and :3002."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(verbose=verbose, *args, **kwargs)

class DataflowSystemGPT5MiniA1RolePolicyReplicate1(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate1"
class DataflowSystemGPT5MiniA1RolePolicyReplicate2(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate2"
class DataflowSystemGPT5MiniA1RolePolicyReplicate3(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate3"
class DataflowSystemGPT5MiniA0ControlReplicate1(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate1"
class DataflowSystemGPT5MiniA0ControlReplicate2(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate2"
class DataflowSystemGPT5MiniA0ControlReplicate3(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate3"


# ---------------------------------------------------------------------------
#  A2 / A3 — is the SOURCE stats block worth its bytes?
#  Rule A bundles two levers (row capping + a source stats block). Measured
#  waste inside that block: null=0 on 32.1% of column entries, distinct==rows on
#  5.9%, the whole block redundant on 22.4% of blocks (sample already shows every
#  row), and a `Schema (N cols):` echo on 91.6% of blocks. Split the bundle:
#    A2  stats ON but ANOMALY density  -> keeps the facts, drops the ceremony
#    A3  stats OFF                     -> binary control
#  Everything else identical to A1, same :3002 service, so A1/A0 stay the
#  reference pair (default density is "full" and golden parity is unchanged).
# ---------------------------------------------------------------------------
class _A2AnomalyStats(_A1RolePolicy):
    """A2: source stats rendered anomaly-only."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(verbose=verbose, *args, **kwargs)
        self.role_policy_config = dict(self.role_policy_config or {})
        self.role_policy_config["sourceStatsDensity"] = "anomaly"


class _A3NoSourceStats(_A1RolePolicy):
    """A3: row capping kept, source stats block removed entirely."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(verbose=verbose, *args, **kwargs)
        self.role_policy_config = dict(self.role_policy_config or {})
        self.role_policy_config["sourceStats"] = False


class DataflowSystemGPT5MiniA2AnomalyStatsReplicate1(_A2AnomalyStats):
    _NAME = "DataflowSystemGPT5MiniA2AnomalyStatsReplicate1"
class DataflowSystemGPT5MiniA2AnomalyStatsReplicate2(_A2AnomalyStats):
    _NAME = "DataflowSystemGPT5MiniA2AnomalyStatsReplicate2"
class DataflowSystemGPT5MiniA2AnomalyStatsReplicate3(_A2AnomalyStats):
    _NAME = "DataflowSystemGPT5MiniA2AnomalyStatsReplicate3"
class DataflowSystemGPT5MiniA3NoSourceStatsReplicate1(_A3NoSourceStats):
    _NAME = "DataflowSystemGPT5MiniA3NoSourceStatsReplicate1"
class DataflowSystemGPT5MiniA3NoSourceStatsReplicate2(_A3NoSourceStats):
    _NAME = "DataflowSystemGPT5MiniA3NoSourceStatsReplicate2"
class DataflowSystemGPT5MiniA3NoSourceStatsReplicate3(_A3NoSourceStats):
    _NAME = "DataflowSystemGPT5MiniA3NoSourceStatsReplicate3"
# Sentinel: a 4th A0 rep on the NEW sha. Golden parity proves the render is
# byte-identical, but it does not measure the run-level offset (fresh stack vs
# deep-in-pool) that dominates this benchmark's variance. This rep does.
class DataflowSystemGPT5MiniA0ControlReplicate4(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate4"


# Rep expansion: resolve A1-vs-A0 (+5.0 at 3 reps, inside the +-4-5pt floor).
# 8 reps/arm gives ~+-3pt SEM on a ~8.5pt rep std. Reps 1-3 ran on 4af1e98da,
# these run on 9d60d01dc; golden parity holds for both configs (default "full"
# density renders byte-identical) and A0ControlReplicate4 is the cross-sha
# sentinel that measures the run-level offset directly.
class DataflowSystemGPT5MiniA1RolePolicyReplicate4(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate4"
class DataflowSystemGPT5MiniA1RolePolicyReplicate5(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate5"
class DataflowSystemGPT5MiniA1RolePolicyReplicate6(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate6"
class DataflowSystemGPT5MiniA1RolePolicyReplicate7(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate7"
class DataflowSystemGPT5MiniA1RolePolicyReplicate8(_A1RolePolicy):
    _NAME = "DataflowSystemGPT5MiniA1RolePolicyReplicate8"
class DataflowSystemGPT5MiniA0ControlReplicate5(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate5"
class DataflowSystemGPT5MiniA0ControlReplicate6(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate6"
class DataflowSystemGPT5MiniA0ControlReplicate7(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate7"
class DataflowSystemGPT5MiniA0ControlReplicate8(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate8"


# ---------------------------------------------------------------------------
#  A4 — source provenance principle on top of A_win (= A1 full config).
#  Gold-solution trace dive: per-file identity is a load-time fact erased by
#  concat; suffix-regex derivation = 0%-pass trap (legal-hard-29 n=26,
#  legal-hard-16 n=108). The flag appends ONE prompt principle: multi-file
#  loaders add a __source_file column. Byte-identical prompt when off.
#  Falsifiable: must lift legal-hard-29 + legal-hard-16 specifically.
# ---------------------------------------------------------------------------
class _A4SourceProv(_A1RolePolicy):
    """A4: A1 render policy + sourceProvenanceHint prompt principle."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("source_provenance_hint", True)
        super().__init__(verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniA4SourceProvReplicate1(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate1"
class DataflowSystemGPT5MiniA4SourceProvReplicate2(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate2"
class DataflowSystemGPT5MiniA4SourceProvReplicate3(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate3"


class _A5B2Combo(_A1RolePolicy):
    """A5: A1 render policy + B2 data history (1 shape-rendered prior result/op).
    B2 alone: -10.4% cost, rep std 1.9, -3.2 acc (inside noise). Disjoint fields
    from A1 (A shapes the LATEST render per role; B2 appends history), so the
    combo tests whether the savings stack without an accuracy tax."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("summarize_params", {"operators": {"defaults": {"result": {"history": {
            "lastK": 1, "render": {"detail": "shape"}}}}}})
        super().__init__(verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniA5B2ComboReplicate1(_A5B2Combo):
    _NAME = "DataflowSystemGPT5MiniA5B2ComboReplicate1"
class DataflowSystemGPT5MiniA5B2ComboReplicate2(_A5B2Combo):
    _NAME = "DataflowSystemGPT5MiniA5B2ComboReplicate2"
class DataflowSystemGPT5MiniA5B2ComboReplicate3(_A5B2Combo):
    _NAME = "DataflowSystemGPT5MiniA5B2ComboReplicate3"


# A4 validation reps: 8-rep footing for the one arm that cleared 2x SEM at 3
# reps (71.7 +-2.4 vs A0 59.1 +-12.3). Also watches the archeology-hard-7 flag.
class DataflowSystemGPT5MiniA4SourceProvReplicate4(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate4"
class DataflowSystemGPT5MiniA4SourceProvReplicate5(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate5"
class DataflowSystemGPT5MiniA4SourceProvReplicate6(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate6"
class DataflowSystemGPT5MiniA4SourceProvReplicate7(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate7"
class DataflowSystemGPT5MiniA4SourceProvReplicate8(_A4SourceProv):
    _NAME = "DataflowSystemGPT5MiniA4SourceProvReplicate8"


# ---------------------------------------------------------------------------
#  A6 — ISOLATE the structuralHints leg. A0 control + `Output Table profile:`
#  facts on sources, nothing else changed (no row cap, no stats, no interior
#  trim). Those facts render in 62-64% of every A arm and 0% of A0, so they are
#  a confound in the whole A-series; this is the missing single-leg control.
# ---------------------------------------------------------------------------
class _A6HintsOnly(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """A6: A0 base + sourceStructuralHints only."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(role_policy_config={"hintsOnly": True}, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniA6HintsOnlyReplicate1(_A6HintsOnly):
    _NAME = "DataflowSystemGPT5MiniA6HintsOnlyReplicate1"
class DataflowSystemGPT5MiniA6HintsOnlyReplicate2(_A6HintsOnly):
    _NAME = "DataflowSystemGPT5MiniA6HintsOnlyReplicate2"
class DataflowSystemGPT5MiniA6HintsOnlyReplicate3(_A6HintsOnly):
    _NAME = "DataflowSystemGPT5MiniA6HintsOnlyReplicate3"
class DataflowSystemGPT5MiniA6HintsOnlyReplicate4(_A6HintsOnly):
    _NAME = "DataflowSystemGPT5MiniA6HintsOnlyReplicate4"


# Fresh A0 control on the CURRENT sha. The stats-bound commit (81dc518be)
# changed the DEFAULT render — proof-based suppression now fires without any
# flag — so A0 reps 1-8 (4af1e98da / 9d60d01dc) are a different vintage and
# cannot serve as A6's control. Same config as _A0Control, new names so the runs
# land in fresh scratch dirs.
class DataflowSystemGPT5MiniA0ControlReplicate9(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate9"
class DataflowSystemGPT5MiniA0ControlReplicate10(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate10"
class DataflowSystemGPT5MiniA0ControlReplicate11(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate11"
class DataflowSystemGPT5MiniA0ControlReplicate12(_A0Control):
    _NAME = "DataflowSystemGPT5MiniA0ControlReplicate12"


# ===========================================================================
#  C9 / C10 / C11 — per-operator CHAR budget at the raw-data boundary, full 104.
#  Uses the new render-time `tuple.maxChars` (the agent-level
#  max_operator_result_char_limit is one engine-side number for the whole
#  execution and cannot vary per operator). Global budget stays 5k so the engine
#  returns 5k for every operator; the per-op budget can only reduce from there.
#    C9  LATEST+code : sources 5k + stats, every downstream op 1k + no stats
#    C10 DELTA       : same split (char-budget leg binds on DELTA; caps each
#                      event's render of that operator)
#    C11 LATEST+code : 5k + stats for ALL ops (the no-policy reference)
#  All three on the CURRENT sha — the existing LatestStats5kD2Code reps predate
#  the stats-bound + provenance commits and are a different vintage.
# ===========================================================================
_CHAR_SPLIT = {
    "sourceMaxChars": 5000,
    "sourceStats": True,
    "sourceStructuralHints": True,
    "nonSourceMaxChars": 1000,
    "nonSourceStats": False,
}


class _C9SourceRichLatest(_GPT5MiniSweepD2):
    """C9: LATEST + code, sources 5k + stats, downstream 1k no stats."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "_C9SourceRichLatest"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(
            enable_code_in_snapshot=True,
            role_policy_config=dict(_CHAR_SPLIT),
            verbose=verbose, *args, **kwargs)


class _C10SourceRichDelta(_GPT5MiniSweepD2):
    """C10: DELTA, sources 5k + stats, downstream 1k no stats."""
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "_C10SourceRichDelta"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(
            role_policy_config=dict(_CHAR_SPLIT),
            verbose=verbose, *args, **kwargs)


class _C11UniformRichLatest(_GPT5MiniSweepD2):
    """C11: LATEST + code, 5k + stats for every operator (no per-op policy)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "_C11UniformRichLatest"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniC9SourceRichLatestReplicate1(_C9SourceRichLatest):
    _NAME = "DataflowSystemGPT5MiniC9SourceRichLatestReplicate1"
class DataflowSystemGPT5MiniC9SourceRichLatestReplicate2(_C9SourceRichLatest):
    _NAME = "DataflowSystemGPT5MiniC9SourceRichLatestReplicate2"
class DataflowSystemGPT5MiniC9SourceRichLatestReplicate3(_C9SourceRichLatest):
    _NAME = "DataflowSystemGPT5MiniC9SourceRichLatestReplicate3"
class DataflowSystemGPT5MiniC10SourceRichDeltaReplicate1(_C10SourceRichDelta):
    _NAME = "DataflowSystemGPT5MiniC10SourceRichDeltaReplicate1"
class DataflowSystemGPT5MiniC10SourceRichDeltaReplicate2(_C10SourceRichDelta):
    _NAME = "DataflowSystemGPT5MiniC10SourceRichDeltaReplicate2"
class DataflowSystemGPT5MiniC10SourceRichDeltaReplicate3(_C10SourceRichDelta):
    _NAME = "DataflowSystemGPT5MiniC10SourceRichDeltaReplicate3"
class DataflowSystemGPT5MiniC11UniformRichLatestReplicate1(_C11UniformRichLatest):
    _NAME = "DataflowSystemGPT5MiniC11UniformRichLatestReplicate1"
class DataflowSystemGPT5MiniC11UniformRichLatestReplicate2(_C11UniformRichLatest):
    _NAME = "DataflowSystemGPT5MiniC11UniformRichLatestReplicate2"
class DataflowSystemGPT5MiniC11UniformRichLatestReplicate3(_C11UniformRichLatest):
    _NAME = "DataflowSystemGPT5MiniC11UniformRichLatestReplicate3"


# ===========================================================================
#  C12 — the missing cell: LATEST 1k + code + stats/D2.
#  C6 (LatestStats1kD2) is LATEST 1k + stats WITHOUT code and is the worst arm
#  on the board (63.1). C3 is LATEST 1k + code WITHOUT stats (68.7). This is the
#  1k twin of C8s/C11.
# ===========================================================================
class _C12LatestStats1kCode(_GPT5MiniSweepD2):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "_C12LatestStats1kCode"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


# ===========================================================================
#  A7 — engine-side multi-file load provenance (`files read: N [names]`).
#  Config-identical to C11 (LATEST 5k + code + stats/D2, the uniform-rich arm)
#  so the ONLY difference is the new __file_io__ fact the worker publishes. The
#  fact renders on the structural-hints rung, which stats/D2 already enables —
#  hence a stats-carrying base is required.
#  Unlike A4 (a prompt principle the agent obeys 40% of the time) this is an
#  observation, so coverage is 100% of multi-file loads.
# ===========================================================================
class _A7FileIOFact(_C11UniformRichLatest):
    """A7: C11 + worker-published `files read:` fact."""
    pass


class DataflowSystemGPT5MiniC12LatestStats1kCodeReplicate1(_C12LatestStats1kCode):
    _NAME = "DataflowSystemGPT5MiniC12LatestStats1kCodeReplicate1"
class DataflowSystemGPT5MiniC12LatestStats1kCodeReplicate2(_C12LatestStats1kCode):
    _NAME = "DataflowSystemGPT5MiniC12LatestStats1kCodeReplicate2"
class DataflowSystemGPT5MiniC12LatestStats1kCodeReplicate3(_C12LatestStats1kCode):
    _NAME = "DataflowSystemGPT5MiniC12LatestStats1kCodeReplicate3"
class DataflowSystemGPT5MiniA7FileIOFactReplicate1(_A7FileIOFact):
    _NAME = "DataflowSystemGPT5MiniA7FileIOFactReplicate1"
class DataflowSystemGPT5MiniA7FileIOFactReplicate2(_A7FileIOFact):
    _NAME = "DataflowSystemGPT5MiniA7FileIOFactReplicate2"
class DataflowSystemGPT5MiniA7FileIOFactReplicate3(_A7FileIOFact):
    _NAME = "DataflowSystemGPT5MiniA7FileIOFactReplicate3"


# ===========================================================================
#  PAIRED A7 MATRIX (new engine era, post-restart 2026-07-29 13:2x).
#  The A7 `Files read: N [names]` fact now has its OWN render gate
#  (`column.fileIoFacts`, commit 5132cbe1a) instead of riding the stats/D2
#  structuralHints rung — so it composes with schema-only arms, including the
#  best-measured config (D8 = LATEST 5k + code, 69.0 +-3.0), which the old
#  coupling locked out entirely.
#
#  Four arms, each 3 reps, paired so every A7 variant has a same-era control:
#    D8      LATEST 5k + code                  (control, = C8 config)
#    D8F     LATEST 5k + code + files-read      (A7 on the best arm)
#    D12     LATEST 1k + code + stats/D2        (control, = C12 config)
#    D12F    LATEST 1k + code + stats + files-read
#  New names (D*) because the engine was restarted: every earlier arm is a
#  different engine era and cannot serve as a control here.
# ===========================================================================
_FILEIO_PATCH = {"operators": {"defaults": {"result": {"latest": {"column": {"fileIoFacts": True}}}}}}


class _D8Latest5kCode(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """D8: same config as C8 (best arm), fresh engine era."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(verbose=verbose, *args, **kwargs)


class _D8FileIO(_D8Latest5kCode):
    """D8F: D8 + the engine-side `Files read:` fact."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("summarize_params", dict(_FILEIO_PATCH))
        super().__init__(verbose=verbose, *args, **kwargs)


class _D12LatestStats1kCode(_GPT5MiniSweepD2):
    """D12: LATEST 1k + code + stats/D2 (the missing cell), fresh era."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "_D12LatestStats1kCode"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class _D12FileIO(_D12LatestStats1kCode):
    """D12F: D12 + the engine-side `Files read:` fact."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("summarize_params", dict(_FILEIO_PATCH))
        super().__init__(verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniD8Latest5kCodeReplicate1(_D8Latest5kCode):
    _NAME = "DataflowSystemGPT5MiniD8Latest5kCodeReplicate1"
class DataflowSystemGPT5MiniD8Latest5kCodeReplicate2(_D8Latest5kCode):
    _NAME = "DataflowSystemGPT5MiniD8Latest5kCodeReplicate2"
class DataflowSystemGPT5MiniD8Latest5kCodeReplicate3(_D8Latest5kCode):
    _NAME = "DataflowSystemGPT5MiniD8Latest5kCodeReplicate3"
class DataflowSystemGPT5MiniD8FileIOReplicate1(_D8FileIO):
    _NAME = "DataflowSystemGPT5MiniD8FileIOReplicate1"
class DataflowSystemGPT5MiniD8FileIOReplicate2(_D8FileIO):
    _NAME = "DataflowSystemGPT5MiniD8FileIOReplicate2"
class DataflowSystemGPT5MiniD8FileIOReplicate3(_D8FileIO):
    _NAME = "DataflowSystemGPT5MiniD8FileIOReplicate3"
class DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate1(_D12LatestStats1kCode):
    _NAME = "DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate1"
class DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate2(_D12LatestStats1kCode):
    _NAME = "DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate2"
class DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate3(_D12LatestStats1kCode):
    _NAME = "DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate3"
class DataflowSystemGPT5MiniD12FileIOReplicate1(_D12FileIO):
    _NAME = "DataflowSystemGPT5MiniD12FileIOReplicate1"
class DataflowSystemGPT5MiniD12FileIOReplicate2(_D12FileIO):
    _NAME = "DataflowSystemGPT5MiniD12FileIOReplicate2"
class DataflowSystemGPT5MiniD12FileIOReplicate3(_D12FileIO):
    _NAME = "DataflowSystemGPT5MiniD12FileIOReplicate3"


# ===========================================================================
#  N-SERIES (era 2, sha 589b08967+). `stats on` now IMPLIES the `Files read:`
#  fact (coupled in resolveOperatorParams), so every N arm carries it.
#    N1  LATEST 5k + code + stats            (A7's config, re-run clean in era 2)
#    N2  DELTA  5k + stats                   (C4's config + the fact — never run)
#    N3  LATEST src 5k / downstream 2k, stats BOTH SIDES (never run: C9/C10 used
#        1k downstream WITH STATS OFF)
#  3 reps each. Era 1 arms are NOT comparable (identical config scored 69.0 vs
#  71.3 across the engine restart).
# ===========================================================================
class _N1Latest5kStats(_GPT5MiniSweepD2):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "_N1Latest5kStats"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class _N2Delta5kStats(_GPT5MiniSweepD2):
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 5000
    _NAME = "_N2Delta5kStats"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(verbose=verbose, *args, **kwargs)


class _N3SrcRich5k2k(_GPT5MiniSweepD2):
    """N3: sources 5k, every downstream op 2k, stats ON BOTH SIDES."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "_N3SrcRich5k2k"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(
            enable_code_in_snapshot=True,
            role_policy_config={
                "sourceMaxChars": 5000,
                "sourceStats": True,
                "sourceStructuralHints": True,
                "nonSourceMaxChars": 2000,
                "nonSourceStats": True,
            },
            verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniN1Latest5kStatsReplicate1(_N1Latest5kStats):
    _NAME = "DataflowSystemGPT5MiniN1Latest5kStatsReplicate1"
class DataflowSystemGPT5MiniN1Latest5kStatsReplicate2(_N1Latest5kStats):
    _NAME = "DataflowSystemGPT5MiniN1Latest5kStatsReplicate2"
class DataflowSystemGPT5MiniN1Latest5kStatsReplicate3(_N1Latest5kStats):
    _NAME = "DataflowSystemGPT5MiniN1Latest5kStatsReplicate3"
class DataflowSystemGPT5MiniN2Delta5kStatsReplicate1(_N2Delta5kStats):
    _NAME = "DataflowSystemGPT5MiniN2Delta5kStatsReplicate1"
class DataflowSystemGPT5MiniN2Delta5kStatsReplicate2(_N2Delta5kStats):
    _NAME = "DataflowSystemGPT5MiniN2Delta5kStatsReplicate2"
class DataflowSystemGPT5MiniN2Delta5kStatsReplicate3(_N2Delta5kStats):
    _NAME = "DataflowSystemGPT5MiniN2Delta5kStatsReplicate3"
class DataflowSystemGPT5MiniN3SrcRich5k2kReplicate1(_N3SrcRich5k2k):
    _NAME = "DataflowSystemGPT5MiniN3SrcRich5k2kReplicate1"
class DataflowSystemGPT5MiniN3SrcRich5k2kReplicate2(_N3SrcRich5k2k):
    _NAME = "DataflowSystemGPT5MiniN3SrcRich5k2kReplicate2"
class DataflowSystemGPT5MiniN3SrcRich5k2kReplicate3(_N3SrcRich5k2k):
    _NAME = "DataflowSystemGPT5MiniN3SrcRich5k2kReplicate3"


# ===========================================================================
#  N4 / N5 — probe the 2k sampling tier with code + stats (+fact by coupling).
#  Context: sampling saturates early on DELTA (1k 63.3 -> 2k 66.2 -> 5k 66.6),
#  but the 2k tier was never tried on LATEST-with-code, and the source-rich split
#  was only tested at 5k/1k (C9/C10, stats off downstream) and 5k/2k (N3).
#    N4  LATEST 2k + code + stats            — the 2k twin of N1 (5k) / D12 (1k)
#    N5  LATEST src 2k / downstream 1k + code + stats, stats BOTH sides
#  Both era 2, 3 reps.
# ===========================================================================
class _N4Latest2kStats(_GPT5MiniSweepD2):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 2000
    _NAME = "_N4Latest2kStats"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class _N5SrcRich2k1k(_GPT5MiniSweepD2):
    """N5: sources 2k, every downstream op 1k, stats ON BOTH SIDES."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 2000
    _NAME = "_N5SrcRich2k1k"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(
            enable_code_in_snapshot=True,
            role_policy_config={
                "sourceMaxChars": 2000,
                "sourceStats": True,
                "sourceStructuralHints": True,
                "nonSourceMaxChars": 1000,
                "nonSourceStats": True,
            },
            verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniN4Latest2kStatsReplicate1(_N4Latest2kStats):
    _NAME = "DataflowSystemGPT5MiniN4Latest2kStatsReplicate1"
class DataflowSystemGPT5MiniN4Latest2kStatsReplicate2(_N4Latest2kStats):
    _NAME = "DataflowSystemGPT5MiniN4Latest2kStatsReplicate2"
class DataflowSystemGPT5MiniN4Latest2kStatsReplicate3(_N4Latest2kStats):
    _NAME = "DataflowSystemGPT5MiniN4Latest2kStatsReplicate3"
class DataflowSystemGPT5MiniN5SrcRich2k1kReplicate1(_N5SrcRich2k1k):
    _NAME = "DataflowSystemGPT5MiniN5SrcRich2k1kReplicate1"
class DataflowSystemGPT5MiniN5SrcRich2k1kReplicate2(_N5SrcRich2k1k):
    _NAME = "DataflowSystemGPT5MiniN5SrcRich2k1kReplicate2"
class DataflowSystemGPT5MiniN5SrcRich2k1kReplicate3(_N5SrcRich2k1k):
    _NAME = "DataflowSystemGPT5MiniN5SrcRich2k1kReplicate3"


# ===========================================================================
#  N6 — the 3K tier: LATEST 3k + code + stats (+ files-read fact, now default-on).
#  Fills the gap between N4 (2k, 67.8) and N1 (5k, 70.1); the measured tiers so
#  far are 1k 63.8 / 1k+fact 68.8 / 2k 67.8 / 5k 70.1-71.3.
#
#  D8F reps 4-5 — more data on the best arm (LATEST 5k + code + fact, 71.2 at
#  $0.0154). NOTE: these run on a LATER RENDER VINTAGE than reps 1-3 — the
#  `Files read:` line moved from inside `Result:` to above `Code:` (commit
#  23a5325fc / f31a017cc). Report reps 4-5 separately before pooling with 1-3.
# ===========================================================================
class _N6Latest3kStats(_GPT5MiniSweepD2):
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 3000
    _NAME = "_N6Latest3kStats"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", ROLE_POLICY_ENDPOINT)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniN6Latest3kStatsReplicate1(_N6Latest3kStats):
    _NAME = "DataflowSystemGPT5MiniN6Latest3kStatsReplicate1"
class DataflowSystemGPT5MiniN6Latest3kStatsReplicate2(_N6Latest3kStats):
    _NAME = "DataflowSystemGPT5MiniN6Latest3kStatsReplicate2"
class DataflowSystemGPT5MiniN6Latest3kStatsReplicate3(_N6Latest3kStats):
    _NAME = "DataflowSystemGPT5MiniN6Latest3kStatsReplicate3"
class DataflowSystemGPT5MiniD8FileIOReplicate4(_D8FileIO):
    _NAME = "DataflowSystemGPT5MiniD8FileIOReplicate4"
class DataflowSystemGPT5MiniD8FileIOReplicate5(_D8FileIO):
    _NAME = "DataflowSystemGPT5MiniD8FileIOReplicate5"


# ===========================================================================
#  LAYOUT A/B — 2026-07-30, fresh engine, both arms interleaved in ONE pool.
#
#  Why: D8F reps 4-5 (post-layout) scored 65.6 vs reps 1-3 (pre-layout) 71.2,
#  = -5.6 pt at 2.90x SE. But reps 1-3 ran on a 1-3 h old engine and reps 4-5
#  on a 10 h old engine that died minutes later, and engine age alone is worth
#  ~2.3 pt (C8 era1 69.0 vs D8 era2 71.3). That comparison cannot separate
#  layout from senescence, so it is re-run here properly:
#
#    LOld  `Files read:` INSIDE `Result:`         agent-service 311ddd646, :3003
#    LNew  `Files read:` above `Code:` w/`Inputs:` agent-service c516d800f, :3002
#
#  Both endpoints serve the SAME engine at the SAME time and the pool
#  interleaves the arms, so engine age is held constant by construction
#  instead of being corrected for afterwards.
#
#  Config is D8F's exactly: LATEST 5k + code + files-read, no stats — the arm
#  the discrepancy showed up on. NOTE the contrast is the render VINTAGE, not
#  a single line: 311ddd646..c516d800f also carries the DELTA legacyFormatOptions
#  fix (irrelevant to a LATEST arm) and the prompt's layout example. So this
#  measures the layout PACKAGE, which is the shipped unit.
# ===========================================================================
LAYOUT_OLD_ENDPOINT = "http://localhost:3003"


class _LayoutOld(_D8FileIO):
    """`Files read:` inside `Result:` — the render D8F reps 1-3 ran on."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        # assignment, not setdefault: must beat _D8Latest5kCode's :3002 default
        kwargs["agent_service_endpoint"] = LAYOUT_OLD_ENDPOINT
        super().__init__(verbose=verbose, *args, **kwargs)


class _LayoutNew(_D8FileIO):
    """`Files read:` above `Code:`, grouped with `Inputs:` — current main."""


class DataflowSystemGPT5MiniLayoutOldReplicate1(_LayoutOld):
    _NAME = "DataflowSystemGPT5MiniLayoutOldReplicate1"
class DataflowSystemGPT5MiniLayoutOldReplicate2(_LayoutOld):
    _NAME = "DataflowSystemGPT5MiniLayoutOldReplicate2"
class DataflowSystemGPT5MiniLayoutOldReplicate3(_LayoutOld):
    _NAME = "DataflowSystemGPT5MiniLayoutOldReplicate3"
class DataflowSystemGPT5MiniLayoutNewReplicate1(_LayoutNew):
    _NAME = "DataflowSystemGPT5MiniLayoutNewReplicate1"
class DataflowSystemGPT5MiniLayoutNewReplicate2(_LayoutNew):
    _NAME = "DataflowSystemGPT5MiniLayoutNewReplicate2"
class DataflowSystemGPT5MiniLayoutNewReplicate3(_LayoutNew):
    _NAME = "DataflowSystemGPT5MiniLayoutNewReplicate3"


# ===========================================================================
#  RULE B — versions/history on a LATEST core (config-only; no service change).
#  Base = C8 Latest5k+code (best arm). Each ray adds ONE history channel to
#  isolate which kind of memory repays the re-derivation tax:
#    B1 codeHistory=1    -> prior CODE version per operator
#    B2 resultHistory=1  -> prior RESULT version per operator (shape-rendered)
#    B3 reasoningReplayK -> last-3 thoughts (`# Reasoning`), no per-op history
#  3 reps each; run on the 20-task discriminating hard subset.
# ===========================================================================
class _B1CodeHist(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """Rule B1: latest5k+code + 1 prior code version per operator."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            summarize_params={"operators": {"defaults": {"property": {"codeHistory": 1}}}},
            verbose=verbose, *args, **kwargs)

class _B2ResultHist(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """Rule B2: latest5k+code + 1 prior result version per operator (shape only)."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            summarize_params={"operators": {"defaults": {"result": {"history": {
                "lastK": 1, "render": {"detail": "shape"}}}}}},
            verbose=verbose, *args, **kwargs)

class _B3Replay(DataflowSystemGPT5MiniLatest5kCodeInSnap):
    """Rule B3: latest5k+code + last-3 thought replay (targets reasoning tax)."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            summarize_params={"trajectory": {"reasoningReplayK": 3}},
            verbose=verbose, *args, **kwargs)

class DataflowSystemGPT5MiniB1CodeHistReplicate1(_B1CodeHist):
    _NAME = "DataflowSystemGPT5MiniB1CodeHistReplicate1"
class DataflowSystemGPT5MiniB1CodeHistReplicate2(_B1CodeHist):
    _NAME = "DataflowSystemGPT5MiniB1CodeHistReplicate2"
class DataflowSystemGPT5MiniB1CodeHistReplicate3(_B1CodeHist):
    _NAME = "DataflowSystemGPT5MiniB1CodeHistReplicate3"
class DataflowSystemGPT5MiniB2ResultHistReplicate1(_B2ResultHist):
    _NAME = "DataflowSystemGPT5MiniB2ResultHistReplicate1"
class DataflowSystemGPT5MiniB2ResultHistReplicate2(_B2ResultHist):
    _NAME = "DataflowSystemGPT5MiniB2ResultHistReplicate2"
class DataflowSystemGPT5MiniB2ResultHistReplicate3(_B2ResultHist):
    _NAME = "DataflowSystemGPT5MiniB2ResultHistReplicate3"
class DataflowSystemGPT5MiniB3ReplayReplicate1(_B3Replay):
    _NAME = "DataflowSystemGPT5MiniB3ReplayReplicate1"
class DataflowSystemGPT5MiniB3ReplayReplicate2(_B3Replay):
    _NAME = "DataflowSystemGPT5MiniB3ReplayReplicate2"
class DataflowSystemGPT5MiniB3ReplayReplicate3(_B3Replay):
    _NAME = "DataflowSystemGPT5MiniB3ReplayReplicate3"


# --- Rep5-7: post-prompt-change replicates (2026-07-28) --------------------
# The agent-service prompt/tool change of 2026-07-28 made "code is visible" the
# default, which flipped the DELTA arms from the verbose summary instruction to
# the terse one (measured: operator summaries 136-141 -> ~63 chars). Rep0-4 of
# these three arms were produced under the OLD wording and Rep5-7 under the NEW
# one, so the two blocks are a paired before/after on the same configs. Keep
# both; never pool them.
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate5(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate5"
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate6(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate6"
class DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate7(DataflowSystemGPT5MiniDelta1kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate7"

class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate5(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate5"
class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate6(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate6"
class DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate7(DataflowSystemGPT5MiniDelta5kSchemaOnly):
    _NAME = "DataflowSystemGPT5MiniDelta5kSchemaOnlyReplicate7"

class DataflowSystemGPT5MiniDeltaStats1kD2Replicate5(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate5"
class DataflowSystemGPT5MiniDeltaStats1kD2Replicate6(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate6"
class DataflowSystemGPT5MiniDeltaStats1kD2Replicate7(DataflowSystemGPT5MiniDeltaStats1kD2):
    _NAME = "DataflowSystemGPT5MiniDeltaStats1kD2Replicate7"

# --- C9: latest + 5k + code-in-snapshot + column stats ---------------------
# The missing cell of the knob star: C8 (latest 5k + code) with stats turned on,
# i.e. the latest-mode twin of C4 that also shows the agent its own code.
class DataflowSystemGPT5MiniLatestStats5kD2Code(_GPT5MiniSweepD2):
    """gpt-5-mini, LATEST, 5k, column stats ON + data_level=2 + code in snapshot
    (C9 — C8 plus stats / C4's latest+code twin)."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "DataflowSystemGPT5MiniLatestStats5kD2Code"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate0(DataflowSystemGPT5MiniLatestStats5kD2Code):
    _NAME = "DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate0"
class DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate1(DataflowSystemGPT5MiniLatestStats5kD2Code):
    _NAME = "DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate1"
class DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate2(DataflowSystemGPT5MiniLatestStats5kD2Code):
    _NAME = "DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate2"
class DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate3(DataflowSystemGPT5MiniLatestStats5kD2Code):
    _NAME = "DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate3"
class DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate4(DataflowSystemGPT5MiniLatestStats5kD2Code):
    _NAME = "DataflowSystemGPT5MiniLatestStats5kD2CodeReplicate4"


# ===========================================================================
#  P-SERIES — the CODE budget. 2026-07-30.
#
#  Every context knob to date aimed at the DATA render (rows, stats, schema, chars).
#  Byte accounting over ~100 traces/arm (judgment_runs/mini_star/byte_accounting.py)
#  says that was the wrong target:
#
#      component share of the rendered dataflow
#      arm         code    rows   stats  schema
#      D8  5k     39.4%   46.2%      -    7.1%
#      D12 1k     33.9%   20.8%  27.7%    9.1%   <- code is the LARGEST component
#
#  `max_operator_result_char_limit` clamps table ROWS only, so code has never been
#  under any budget. This also explains mechanically why source-rich/downstream-lean
#  lost (C9 67.5 / C10 67.1 / N3 68.4 / N5 67.9 vs uniform-5k 70.1-71.3): on a C9
#  DataProcessing block the split is code 1,302 B (67%) vs rows 356 B (18%), so the
#  lever was aimed at 18% of the block.
#
#  A CAP, not a drop. Two measurements decided that:
#   * code size is long-tailed — p50 286 B, p90 2,014 B, p99 6,081 B, max 16,400 B —
#     so an 800 B cap removes ~49% of code bytes while leaving 76.5% of blocks
#     untouched, and 400 B removes ~64%;
#   * dropping code by operator ROLE barely binds. Tried first and smoke-tested: on
#     these 2-5 operator pipelines `frontier` + `near-frontier` already covers nearly
#     every operator (near-frontier IS the frontier's upstream), so a role-keyed drop
#     left 4 of 5 code blocks intact. The `codeBudget` leg in role-policy.ts remains
#     for longer pipelines but is not what these arms test.
#
#  Base config is D8F's (the best arm): LATEST 5k + code + files-read, no stats.
#  P0 is a same-pool control, so nothing rests on cross-pool engine age.
#
#    P0  control, code uncapped (= D8F)
#    P1  codeMaxChars 800   (~49% of code bytes, ~17% of the dataflow render)
#    P2  codeMaxChars 400   (~64% of code bytes) — is there a cliff between them?
#
#  Served from the code-lean worktree on :3004.
# ===========================================================================
CODE_LEAN_ENDPOINT = "http://localhost:3004"


def _code_cap_patch(n):
    """_FILEIO_PATCH plus a code budget. Merged by hand rather than dict-updated:
    both live under operators.defaults and a shallow update would drop one."""
    return {
        "operators": {
            "defaults": {
                "result": {"latest": {"column": {"fileIoFacts": True}}},
                "property": {"codeMaxChars": n},
            }
        }
    }


class _PBase(_D8FileIO):
    """D8F config, pointed at the :3004 build carrying the codeMaxChars param."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["agent_service_endpoint"] = CODE_LEAN_ENDPOINT
        super().__init__(verbose=verbose, *args, **kwargs)


class _P0CodeControl(_PBase):
    """Control: no code cap, so this is byte-parity D8F on the :3004 build."""


class _P1Code800(_PBase):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["summarize_params"] = _code_cap_patch(800)
        super().__init__(verbose=verbose, *args, **kwargs)


class _P2Code400(_PBase):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["summarize_params"] = _code_cap_patch(400)
        super().__init__(verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniP0CodeControlReplicate1(_P0CodeControl):
    _NAME = "DataflowSystemGPT5MiniP0CodeControlReplicate1"
class DataflowSystemGPT5MiniP0CodeControlReplicate2(_P0CodeControl):
    _NAME = "DataflowSystemGPT5MiniP0CodeControlReplicate2"
class DataflowSystemGPT5MiniP0CodeControlReplicate3(_P0CodeControl):
    _NAME = "DataflowSystemGPT5MiniP0CodeControlReplicate3"
class DataflowSystemGPT5MiniP1Code800Replicate1(_P1Code800):
    _NAME = "DataflowSystemGPT5MiniP1Code800Replicate1"
class DataflowSystemGPT5MiniP1Code800Replicate2(_P1Code800):
    _NAME = "DataflowSystemGPT5MiniP1Code800Replicate2"
class DataflowSystemGPT5MiniP1Code800Replicate3(_P1Code800):
    _NAME = "DataflowSystemGPT5MiniP1Code800Replicate3"
class DataflowSystemGPT5MiniP2Code400Replicate1(_P2Code400):
    _NAME = "DataflowSystemGPT5MiniP2Code400Replicate1"
class DataflowSystemGPT5MiniP2Code400Replicate2(_P2Code400):
    _NAME = "DataflowSystemGPT5MiniP2Code400Replicate2"
class DataflowSystemGPT5MiniP2Code400Replicate3(_P2Code400):
    _NAME = "DataflowSystemGPT5MiniP2Code400Replicate3"


# ===========================================================================
#  Q-SERIES — the missing cell: 5k + STATS + LATEST + fact on the CURRENT render.
#
#  A7 (70.5) and N1 (70.1) already tested 5k+stats+latest+fact, but both finished
#  before the layout commit 23a5325fc (21:03 Jul 29) and before 6f544c4c1 (20:52)
#  made `fileIoFacts` an independent default-on flag — they picked the fact up via
#  the original stats-coupling. So no arm has ever run stats-on against the current
#  render, where `Files read:` sits above `Code:` with `Inputs:`.
#
#  Q0 is a co-run control (= P0: 5k, no stats, fact) rather than a reuse of P0's
#  reps, so the pair shares engine age exactly.
#
#  Prior: stats have never helped LATEST — D8 71.3 (no stats) vs N1 70.1 (stats),
#  C8 69.0 vs C8s 68.6 — and cost more. Stats only paid at a starved 1k budget
#  (D12 63.8 -> D12F 68.8), and even there the fact did the work. Expectation is
#  therefore parity-or-worse; this closes the cell rather than chasing a win.
# ===========================================================================
_Q_STATS_PATCH = {
    "operators": {"defaults": {"result": {"latest": {"column": {"fileIoFacts": True}}}}}
}


class _Q0Control(_PBase):
    """= P0 exactly: 5k, code, fact, NO stats. Co-run control on :3004."""


class _Q1Stats(_GPT5MiniSweepD2):
    """5k + code + fact + per-column STATS, on the CURRENT render (:3004).

    Derived from _GPT5MiniSweepD2 (which is where data_level=2 / column_stats=True
    are set) rather than from _PBase — the D8F chain hardcodes stats OFF, so passing
    column_stats as a kwarg collided with it. Same parent N1 uses; the differences
    from N1 are the endpoint (current render vintage) and an EXPLICIT fileIoFacts
    patch instead of relying on the old stats-coupling.
    """
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 5000
    _NAME = "_Q1Stats"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["agent_service_endpoint"] = CODE_LEAN_ENDPOINT
        kwargs["summarize_params"] = dict(_Q_STATS_PATCH)
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)


class DataflowSystemGPT5MiniQ0ControlReplicate1(_Q0Control):
    _NAME = "DataflowSystemGPT5MiniQ0ControlReplicate1"
class DataflowSystemGPT5MiniQ0ControlReplicate2(_Q0Control):
    _NAME = "DataflowSystemGPT5MiniQ0ControlReplicate2"
class DataflowSystemGPT5MiniQ0ControlReplicate3(_Q0Control):
    _NAME = "DataflowSystemGPT5MiniQ0ControlReplicate3"
class DataflowSystemGPT5MiniQ1StatsReplicate1(_Q1Stats):
    _NAME = "DataflowSystemGPT5MiniQ1StatsReplicate1"
class DataflowSystemGPT5MiniQ1StatsReplicate2(_Q1Stats):
    _NAME = "DataflowSystemGPT5MiniQ1StatsReplicate2"
class DataflowSystemGPT5MiniQ1StatsReplicate3(_Q1Stats):
    _NAME = "DataflowSystemGPT5MiniQ1StatsReplicate3"


# ===========================================================================
#  R-SERIES — does the `Files read:` fact RESCUE the lean-downstream split?
#
#  The gap: src 5k / down 1k + LATEST + fact was never run. C9 is that exact split
#  (sourceMaxChars 5000 / nonSourceMaxChars 1000) but ran 2026-07-29 01:59, before
#  the A7 fact existed — 0 of its 103 traces contain `Files read:`. N3 (5k/2k) and
#  N5 (2k/1k) both carry the fact; the 5k/1k cell does not.
#
#  Why it is worth a pool rather than an assumption: the fact's one large win was on
#  a STARVED budget — D12 63.8 -> D12F 68.8, +5.0 pt at 3.30x SE, when downstream had
#  1k. C9 is exactly a starved-downstream arm, so this is the untested mechanism that
#  could still justify the split family (C9 67.5 / C10 67.1 / N3 68.4 / N5 67.9, all
#  below uniform-5k's 71.2).
#
#  Against it: byte accounting says the split lever aims at `rows` (18% of a
#  downstream block) while `code` sits at 67% untouched, and the cost rule says only
#  step reduction pays. The fact does reduce steps, so the prior is genuinely open.
#
#    R0  src 5k / down 1k, stats, LATEST, fact OFF   (= C9 on the current render)
#    R1  src 5k / down 1k, stats, LATEST, fact ON
#
#  fileIoFacts defaults ON since 6f544c4c1, so R0 must switch it off EXPLICITLY.
# ===========================================================================
_SPLIT_5K_1K = {
    # NO stats on either side: this isolates the char-budget split itself, on the
    # same no-stats base as D8F (the best arm). structuralHints off too — A6 showed
    # the hints leg is its own confound and it was never a stats-free test.
    "sourceMaxChars": 5000, "sourceStats": False, "sourceStructuralHints": False,
    "nonSourceMaxChars": 1000, "nonSourceStats": False,
}


# ===========================================================================
#  S-SERIES — isolate HINTS, and re-test the split with neither stats nor hints.
#
#  Why: every split arm ever run (C9 67.5 / C10 67.1 / N3 68.4 / N5 67.9) carried
#  THREE things at once — the char split, per-column stats, AND structuralHints —
#  because all three ride the _GPT5MiniSweepD2 base. The top arms (D8 71.3, D8F 71.2,
#  P0 70.8) carry NONE of them. Verified by grepping traces for `Output Table
#  profile`: 0 occurrences in D8/D8F/P0, 52-56 in A7/N1/C9/Q1. So "splits lose" was
#  never a clean measurement; stats and hints were confounded into it.
#
#  Hints are engine-observed load-quality facts, distinct from stats:
#      Output Table profile:
#        - empty rows: 2 of 8365 rows are entirely null
#        - duplicate rows: 1 of 8365 (0%)
#        - headers: 29 of 30 columns are unnamed (...)
#  The engine publishes them unconditionally (`_publish_column_stats` is ungated), so
#  rendering hints WITHOUT stats is possible — which no arm has ever done except A6's
#  hintsOnly, and that was on a stats-on base.
#
#    S0  uniform 5k, no stats, no hints, +fact      (co-run control = D8F/P0)
#    S1  uniform 5k, no stats, +HINTS,   +fact      (hints isolated at last)
#    S2  src 5k / down 2k, no stats, no hints, +fact  (split, finally clean)
#    S3  src 5k / down 2k, no stats, +HINTS,   +fact
#
#  S0 is co-run because engine drift between pools is large: P0 scored 70.8 and Q0
#  66.4 on IDENTICAL config two hours apart. Cross-pool controls are not usable.
# ===========================================================================
def _sp(fact=True, hints=False):
    col = {"fileIoFacts": fact}
    if hints:
        col["structuralHints"] = True
    return {"operators": {"defaults": {"result": {"latest": {"column": col}}}}}


def _split_5k_2k(hints):
    return {
        "sourceMaxChars": 5000, "sourceStats": False, "sourceStructuralHints": hints,
        "nonSourceMaxChars": 2000, "nonSourceStats": False,
    }


class _S0Control(_PBase):
    """Uniform 5k, no stats, no hints, +fact. Byte-parity with D8F/P0."""


class _S1Hints(_PBase):
    """Uniform 5k, no stats, +hints, +fact — the first stats-free hints arm."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["summarize_params"] = _sp(fact=True, hints=True)
        super().__init__(verbose=verbose, *args, **kwargs)


class _S2Split(_PBase):
    """src 5k / down 2k, no stats, no hints, +fact."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["role_policy_config"] = _split_5k_2k(False)
        kwargs["summarize_params"] = _sp(fact=True, hints=False)
        super().__init__(verbose=verbose, *args, **kwargs)


class _S3SplitHints(_PBase):
    """src 5k / down 2k, no stats, +hints, +fact."""
    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs["role_policy_config"] = _split_5k_2k(True)
        kwargs["summarize_params"] = _sp(fact=True, hints=True)
        super().__init__(verbose=verbose, *args, **kwargs)


for _i in (1, 2, 3):
    for _cls, _tag in ((_S0Control, "S0Control"), (_S1Hints, "S1Hints"),
                       (_S2Split, "S2Split"), (_S3SplitHints, "S3SplitHints")):
        _n = f"DataflowSystemGPT5Mini{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  LUNA SERIES — gpt-5.6-luna baseline sweep (first non-gpt-5-mini model in the
#  current render era). Mirrors the gpt-5-mini anchor/C1-C4 factorial so the two
#  models are directly comparable knob-for-knob.
#
#    LunaAnchor  1K, DELTA,  no stats
#    LunaC1      5K, DELTA,  no stats            (+sampling)
#    LunaC2      1K, DELTA,  stats + hints       (+stats)
#    LunaC3      1K, LATEST, no stats, +code     (+latest)
#    LunaC4      5K, LATEST, stats + hints, +code (all three)
#
#  All carry the `Files read:` fact, which is default-on since 6f544c4c1. NOTE: for
#  a plain DELTA arm `server.ts` never enters the summarize-params branch (it is
#  gated on frontierDecay / probeRetirement / rolePolicy / renderPrefs /
#  enableCodeInSnapshot&&LATEST), so setting `summarize_params` here would be a
#  SILENT NO-OP. DELTA picks the fact up via the legacy default instead
#  (legacyFormatOptions, fixed in f7cabbe43) — verified by grepping traces, not
#  assumed.
#
#  `stats` and `hints` are deliberately bundled: data_level=2 + column_stats=True is
#  what produces both the per-column block and the `Output Table profile` hints, and
#  that bundling is what the gpt-5-mini C2/C8s arms used. On gpt-5-mini the two were
#  separated later (S-series) and hints turned out to be the cheap half.
#
#  LATEST arms carry code-in-snapshot, matching gpt-5-mini's C3/C8s and the current
#  default (64f5ea4dc).
# ===========================================================================
class _LunaBase(DataflowSystem):
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _STATS = False
    _CODE = False
    _NAME = "_LunaBase"
    _MODEL = "gpt-5.6-luna"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", CODE_LEAN_ENDPOINT)
        super().__init__(
            model_type=self._MODEL,
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=2 if self._STATS else 1,
            column_stats=self._STATS,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            enable_code_in_snapshot=self._CODE,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class _LunaAnchor(_LunaBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 1000; _STATS = False; _CODE = False
    _NAME = "_LunaAnchor"


class _LunaC1(_LunaBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 5000; _STATS = False; _CODE = False
    _NAME = "_LunaC1"


class _LunaC2(_LunaBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 1000; _STATS = True; _CODE = False
    _NAME = "_LunaC2"


class _LunaC3(_LunaBase):
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 1000; _STATS = False; _CODE = True
    _NAME = "_LunaC3"


class _LunaC4(_LunaBase):
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 5000; _STATS = True; _CODE = True
    _NAME = "_LunaC4"


for _i in (1, 2, 3):
    for _cls, _tag in ((_LunaAnchor, "LunaAnchor"), (_LunaC1, "LunaC1"), (_LunaC2, "LunaC2"),
                       (_LunaC3, "LunaC3"), (_LunaC4, "LunaC4")):
        _n = f"DataflowSystem{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  TERRA SERIES — gpt-5.6-terra, identical five-arm factorial to the luna sweep so
#  the two 5.6 models compare knob-for-knob and against gpt-5-mini.
#  Same Responses-API routing and reasoning_effort=medium (terra rejects function
#  tools on /v1/chat/completions with any effort other than "none", exactly as luna
#  does — verified directly against the API).
# ===========================================================================
class _TerraBase(_LunaBase):
    _MODEL = "gpt-5.6-terra"
    _NAME = "_TerraBase"


class _TerraAnchor(_TerraBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 1000; _STATS = False; _CODE = False
    _NAME = "_TerraAnchor"


class _TerraC1(_TerraBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 5000; _STATS = False; _CODE = False
    _NAME = "_TerraC1"


class _TerraC2(_TerraBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 1000; _STATS = True; _CODE = False
    _NAME = "_TerraC2"


class _TerraC3(_TerraBase):
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 1000; _STATS = False; _CODE = True
    _NAME = "_TerraC3"


class _TerraC4(_TerraBase):
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 5000; _STATS = True; _CODE = True
    _NAME = "_TerraC4"


for _i in (1, 2, 3):
    for _cls, _tag in ((_TerraAnchor, "TerraAnchor"), (_TerraC1, "TerraC1"), (_TerraC2, "TerraC2"),
                       (_TerraC3, "TerraC3"), (_TerraC4, "TerraC4")):
        _n = f"DataflowSystem{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  T-SERIES — terra round 2, built from the round-1 trace analysis.
#
#  Terra's factorial left exactly two positive signals: sampling (C1 +1.7 at
#  1.19x SE) and the C4 bundle (+2.8 at 1.83x SE). But C4 bundles three knobs and
#  LATEST alone HURTS terra (C3 -1.4), and both of C4's clean task-level wins over
#  C3 (environment-hard-18, biomedical-hard-5) look sampling-driven — C1 also wins
#  environment-hard-18. So the bundle's active ingredients are plausibly
#  sampling+stats, with LATEST dead weight.
#
#    T0  = C4 exactly (5K LATEST stats+hints +code)   — co-run control
#    T1  = 5K DELTA stats+hints                        — the missing C1xC2 cell;
#          if T1 >= T0 then DELTA is terra's true mode and 76.0 isn't the ceiling
#    T2  = C4 with 10K sampling                        — terra is the first model
#          where sampling clearly pays; every earlier model saturated by 5K.
#          10K tests whether 76.0 is sampling-limited.
# ===========================================================================
class _T0Control(_TerraC4):
    _NAME = "_T0Control"


class _T1Delta5kStats(_TerraBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 5000; _STATS = True; _CODE = False
    _NAME = "_T1Delta5kStats"


class _T2Latest10k(_TerraBase):
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 10000; _STATS = True; _CODE = True
    _NAME = "_T2Latest10k"


for _i in (1, 2, 3):
    for _cls, _tag in ((_T0Control, "T0Control"), (_T1Delta5kStats, "T1Delta5kStats"),
                       (_T2Latest10k, "T2Latest10k")):
        _n = f"DataflowSystem{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  E-SERIES — reasoning EFFORT, the axis no arm has tested. Every 5.6 arm so far
#  ran at effort=medium. Two facts motivate this: (1) render knobs are exhausted —
#  luna's factorial spans 2.6 pt, terra's 4.1, T-round-2 killed the 10K-sampling
#  and DELTA-mode hypotheses (both <1x SE vs the co-run control); (2) failing runs
#  burn ~2x the reasoning tokens of passing runs on BOTH models (terra 551 vs
#  1,222; luna 691 vs 1,391), so the model already modulates effort with
#  difficulty — the question is whether a higher ceiling converts grind into
#  correct answers, especially on hard tasks.
#
#    E0  terra, 5K DELTA stats+hints, effort=medium   (co-run control = T1 config)
#    E1  same, effort=high (via the gpt-5.6-terra-high litellm alias)
#
#  T1's config is the base because it was round 2's numerically best arm (72.8).
# ===========================================================================
class _E0Medium(_T1Delta5kStats):
    _NAME = "_E0Medium"


class _E1High(_T1Delta5kStats):
    _MODEL = "gpt-5.6-terra-high"
    _NAME = "_E1High"


for _i in (1, 2, 3):
    for _cls, _tag in ((_E0Medium, "E0Medium"), (_E1High, "E1High")):
        _n = f"DataflowSystem{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  LS/TS SERIES — the mini S-series 2x2 (hints x split) replicated on BOTH 5.6
#  models, 5 reps, ONE combined pool so luna and terra share engine age.
#
#  Why: the per-operator family was never tested on 5.6. On mini, the clean S2
#  split was the best arm in its pool (+1.4 at 1.87x SE, -12.4% cost) and S1
#  hints-only was the largest cost lever found (-24% at parity, via one fewer
#  step). Terra's render is rows-dominated (52% vs mini's 21-46%), so the split
#  attacks a bigger share of the budget than it ever did on mini. Against: both
#  5.6 models were knob-insensitive on global knobs.
#
#  Base = 5K LATEST + code, NO stats, fact on (matches mini S-series exactly):
#    *S0  uniform, no hints          (control)
#    *S1  uniform, +HINTS            (structuralHints without stats)
#    *S2  src 5K / down 2K, no hints (the clean split)
#    *S3  split + hints              (interaction)
# ===========================================================================
_S56_SPLIT = lambda hints: {
    "sourceMaxChars": 5000, "sourceStats": False, "sourceStructuralHints": hints,
    "nonSourceMaxChars": 2000, "nonSourceStats": False,
}
_S56_SP = lambda hints: {"operators": {"defaults": {"result": {"latest": {"column": (
    {"fileIoFacts": True, "structuralHints": True} if hints else {"fileIoFacts": True}
)}}}}}


def _mk56(base, tag, hints, split):
    class _A(base):
        _CONTEXT_MODE = "latest"; _RESULT_CHARS = 5000; _STATS = False; _CODE = True
        _NAME = f"_{tag}"

        def __init__(self, verbose: bool = False, *args, **kwargs):
            if split:
                kwargs["role_policy_config"] = _S56_SPLIT(hints)
            kwargs["summarize_params"] = _S56_SP(hints)
            super().__init__(verbose=verbose, *args, **kwargs)
    _A.__name__ = f"_{tag}"
    return _A


for _model_base, _pref in ((_LunaBase, "LS"), (_TerraBase, "TS")):
    for _suffix, _hints, _split in (("0Control", False, False), ("1Hints", True, False),
                                    ("2Split", False, True), ("3SplitHints", True, True)):
        _tag = f"{_pref}{_suffix}"
        _cls = _mk56(_model_base, _tag, _hints, _split)
        globals()[f"_{_tag}"] = _cls
        for _i in (1, 2, 3, 4, 5):
            _n = f"DataflowSystem{_tag}Replicate{_i}"
            globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# LS/TS reps 6-7 (user extended the pool to 7 reps per arm).
for _pref in ("LS", "TS"):
    for _suffix in ("0Control", "1Hints", "2Split", "3SplitHints"):
        _tag = f"{_pref}{_suffix}"
        _cls = globals()[f"_{_tag}"]
        for _i in (6, 7):
            _n = f"DataflowSystem{_tag}Replicate{_i}"
            globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# Round-1 factorial reps 4-5 (both models). Run co-run in ONE pool; do NOT blindly
# pool with reps 1-3 — different engine era (measured drift -4.5 pt on identical
# config). Pool across eras only if per-arm levels match; else report separately.
for _pref, _arms in (("Luna", ("_LunaAnchor", "_LunaC1", "_LunaC2", "_LunaC3", "_LunaC4")),
                     ("Terra", ("_TerraAnchor", "_TerraC1", "_TerraC2", "_TerraC3", "_TerraC4"))):
    for _cn in _arms:
        _cls = globals()[_cn]
        _tag = _cn[1:]
        for _i in (4, 5):
            _n = f"DataflowSystem{_tag}Replicate{_i}"
            globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  LUNA C5 — 5K DELTA + stats/hints: the missing C1xC2 cell on luna (terra's was
#  T1). Notes: (1) DELTA carries operator code inline per event by design, so
#  "+code" is inherent and enable_code_in_snapshot stays False (the server gate
#  is LATEST-only); (2) the `Files read:` fact is expected NOT to render — known
#  5.6+DELTA defect, 0/104 across every luna/terra DELTA arm. Stats+hints do.
#  Co-run control: fresh LunaC1 reps 6-8 (5K DELTA no-stats), so C5-C1 = the
#  stats effect measured within-pool rather than against last era's C1.
# ===========================================================================
class _LunaC5(_LunaBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 5000; _STATS = True; _CODE = False
    _NAME = "_LunaC5"


for _i in (1, 2, 3, 4, 5):
    _n = f"DataflowSystemLunaC5Replicate{_i}"
    globals()[_n] = type(_n, (_LunaC5,), {"_NAME": _n})
for _i in (6, 7, 8):
    _n = f"DataflowSystemLunaC1Replicate{_i}"
    globals()[_n] = type(_n, (_LunaC1,), {"_NAME": _n})


# TERRA C5 — same cell as luna C5 (5K DELTA + stats/hints; config identical to
# _T1Delta5kStats, renamed for the paired 5v3 test). Luna's within-pool result was
# +2.5 at 3.23x SE (hard +2.5 at 3.46x) — the first 5.6 render knob past 2x SE, an
# interaction (stats needs the 5K sample; each half alone was noise). Terra's T1 at
# 3v3 showed +1.3 at 0.93x; this rerun at 5v3 with a co-run C1 control tests whether
# the interaction generalizes across the 5.6 pair.
class _TerraC5(_TerraBase):
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 5000; _STATS = True; _CODE = False
    _NAME = "_TerraC5"


for _i in (1, 2, 3, 4, 5):
    _n = f"DataflowSystemTerraC5Replicate{_i}"
    globals()[_n] = type(_n, (_TerraC5,), {"_NAME": _n})
for _i in (6, 7, 8):
    _n = f"DataflowSystemTerraC1Replicate{_i}"
    globals()[_n] = type(_n, (_TerraC1,), {"_NAME": _n})


# Terra C5 pool: control extended to 5 reps (5v5).
for _i in (9, 10):
    _n = f"DataflowSystemTerraC1Replicate{_i}"
    globals()[_n] = type(_n, (_TerraC1,), {"_NAME": _n})


# ─────────────────────────────────────────────────────────────────────────
# Stable-branch verification arm: claude-haiku-4.5 (cheapest gateway model)
# on the converge stack. Used by the dataflow-agent repo's benchmark/bench.sh
# smoke runs; parameters mirror DataflowSystemGPT54LatestSchemaConverge with
# only the model swapped.
# ─────────────────────────────────────────────────────────────────────────
class DataflowSystemStableHaiku(DataflowSystem):
    """claude-haiku-4.5 converge stack (LATEST, flow_level=1, data_level=1)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=1,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name="DataflowSystemStableHaiku",
            verbose=verbose,
            *args,
            **kwargs,
        )


class _GPT52FactorialBase(DataflowSystem):
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 1000
    _STATS = False
    _CODE = False
    _NAME = "_GPT52FactorialBase"
    _MODEL = "gpt-5.2"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        # `enable_code_in_snapshot` is passed ONLY by the arms that want it on,
        # which is what the mini classes do (C3/C4 pass True; the DELTA arms
        # never mention it and inherit the True default). It is a no-op on
        # DELTA either way — `isCodeShownToAgent()` is
        # `contextMode === DELTA || enableCodeInSnapshot`, so DELTA short-
        # circuits it in the tool schema, the prompt fragment and the renderer
        # alike — but passing `False` there would leave the recorded config
        # differing from mini's for no behavioral reason, and someone would
        # eventually have to re-derive that it did not matter.
        # NOTE: a future LATEST-without-code arm must pass False EXPLICITLY;
        # the default is True and this base will not infer it from `_CODE`.
        if self._CODE:
            kwargs.setdefault("enable_code_in_snapshot", True)
        super().__init__(
            model_type=self._MODEL,
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=1,
            data_level=2 if self._STATS else 1,
            column_stats=self._STATS,
            attempt_reflection=True,
            max_operator_result_char_limit=self._RESULT_CHARS,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class _GPT52Anchor(_GPT52FactorialBase):
    """gpt-5.2, DELTA, 1k, schema-only. Mimics DataflowSystemGPT5MiniDelta1kSchemaOnly."""
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 1000; _STATS = False; _CODE = False
    _NAME = "_GPT52Anchor"


class _GPT52C1(_GPT52FactorialBase):
    """+sampling. gpt-5.2, DELTA, 5k, schema-only. Mimics DataflowSystemGPT5MiniDelta5kSchemaOnly."""
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 5000; _STATS = False; _CODE = False
    _NAME = "_GPT52C1"


class _GPT52C2(_GPT52FactorialBase):
    """+stats. gpt-5.2, DELTA, 1k, stats + hints. Mimics DataflowSystemGPT5MiniDeltaStats1kD2."""
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 1000; _STATS = True; _CODE = False
    _NAME = "_GPT52C2"


class _GPT52C3(_GPT52FactorialBase):
    """+latest. gpt-5.2, LATEST, 1k, schema-only, code in snapshot.
    Mimics DataflowSystemGPT5MiniLatest1kCodeInSnap.

    `_CODE = True` is not an extra knob: DELTA carries operator code inline per
    event, so a code-blind LATEST cell would confound the mode axis with a code
    axis. Same reasoning the luna/terra C3 cells use."""
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 1000; _STATS = False; _CODE = True
    _NAME = "_GPT52C3"


class _GPT52C4(_GPT52FactorialBase):
    """All three. gpt-5.2, LATEST, 5k, stats + hints, code in snapshot.
    Mimics DataflowSystemGPT5MiniLatestStats5kD2Code."""
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 5000; _STATS = True; _CODE = True
    _NAME = "_GPT52C4"


# Five replicates, numbered from 0. Each name gets its own scratch dir, so a
# replicate is an independent single-shot run, not a re-score of the same one;
# 5 is what the post-repair luna/terra table used and what the +-across-reps
# std in that table is computed over.
for _i in (0, 1, 2, 3, 4):
    for _cls, _tag in ((_GPT52Anchor, "Anchor"), (_GPT52C1, "C1"), (_GPT52C2, "C2"),
                       (_GPT52C3, "C3"), (_GPT52C4, "C4")):
        _n = f"DataflowSystemGPT52{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  GPT-5.2 — the 2k sampling cells, ADDED alongside C1/C4 rather than replacing
#  them. Anchor/C2/C3 already pin 1k and C1/C4 pin 5k, so adding 2k turns the
#  sampling knob into three points (1k / 2k / 5k) at both ends of the mode axis
#  while leaving the anchor/C1-C4 factorial byte-identical to the gpt-5-mini
#  mirror it was built to be.
#
#    C1Sample2k   2K, DELTA,  no stats                 (C1 at 2k)
#    C4Sample2k   2K, LATEST, stats + hints, +code     (C4 at 2k)
#
#  Mini twins, for the cross-model read:
#    C1Sample2k <- DataflowSystemGPT5MiniDelta2kSchemaOnly  (mini's C7, reps 0-4)
#    C4Sample2k <- NONE. Mini's LATEST+stats+code cell exists only at 5k
#      (LatestStats5kD2Code); its LATEST+stats arms without code are 1k and 3k.
#      So C4Sample2k is a NEW cell, not a mirror — read it against gpt-5.2's own
#      C4 (same config at 5k) and C3 (LATEST at 1k), not against mini.
#
#  Why 2k specifically: it is the observation channel LongDS standardized on
#  (`_LongDS2kD2` in longds/arms.py), so a 2k point here is directly readable
#  against the LongDS layout results instead of needing a knob translation.
# ===========================================================================
class _GPT52C1Sample2k(_GPT52FactorialBase):
    """C1 at 2k. gpt-5.2, DELTA, 2k, schema-only. Mimics DataflowSystemGPT5MiniDelta2kSchemaOnly."""
    _CONTEXT_MODE = "delta"; _RESULT_CHARS = 2000; _STATS = False; _CODE = False
    _NAME = "_GPT52C1Sample2k"


class _GPT52C4Sample2k(_GPT52FactorialBase):
    """C4 at 2k. gpt-5.2, LATEST, 2k, stats + hints, code in snapshot.
    No gpt-5-mini twin — see the block comment above."""
    _CONTEXT_MODE = "latest"; _RESULT_CHARS = 2000; _STATS = True; _CODE = True
    _NAME = "_GPT52C4Sample2k"


for _i in (0, 1, 2, 3, 4):
    for _cls, _tag in ((_GPT52C1Sample2k, "C1Sample2k"), (_GPT52C4Sample2k, "C4Sample2k")):
        _n = f"DataflowSystemGPT52{_tag}Replicate{_i}"
        globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ===========================================================================
#  FILE-IO E2E PAIR — verifies the `Files read:` DELTA render fix end to end.
#
#  Both are the gpt-5.2 Anchor config (1K DELTA, schema-only). The ONLY
#  difference is which agent-service they talk to:
#    FileIOOld -> :3004  worktree code-lean  @ afc64b980           (unfixed)
#    FileIONew -> :3005  worktree prompt-fix @ afc64b980 + fix     (fixed)
#  prompt-fix branched from code-lean's exact HEAD, so this is a one-variable
#  comparison of the render change rather than a cross-vintage one.
#
#  Run against a MULTI-FILE task (e.g. environment-easy-4, which loads five
#  water-body-testing CSVs). On a single-file load `formatFileIo` correctly
#  returns nothing, so a single-file task cannot distinguish fixed from broken —
#  which is exactly why the earlier archeology-easy-3 smoke showed 0 on every
#  arm and proved nothing.
#
#  Throwaway: delete once the fix lands. Not part of any factorial.
# ===========================================================================
class _GPT52FileIOOld(_GPT52Anchor):
    _NAME = "_GPT52FileIOOld"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", CODE_LEAN_ENDPOINT)
        super().__init__(verbose=verbose, *args, **kwargs)


class _GPT52FileIONew(_GPT52Anchor):
    _NAME = "_GPT52FileIONew"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", PROMPT_FIX_ENDPOINT)
        super().__init__(verbose=verbose, *args, **kwargs)


for _cls, _tag in ((_GPT52FileIOOld, "FileIOOld"), (_GPT52FileIONew, "FileIONew")):
    _n = f"DataflowSystemGPT52{_tag}E2E"
    globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# Throwaway smoke: the gpt-5.2-medium litellm alias (reasoning_effort pinned to
# medium; the bare gpt-5.2 alias defaults to NO reasoning on chat/completions).
# One task, checks reasoning_tokens lands in stats.json. Delete after use.
class _GPT52MediumSmoke(_GPT52Anchor):
    _MODEL = "gpt-5.2-medium"
    _NAME = "_GPT52MediumSmoke"


DataflowSystemGPT52MediumSmokeE2E = type(
    "DataflowSystemGPT52MediumSmokeE2E", (_GPT52MediumSmoke,), {"_NAME": "DataflowSystemGPT52MediumSmokeE2E"}
)


# ===========================================================================
#  GPT-5.2-MEDIUM — the same seven cells, but with reasoning actually ON.
#
#  Every gpt-5.2 arm before 2026-08-09 ran through the bare `gpt-5.2` litellm
#  alias, which sends no `reasoning_effort`; probed directly, that is byte-
#  identical to `reasoning_effort:"none"` (0 reasoning tokens), so the whole
#  gpt-5.2 family to date is a NO-REASONING baseline. The `gpt-5.2-medium`
#  alias pins medium. Measured on archeology-easy-3, Anchor config:
#      bare   -> 0 reasoning tok,   340 out,  $0.0199
#      medium -> 952 reasoning tok, 1527 out, $0.0391
#  so expect roughly 2x cost per task.
#
#  Only the model changes; every render knob is inherited unchanged from the
#  corresponding cell, so `GPT52MediumC4Sample2k` vs `GPT52C4Sample2k` is a
#  clean one-variable test of what reasoning buys.
# ===========================================================================
class _GPT52MediumC4Sample2k(_GPT52C4Sample2k):
    _MODEL = "gpt-5.2-medium"
    _NAME = "_GPT52MediumC4Sample2k"


for _i in (0, 1, 2, 3, 4):
    _n = f"DataflowSystemGPT52MediumC4Sample2kReplicate{_i}"
    globals()[_n] = type(_n, (_GPT52MediumC4Sample2k,), {"_NAME": _n})


# ===========================================================================
#  HAIKU-4.5 2k/D2 PAIR — DELTA vs LATEST, both with stats, on the PROMPT-FIX
#  agent-service (:3005).
#
#    Haiku2kDeltaStatsD2   2K, DELTA,  stats + hints, data_level=2
#    Haiku2kLatestStatsD2  2K, LATEST, stats + hints, data_level=2, +code
#
#  ENDPOINT IS THE POINT: these run against :3005 (worktree `prompt-fix`,
#  branch fix/context-format-delta), NOT :3001. That branch carries the three
#  prompt/render fixes, and one of them is load-bearing for the DELTA arm here:
#
#    * `Files read:` renders in DELTA at all. On every other service the fact is
#      computed and discarded, so a DELTA arm shows it 0/104 tasks. The LATEST
#      arm would get it either way — this pair is the first time both modes do.
#    * context-format.delta.md no longer describes a `# Current Dataflow`
#      section that a lossless DELTA arm never renders, and no longer claims
#      operator code is hidden when every Action carries it.
#    * the Key Principles no longer tell a code-visible arm its code was
#      discarded (applies to BOTH arms: DELTA shows code inline, LATEST via
#      enable_code_in_snapshot).
#
#  So these are NOT comparable with any gpt-5.2 / luna / mini arm — those all
#  ran against services without these fixes. Read them only against each other.
#
#  `_CODE = True` on the LATEST arm is not an extra knob: DELTA carries code
#  inline per event, so a code-blind LATEST cell would confound the mode axis
#  with a code axis — same reasoning the luna/terra C3 cells use.
# ===========================================================================
PROMPT_FIX_ENDPOINT_HAIKU = "http://localhost:3005"


class _Haiku2kD2Base(DataflowSystem):
    _CONTEXT_MODE = "delta"
    _RESULT_CHARS = 2000
    _CODE = False
    _NAME = "_Haiku2kD2Base"
    _MODEL = "claude-haiku-4.5"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault("agent_service_endpoint", PROMPT_FIX_ENDPOINT_HAIKU)
        if self._CODE:
            kwargs.setdefault("enable_code_in_snapshot", True)
        super().__init__(
            model_type=self._MODEL,
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            # flow_level=0: NO `# Operators needing attention` section. Raw
            # errors are unaffected — formatOperatorResult renders
            # `[ERROR] <engine message>` + the failing operator's code purely on
            # `opInfo.error`, with no flow gate. What is given up is the triage
            # layer: topological (root-cause-first) ordering, the
            # `blocked — upstream X errored; fix those first` line for operators
            # that never ran (those render nothing per-operator, so that link is
            # stated nowhere else), and the exception-keyed loader remediation
            # hints. Measured incidence of the section on comparable arms: ~30%
            # of tasks (33/104, 23/102, 32/104).
            flow_level=0,
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


class _Haiku2kDeltaStatsD2(_Haiku2kD2Base):
    """2K, DELTA, stats + hints, d=2. The arm the `Files read:` fix unblocks."""
    _CONTEXT_MODE = "delta"; _CODE = False
    _NAME = "_Haiku2kDeltaStatsD2"


class _Haiku2kLatestStatsD2(_Haiku2kD2Base):
    """2K, LATEST, stats + hints, d=2, code in snapshot."""
    _CONTEXT_MODE = "latest"; _CODE = True
    _NAME = "_Haiku2kLatestStatsD2"


for _cls, _tag in ((_Haiku2kDeltaStatsD2, "Haiku2kDeltaStatsD2"), (_Haiku2kLatestStatsD2, "Haiku2kLatestStatsD2")):
    _n = f"DataflowSystem{_tag}Rep0"
    globals()[_n] = type(_n, (_cls,), {"_NAME": _n})


# ─────────────────────────────────────────────────────────────────────────
# Context-matrix smoke arms for the dataflow-agent repo's stack verification:
# claude-haiku-4.5 × {LATEST, DELTA} × {bare, decorated}. The four arms vary
# ONLY the context knobs, so running all four proves both context modes and
# both DECORATE ladders assemble and render.
#   bare      — flow_level=0, data_level=0, no column stats
#   decorated — flow_level=1 (loader hints + `Operators needing attention`),
#               data_level=2 (typed `Schema (N cols)` + `Output Table profile`),
#               column_stats (`Column Schema and stats:` per result)
# Everything else mirrors DataflowSystemStableHaiku.
# ─────────────────────────────────────────────────────────────────────────
class _HaikuContextMatrix(DataflowSystem):
    _CONTEXT_MODE = "latest"; _FLOW = 0; _DATA = 0; _STATS = False
    _NAME = "_HaikuContextMatrix"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode=self._CONTEXT_MODE,
            max_steps=25,
            flow_level=self._FLOW,
            data_level=self._DATA,
            column_stats=self._STATS,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaikuLatestBare(_HaikuContextMatrix):
    """LATEST × flow_level=0, data_level=0 — leaf snapshot, no decoration."""
    _NAME = "DataflowSystemHaikuLatestBare"


class DataflowSystemHaikuLatestRich(_HaikuContextMatrix):
    """LATEST × flow_level=1, data_level=2 + column stats."""
    _FLOW = 1; _DATA = 2; _STATS = True
    _NAME = "DataflowSystemHaikuLatestRich"


class DataflowSystemHaikuDeltaBare(_HaikuContextMatrix):
    """DELTA × flow_level=0, data_level=0 — event trajectory, no decoration."""
    _CONTEXT_MODE = "delta"
    _NAME = "DataflowSystemHaikuDeltaBare"


class DataflowSystemHaikuDeltaRich(_HaikuContextMatrix):
    """DELTA × flow_level=1, data_level=2 + column stats — the shipped shape."""
    _CONTEXT_MODE = "delta"; _FLOW = 1; _DATA = 2; _STATS = True
    _NAME = "DataflowSystemHaikuDeltaRich"

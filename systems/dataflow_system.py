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
        stat_scopes: Optional[Dict[str, str]] = None,
        message_layout: Optional[str] = None,
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
        cache_aligned_context: Optional[bool] = None,
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
        # WHERE each data channel renders (source / nonsource / all / off).
        self.stat_scopes = stat_scopes
        # block (legacy) | native (real tool-calling transcript)
        self.message_layout = message_layout
        self.versioned_mode = versioned_mode
        self.session_turns = session_turns
        self.recall_max_result_chars = recall_max_result_chars
        self.recall_operator_level = recall_operator_level
        self.spec_audit = spec_audit
        self.versioned_heads = versioned_heads
        self.index_rich_tables = index_rich_tables
        self.index_detailed_operators = index_detailed_operators
        self.index_thin_observations = index_thin_observations
        self.cache_aligned_context = cache_aligned_context
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
            stat_scopes=self.stat_scopes,
            message_layout=self.message_layout,
            versioned_mode=self.versioned_mode,
            session_turns=self.session_turns,
            recall_max_result_chars=self.recall_max_result_chars,
            recall_operator_level=self.recall_operator_level,
            spec_audit=self.spec_audit,
            versioned_heads=self.versioned_heads,
            index_rich_tables=self.index_rich_tables,
            index_detailed_operators=self.index_detailed_operators,
            index_thin_observations=self.index_thin_observations,
            cache_aligned_context=self.cache_aligned_context,
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
        # Anthropic bills cache WRITES at 1.25x input. The @ai-sdk/openai usage
        # object does not expose litellm's cache_creation_tokens extension today,
        # so this stays 0 on the dataflow path until agent-service surfaces it —
        # the read is here so stats and cost pick it up the moment that lands.
        token_usage_cache_creation = (
            usage.get("cache_creation_input_tokens", 0)
            or usage.get("cacheCreationInputTokens", 0)
            or usage.get("cache_creation_tokens", 0)
        )
        cost_usd = 0.0
        try:
            from systems.cost_utils import compute_cost
            c = compute_cost(
                self.model_type,
                input_tokens=token_usage_input,
                output_tokens=token_usage_output,
                cached_tokens=token_usage_cached,
                cache_creation_tokens=token_usage_cache_creation,
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
            "cache_creation_tokens": token_usage_cache_creation,
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






# ─────────────────────────────────────────────────────────────────────────
# The "converge" stack = the cost-minimized DataFlow configuration:
# LATEST context (aggregated history) + flow_level=1 (loader-remediation) +
# data_level=1 (compact typed Schema line) + a loader-proliferation budget and
# attempt-reflection (ACT-side convergence guards). Expressed via the two
# ordinal DECORATE levels (see claude/CONTEXT-DESIGN.md §5/§8b).
# ─────────────────────────────────────────────────────────────────────────
















# Schema-only twins of the 5k stats arms: column stats OFF, only the Schema line
# (output column names + types, data_level=1) kept. Isolates "just the schema" vs
# "schema + full per-column stats" at the 5k operating point, for latest & delta.










# ---------------------------------------------------------------------------
# Probe-prompt fresh controls. Config-identical to the C1/C2/C3 base arms;
# NEW names so runs land in fresh scratch dirs (the base arms' folders keep
# their pre-probe vintage). The knob is the agent-service prompt itself: the
# raw-probe principles + worked-example beats are PERMANENT in the service
# since dataflow-agent acf87127f (+ 5c10913e6, 57bd2fd0a), so any run of
# these classes carries them. Rerun set: the probing-issue tasks from the
# deep dives (format-blinded loads, dirty headers, key traps) + controls.
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# data_level=2 + result-char sweep (gpt-5.2). Same recipe as _GPT52StatsSweep
# (column_stats ON, flow_level=1, max_steps=25, attempt_reflection) but with
# data_level=2 — the `Output Table profile:` block (all-null rows/cols by name,
# duplicate-row count, unnamed-header count) ON. Two result-char points: 5k
# (recovery test vs the data_level=1 5k arms) and 10k (full runs). Optional
# static_compaction demonstrates the DELTA-only auto-fold flag in isolation.
# ---------------------------------------------------------------------------


































# ---------------------------------------------------------------------------
# gpt-5-mini 3k replica of the gpt-5.2 data-context sweep. These hold the
# DataflowAgent knobs constant with the gpt-5.2 arms and only change model_type.
# ---------------------------------------------------------------------------










# --- gpt-5-mini C1/C2 knob arms (subtask-eval study) ---
# C1 char cap: Delta1k vs Delta5k (schema-only). C2 profile: Delta1k schema vs
# DeltaStats1kD2 (both 1k). Matched one-knob pairs on the mini substrate.








# --- code-in-snapshot experiment (LATEST, 1k, gpt-5-mini) ---
# Does showing the agent its OWN code in the snapshot (with short summaries) help?
# Baseline = plain LATEST-1k schema-only; ray = same + enableCodeInSnapshot.




# --- Variance replicates (gpt-5-mini): config-identical to anchor + C1..C6,
# NEW names so each lands in its own scratch dir = independent single-shot run.
# Two per base arm -> with the original, 3 independent samples per knob to
# estimate the run-to-run randomness floor. Placed AFTER all base arms so the
# subclass references resolve. (No recovery rounds when run — raw single-shot.)








# --- Replicate0: clean single-shot re-run of the 7 base arms (anchor+C1-C6),
# because the base arms' round0 traces were overwritten in-place by their 2
# recovery rounds. Config-identical, new names, NO retries when run -> a 3rd
# clean single-shot trace set per knob (variance triple = Rep0/Rep1/Rep2).


# --- Replicate3/Replicate4: extend every knob to 5 clean single-shot reps.
# --- C7 (Delta2kSchemaOnly) Replicate0-4: new knob, 5 reps from scratch.









# --- C8: LATEST 5k + code-in-snapshot (wide-sampling twin of C3), 5 reps.



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




# Sentinel: a 4th A0 rep on the NEW sha. Golden parity proves the render is
# byte-identical, but it does not measure the run-level offset (fresh stack vs
# deep-in-pool) that dominates this benchmark's variance. This rep does.


# Rep expansion: resolve A1-vs-A0 (+5.0 at 3 reps, inside the +-4-5pt floor).
# 8 reps/arm gives ~+-3pt SEM on a ~8.5pt rep std. Reps 1-3 ran on 4af1e98da,
# these run on 9d60d01dc; golden parity holds for both configs (default "full"
# density renders byte-identical) and A0ControlReplicate4 is the cross-sha
# sentinel that measures the run-level offset directly.


# ---------------------------------------------------------------------------
#  A4 — source provenance principle on top of A_win (= A1 full config).
#  Gold-solution trace dive: per-file identity is a load-time fact erased by
#  concat; suffix-regex derivation = 0%-pass trap (legal-hard-29 n=26,
#  legal-hard-16 n=108). The flag appends ONE prompt principle: multi-file
#  loaders add a __source_file column. Byte-identical prompt when off.
#  Falsifiable: must lift legal-hard-29 + legal-hard-16 specifically.
# ---------------------------------------------------------------------------








# A4 validation reps: 8-rep footing for the one arm that cleared 2x SEM at 3
# reps (71.7 +-2.4 vs A0 59.1 +-12.3). Also watches the archeology-hard-7 flag.


# ---------------------------------------------------------------------------
#  A6 — ISOLATE the structuralHints leg. A0 control + `Output Table profile:`
#  facts on sources, nothing else changed (no row cap, no stats, no interior
#  trim). Those facts render in 62-64% of every A arm and 0% of A0, so they are
#  a confound in the whole A-series; this is the missing single-leg control.
# ---------------------------------------------------------------------------




# Fresh A0 control on the CURRENT sha. The stats-bound commit (81dc518be)
# changed the DEFAULT render — proof-based suppression now fires without any
# flag — so A0 reps 1-8 (4af1e98da / 9d60d01dc) are a different vintage and
# cannot serve as A6's control. Same config as _A0Control, new names so the runs
# land in fresh scratch dirs.


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










# ===========================================================================
#  C12 — the missing cell: LATEST 1k + code + stats/D2.
#  C6 (LatestStats1kD2) is LATEST 1k + stats WITHOUT code and is the worst arm
#  on the board (63.1). C3 is LATEST 1k + code WITHOUT stats (68.7). This is the
#  1k twin of C8s/C11.
# ===========================================================================


# ===========================================================================
#  A7 — engine-side multi-file load provenance (`files read: N [names]`).
#  Config-identical to C11 (LATEST 5k + code + stats/D2, the uniform-rich arm)
#  so the ONLY difference is the new __file_io__ fact the worker publishes. The
#  fact renders on the structural-hints rung, which stats/D2 already enables —
#  hence a stats-carrying base is required.
#  Unlike A4 (a prompt principle the agent obeys 40% of the time) this is an
#  observation, so coverage is 100% of multi-file loads.
# ===========================================================================




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








# ===========================================================================
#  N4 / N5 — probe the 2k sampling tier with code + stats (+fact by coupling).
#  Context: sampling saturates early on DELTA (1k 63.3 -> 2k 66.2 -> 5k 66.6),
#  but the 2k tier was never tried on LATEST-with-code, and the source-rich split
#  was only tested at 5k/1k (C9/C10, stats off downstream) and 5k/2k (N3).
#    N4  LATEST 2k + code + stats            — the 2k twin of N1 (5k) / D12 (1k)
#    N5  LATEST src 2k / downstream 1k + code + stats, stats BOTH sides
#  Both era 2, 3 reps.
# ===========================================================================






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








# ===========================================================================
#  RULE B — versions/history on a LATEST core (config-only; no service change).
#  Base = C8 Latest5k+code (best arm). Each ray adds ONE history channel to
#  isolate which kind of memory repays the re-derivation tax:
#    B1 codeHistory=1    -> prior CODE version per operator
#    B2 resultHistory=1  -> prior RESULT version per operator (shape-rendered)
#    B3 reasoningReplayK -> last-3 thoughts (`# Reasoning`), no per-op history
#  3 reps each; run on the 20-task discriminating hard subset.
# ===========================================================================





# --- Rep5-7: post-prompt-change replicates (2026-07-28) --------------------
# The agent-service prompt/tool change of 2026-07-28 made "code is visible" the
# default, which flipped the DELTA arms from the verbose summary instruction to
# the terse one (measured: operator summaries 136-141 -> ~63 chars). Rep0-4 of
# these three arms were produced under the OLD wording and Rep5-7 under the NEW
# one, so the two blocks are a paired before/after on the same configs. Keep
# both; never pool them.



# --- C9: latest + 5k + code-in-snapshot + column stats ---------------------
# The missing cell of the knob star: C8 (latest 5k + code) with stats turned on,
# i.e. the latest-mode twin of C4 that also shows the agent its own code.




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














# ===========================================================================
#  TERRA SERIES — gpt-5.6-terra, identical five-arm factorial to the luna sweep so
#  the two 5.6 models compare knob-for-knob and against gpt-5-mini.
#  Same Responses-API routing and reasoning_effort=medium (terra rejects function
#  tools on /v1/chat/completions with any effort other than "none", exactly as luna
#  does — verified directly against the API).
# ===========================================================================














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




# LS/TS reps 6-7 (user extended the pool to 7 reps per arm).


# Round-1 factorial reps 4-5 (both models). Run co-run in ONE pool; do NOT blindly
# pool with reps 1-3 — different engine era (measured drift -4.5 pt on identical
# config). Pool across eras only if per-arm levels match; else report separately.


# ===========================================================================
#  LUNA C5 — 5K DELTA + stats/hints: the missing C1xC2 cell on luna (terra's was
#  T1). Notes: (1) DELTA carries operator code inline per event by design, so
#  "+code" is inherent and enable_code_in_snapshot stays False (the server gate
#  is LATEST-only); (2) the `Files read:` fact is expected NOT to render — known
#  5.6+DELTA defect, 0/104 across every luna/terra DELTA arm. Stats+hints do.
#  Co-run control: fresh LunaC1 reps 6-8 (5K DELTA no-stats), so C5-C1 = the
#  stats effect measured within-pool rather than against last era's C1.
# ===========================================================================




# TERRA C5 — same cell as luna C5 (5K DELTA + stats/hints; config identical to
# _T1Delta5kStats, renamed for the paired 5v3 test). Luna's within-pool result was
# +2.5 at 3.23x SE (hard +2.5 at 3.46x) — the first 5.6 render knob past 2x SE, an
# interaction (stats needs the 5K sample; each half alone was noise). Terra's T1 at
# 3v3 showed +1.3 at 0.93x; this rerun at 5v3 with a co-run C1 control tests whether
# the interaction generalizes across the 5.6 pair.




# Terra C5 pool: control extended to 5 reps (5v5).


# ─────────────────────────────────────────────────────────────────────────
# Stable-branch verification arm: claude-haiku-4.5 (cheapest gateway model)
# on the converge stack. Used by the dataflow-agent repo's benchmark/bench.sh
# smoke runs; parameters mirror DataflowSystemGPT54LatestSchemaConverge with
# only the model swapped.
# ─────────────────────────────────────────────────────────────────────────














# Five replicates, numbered from 0. Each name gets its own scratch dir, so a
# replicate is an independent single-shot run, not a re-score of the same one;
# 5 is what the post-repair luna/terra table used and what the +-across-reps
# std in that table is computed over.


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






# Throwaway smoke: the gpt-5.2-medium litellm alias (reasoning_effort pinned to
# medium; the bare gpt-5.2 alias defaults to NO reasoning on chat/completions).
# One task, checks reasoning_tokens lands in stats.json. Delete after use.




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










# ===========================================================================
#  GPT-5.2-MEDIUM — C4 and C5 with reasoning actually ON.
#
#  Same render knobs as the corresponding gpt-5.2 cells; only the model alias
#  changes, so each is a clean one-variable read of what medium reasoning buys.
#  The bare `gpt-5.2` alias sends no reasoning_effort — probed through the proxy
#  it returns 0 reasoning tokens against gpt-5.2-medium's 2302 — so the existing
#  C4 cell is the no-reasoning control for C4 here.
#
#    MediumC4   5K, LATEST, stats + hints, +code   (all three knobs)
#    MediumC5   5K, DELTA,  stats + hints          (sampling + stats, no mode flip)
#
#  C5 is a NEW cell for gpt-5.2: its factorial stops at C4. The definition
#  follows luna/terra's C5 (5K DELTA w stats, LUNA_TERRA_FINAL_TABLE.md), which
#  also makes it the direct peer of gpt-5-mini's C4 (DeltaStats5kD2) — so the
#  same config is readable across all three models.
# ===========================================================================






# ===========================================================================
#  SCOPED-STATS MATRIX (claude-haiku-4.5) — Idea 1.
#
#  `data_level` says WHICH channels are on; `stat_scopes` says WHERE each one
#  renders. The asymmetry being tested: a SOURCE operator can only lose data at
#  parse time (coercion NaNs, dropped values, partial file reads); a CONSUMER
#  can only lose it inside its own transform (a join dropping keys). A channel
#  aimed at one is noise on the other.
#
#  All arms share the same channel set (data_level=2 + column_stats + coercion +
#  lineage + file-IO); ONLY the scoping differs, so any delta is attributable to
#  placement rather than to how much evidence exists.
#
#    ScopedControl    everything everywhere (the v8 "rich" shape) — the control
#    ScopedSplit      coercion+fileIO at sources, lineage+colstats downstream
#    ScopedLean       as Split, but column stats OFF entirely (v8 measured
#                     full-column stats changing join decomposition on passnyc)
#    ScopedSrcStats   column stats ONLY at sources (schema discovery at the lake
#                     edge, silence downstream)
# ===========================================================================
class _HaikuScopedBase(DataflowSystem):
    _SCOPES: Optional[Dict[str, str]] = None
    _NAME = "_HaikuScopedBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            context_mode="delta",
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            coercion_facts=True,
            row_lineage=True,
            attempt_reflection=True,
            max_operator_result_char_limit=2000,
            max_operator_result_cell_char_limit=3000,
            stat_scopes=self._SCOPES,
            name=self._NAME,
            verbose=verbose,
            *args,
            **kwargs,
        )


class DataflowSystemHaikuScopedControl(_HaikuScopedBase):
    """Every channel on every operator — the unscoped control."""
    _SCOPES = None
    _NAME = "DataflowSystemHaikuScopedControl"


class DataflowSystemHaikuScopedSplit(_HaikuScopedBase):
    """Parse-time evidence at sources, transform evidence downstream."""
    _SCOPES = {
        "coercion": "source",
        "fileIo": "source",
        "valueFormat": "source",
        "lineage": "nonsource",
        "columnStats": "nonsource",
        "structural": "all",
    }
    _NAME = "DataflowSystemHaikuScopedSplit"


class DataflowSystemHaikuScopedLean(_HaikuScopedBase):
    """Split, minus column stats entirely — tests whether the stats flood (not
    the stats themselves) is what hurt the v8 passnyc arms."""
    _SCOPES = {
        "coercion": "source",
        "fileIo": "source",
        "valueFormat": "source",
        "lineage": "nonsource",
        "columnStats": "off",
        "structural": "all",
    }
    _NAME = "DataflowSystemHaikuScopedLean"


class DataflowSystemHaikuScopedSrcStats(_HaikuScopedBase):
    """Column stats only where the data enters — schema discovery at the lake
    edge, silence once the shape is known."""
    _SCOPES = {
        "coercion": "source",
        "fileIo": "source",
        "columnStats": "source",
        "lineage": "nonsource",
        "structural": "all",
    }
    _NAME = "DataflowSystemHaikuScopedSrcStats"


# ===========================================================================
#  MESSAGE-FRAMING PAIR (claude-haiku-4.5) — Idea 2.
#
#  Identical rendering and knobs; ONLY the wire framing differs.
#    LayoutBlock   the whole rendered trajectory as one user message per step
#    LayoutNative  the same render as a real tool-calling transcript: assistant
#                  turns carry actual tool_calls, each tool result carries OUR
#                  rendered execution evidence (schema/stats/lineage/coercion)
#
#  Motivation: block re-renders every step so nothing prefix-matches — measured
#  56.6% of 2.93M input tokens spent as cache WRITES (1.25x) instead of READS
#  (0.10x) on `environment`. Native makes the prefix append-only.
#
#  NOTE the earlier prose variant (rendered `Action:` text in assistant turns)
#  scored 5.0% vs 55.3%: the model narrated calls instead of making them. Native
#  frames exist precisely to remove that ambiguity.
# ===========================================================================
class _HaikuLayoutBase(DataflowSystem):
    _LAYOUT = "block"
    _NAME = "_HaikuLayoutBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5", context_mode="delta", max_steps=25,
            flow_level=1, data_level=2, column_stats=True, attempt_reflection=True,
            max_operator_result_char_limit=2000, max_operator_result_cell_char_limit=3000,
            message_layout=self._LAYOUT,
            name=self._NAME, verbose=verbose, *args, **kwargs,
        )


class DataflowSystemHaikuLayoutBlock(_HaikuLayoutBase):
    """One user block per step — the legacy framing."""
    _LAYOUT = "block"
    _NAME = "DataflowSystemHaikuLayoutBlock"


class DataflowSystemHaikuLayoutBlockSplit(_HaikuLayoutBase):
    """Block bytes, cut into [history][current state] so the history can cache.

    Isolates framing from cost: this and LayoutNative are both cacheable, so a
    score gap between them is the framing alone, not one arm paying 1.25x cache
    writes while the other pays 0.10x reads.
    """
    _LAYOUT = "blockSplit"
    _NAME = "DataflowSystemHaikuLayoutBlockSplit"


class DataflowSystemHaikuLayoutNative(_HaikuLayoutBase):
    """Same render, framed as a real tool-calling transcript."""
    _LAYOUT = "native"
    _NAME = "DataflowSystemHaikuLayoutNative"


# ===========================================================================
# CANONICAL CONFIG GRID  (Anchor + C1..C5) x model x replicate
# ===========================================================================
# One generated family replacing the ad-hoc hand-written `*Replicate*` classes.
# Every arm carries `message_layout="blockSplit"`: the rendered context is
# byte-identical to the legacy single block, but it is sent as
# [history][current state] so the history is a stable prefix that prompt
# caching can actually reuse. This is a wire-framing change only — the agent
# sees the same text it always did (summarize.ts verifies the two halves
# reconstruct the render before using them).
#
# Axes, matching the original campaign:
#   Anchor  delta  1k  schema-only     C3  latest 1k  schema-only
#   C1      delta  5k  schema-only     C4  latest 5k  stats
#   C2      delta  1k  stats           C5  latest 5k  schema-only
_GRID_CONFIGS = {
    "Anchor": ("delta", 1000, False),
    "C1": ("delta", 5000, False),
    "C2": ("delta", 1000, True),
    "C3": ("latest", 1000, False),
    "C4": ("latest", 5000, True),
    "C5": ("latest", 5000, False),
}
_GRID_MODELS = {"Haiku": "claude-haiku-4.5", "Sonnet": "claude-sonnet-5",
                "GPT5Mini": "gpt-5-mini-medium"}
_GRID_REPS = (1, 2, 3)


def _make_grid_system(model_tag, model_type, cfg_tag, mode, chars, stats, rep):
    name = f"DataflowSystem{model_tag}{cfg_tag}Rep{rep}"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        DataflowSystem.__init__(
            self,
            model_type=model_type,
            context_mode=mode,
            max_steps=25,
            flow_level=1,
            data_level=2 if stats else 1,
            column_stats=stats,
            stats_enabled=stats,
            attempt_reflection=True,
            max_operator_result_char_limit=chars,
            max_operator_result_cell_char_limit=3000,
            message_layout="blockSplit",
            name=name,
            verbose=verbose,
            *args,
            **kwargs,
        )

    doc = (f"{model_type}, {mode.upper()}, {chars // 1000}k result chars, "
           f"{'stats' if stats else 'schema-only'} (rep {rep}). Cached via blockSplit.")
    return type(name, (DataflowSystem,), {"__init__": __init__, "__doc__": doc})


_GRID_SYSTEMS = {}
for _mtag, _mtype in _GRID_MODELS.items():
    for _ctag, (_mode, _chars, _stats) in _GRID_CONFIGS.items():
        for _rep in _GRID_REPS:
            _cls = _make_grid_system(_mtag, _mtype, _ctag, _mode, _chars, _stats, _rep)
            _GRID_SYSTEMS[_cls.__name__] = _cls
            globals()[_cls.__name__] = _cls

GRID_SYSTEM_NAMES = sorted(_GRID_SYSTEMS)


# ===========================================================================
# BEST-KNOWN CONFIG  (delta, 5k, stats)  x model x replicate
# ===========================================================================
# Reconstructed from the gpt-5.2-medium campaign (104 tasks x 6 workloads):
#   DataflowSystemGPT52MediumC5Replicate0  delta 5k stats  79.1%   <- winner
#   DataflowSystemGPT52MediumC4Replicate0  latest 5k stats 76.2%
# Registered under `Best` rather than `C5` on purpose: the campaign's C5 and
# the Anchor/C1..C5 grid's C5 (latest 5k, schema-only) are DIFFERENT configs
# that happened to share a number, and collapsing them would silently redefine
# the grid axis.
#
# gpt-5-mini carries this config as a PROXY: its own campaign results are not
# in this checkout (scratch dirs are empty, no measures CSVs), so its true best
# config is unverified here and inherited from gpt-5.2-medium.
_BEST_MODELS = {
    "GPT5Mini": "gpt-5-mini-medium",
    "GPT52Medium": "gpt-5.2-medium",
    "Luna": "gpt-5.6-luna",
    "Terra": "gpt-5.6-terra",
}


def _make_best_system(model_tag, model_type, rep):
    name = f"DataflowSystem{model_tag}BestRep{rep}"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        DataflowSystem.__init__(
            self,
            model_type=model_type,
            context_mode="delta",
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            stats_enabled=True,
            attempt_reflection=True,
            max_operator_result_char_limit=5000,
            max_operator_result_cell_char_limit=3000,
            message_layout="blockSplit",
            name=name,
            verbose=verbose,
            *args,
            **kwargs,
        )

    doc = f"{model_type}, DELTA, 5k, stats — best-known config (rep {rep}). Cached via blockSplit."
    return type(name, (DataflowSystem,), {"__init__": __init__, "__doc__": doc})


for _btag, _btype in _BEST_MODELS.items():
    for _rep in _GRID_REPS:
        _cls = _make_best_system(_btag, _btype, _rep)
        _GRID_SYSTEMS[_cls.__name__] = _cls
        globals()[_cls.__name__] = _cls

GRID_SYSTEM_NAMES = sorted(_GRID_SYSTEMS)


# ===========================================================================
# LUNA DELTA 2K + STATS — the missing rows-axis midpoint on the stats ray
# ===========================================================================
# The luna factorial (bobflow campaign, 104 tasks x 6 workloads) sampled the
# stats ray at only two char budgets:
#
#   LunaC2   delta 1k stats+hints   72.4   <- this arm sits between them
#   LunaC5   delta 5k stats+hints   74.0
#
# so the 1.6 pt C2->C5 gap cannot be attributed to the rows budget without a
# midpoint. This arm is that midpoint.
#
# Constructed to match bobflow's `_LunaBase` argument-for-argument, NOT the
# Anchor/C1..C5 `_GRID_CONFIGS` family: the grid pins
# `message_layout="blockSplit"` and the historical luna arms left it unset
# (None), so inheriting the grid would add a wire-framing change on top of the
# char-budget change and make the comparison two-variable.
#
# `enable_code_in_snapshot=False` is explicit for self-documentation only —
# server.ts:325/336 gate code-in-snapshot on `contextMode === LATEST`, so it is
# already a no-op in DELTA. `stats_enabled` is likewise omitted to match
# `_LunaBase`: it only lifts data_level to >=1 and data_level=2 is set here.
#
# CAVEAT for whoever reads the number: this runs on a LATER agent-service build
# than the luna factorial (which ran at sha 1075905b5 from the bobflow
# code-lean worktree). No co-run control was recorded, so a 2k-vs-C2/C5 read
# carries the build delta as a confound.
# ===========================================================================
class DataflowSystemLunaDeltaStats2kRep1(DataflowSystem):
    """gpt-5.6-luna, DELTA, 2k result chars, stats + hints, no code-in-snapshot."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.6-luna",
            context_mode="delta",
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            attempt_reflection=True,
            max_operator_result_char_limit=2000,
            max_operator_result_cell_char_limit=3000,
            enable_code_in_snapshot=False,
            name="DataflowSystemLunaDeltaStats2kRep1",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ===========================================================================
# LUNA DELTA 2K + STATS, CACHE-ALIGNED — same knobs as
# DataflowSystemLunaDeltaStats2kRep1, plus append-only message packaging.
# ===========================================================================
# Byte-for-byte the 2k arm above with ONE difference: cache_aligned_context.
# Measured motivation (24,775 task-runs, block layout): gpt-5.6-luna/terra and
# claude-* keep cached-input PINNED (cv 0.007-0.011, cached last/first 0.98-1.00
# while input grows 1.84-2.45x), because the legacy loop re-renders the whole
# trajectory into one fresh user message per step. gpt-5.2 / gpt-5-mini instead
# GROW (cv 0.23, cached tracks input, 100% 128-token aligned) — they do true
# intra-message prefix caching, which the 5.6 deployment does not:
# gpt-5.6-luna returns 0 cache for "text appended inside a message" where
# gpt-5.2 serves a 124.8k prefix hit (gateway probe, BUSINESS_CAMPAIGN.md).
#
# cacheAlignedContext renders once at step 1 and lets the SDK append
# assistant/tool messages for steps 2..N, so earlier messages stay
# byte-untouched — the one shape luna DOES credit.
#
# WATCH ITEM: putting rendered `Action:` prose in an assistant turn makes the
# model imitate it and narrate tool calls instead of calling them (measured
# 5.0% vs 55.3% tool-call rate, types/agent.ts:448-452). cacheAlignedContext
# carries real tool-call/tool-result parts rather than prose, so it should not
# trip that — but the tool-call rate is the primary guard on this arm, not cost.
# ===========================================================================
class DataflowSystemLunaDeltaStats2kCacheRep1(DataflowSystem):
    """gpt-5.6-luna, DELTA, 2k chars, stats + hints, append-only message packaging."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.6-luna",
            context_mode="delta",
            max_steps=25,
            flow_level=1,
            data_level=2,
            column_stats=True,
            attempt_reflection=True,
            max_operator_result_char_limit=2000,
            max_operator_result_cell_char_limit=3000,
            enable_code_in_snapshot=False,
            cache_aligned_context=True,
            name="DataflowSystemLunaDeltaStats2kCacheRep1",
            verbose=verbose,
            *args,
            **kwargs,
        )


# ===========================================================================
# LUNA LATEST 1K — opSplit experiment pair (C3 config: latest, 1k, no stats, +code)
# ===========================================================================
# Single-knob pair. Both are the bobflow LunaC3 config; they differ only in
# message framing, which is a wire change guarded by a byte-parity check
# (summarize.ts) — the model sees the same render either way.
#
#   ...Latest1kRep1          message_layout unset -> "block": one user message
#   ...Latest1kOpSplitRep1   message_layout="opSplit": one message per operator
#
# Why: LATEST snapshots are already 85% mean / 100% median byte-stable
# step-to-step (measured, luna C3 578 step-pairs / C4 495), but shipping them as
# one message earns luna nothing (cached pinned at ~5.3k = system+tools only).
# opSplit localises invalidation to the revised operator and those after it.
# Order is left as the renderer produced it — last-modified reordering would
# cache better but breaks dataflow reading order, a separate experiment.
# ===========================================================================
def _mk_luna_latest_1k(name, layout):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        DataflowSystem.__init__(
            self,
            model_type="gpt-5.6-luna",
            context_mode="latest",
            max_steps=25,
            flow_level=1,
            data_level=1,
            column_stats=False,
            attempt_reflection=True,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            enable_code_in_snapshot=True,
            message_layout=layout,
            name=name,
            verbose=verbose,
            *args,
            **kwargs,
        )
    doc = (f"gpt-5.6-luna, LATEST, 1k chars, schema-only, code-in-snapshot; "
           f"message framing = {layout or 'block'}.")
    return type(name, (DataflowSystem,), {"__init__": __init__, "__doc__": doc})


DataflowSystemLunaLatest1kRep1 = _mk_luna_latest_1k("DataflowSystemLunaLatest1kRep1", None)
DataflowSystemLunaLatest1kOpSplitRep1 = _mk_luna_latest_1k("DataflowSystemLunaLatest1kOpSplitRep1", "opSplit")

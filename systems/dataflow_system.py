# -*- coding: utf-8 -*-
"""
DataflowSystem - KramaBench System wrapper for Texera DataflowAgent.

This module provides a System implementation that uses the Texera Agent Service
to solve benchmark tasks via dataflow-based agents.
"""

import os
import fnmatch
import json
import re
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
        result_rendering: str = None,
        context_scope: str = None,
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
        max_unexecuted_code_edits: Optional[int] = None,
        source_manifest_enabled: bool = False,
        source_manifest_max_files: int = 80,
        source_manifest_max_related_per_source: int = 40,
        metric_evidence_guidance_enabled: bool = False,
        schema_first_code_mode_enabled: bool = False,
        table_structure_hints_enabled: bool = False,
        raw_loader_provenance_enabled: bool = False,
        bounded_execution_guidance_enabled: bool = False,
        cardinality_pressure_guidance_enabled: bool = False,
        entity_key_hygiene_guidance_enabled: bool = False,
        component_grain_guidance_enabled: bool = False,
        key_grain_comparison_guidance_enabled: bool = False,
        key_grain_evidence_contract_enabled: bool = False,
        label_component_profile_contract_enabled: bool = False,
        observed_component_inventory_contract_enabled: bool = False,
        data_discovered_component_inventory_contract_enabled: bool = False,
        boundary_token_inventory_contract_enabled: bool = False,
        flow_progress_digest_enabled: bool = False,
        candidate_selection_impact_contract_enabled: bool = False,
        evidence_dependency_gate_enabled: bool = False,
        execution_safe_operator_ids_enabled: bool = False,
        fallback_contract_guidance_enabled: bool = False,
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
            result_rendering: Successful result context rendering
                ("rows", "digest", or "adaptive"); None uses server default
            context_scope: Operator detail scope
                ("all" or "active-lineage"); None uses server default
            tool_timeout_seconds: Tool timeout (default: 240)
            execution_timeout_minutes: Execution timeout (default: 4)
            agent_mode: Agent mode (default: code)
            context_mode: Context selection policy (default: latest)
            parallel_tool_calls: Allow parallel tool calls (default: True)
            allowed_operator_types: Optional whitelist of operator types; None uses server default
            disabled_tools: Optional list of tool names to disable
            max_operator_edits: Optional convergence guard cap; None uses server default
            lineage_hint_on_stall: Optional lineage hint toggle for convergence guard stalls
            max_unexecuted_code_edits: Optional CODE-mode execution-evidence
                cadence cap; None uses server default
            source_manifest_enabled: Add a compact source-planning manifest to the prompt
            source_manifest_max_files: Maximum files to list in each manifest section
            source_manifest_max_related_per_source: Maximum related siblings per listed file
            metric_evidence_guidance_enabled: Enable server-side CODE-mode
                guidance for provenance-preserving final metric evidence
            schema_first_code_mode_enabled: Enable server-side compiled
                schema rendering in CODE-mode context
            table_structure_hints_enabled: Enable server-side DataLoading
                structure hints for likely metadata/header/footer rows
            raw_loader_provenance_enabled: Enable server-side compact raw-source
                provenance for DataLoading operators with literal relative paths
            bounded_execution_guidance_enabled: Enable server-side CODE-mode
                guidance to validate expensive full-table operators on bounded
                probes before scaling execution
            cardinality_pressure_guidance_enabled: Enable server-side CODE-mode
                guidance and context hints for large/wide intermediate outputs
                so downstream execution uses the task-required row grain
            entity_key_hygiene_guidance_enabled: Enable server-side CODE-mode
                guidance and context hints for high-cardinality string entity
                keys before grouping, joining, deduplicating, or distinct counts
            component_grain_guidance_enabled: Enable server-side CODE-mode
                guidance and context hints for high-cardinality string labels
                whose values may encode sub-entity or sampling-location
                component grains before entity-level counts
            key_grain_comparison_guidance_enabled: Enable server-side CODE-mode
                guidance and context hints that require a candidate key-grain
                comparison table before final entity-level counts
            key_grain_evidence_contract_enabled: Enable server-side LATEST
                context notices that treat candidate key-grain comparison as
                an explicit executed-table evidence contract
            label_component_profile_contract_enabled: Enable server-side
                LATEST context notices that require an all-value
                label-component profile before final entity-key selection
            observed_component_inventory_contract_enabled: Enable server-side
                LATEST context notices that require a data-driven inventory of
                observed label component/separator structure before component
                profiling or candidate-key evidence
            data_discovered_component_inventory_contract_enabled: Require
                observed inventory evidence to include data-discovered
                token/transform columns before it satisfies in LATEST context
            boundary_token_inventory_contract_enabled: Require observed
                inventory evidence to prove complete boundary-token enumeration
                and downstream candidate coverage before it satisfies in LATEST
                context
            flow_progress_digest_enabled: Enable server-side LATEST context
                digest of recent actions, repeated edits, current failures, and
                unexecuted terminal operators
            candidate_selection_impact_contract_enabled: Enable server-side
                LATEST context notices that require an executed impact table
                comparing key candidates against the downstream entity measure
            evidence_dependency_gate_enabled: Enable server-side LATEST
                dependency checks that validate typed evidence artifacts through
                workflow links to their prerequisite artifacts
            execution_safe_operator_ids_enabled: Enable server-side CODE-mode
                guidance and tool-boundary validation for operator IDs that
                are compatible with workflow execution persistence
            fallback_contract_guidance_enabled: Enable server-side CODE-mode
                guidance and context hints for dependency/runtime capability
                failures so fallback answers preserve the requested contract
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
        self.result_rendering = result_rendering
        self.context_scope = context_scope
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
        self.max_unexecuted_code_edits = max_unexecuted_code_edits
        self.source_manifest_enabled = source_manifest_enabled
        self.source_manifest_max_files = source_manifest_max_files
        self.source_manifest_max_related_per_source = source_manifest_max_related_per_source
        self.metric_evidence_guidance_enabled = metric_evidence_guidance_enabled
        self.schema_first_code_mode_enabled = schema_first_code_mode_enabled
        self.table_structure_hints_enabled = table_structure_hints_enabled
        self.raw_loader_provenance_enabled = raw_loader_provenance_enabled
        self.bounded_execution_guidance_enabled = bounded_execution_guidance_enabled
        self.cardinality_pressure_guidance_enabled = cardinality_pressure_guidance_enabled
        self.entity_key_hygiene_guidance_enabled = entity_key_hygiene_guidance_enabled
        self.component_grain_guidance_enabled = component_grain_guidance_enabled
        self.key_grain_comparison_guidance_enabled = key_grain_comparison_guidance_enabled
        self.key_grain_evidence_contract_enabled = key_grain_evidence_contract_enabled
        self.label_component_profile_contract_enabled = label_component_profile_contract_enabled
        self.observed_component_inventory_contract_enabled = observed_component_inventory_contract_enabled
        self.data_discovered_component_inventory_contract_enabled = (
            data_discovered_component_inventory_contract_enabled
        )
        self.boundary_token_inventory_contract_enabled = boundary_token_inventory_contract_enabled
        self.flow_progress_digest_enabled = flow_progress_digest_enabled
        self.candidate_selection_impact_contract_enabled = candidate_selection_impact_contract_enabled
        self.evidence_dependency_gate_enabled = evidence_dependency_gate_enabled
        self.execution_safe_operator_ids_enabled = execution_safe_operator_ids_enabled
        self.fallback_contract_guidance_enabled = fallback_contract_guidance_enabled

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
            result_rendering=self.result_rendering,
            context_scope=self.context_scope,
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
            max_unexecuted_code_edits=self.max_unexecuted_code_edits,
            metric_evidence_guidance=self.metric_evidence_guidance_enabled,
            schema_first_code_mode=self.schema_first_code_mode_enabled,
            table_structure_hints=self.table_structure_hints_enabled,
            raw_loader_provenance=self.raw_loader_provenance_enabled,
            bounded_execution_guidance=self.bounded_execution_guidance_enabled,
            cardinality_pressure_guidance=self.cardinality_pressure_guidance_enabled,
            entity_key_hygiene_guidance=self.entity_key_hygiene_guidance_enabled,
            component_grain_guidance=self.component_grain_guidance_enabled,
            key_grain_comparison_guidance=self.key_grain_comparison_guidance_enabled,
            key_grain_evidence_contract=self.key_grain_evidence_contract_enabled,
            label_component_profile_contract=self.label_component_profile_contract_enabled,
            observed_component_inventory_contract=self.observed_component_inventory_contract_enabled,
            data_discovered_component_inventory_contract=self.data_discovered_component_inventory_contract_enabled,
            boundary_token_inventory_contract=self.boundary_token_inventory_contract_enabled,
            flow_progress_digest=self.flow_progress_digest_enabled,
            candidate_selection_impact_contract=self.candidate_selection_impact_contract_enabled,
            evidence_dependency_gate=self.evidence_dependency_gate_enabled,
            execution_safe_operator_ids=self.execution_safe_operator_ids_enabled,
            fallback_contract_guidance=self.fallback_contract_guidance_enabled,
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()

    def _build_prompt(
        self,
        query: str,
        file_paths: List[str],
        format_hint: str = "",
        source_manifest: str = "",
    ) -> str:
        """
        Build the prompt for the agent.

        Args:
            query: The natural language query
            file_paths: List of file paths available for the query
            format_hint: Optional format hint for the expected answer format
            source_manifest: Optional compact manifest of related dataset sources

        Returns:
            Formatted prompt string
        """
        manifest_block = ""
        if source_manifest:
            manifest_block = f"""

Source manifest (related dataset files discovered from the paths above):
{source_manifest}

Source planning rules:
- Treat the listed data files as starting points, not proof that sibling files are irrelevant.
- If the task or an intermediate result derives a key such as a year, state, month, satellite id, or entity, and a listed filename/path carries a different key, inspect the manifest and load the matching sibling file or pattern.
- Preserve filename/path-derived metadata when it identifies an entity, state, year, or other grouping key.
- If a filter on a derived target value returns zero rows, re-check source selection before falling back to another value. Do not silently substitute another year/entity.
"""

        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Note: All paths are relative. Some paths may contain wildcards (e.g., "folder/*" or "file-*.csv"). Use glob patterns to match and read those files.
{manifest_block}

Question: {query}

Answer format: {format_hint}

Your last line MUST BE: **Final Answer: <value>**"""

        return prompt

    def _to_dataset_relative(self, path: str) -> str:
        """Convert a prompt path back to a path relative to dataset_directory."""
        normalized = os.path.normpath(path)
        dataset_rel = os.path.normpath(os.path.relpath(self.dataset_directory))
        if normalized == dataset_rel:
            return ""
        prefix = dataset_rel + os.sep
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
        return normalized

    def _format_manifest_file_list(self, files: List[str], limit: int) -> str:
        """Format dataset-relative file paths as prompt-relative paths with a cap."""
        if not files:
            return "(none)"
        shown = files[:limit]
        paths = [
            os.path.relpath(os.path.join(self.dataset_directory, file_path))
            for file_path in shown
        ]
        rendered = json.dumps(paths, indent=2)
        remaining = len(files) - len(shown)
        if remaining > 0:
            rendered += f"\n  ... {remaining} more not shown"
        return rendered

    def _match_dataset_pattern(self, pattern: str) -> List[str]:
        """Match a dataset-relative glob pattern against known dataset files."""
        normalized = pattern.replace(os.sep, "/")
        return sorted(
            file_path
            for file_path in self.dataset.keys()
            if fnmatch.fnmatch(file_path.replace(os.sep, "/"), normalized)
        )

    def _numeric_family_pattern(self, file_path: str) -> Optional[str]:
        """Return a sibling glob by replacing numeric runs in the basename."""
        directory = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        if not re.search(r"\d", basename):
            return None
        family_basename = re.sub(r"\d+", "*", basename)
        if family_basename == basename:
            return None
        return os.path.join(directory, family_basename) if directory else family_basename

    def _build_source_manifest(self, file_paths: List[str]) -> str:
        """
        Build a compact manifest from listed sources and nearby dataset siblings.

        This is intentionally mechanical and domain-agnostic: it exposes wildcard
        expansions and numeric filename families, which covers source-selection
        tasks without hard-coding task ids or answer-specific filenames.
        """
        if not file_paths:
            return ""

        listed_files: set[str] = set()
        wildcard_sections: List[tuple[str, List[str]]] = []

        for path in file_paths:
            dataset_path = self._to_dataset_relative(path)
            if not dataset_path:
                continue
            if "*" in dataset_path or "?" in dataset_path:
                matches = self._match_dataset_pattern(dataset_path)
                if matches:
                    wildcard_sections.append((path, matches))
                    listed_files.update(matches)
            elif dataset_path in self.dataset:
                listed_files.add(dataset_path)

        sections: List[str] = []

        if wildcard_sections:
            sections.append("Wildcard expansions:")
            for pattern, matches in wildcard_sections:
                sections.append(f"- {pattern} -> {len(matches)} files:")
                sections.append(
                    self._format_manifest_file_list(
                        matches,
                        min(self.source_manifest_max_files, len(matches)),
                    )
                )

        related_sections: List[tuple[str, str, List[str]]] = []
        seen_patterns: set[str] = set()
        for file_path in sorted(listed_files):
            family_pattern = self._numeric_family_pattern(file_path)
            if not family_pattern or family_pattern in seen_patterns:
                continue
            seen_patterns.add(family_pattern)
            family_matches = self._match_dataset_pattern(family_pattern)
            if len(family_matches) <= 1:
                continue
            related_sections.append((file_path, family_pattern, family_matches))

        if related_sections:
            sections.append("Related sibling file families:")
            for source_file, pattern, matches in related_sections:
                prompt_source = os.path.relpath(os.path.join(self.dataset_directory, source_file))
                prompt_pattern = os.path.relpath(os.path.join(self.dataset_directory, pattern))
                sections.append(f"- From {prompt_source}: {prompt_pattern} -> {len(matches)} files:")
                sections.append(
                    self._format_manifest_file_list(
                        matches,
                        min(self.source_manifest_max_related_per_source, len(matches)),
                    )
                )

        if not sections:
            return ""

        return "\n".join(sections)

    def _normalized_path_part(self, value: str) -> str:
        return value.lower().replace(" ", "_").replace("-", "_")

    def _find_same_named_dataset_dirs(self, dir_part: str, wildcard_part: str) -> List[str]:
        """Find dataset directories with the same basename and matching files."""
        dir_basename = os.path.basename(dir_part)
        if not dir_basename:
            return []
        normalized_basename = self._normalized_path_part(dir_basename)
        found_dirs: set[str] = set()
        for file_path in self.dataset.keys():
            file_dir = os.path.dirname(file_path)
            if not file_dir:
                continue
            if self._normalized_path_part(os.path.basename(file_dir)) != normalized_basename:
                continue
            if fnmatch.fnmatch(os.path.basename(file_path), wildcard_part):
                found_dirs.add(file_dir)
        return sorted(found_dirs)

    def _augment_file_paths_with_manifest_resolutions(self, file_paths: List[str]) -> List[str]:
        """
        Add resolved wildcard alternatives when a prompt path matches no files.

        This is enabled only for source-manifest systems. It handles cases where
        the truth subset names a directory by basename while the actual files live
        under a nested vendor/export directory.
        """
        augmented: List[str] = []
        seen: set[str] = set()

        def add(path: str) -> None:
            if path not in seen:
                seen.add(path)
                augmented.append(path)

        for path in file_paths:
            add(path)
            dataset_path = self._to_dataset_relative(path)
            if not dataset_path or ("*" not in dataset_path and "?" not in dataset_path):
                continue
            if self._match_dataset_pattern(dataset_path):
                continue
            parts = dataset_path.rsplit(os.sep, 1)
            if len(parts) == 1:
                parts = dataset_path.rsplit("/", 1)
            if len(parts) != 2:
                continue
            dir_part, wildcard_part = parts
            for found_dir in self._find_same_named_dataset_dirs(dir_part, wildcard_part):
                resolved = os.path.relpath(
                    os.path.join(self.dataset_directory, found_dir, wildcard_part)
                )
                add(resolved)

        return augmented

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

        if self.source_manifest_enabled:
            file_paths = self._augment_file_paths_with_manifest_resolutions(file_paths)

        if self.verbose:
            print(f"[DataflowSystem] Processing query: {query_id}")
            print(f"[DataflowSystem] Using {len(file_paths)} files")

        # Build prompt with file paths and format hint
        format_hint = self.format_hints.get(query_id, "")
        source_manifest = (
            self._build_source_manifest(file_paths)
            if self.source_manifest_enabled
            else ""
        )
        prompt = self._build_prompt(
            query,
            file_paths,
            format_hint=format_hint,
            source_manifest=source_manifest,
        )

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
                "result_rendering": self.result_rendering,
                "context_scope": self.context_scope,
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
                "max_unexecuted_code_edits": self.max_unexecuted_code_edits,
                "source_manifest_enabled": self.source_manifest_enabled,
                "source_manifest_max_files": self.source_manifest_max_files,
                "source_manifest_max_related_per_source": self.source_manifest_max_related_per_source,
                "metric_evidence_guidance_enabled": self.metric_evidence_guidance_enabled,
                "schema_first_code_mode_enabled": self.schema_first_code_mode_enabled,
                "table_structure_hints_enabled": self.table_structure_hints_enabled,
                "raw_loader_provenance_enabled": self.raw_loader_provenance_enabled,
                "bounded_execution_guidance_enabled": self.bounded_execution_guidance_enabled,
                "cardinality_pressure_guidance_enabled": self.cardinality_pressure_guidance_enabled,
                "entity_key_hygiene_guidance_enabled": self.entity_key_hygiene_guidance_enabled,
                "component_grain_guidance_enabled": self.component_grain_guidance_enabled,
                "key_grain_comparison_guidance_enabled": self.key_grain_comparison_guidance_enabled,
                "key_grain_evidence_contract_enabled": self.key_grain_evidence_contract_enabled,
                "label_component_profile_contract_enabled": self.label_component_profile_contract_enabled,
                "observed_component_inventory_contract_enabled": self.observed_component_inventory_contract_enabled,
                "data_discovered_component_inventory_contract_enabled": self.data_discovered_component_inventory_contract_enabled,
                "boundary_token_inventory_contract_enabled": self.boundary_token_inventory_contract_enabled,
                "flow_progress_digest_enabled": self.flow_progress_digest_enabled,
                "candidate_selection_impact_contract_enabled": self.candidate_selection_impact_contract_enabled,
                "evidence_dependency_gate_enabled": self.evidence_dependency_gate_enabled,
                "execution_safe_operator_ids_enabled": self.execution_safe_operator_ids_enabled,
                "fallback_contract_guidance_enabled": self.fallback_contract_guidance_enabled,
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
# plan2: latest-mode source manifest. These variants isolate prompt-level
# source planning against the existing LatestStatsOn systems. The manifest
# lists wildcard expansions and numeric sibling file families so the agent
# can load a derived source file instead of silently falling back.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestSourceManifestStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _SOURCE_MANIFEST_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            source_manifest_enabled=self._SOURCE_MANIFEST_ENABLED,
            source_manifest_max_files=80,
            source_manifest_max_related_per_source=40,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSourceManifestStatsOn(
    _GPTLatestSourceManifestStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestSourceManifestStatsOn"


class DataflowSystemGPT52LatestSourceManifestStatsOn(
    _GPTLatestSourceManifestStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestSourceManifestStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan3: latest-mode metric evidence guidance. These variants isolate a
# server-side CODE prompt rule against LatestStatsOn: preserve provenance keys
# through the DAG and materialize final metric evidence before answering.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestSemanticStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _METRIC_EVIDENCE_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            metric_evidence_guidance_enabled=self._METRIC_EVIDENCE_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSemanticStatsOn(_GPTLatestSemanticStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestSemanticStatsOn"


class DataflowSystemGPT52LatestSemanticStatsOn(_GPTLatestSemanticStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestSemanticStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan4: latest-mode schema-first CODE context. These variants isolate a
# context-compiler rule against LatestStatsOn: render compiled current-snapshot
# input/output schemas in CODE mode, not only GENERAL mode.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestSchemaStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _SCHEMA_FIRST_CODE_MODE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            schema_first_code_mode_enabled=self._SCHEMA_FIRST_CODE_MODE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestSchemaStatsOn(_GPTLatestSchemaStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestSchemaStatsOn"


class DataflowSystemGPT52LatestSchemaStatsOn(_GPTLatestSchemaStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestSchemaStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan5: latest-mode table-structure hints for suspicious DataLoading
# results. These variants isolate a context-compiler rule against
# LatestStatsOn: render compact generic evidence when a loader appears to
# contain metadata/header/footer rows rather than one clean table.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestTableHintsStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _TABLE_STRUCTURE_HINTS_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            table_structure_hints_enabled=self._TABLE_STRUCTURE_HINTS_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestTableHintsStatsOn(_GPTLatestTableHintsStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestTableHintsStatsOn"


class DataflowSystemGPT52LatestTableHintsStatsOn(_GPTLatestTableHintsStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestTableHintsStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan6: latest-mode raw loader provenance for DataLoading sources.
# These variants isolate a context-compiler rule against LatestStatsOn:
# render compact raw line/block evidence for literal relative source files.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestRawProvenanceStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _RAW_LOADER_PROVENANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            raw_loader_provenance_enabled=self._RAW_LOADER_PROVENANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestRawProvenanceStatsOn(_GPTLatestRawProvenanceStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestRawProvenanceStatsOn"


class DataflowSystemGPT52LatestRawProvenanceStatsOn(_GPTLatestRawProvenanceStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestRawProvenanceStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan7: latest-mode adaptive result digests. These variants isolate a
# result-rendering context rule against LatestStatsOn: keep row previews for
# the latest frontier operators and render stable successful operators as
# shape/column/stat digests.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestAdaptiveDigestStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _RESULT_RENDERING = "adaptive"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            result_rendering=self._RESULT_RENDERING,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestAdaptiveDigestStatsOn(_GPTLatestAdaptiveDigestStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestAdaptiveDigestStatsOn"


class DataflowSystemGPT52LatestAdaptiveDigestStatsOn(_GPTLatestAdaptiveDigestStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestAdaptiveDigestStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan8: latest-mode active-lineage context scope. These variants isolate a
# dataflow-lineage context rule against LatestStatsOn: keep the latest
# frontier and its transitive upstream operators detailed, while unrelated
# branches render as concise operator digests.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestActiveLineageScopeStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _CONTEXT_SCOPE = "active-lineage"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            context_scope=self._CONTEXT_SCOPE,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestActiveLineageScopeStatsOn(_GPTLatestActiveLineageScopeStatsOnVariant):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestActiveLineageScopeStatsOn"


class DataflowSystemGPT52LatestActiveLineageScopeStatsOn(_GPTLatestActiveLineageScopeStatsOnVariant):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestActiveLineageScopeStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan9: latest-mode bounded execution probe guidance. These variants isolate
# a general execution-planning rule against LatestStatsOn: validate expensive
# row-wise, external-model, or broad-join operators on bounded probes before
# scaling to full-table execution.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestBoundedExecutionProbeStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _BOUNDED_EXECUTION_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            bounded_execution_guidance_enabled=self._BOUNDED_EXECUTION_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestBoundedExecutionProbeStatsOn(
    _GPTLatestBoundedExecutionProbeStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestBoundedExecutionProbeStatsOn"


class DataflowSystemGPT52LatestBoundedExecutionProbeStatsOn(
    _GPTLatestBoundedExecutionProbeStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestBoundedExecutionProbeStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan10: latest-mode fallback contract guidance. These variants isolate a
# general runtime-capability rule against LatestStatsOn: dependency/import/
# package failures should either be repaired with an exact equivalent and
# validation evidence, or surfaced as a diagnostic DataFrame rather than
# answered through an unvalidated proxy.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestFallbackContractStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _FALLBACK_CONTRACT_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            fallback_contract_guidance_enabled=self._FALLBACK_CONTRACT_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestFallbackContractStatsOn(
    _GPTLatestFallbackContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestFallbackContractStatsOn"


class DataflowSystemGPT52LatestFallbackContractStatsOn(
    _GPTLatestFallbackContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestFallbackContractStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan11: latest-mode cardinality-pressure guidance. These variants isolate a
# general context-compiler rule against LatestStatsOn: when typed intermediate
# output shape is large/wide, surface it as execution-risk evidence before
# downstream full-table row-wise calls, broad joins, or simulations.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestCardinalityPressureStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _CARDINALITY_PRESSURE_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            cardinality_pressure_guidance_enabled=self._CARDINALITY_PRESSURE_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestCardinalityPressureStatsOn(
    _GPTLatestCardinalityPressureStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestCardinalityPressureStatsOn"


class DataflowSystemGPT52LatestCardinalityPressureStatsOn(
    _GPTLatestCardinalityPressureStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestCardinalityPressureStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan12: latest-mode entity-key hygiene guidance. These variants isolate a
# general context-compiler rule against LatestStatsOn: when typed column stats
# expose high-cardinality string fields, surface them as candidate entity-key
# evidence before grouping, joining, deduplicating, or distinct counting.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestEntityKeyHygieneStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _ENTITY_KEY_HYGIENE_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            entity_key_hygiene_guidance_enabled=self._ENTITY_KEY_HYGIENE_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestEntityKeyHygieneStatsOn(
    _GPTLatestEntityKeyHygieneStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestEntityKeyHygieneStatsOn"


class DataflowSystemGPT52LatestEntityKeyHygieneStatsOn(
    _GPTLatestEntityKeyHygieneStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestEntityKeyHygieneStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan13: latest-mode component-grain guidance. These variants isolate a
# general context-compiler rule against LatestStatsOn: high-cardinality string
# labels may encode base entities plus sub-entity, sampling-location, suffix,
# prefix, or other component grains, so entity counts should compare
# whole-label and component-derived candidate keys before finalizing.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestComponentGrainStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _COMPONENT_GRAIN_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            component_grain_guidance_enabled=self._COMPONENT_GRAIN_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestComponentGrainStatsOn(
    _GPTLatestComponentGrainStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestComponentGrainStatsOn"


class DataflowSystemGPT52LatestComponentGrainStatsOn(
    _GPTLatestComponentGrainStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestComponentGrainStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan14: latest-mode key-grain comparison guidance. These variants isolate a
# stricter general context-compiler rule against LatestStatsOn: when
# high-cardinality string labels expose multiple plausible entity grains, final
# entity counts should be preceded by an explicit candidate-key comparison
# table rather than a single un-audited normalization choice.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestKeyGrainComparisonStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _KEY_GRAIN_COMPARISON_GUIDANCE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            key_grain_comparison_guidance_enabled=self._KEY_GRAIN_COMPARISON_GUIDANCE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestKeyGrainComparisonStatsOn(
    _GPTLatestKeyGrainComparisonStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestKeyGrainComparisonStatsOn"


class DataflowSystemGPT52LatestKeyGrainComparisonStatsOn(
    _GPTLatestKeyGrainComparisonStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestKeyGrainComparisonStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan15: execution-safe operator ID guidance. These variants isolate a
# general harness contract fix: CODE-mode operator IDs must remain valid
# Python parameters while also avoiding logicalOpId characters that workflow
# execution persistence rejects.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestExecutionSafeIdsStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _EXECUTION_SAFE_OPERATOR_IDS_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            execution_safe_operator_ids_enabled=self._EXECUTION_SAFE_OPERATOR_IDS_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestExecutionSafeIdsStatsOn(
    _GPTLatestExecutionSafeIdsStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestExecutionSafeIdsStatsOn"


class DataflowSystemGPT52LatestExecutionSafeIdsStatsOn(
    _GPTLatestExecutionSafeIdsStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestExecutionSafeIdsStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan16: execution-evidence cadence guard. These variants isolate a general
# harness rule: CODE-mode edits should be followed by explicit result
# inspection before the agent creates more parallel/final variants.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestExecutionCadenceStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _MAX_UNEXECUTED_CODE_EDITS = 3

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            max_unexecuted_code_edits=self._MAX_UNEXECUTED_CODE_EDITS,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestExecutionCadenceStatsOn(
    _GPTLatestExecutionCadenceStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestExecutionCadenceStatsOn"


class DataflowSystemGPT52LatestExecutionCadenceStatsOn(
    _GPTLatestExecutionCadenceStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestExecutionCadenceStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan17: key-grain evidence contract in LATEST. These variants isolate a
# general context-compiler rule: ambiguous high-cardinality entity labels are
# not just a prompt hint; they create an auditable evidence contract that is
# satisfied only by an executed candidate-key table with the required schema.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestKeyGrainEvidenceContractStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestKeyGrainEvidenceContractStatsOn(
    _GPTLatestKeyGrainEvidenceContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestKeyGrainEvidenceContractStatsOn"


class DataflowSystemGPT52LatestKeyGrainEvidenceContractStatsOn(
    _GPTLatestKeyGrainEvidenceContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestKeyGrainEvidenceContractStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan18: label-component profile contract in LATEST. These variants build on
# the Plan17 key-grain evidence contract with a general context-compiler rule:
# high-cardinality labels need an all-value profile of observed component
# structure before the model selects which candidate key grains to compare.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestLabelComponentProfileContractStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True
    _LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            label_component_profile_contract_enabled=self._LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestLabelComponentProfileContractStatsOn(
    _GPTLatestLabelComponentProfileContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestLabelComponentProfileContractStatsOn"


class DataflowSystemGPT52LatestLabelComponentProfileContractStatsOn(
    _GPTLatestLabelComponentProfileContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestLabelComponentProfileContractStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan19: candidate-selection impact contract in LATEST. These variants build
# on Plan17/18 evidence artifacts with a general dataflow rule: the selected
# entity key remains provisional until candidate keys are compared against the
# downstream entity-level predicate or final measure in an executed table.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestCandidateSelectionImpactContractStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True
    _LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED = True
    _CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            label_component_profile_contract_enabled=self._LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED,
            candidate_selection_impact_contract_enabled=self._CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestCandidateSelectionImpactContractStatsOn(
    _GPTLatestCandidateSelectionImpactContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestCandidateSelectionImpactContractStatsOn"


class DataflowSystemGPT52LatestCandidateSelectionImpactContractStatsOn(
    _GPTLatestCandidateSelectionImpactContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestCandidateSelectionImpactContractStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan20: observed component inventory contract in LATEST. These variants
# require candidate generation to begin from a data-driven inventory of
# observed label component/separator structure before Plan18/17/19 evidence
# artifacts use or compare component-derived keys.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestObservedComponentInventoryContractStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED = True
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True
    _LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED = True
    _CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            observed_component_inventory_contract_enabled=self._OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            label_component_profile_contract_enabled=self._LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED,
            candidate_selection_impact_contract_enabled=self._CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestObservedComponentInventoryContractStatsOn(
    _GPTLatestObservedComponentInventoryContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestObservedComponentInventoryContractStatsOn"


class DataflowSystemGPT52LatestObservedComponentInventoryContractStatsOn(
    _GPTLatestObservedComponentInventoryContractStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestObservedComponentInventoryContractStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan21: evidence dependency gate in LATEST. These variants build on the
# typed evidence contracts and require the workflow DAG to prove each evidence
# artifact consumes its prerequisite artifact through dataflow links.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestEvidenceDependencyGateStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED = True
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True
    _LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED = True
    _CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED = True
    _EVIDENCE_DEPENDENCY_GATE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            observed_component_inventory_contract_enabled=self._OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            label_component_profile_contract_enabled=self._LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED,
            candidate_selection_impact_contract_enabled=self._CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED,
            evidence_dependency_gate_enabled=self._EVIDENCE_DEPENDENCY_GATE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestEvidenceDependencyGateStatsOn(
    _GPTLatestEvidenceDependencyGateStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestEvidenceDependencyGateStatsOn"


class DataflowSystemGPT52LatestEvidenceDependencyGateStatsOn(
    _GPTLatestEvidenceDependencyGateStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestEvidenceDependencyGateStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan22: data-discovered component inventory in LATEST. These variants keep
# the Plan21 typed dependency graph, but strengthen observed inventory evidence
# so it must expose candidate tokens/transforms discovered from data values.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestDataDiscoveredComponentInventoryStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED = True
    _DATA_DISCOVERED_COMPONENT_INVENTORY_CONTRACT_ENABLED = True
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True
    _LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED = True
    _CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED = True
    _EVIDENCE_DEPENDENCY_GATE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            observed_component_inventory_contract_enabled=self._OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED,
            data_discovered_component_inventory_contract_enabled=self._DATA_DISCOVERED_COMPONENT_INVENTORY_CONTRACT_ENABLED,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            label_component_profile_contract_enabled=self._LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED,
            candidate_selection_impact_contract_enabled=self._CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED,
            evidence_dependency_gate_enabled=self._EVIDENCE_DEPENDENCY_GATE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestDataDiscoveredComponentInventoryStatsOn(
    _GPTLatestDataDiscoveredComponentInventoryStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestDataDiscoveredComponentInventoryStatsOn"


class DataflowSystemGPT52LatestDataDiscoveredComponentInventoryStatsOn(
    _GPTLatestDataDiscoveredComponentInventoryStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestDataDiscoveredComponentInventoryStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan23: boundary-token inventory completeness in LATEST. These variants
# keep the typed Plan22 dependency chain, but require observed inventory to
# prove broad boundary-token enumeration and downstream candidate coverage.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestBoundaryTokenInventoryStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED = True
    _DATA_DISCOVERED_COMPONENT_INVENTORY_CONTRACT_ENABLED = True
    _BOUNDARY_TOKEN_INVENTORY_CONTRACT_ENABLED = True
    _KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED = True
    _LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED = True
    _CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED = True
    _EVIDENCE_DEPENDENCY_GATE_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            observed_component_inventory_contract_enabled=self._OBSERVED_COMPONENT_INVENTORY_CONTRACT_ENABLED,
            data_discovered_component_inventory_contract_enabled=self._DATA_DISCOVERED_COMPONENT_INVENTORY_CONTRACT_ENABLED,
            boundary_token_inventory_contract_enabled=self._BOUNDARY_TOKEN_INVENTORY_CONTRACT_ENABLED,
            key_grain_evidence_contract_enabled=self._KEY_GRAIN_EVIDENCE_CONTRACT_ENABLED,
            label_component_profile_contract_enabled=self._LABEL_COMPONENT_PROFILE_CONTRACT_ENABLED,
            candidate_selection_impact_contract_enabled=self._CANDIDATE_SELECTION_IMPACT_CONTRACT_ENABLED,
            evidence_dependency_gate_enabled=self._EVIDENCE_DEPENDENCY_GATE_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestBoundaryTokenInventoryStatsOn(
    _GPTLatestBoundaryTokenInventoryStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestBoundaryTokenInventoryStatsOn"


class DataflowSystemGPT52LatestBoundaryTokenInventoryStatsOn(
    _GPTLatestBoundaryTokenInventoryStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestBoundaryTokenInventoryStatsOn"


# ────────────────────────────────────────────────────────────────────────
# plan24: flow-progress digest in LATEST. These variants isolate compact
# ReAct/dataflow progress memory without enabling FULL/DELTA history.
# ────────────────────────────────────────────────────────────────────────


class _GPTLatestFlowProgressDigestStatsOnVariant(_GPTStatsOnVariant):
    _CONTEXT_MODE = "latest"
    _FLOW_PROGRESS_DIGEST_ENABLED = True

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            verbose=verbose,
            flow_progress_digest_enabled=self._FLOW_PROGRESS_DIGEST_ENABLED,
            *args,
            **kwargs,
        )


class DataflowSystemGPT5MiniLatestFlowProgressDigestStatsOn(
    _GPTLatestFlowProgressDigestStatsOnVariant
):
    _MODEL_TYPE = "gpt-5-mini"
    _NAME = "DataflowSystemGPT5MiniLatestFlowProgressDigestStatsOn"


class DataflowSystemGPT52LatestFlowProgressDigestStatsOn(
    _GPTLatestFlowProgressDigestStatsOnVariant
):
    _MODEL_TYPE = "gpt-5.2"
    _NAME = "DataflowSystemGPT52LatestFlowProgressDigestStatsOn"


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

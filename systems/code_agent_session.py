# -*- coding: utf-8 -*-
"""
Code Agent Session Runner for session-based benchmarking.

This module wraps CodeAgentWrapper to support multi-query sessions
where agent state is maintained between queries.
"""

import fnmatch
import glob
import json
import os
from typing import Any, Optional, List

from benchmark.session_runner import SessionRunner
from code_agent import CodeAgentWrapper, AUTHORIZED_IMPORTS

# Default settings - use o4-mini by default for session benchmarks
DEFAULT_MODEL_TYPE = os.environ.get("CODE_AGENT_MODEL", "o4-mini")
DEFAULT_MAX_STEPS = int(os.environ.get("CODE_AGENT_MAX_STEPS", 50))
DEFAULT_API_BASE = os.environ.get("CODE_AGENT_API_BASE", "http://localhost:9096/api")
DEFAULT_API_KEY = os.environ.get("CODE_AGENT_API_KEY", "dummy")


class CodeAgentSessionRunner(SessionRunner):
    """
    Session runner using CodeAgentWrapper.

    Maintains agent state between queries within a session by NOT calling
    reset() between queries. Each session starts with a fresh agent.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        max_steps: int = DEFAULT_MAX_STEPS,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        authorized_imports: Optional[List[str]] = None,
        verbosity_level: int = 1,
        dataset_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the session runner.

        Args:
            model_type: LLM model type to use
            max_steps: Maximum steps per query
            api_base: API endpoint for the model
            api_key: API key for authentication
            authorized_imports: List of allowed imports
            verbosity_level: Logging verbosity (0=quiet, 1=normal, 2=verbose)
            dataset_directory: Path to dataset directory for file access
            **kwargs: Additional arguments passed to CodeAgentWrapper
        """
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.authorized_imports = authorized_imports or AUTHORIZED_IMPORTS
        self.verbosity_level = verbosity_level
        self.dataset_directory = dataset_directory
        self.kwargs = kwargs

        self._agent: Optional[CodeAgentWrapper] = None
        self._session_trace: List[dict] = []
        self._is_first_query: bool = True

    def setup(self) -> None:
        """
        Initialize agent for a new session.

        Creates a fresh CodeAgentWrapper instance.
        """
        self._session_trace = []
        self._is_first_query = True

        self._agent = CodeAgentWrapper(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            api_key=self.api_key,
            authorized_imports=self.authorized_imports,
            verbosity_level=self.verbosity_level,
        )
        self._agent.setup()

    def _expand_data_sources(self, data_sources: List[str]) -> List[str]:
        """
        Expand wildcard patterns in data_sources to actual file paths.

        Handles:
        - Wildcards like "State MSA Identity Theft Data/*" or "file-*.csv"
        - Empty string or "./" meaning all files
        - Directory paths ending with "/"
        - Fuzzy names like "Constitution Beach" matching "constitution_beach_datasheet.csv"

        Args:
            data_sources: List of file patterns (may contain wildcards)

        Returns:
            List of actual file paths (relative to current working directory)
        """
        if not self.dataset_directory:
            return []

        # Get all files in dataset directory recursively
        all_files = []
        for root, _, files in os.walk(self.dataset_directory):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, self.dataset_directory)
                all_files.append(rel_path)

        # Handle empty data_sources or special "all files" patterns
        if not data_sources:
            return []

        expanded_paths = []

        for pattern in data_sources:
            # Handle empty string or "./" as "all files"
            if pattern == "" or pattern == "./" or pattern == ".":
                expanded_paths.extend(all_files)
                continue

            # Handle directory paths ending with "/" - get all files in that directory
            if pattern.endswith('/'):
                dir_pattern = pattern.rstrip('/')
                for f in all_files:
                    if f.startswith(dir_pattern + '/') or dir_pattern in f:
                        expanded_paths.append(f)
                continue

            # Check if pattern contains wildcards
            if '*' in pattern or '?' in pattern:
                # Try to match against all files using fnmatch (case-insensitive)
                matched = []
                pattern_lower = pattern.lower()
                pattern_dir_lower = os.path.dirname(pattern).lower()
                pattern_base_lower = os.path.basename(pattern).lower()

                for f in all_files:
                    f_lower = f.lower()
                    # Match against full relative path (case-insensitive)
                    if fnmatch.fnmatch(f_lower, pattern_lower) or fnmatch.fnmatch(f_lower, f"**/{pattern_lower}"):
                        matched.append(f)
                    # Also try matching basename against pattern's basename (case-insensitive)
                    elif fnmatch.fnmatch(os.path.basename(f_lower), pattern_base_lower):
                        # Check if parent directory matches too (case-insensitive)
                        if not pattern_dir_lower or pattern_dir_lower in f_lower:
                            matched.append(f)

                if matched:
                    expanded_paths.extend(matched)
                else:
                    # Fallback: try glob with recursive search
                    glob_pattern = os.path.join(self.dataset_directory, "**", pattern)
                    glob_matches = glob.glob(glob_pattern, recursive=True)
                    for match in glob_matches:
                        expanded_paths.append(os.path.relpath(match, self.dataset_directory))

                    if not glob_matches:
                        print(f"WARNING: No files matched pattern '{pattern}'")
            else:
                # No wildcards - treat as exact path or fuzzy match
                exact_path = os.path.join(self.dataset_directory, pattern)
                if os.path.exists(exact_path):
                    # Check if it's a directory
                    if os.path.isdir(exact_path):
                        for f in all_files:
                            if f.startswith(pattern + '/') or f.startswith(pattern + os.sep):
                                expanded_paths.append(f)
                    else:
                        expanded_paths.append(pattern)
                else:
                    # Search for file anywhere in dataset with fuzzy matching
                    found = False
                    # Normalize pattern for fuzzy matching (e.g., "Constitution Beach" -> "constitution_beach")
                    pattern_normalized = pattern.lower().replace(' ', '_').replace('-', '_')

                    for f in all_files:
                        # Exact suffix match
                        if f.endswith(pattern) or os.path.basename(f) == pattern:
                            expanded_paths.append(f)
                            found = True
                        # Fuzzy match: check if normalized pattern is in the file path
                        elif pattern_normalized in f.lower().replace(' ', '_').replace('-', '_'):
                            expanded_paths.append(f)
                            found = True

                    if not found:
                        print(f"WARNING: File not found '{pattern}'")

        # Convert to paths relative to cwd for agent use
        result = []
        for p in expanded_paths:
            full_path = os.path.join(self.dataset_directory, p)
            result.append(os.path.relpath(full_path))

        return list(set(result))  # Remove duplicates

    def _build_prompt(self, query: str, data_sources: List[str], is_first: bool = False) -> str:
        """
        Build a prompt with file context.

        Args:
            query: The query/question
            data_sources: List of relative file paths for this query
            is_first: Whether this is the first query in the session

        Returns:
            Formatted prompt string
        """
        # Expand wildcards and build file paths
        file_paths = self._expand_data_sources(data_sources)

        if is_first and file_paths:
            # First query gets full context
            prompt = f"""You are a data scientist working on a multi-step analysis task.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Question: {query}

Instructions:
1. Read the relevant data files using the provided paths.
2. Answer the question step by step.
3. Your final answer should be on the last line in the format: **Final Answer: <value>**
   - For numeric questions: just the number (e.g., "15")
   - For list questions: a JSON array (e.g., ["Tokyo", "London"])
   - For string questions: just the value (e.g., "file.txt")"""
        elif file_paths:
            # Subsequent queries mention available files
            prompt = f"""Continue the analysis. Additional files available if needed:
{json.dumps(file_paths, indent=2)}

Question: {query}

Provide your answer on the last line as: **Final Answer: <value>**"""
        else:
            # No specific files - just the query
            prompt = f"""{query}

Provide your answer on the last line as: **Final Answer: <value>**"""

        return prompt

    def run_query(self, query: str, data_sources: Optional[List[str]] = None) -> str:
        """
        Run a single query, maintaining session state.

        IMPORTANT: This method does NOT reset the agent between queries.
        The agent's memory and state persist across queries within a session.

        Args:
            query: The query/question to process
            data_sources: Optional list of file paths for this query

        Returns:
            The agent's response as a string
        """
        if self._agent is None:
            raise RuntimeError("Agent not set up. Call setup() first.")

        # Build prompt with file context
        prompt = self._build_prompt(query, data_sources or [], is_first=self._is_first_query)
        self._is_first_query = False

        # Run the query (agent maintains state)
        result = self._agent.run(prompt)

        # Accumulate trace
        if result.reasoning_trace:
            self._session_trace.extend(result.reasoning_trace)

        return result.response

    def get_reasoning_trace(self) -> List[dict]:
        """
        Get the accumulated reasoning trace for the session.

        Returns:
            List of trace entries from all queries in the session
        """
        return self._session_trace

    def cleanup(self) -> None:
        """
        Cleanup agent resources.
        """
        if self._agent:
            self._agent.cleanup()
            self._agent = None


class CodeAgentSessionSystem:
    """
    System wrapper for session-based CodeAgent benchmarking.

    This class provides a System-like interface for integration with
    the existing KramaBench infrastructure.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        max_steps: int = DEFAULT_MAX_STEPS,
        api_base: str = DEFAULT_API_BASE,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the system.

        Args:
            model_type: LLM model type to use
            max_steps: Maximum steps per query
            api_base: API endpoint
            verbose: Enable verbose logging
            **kwargs: Additional arguments
        """
        self.name = f"CodeAgentSession_{model_type}"
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.verbose = verbose
        self.kwargs = kwargs

        self.dataset_directory: Optional[str] = None

    def process_dataset(self, dataset_directory: str) -> None:
        """
        Set the dataset directory.

        For CodeAgent, this just stores the path - the agent handles
        file access directly.
        """
        self.dataset_directory = dataset_directory

    def create_runner(self) -> CodeAgentSessionRunner:
        """
        Create a new session runner instance.

        Returns:
            CodeAgentSessionRunner configured with this system's settings
        """
        return CodeAgentSessionRunner(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            verbosity_level=2 if self.verbose else 1,
            dataset_directory=self.dataset_directory,
            **self.kwargs
        )

# -*- coding: utf-8 -*-
"""
CodeAgentSystem - KramaBench System wrapper for smolagents CodeAgent.
"""

import fnmatch
import glob
import os
import json
from typing import Dict, List, Optional

from benchmark.benchmark_api import System
from code_agent import CodeAgentWrapper, CodeAgentResult


# Default max steps (can be overridden by CODE_AGENT_MAX_STEPS env var)
DEFAULT_MAX_STEPS = int(os.environ.get("CODE_AGENT_MAX_STEPS", 50))


class CodeAgentSystem(System):
    """KramaBench System using smolagents CodeAgent."""

    def __init__(
        self,
        model_type: str = "claude-haiku-4.5",
        max_steps: int = DEFAULT_MAX_STEPS,
        api_base: str = "http://localhost:9096/api",
        api_key: str = "dummy",
        verbose: bool = False,
        name: str = "CodeAgentSystem",
        *args, **kwargs
    ):
        super().__init__(name, verbose=verbose, *args, **kwargs)
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.agent: Optional[CodeAgentWrapper] = None
        self.output_dir = f"./system_scratch/{name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """Process the dataset directory."""
        self.dataset_directory = dataset_directory
        self.dataset = {}
        for dirpath, _, filenames in os.walk(dataset_directory):
            for fname in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, fname), dataset_directory)
                self.dataset[rel_path] = None

        if self.verbose:
            print(f"[{self.name}] Found {len(self.dataset)} files in {dataset_directory}")

        # Setup agent
        self.agent = CodeAgentWrapper(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            api_key=self.api_key,
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()

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

        all_files = list(self.dataset.keys())

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

    def serve_query(
        self,
        query: str,
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None
    ) -> Dict:
        """Serve a query using the CodeAgent."""
        if not self.agent:
            raise RuntimeError("Call process_dataset() first.")

        # Expand wildcards and build file paths
        if subset_files:
            file_paths = self._expand_data_sources(subset_files)
        else:
            file_paths = [os.path.relpath(os.path.join(self.dataset_directory, f)) for f in self.dataset.keys()]

        if self.verbose:
            print(f"[{self.name}] Query: {query_id}, Files: {len(file_paths)}")

        # Build prompt
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Question: {query}

Instructions:
1. Read the relevant data files using the provided paths and analyze the data.
2. Compute the answer step by step.
3. IMPORTANT - Your final answer format:
   - For numeric questions: output just the number (e.g., "274")
   - For list questions: output a JSON array (e.g., ["Tokyo", "London", "Paris"])
   - For descriptive/analytical questions: output a complete sentence summarizing your findings (e.g., "The average period is 11 years, with maxima in 1968, 1979, 1989, 2000, and 2014.")
   - For simple string questions: output just the value (e.g., "California")

Example final answers:
- Numeric: "274"
- List: ["Tokyo", "London", "Paris"]
- Descriptive: "The correlation coefficient is 0.85, indicating a strong positive relationship between temperature and sales."
- String: "California"
"""

        # Save outputs
        query_dir = os.path.join(self.output_dir, query_id)
        os.makedirs(query_dir, exist_ok=True)
        with open(os.path.join(query_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        # Reset and run
        self.agent.reset()
        result: CodeAgentResult = self.agent.run(prompt)

        # Save results
        with open(os.path.join(query_dir, "response.txt"), "w") as f:
            f.write(result.response or "(empty)")
        with open(os.path.join(query_dir, "reasoning_trace.json"), "w") as f:
            json.dump(result.reasoning_trace, f, indent=2, default=str)

        # Parse answer
        answer = self._parse_answer(result.response)
        with open(os.path.join(query_dir, "answer.json"), "w") as f:
            json.dump({"answer": answer}, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] Answer: {answer}, Steps: {len(result.reasoning_trace)}, Time: {result.elapsed_seconds:.1f}s")

        return {
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": "",
            "token_usage": 0,
            "token_usage_input": 0,
            "token_usage_output": 0,
        }

    def _parse_answer(self, response: str) -> str:
        """Extract answer from response."""
        if not response:
            return "No response"

        response = str(response).strip()

        # JSON array
        if response.startswith('[') and response.endswith(']'):
            return response

        # Number
        try:
            float(response.replace(',', ''))
            return response
        except ValueError:
            pass

        # Short string
        if len(response) < 100 and '\n' not in response:
            return response

        # Last line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[-1] if lines else response

    def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.agent:
            self.agent.cleanup()
            self.agent = None


# Pre-configured variants
class CodeAgentSystemHaiku(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="claude-haiku-4.5", name="CodeAgentSystemHaiku", verbose=verbose, *args, **kwargs)


class CodeAgentSystemSonnet(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="claude-sonnet-4-5", name="CodeAgentSystemSonnet", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGPT(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="gpt-5-mini", name="CodeAgentSystemGPT", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGptO3(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="o3", name="CodeAgentSystemGptO3", verbose=verbose, *args, **kwargs)


class CodeAgentSystemSonnet4(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="claude-sonnet-4", name="CodeAgentSystemSonnet4", verbose=verbose, *args, **kwargs)


class CodeAgentSystemHaiku45(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="claude-haiku-4.5", name="CodeAgentSystemHaiku45", verbose=verbose, *args, **kwargs)


class CodeAgentSystemO4Mini(CodeAgentSystem):
    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="o4-mini", name="CodeAgentSystemO4Mini", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGemini25Pro(CodeAgentSystem):
    """CodeAgentSystem using Google Gemini 2.5 Pro model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="gemini-2.5-pro", name="CodeAgentSystemGemini25Pro", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGpt52(CodeAgentSystem):
    """CodeAgentSystem using GPT-5.2 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="gpt-5.2", name="CodeAgentSystemGpt52", verbose=verbose, *args, **kwargs)

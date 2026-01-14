# -*- coding: utf-8 -*-
"""
CodeAgentSystem - KramaBench System wrapper for smolagents CodeAgent.
"""

import os
import json
import re
import shutil
import time
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
        self.symlink_path: Optional[str] = None  # Shortened path via symlink
        self.symlink_base: Optional[str] = None  # Base directory for symlink cleanup
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
            print(f"[{self.name}] Found {len(self.dataset)} files")

        # Create symlink for shorter paths in prompts
        self._create_symlink(dataset_directory)

        # Setup agent
        self.agent = CodeAgentWrapper(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            api_key=self.api_key,
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()

    def _create_symlink(self, dataset_directory: str) -> None:
        """
        Create a symlink in /tmp for shorter file paths in prompts.

        This reduces token usage by replacing long absolute paths like:
        /Users/.../KramaBench/data/astronomy/input/file.csv
        with shorter paths like:
        /tmp/krama_bench/{timestamp}/astronomy/file.csv

        Each run gets a unique timestamp to avoid conflicts with concurrent runs.
        """
        try:
            # Extract domain name from path (e.g., "astronomy" from ".../data/astronomy/input")
            abs_path = os.path.abspath(dataset_directory)
            parts = abs_path.rstrip('/').split('/')

            # Find domain: look for "data" folder and get next part
            domain = "data"
            if 'data' in parts:
                data_idx = parts.index('data')
                if data_idx + 1 < len(parts):
                    domain = parts[data_idx + 1]

            # Create unique symlink directory with timestamp (milliseconds since epoch)
            timestamp = int(time.time() * 1000)
            symlink_base = f"/tmp/krama_bench/{timestamp}"
            os.makedirs(symlink_base, exist_ok=True)

            symlink_path = os.path.join(symlink_base, domain)

            # Create symlink (no need to check for existing since timestamp is unique)
            os.symlink(abs_path, symlink_path)
            self.symlink_path = symlink_path
            self.symlink_base = symlink_base  # Store base for cleanup

            if self.verbose:
                print(f"[{self.name}] Created symlink: {symlink_path} -> {abs_path}")

        except Exception as e:
            # Fall back to original paths if symlink creation fails
            if self.verbose:
                print(f"[{self.name}] Could not create symlink, using original paths: {e}")
            self.symlink_path = None
            self.symlink_base = None

    def serve_query(
        self,
        query: str,
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None
    ) -> Dict:
        """Serve a query using the CodeAgent."""
        if not self.agent:
            raise RuntimeError("Call process_dataset() first.")

        # Use symlink path for shorter prompts if available
        base_path = self.symlink_path if self.symlink_path else self.dataset_directory

        # Build file paths
        if subset_files:
            file_paths = [os.path.join(base_path, f) for f in subset_files]
        else:
            file_paths = [os.path.join(base_path, f) for f in self.dataset.keys()]

        if self.verbose:
            print(f"[{self.name}] Query: {query_id}, Files: {len(file_paths)} (base: {base_path})")

        # Build prompt
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these absolute paths):
{json.dumps(file_paths, indent=2)}

Question: {query}

Instructions:
1. Read the relevant data files using open() or pandas.read_csv() etc.
2. Compute the answer step by step.
3. IMPORTANT - Your final answer format:
   - For numeric questions: output just the number (e.g., "274")
   - For list questions: output a JSON array (e.g., ["Tokyo", "London", "Paris"])
   - For descriptive/analytical questions: output a complete sentence summarizing your findings (e.g., "The average period is 11 years, with maxima in 1968, 1979, 1989, 2000, and 2014.")
   - For simple string questions: output just the value (e.g., "California")
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
        """Cleanup resources and symlinks."""
        if self.agent:
            self.agent.cleanup()
            self.agent = None

        # Remove symlink directory (entire timestamp-based directory)
        if self.symlink_base and os.path.exists(self.symlink_base):
            try:
                shutil.rmtree(self.symlink_base)
                if self.verbose:
                    print(f"[{self.name}] Removed symlink directory: {self.symlink_base}")
            except Exception as e:
                if self.verbose:
                    print(f"[{self.name}] Could not remove symlink directory: {e}")
            self.symlink_path = None
            self.symlink_base = None


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

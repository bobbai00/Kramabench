# -*- coding: utf-8 -*-
"""
CodeAgentSystem - KramaBench System wrapper for smolagents CodeAgent.
"""

import os
import json
import re
from typing import Dict, List, Optional

from benchmark.benchmark_api import System
from code_agent import CodeAgentWrapper, CodeAgentResult


class CodeAgentSystem(System):
    """KramaBench System using smolagents CodeAgent."""

    def __init__(
        self,
        model_type: str = "claude-haiku-4.5",
        max_steps: int = 50,
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
            print(f"[{self.name}] Found {len(self.dataset)} files")

        # Setup agent
        self.agent = CodeAgentWrapper(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            api_key=self.api_key,
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()

    def serve_query(
        self,
        query: str,
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None
    ) -> Dict:
        """Serve a query using the CodeAgent."""
        if not self.agent:
            raise RuntimeError("Call process_dataset() first.")

        # Build file paths
        if subset_files:
            file_paths = [os.path.join(self.dataset_directory, f) for f in subset_files]
        else:
            file_paths = [os.path.join(self.dataset_directory, f) for f in self.dataset.keys()]

        if self.verbose:
            print(f"[{self.name}] Query: {query_id}, Files: {len(file_paths)}")

        # Build prompt
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these absolute paths):
{json.dumps(file_paths, indent=2)}

Question: {query}

Instructions:
1. Read the relevant data files using open() or pandas.read_csv() etc.
2. Compute the answer step by step.
3. Your final answer should be just the value (number, string, or list).
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
        """Cleanup resources."""
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

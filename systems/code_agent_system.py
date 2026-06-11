# -*- coding: utf-8 -*-
"""
CodeAgentSystem - KramaBench System wrapper for smolagents CodeAgent.
"""

import os
import json
from typing import Dict, List, Optional

from benchmark.benchmark_api import System
from code_agent import CodeAgentWrapper, CodeAgentResult
from systems.data_source_utils import expand_data_sources
from utils.answer_parser import parse_answer


# Default max steps (can be overridden by CODE_AGENT_MAX_STEPS env var)
DEFAULT_MAX_STEPS = int(os.environ.get("CODE_AGENT_MAX_STEPS", 50))


class CodeAgentSystem(System):
    """KramaBench System using smolagents CodeAgent."""

    def __init__(
        self,
        model_type: str = "claude-haiku-4.5",
        max_steps: int = DEFAULT_MAX_STEPS,
        api_base: str = "http://localhost:4000",
        api_key: str = "dummy",
        verbose: bool = False,
        name: str = "CodeAgentSystem",
        use_fine_grained_prompt: bool = None,
        no_action_detail: bool = False,
        *args, **kwargs
    ):
        super().__init__(name, verbose=verbose, *args, **kwargs)
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.use_fine_grained_prompt = use_fine_grained_prompt
        self.no_action_detail = no_action_detail
        self.agent: Optional[CodeAgentWrapper] = None
        self.output_dir = f"./system_scratch/{name}"
        self.format_hints: Dict[str, str] = {}  # Map task_id -> format_hint string
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

        # Load format hints
        self._load_format_hints(dataset_directory)

        # Setup agent
        self.agent = CodeAgentWrapper(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            api_key=self.api_key,
            verbosity_level=2 if self.verbose else 1,
            use_fine_grained_prompt=self.use_fine_grained_prompt,
            no_action_detail=self.no_action_detail,
        )
        self.agent.setup()

    def _load_format_hints(self, dataset_directory: str) -> None:
        """Load format hints for the domain."""
        try:
            parts = str(dataset_directory).rstrip('/').split('/')
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
                        print(f"[{self.name}] Loaded {len(hints)} format hints from {hint_path}")
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] Could not load format hints: {e}")

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
        """Serve a query using the CodeAgent."""
        if not self.agent:
            raise RuntimeError("Call process_dataset() first.")

        # Expand wildcards and build file paths
        if subset_files:
            file_paths = self._expand_data_sources(subset_files)
        else:
            # Use a recursive wildcard instead of listing every file
            file_paths = [os.path.relpath(self.dataset_directory) + "/**/*"]

        if self.verbose:
            print(f"[{self.name}] Query: {query_id}, Files: {len(file_paths)}")

        # Build prompt
        format_hint = self.format_hints.get(query_id, "")
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Note: All paths are relative. Some paths may contain wildcards (e.g., "folder/*" or "file-*.csv"). Use glob patterns to match and read those files.

Question: {query}

Answer format: {format_hint}

Your last line MUST BE: **Final Answer: <value>**"""

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

        # Save stats.json
        stats = {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "cached_tokens": result.cached_tokens,
            "cost_usd": result.cost_usd,
            "num_steps": result.num_steps,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
        }
        with open(os.path.join(query_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        # Parse answer
        answer = parse_answer(result.response)
        with open(os.path.join(query_dir, "answer.json"), "w") as f:
            json.dump({"answer": answer}, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] Answer: {answer}, Steps: {len(result.reasoning_trace)}, Time: {result.elapsed_seconds:.1f}s")

        return {
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": "",
            "token_usage": result.total_tokens,
            "token_usage_input": result.input_tokens,
            "token_usage_output": result.output_tokens,
            "token_usage_reasoning": result.reasoning_tokens,
            "token_usage_cached": result.cached_tokens,
            "cost_usd": result.cost_usd,
        }

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


class CodeAgentSystemGpt52FineGrained(CodeAgentSystem):
    """CodeAgentSystem using GPT-5.2 model with fine-grained (one-line-per-action) prompt."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            name="CodeAgentSystemGpt52FineGrained",
            use_fine_grained_prompt=True,
            verbose=verbose,
            *args, **kwargs
        )


class CodeAgentSystemGpt5MiniHigh(CodeAgentSystem):
    """CodeAgentSystem using GPT-5-mini-high model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="gpt-5-mini-high", name="CodeAgentSystemGpt5MiniHigh", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGpt5MiniMedium(CodeAgentSystem):
    """CodeAgentSystem using GPT-5-mini-medium model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="gpt-5-mini-medium", name="CodeAgentSystemGpt5MiniMedium", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGpt5MiniLow(CodeAgentSystem):
    """CodeAgentSystem using GPT-5-mini-low model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(model_type="gpt-5-mini-low", name="CodeAgentSystemGpt5MiniLow", verbose=verbose, *args, **kwargs)


class CodeAgentSystemGpt52FullInput(CodeAgentSystem):
    """CodeAgentSystem using GPT-5.2 model with full input mode (all dataset files via wildcard)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            name="CodeAgentSystemGpt52FullInput",
            verbose=verbose,
            *args, **kwargs
        )

    def serve_query(self, query, query_id="default-0", subset_files=None):
        """Override to always use full input (ignore subset_files)."""
        return super().serve_query(query, query_id, subset_files=None)


# Model-routing proxy (claude->Anthropic, gpt/o->OpenAI). Code-agent peers that
# hit the SAME upstream as the DataflowSystem so the comparison is apples-to-apples.
PROXY_API_BASE = "http://localhost:8099/v1"


class CodeAgentSystemGpt5MiniProxy(CodeAgentSystem):
    """Code agent on gpt-5-mini via the routing proxy (-> OpenAI). Peer of the
    DataflowSystem gpt-5-mini latest+replay config."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            api_base=PROXY_API_BASE,
            name="CodeAgentSystemGpt5MiniProxy",
            verbose=verbose,
            *args, **kwargs
        )


class CodeAgentSystemGpt54Proxy(CodeAgentSystem):
    """Code agent on gpt-5.4 via the routing proxy (-> OpenAI). Peer of the
    DataflowSystem gpt-5.4 latest+replay config."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.4",
            api_base=PROXY_API_BASE,
            name="CodeAgentSystemGpt54Proxy",
            verbose=verbose,
            *args, **kwargs
        )

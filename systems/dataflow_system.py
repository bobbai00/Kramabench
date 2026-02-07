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
from dataflow_agent import DataflowAgent, MessageResult, get_agent_workflow
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
        max_steps: int = None,
        max_operator_result_char_limit: int = None,
        max_operator_result_cell_char_limit: int = None,
        operator_result_serialization_mode: str = None,
        tool_timeout_seconds: int = None,
        execution_timeout_minutes: int = None,
        agent_mode: str = None,
        fine_grained_prompt: bool = None,
        verbose: bool = False,
        name: str = "DataflowSystem",
        *args,
        **kwargs
    ):
        """
        Initialize the DataflowSystem.

        Parameters can be set via:
        1. Constructor arguments
        2. Environment variables (DATAFLOW_MODEL_TYPE, DATAFLOW_MAX_STEPS, etc.)
        3. Defaults

        Args:
            model_type: LLM model type (env: DATAFLOW_MODEL_TYPE, default: claude-sonnet-4-5)
            max_steps: Maximum steps per query (env: DATAFLOW_MAX_STEPS, default: 100)
            max_operator_result_char_limit: Max chars for operator results (env: DATAFLOW_MAX_RESULT_CHARS, default: 20000)
            max_operator_result_cell_char_limit: Max chars per cell (env: DATAFLOW_MAX_CELL_CHARS, default: 4000)
            operator_result_serialization_mode: Result format (env: DATAFLOW_SERIALIZATION_MODE, default: table)
            tool_timeout_seconds: Tool timeout (env: DATAFLOW_TOOL_TIMEOUT, default: 240)
            execution_timeout_minutes: Execution timeout (env: DATAFLOW_EXEC_TIMEOUT, default: 4)
            agent_mode: Agent mode (env: DATAFLOW_AGENT_MODE, default: code)
            fine_grained_prompt: Use fine-grained prompts (env: DATAFLOW_FINE_GRAINED_PROMPT, default: false)
            verbose: Enable verbose logging
            name: System name for benchmark identification
        """
        super().__init__(name, verbose=verbose, *args, **kwargs)

        # Read from env vars with fallback to defaults
        self.model_type = model_type or os.environ.get("DATAFLOW_MODEL_TYPE", "claude-sonnet-4-5")
        self.max_steps = max_steps or int(os.environ.get("DATAFLOW_MAX_STEPS", "100"))
        self.max_operator_result_char_limit = max_operator_result_char_limit or int(os.environ.get("DATAFLOW_MAX_RESULT_CHARS", "20000"))
        self.max_operator_result_cell_char_limit = max_operator_result_cell_char_limit or int(os.environ.get("DATAFLOW_MAX_CELL_CHARS", "4000"))
        self.operator_result_serialization_mode = operator_result_serialization_mode or os.environ.get("DATAFLOW_SERIALIZATION_MODE", "table")
        self.tool_timeout_seconds = tool_timeout_seconds or int(os.environ.get("DATAFLOW_TOOL_TIMEOUT", "240"))
        self.execution_timeout_minutes = execution_timeout_minutes or int(os.environ.get("DATAFLOW_EXEC_TIMEOUT", "4"))
        self.agent_mode = agent_mode or os.environ.get("DATAFLOW_AGENT_MODE", "code")
        # fine_grained_prompt: if explicitly set use that, otherwise check env var
        if fine_grained_prompt is not None:
            self.fine_grained_prompt = fine_grained_prompt
        else:
            self.fine_grained_prompt = os.environ.get("DATAFLOW_FINE_GRAINED_PROMPT", "false").lower() == "true"

        self.agent: Optional[DataflowAgent] = None
        self.output_dir = kwargs.get("output_dir", f"./system_scratch/{name}")
        self.workload_data: Dict[str, dict] = {}  # Map task_id -> task dict (for ground truth)

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

        # Initialize the agent
        self._setup_agent()

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

    def _setup_agent(self) -> None:
        """Initialize and setup the DataflowAgent."""
        if self.verbose:
            print(f"[DataflowSystem] Setting up agent with model: {self.model_type}")
            print(f"[DataflowSystem] Agent settings: max_steps={self.max_steps}, mode={self.agent_mode}, fine_grained={self.fine_grained_prompt}")

        self.agent = DataflowAgent(
            model_type=self.model_type,
            max_steps=self.max_steps,
            max_operator_result_char_limit=self.max_operator_result_char_limit,
            max_operator_result_cell_char_limit=self.max_operator_result_cell_char_limit,
            operator_result_serialization_mode=self.operator_result_serialization_mode,
            tool_timeout_seconds=self.tool_timeout_seconds,
            execution_timeout_minutes=self.execution_timeout_minutes,
            agent_mode=self.agent_mode,
            fine_grained_prompt=self.fine_grained_prompt,
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()

    def _build_prompt(self, query: str, file_paths: List[str]) -> str:
        """
        Build the prompt for the agent.

        Args:
            query: The natural language query
            file_paths: List of file paths available for the query

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Note: All paths are relative. Some paths may contain wildcards (e.g., "folder/*" or "file-*.csv"). Use glob patterns to match and read those files.

Question: {query}

Instructions:
1. Read the relevant data files using the provided paths and analyze the data. The given data may be raw, and you need to examine carefully and clean the data if needed.
2. Compute the answer step by step.
3. IMPORTANT - Your final answer format:
   - For numeric questions: output just the number (e.g., "274")
   - For list questions: output a JSON array (e.g., ["Tokyo", "London", "Paris"])
   - For descriptive/analytical questions: output a complete sentence summarizing your findings (e.g., "The average period is 11 years, with maxima in 1968, 1979, 1989, 2000, and 2014.")
   - For simple string questions: output just the value (e.g., "California")
4. Numeric format conventions:
   - "percentage" or "rate": output the human-readable value, e.g., 54.03 means 54.03%
   - "proportion" or "fraction": output the decimal, e.g., 0.5403
   - "ratio": output the raw division result, e.g., 2.5 for 5/2

Example final answers:
- Numeric: "274"
- List: ["Tokyo", "London", "Paris"]
- Descriptive: "The correlation coefficient is 0.85, indicating a strong positive relationship between temperature and sales."
- String: "California"

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

        # Build prompt with file paths
        prompt = self._build_prompt(query, file_paths)

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
            "query_id": query_id,
            "dataset_directory": str(self.dataset_directory),
            "num_files": len(file_paths),
            "subset_files": subset_files,
            "agent_settings": {
                "max_steps": self.max_steps,
                "max_operator_result_char_limit": self.max_operator_result_char_limit,
                "max_operator_result_cell_char_limit": self.max_operator_result_cell_char_limit,
                "operator_result_serialization_mode": self.operator_result_serialization_mode,
                "tool_timeout_seconds": self.tool_timeout_seconds,
                "execution_timeout_minutes": self.execution_timeout_minutes,
                "agent_mode": self.agent_mode,
                "fine_grained_prompt": self.fine_grained_prompt,
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

        # Save response and messages for debugging
        response_path = os.path.join(query_output_dir, "response.txt")
        with open(response_path, "w") as f:
            f.write(result.response or "(empty response)")

        messages_path = os.path.join(query_output_dir, "messages.json")
        with open(messages_path, "w") as f:
            json.dump(result.messages, f, indent=2, default=str)

        if self.verbose:
            print(f"[DataflowSystem] Raw response length: {len(result.response) if result.response else 0}")
            print(f"[DataflowSystem] Messages count: {len(result.messages)}")
            print(f"[DataflowSystem] Stopped: {result.stopped}, Error: {result.error}")

        # Extract token usage
        usage = result.usage or {}
        token_usage = usage.get("total_tokens", 0) or usage.get("totalTokens", 0)
        token_usage_input = usage.get("input_tokens", 0) or usage.get("inputTokens", 0)
        token_usage_output = usage.get("output_tokens", 0) or usage.get("outputTokens", 0)

        # Count steps (number of assistant turns with tool calls)
        num_steps = 0
        if result.messages:
            for msg in result.messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    # Count as a step if it has tool-call or tool_use
                    if isinstance(content, list):
                        has_tool_call = any(
                            isinstance(item, dict) and item.get("type") in ("tool-call", "tool_use")
                            for item in content
                        )
                        if has_tool_call:
                            num_steps += 1
                    elif content:
                        num_steps += 1

        # Also check stats from agent service if available
        stats_from_service = result.stats or {}
        if stats_from_service.get("steps"):
            num_steps = stats_from_service.get("steps")

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


# Pre-configured variants for different models
class DataflowSystemHaiku(DataflowSystem):
    """DataflowSystem using Claude Haiku 4.5 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            name="DataflowSystemHaiku",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemSonnet(DataflowSystem):
    """DataflowSystem using Claude Sonnet 4.5 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-sonnet-4-5",
            name="DataflowSystemSonnet",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGPT(DataflowSystem):
    """DataflowSystem using GPT-5-mini model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",
            name="DataflowSystemGPT",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemSonnet37(DataflowSystem):
    """DataflowSystem using Claude Sonnet 3.7 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-sonnet-3.7",
            name="DataflowSystemSonnet37",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemSonnet35(DataflowSystem):
    """DataflowSystem using Claude Sonnet 3.5 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-sonnet-3.5",
            name="DataflowSystemSonnet35",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGptO3(DataflowSystem):
    """DataflowSystem using GPT o3 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="o3",
            name="DataflowSystemGptO3",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemSonnet4(DataflowSystem):
    """DataflowSystem using Claude Sonnet 4 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-sonnet-4",
            name="DataflowSystemSonnet4",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemHaiku45(DataflowSystem):
    """DataflowSystem using Claude Haiku 4.5 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="claude-haiku-4.5",
            name="DataflowSystemHaiku45",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemO4Mini(DataflowSystem):
    """DataflowSystem using OpenAI o4-mini model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="o4-mini",
            name="DataflowSystemO4Mini",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGemini25Pro(DataflowSystem):
    """DataflowSystem using Google Gemini 2.5 Pro model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gemini-2.5-pro",
            name="DataflowSystemGemini25Pro",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGpt52(DataflowSystem):
    """DataflowSystem using GPT-5.2 model."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            name="DataflowSystemGpt52",
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGpt52FineGrained(DataflowSystem):
    """DataflowSystem using GPT-5.2 model with fine-grained (one-line-per-action) prompt."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            name="DataflowSystemGpt52FineGrained",
            fine_grained_prompt=True,
            verbose=verbose,
            *args,
            **kwargs
        )


class DataflowSystemGpt52FullInput(DataflowSystem):
    """DataflowSystem using GPT-5.2 model with full input mode (all dataset files via wildcard)."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            name="DataflowSystemGpt52FullInput",
            verbose=verbose,
            *args,
            **kwargs
        )

    def serve_query(self, query, query_id="default-0", subset_files=None):
        """Override to always use full input (ignore subset_files)."""
        return super().serve_query(query, query_id, subset_files=None)

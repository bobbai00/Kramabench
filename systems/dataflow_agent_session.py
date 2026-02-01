# -*- coding: utf-8 -*-
"""
Dataflow Agent Session Runner for session-based benchmarking.

This module wraps DataflowAgent to support multi-query sessions
where agent state (workflow and conversation history) is maintained
between queries.
"""

import json
import os
from typing import Any, Optional, List, Dict

from benchmark.session_runner import SessionRunner
from dataflow_agent import (
    DataflowAgent,
    AGENT_MAX_STEPS,
    TEXERA_API_ENDPOINT,
    TEXERA_COMPUTING_UNIT_ENDPOINT,
    TEXERA_AGENT_SERVICE_ENDPOINT,
    TEXERA_USERNAME,
    TEXERA_PASSWORD,
    get_agent_workflow,
)

# Default settings (can be overridden via environment variables)
# Default to o4-mini for session benchmarks
DEFAULT_MODEL_TYPE = os.environ.get("DATAFLOW_AGENT_MODEL", "o4-mini")
DEFAULT_MAX_STEPS = int(os.environ.get("DATAFLOW_AGENT_MAX_STEPS", AGENT_MAX_STEPS))


class DataflowAgentSessionRunner(SessionRunner):
    """
    Session runner using DataflowAgent.

    Maintains agent state between queries within a session:
    - Workflow operators persist (can be reused across queries)
    - Conversation history persists (agent remembers prior exchanges)

    Each session starts with a fresh agent and workflow.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        max_steps: int = DEFAULT_MAX_STEPS,
        texera_api_endpoint: str = TEXERA_API_ENDPOINT,
        computing_unit_endpoint: str = TEXERA_COMPUTING_UNIT_ENDPOINT,
        agent_service_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
        username: str = TEXERA_USERNAME,
        password: str = TEXERA_PASSWORD,
        verbosity_level: int = 1,
        dataset_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the session runner.

        Args:
            model_type: LLM model type to use
            max_steps: Maximum steps per query
            texera_api_endpoint: Texera backend API endpoint
            computing_unit_endpoint: Computing unit service endpoint
            agent_service_endpoint: Agent service endpoint
            username: Texera username
            password: Texera password
            verbosity_level: Logging verbosity (0=quiet, 1=normal, 2=verbose)
            dataset_directory: Path to dataset directory for file access
            **kwargs: Additional arguments passed to DataflowAgent
        """
        self.model_type = model_type
        self.max_steps = max_steps
        self.texera_api_endpoint = texera_api_endpoint
        self.computing_unit_endpoint = computing_unit_endpoint
        self.agent_service_endpoint = agent_service_endpoint
        self.username = username
        self.password = password
        self.verbosity_level = verbosity_level
        self.dataset_directory = dataset_directory
        self.kwargs = kwargs

        self._agent: Optional[DataflowAgent] = None
        self._session_messages: List[Dict] = []
        self._is_first_query: bool = True

    def setup(self) -> None:
        """
        Initialize agent for a new session.

        Creates a fresh DataflowAgent with a new workflow.
        """
        self._session_messages = []
        self._is_first_query = True

        self._agent = DataflowAgent(
            model_type=self.model_type,
            max_steps=self.max_steps,
            texera_api_endpoint=self.texera_api_endpoint,
            computing_unit_endpoint=self.computing_unit_endpoint,
            agent_service_endpoint=self.agent_service_endpoint,
            username=self.username,
            password=self.password,
            verbosity_level=self.verbosity_level,
            **self.kwargs
        )
        self._agent.setup()

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
        # Build absolute file paths
        file_paths = []
        if self.dataset_directory and data_sources:
            for f in data_sources:
                file_paths.append(os.path.relpath(os.path.join(self.dataset_directory, f)))

        if is_first and file_paths:
            # First query gets full context about available files
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

        IMPORTANT: This method does NOT reset the agent or clear history.
        - Workflow operators persist (can be reused in subsequent queries)
        - Conversation history persists (agent can reference prior context)

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

        # Run the query (agent maintains workflow and history)
        result = self._agent.run(prompt)

        # Accumulate messages for trace
        if result.messages:
            self._session_messages.extend(result.messages)

        # Parse and return the answer
        return self._parse_answer(result.response)

    def _parse_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        import re

        if not response:
            return ""

        # Look for **Final Answer: xxx** pattern
        match = re.search(r'\*?\*?Final Answer:?\*?\*?\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean markdown formatting
            answer = re.sub(r'^\*\*|\*\*$', '', answer).strip()
            answer = re.sub(r'^`|`$', '', answer).strip()
            return answer

        # Return last non-empty line as fallback
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return response.strip()

    def get_reasoning_trace(self) -> Dict[str, Any]:
        """
        Get the reasoning trace for the session.

        Returns:
            Dictionary with messages and workflow state
        """
        trace = {
            "messages": self._session_messages,
        }

        # Try to get final workflow state
        if self._agent and self._agent.agent_id:
            try:
                workflow = get_agent_workflow(
                    agent_id=self._agent.agent_id,
                    agent_endpoint=self.agent_service_endpoint,
                )
                trace["workflow"] = workflow
            except Exception:
                pass

        return trace

    def cleanup(self) -> None:
        """
        Cleanup agent and workflow resources.
        """
        if self._agent:
            self._agent.cleanup()
            self._agent = None


class DataflowAgentSessionSystem:
    """
    System wrapper for session-based DataflowAgent benchmarking.

    This class provides a System-like interface for integration with
    the existing KramaBench infrastructure.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        max_steps: int = DEFAULT_MAX_STEPS,
        texera_api_endpoint: str = TEXERA_API_ENDPOINT,
        computing_unit_endpoint: str = TEXERA_COMPUTING_UNIT_ENDPOINT,
        agent_service_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the system.

        Args:
            model_type: LLM model type to use
            max_steps: Maximum steps per query
            texera_api_endpoint: Texera API endpoint
            computing_unit_endpoint: Computing unit endpoint
            agent_service_endpoint: Agent service endpoint
            verbose: Enable verbose logging
            **kwargs: Additional arguments
        """
        self.name = f"DataflowAgentSession_{model_type}"
        self.model_type = model_type
        self.max_steps = max_steps
        self.texera_api_endpoint = texera_api_endpoint
        self.computing_unit_endpoint = computing_unit_endpoint
        self.agent_service_endpoint = agent_service_endpoint
        self.verbose = verbose
        self.kwargs = kwargs

        self.dataset_directory: Optional[str] = None

    def process_dataset(self, dataset_directory: str) -> None:
        """
        Set the dataset directory.

        For DataflowAgent, this just stores the path - the agent accesses
        files through the Texera workflow system.
        """
        self.dataset_directory = dataset_directory

    def create_runner(self) -> DataflowAgentSessionRunner:
        """
        Create a new session runner instance.

        Returns:
            DataflowAgentSessionRunner configured with this system's settings
        """
        return DataflowAgentSessionRunner(
            model_type=self.model_type,
            max_steps=self.max_steps,
            texera_api_endpoint=self.texera_api_endpoint,
            computing_unit_endpoint=self.computing_unit_endpoint,
            agent_service_endpoint=self.agent_service_endpoint,
            verbosity_level=2 if self.verbose else 1,
            dataset_directory=self.dataset_directory,
            **self.kwargs
        )

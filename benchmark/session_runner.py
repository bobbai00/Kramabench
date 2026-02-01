# -*- coding: utf-8 -*-
"""
Abstract interface for session-based benchmark runners.

Session runners execute multiple queries sequentially while maintaining
agent state between queries. This enables testing an agent's ability to
build on prior work within a session.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .session import SubtaskResult, SessionResult


class SessionRunner(ABC):
    """
    Abstract interface for running sessions with different agent types.

    A session runner manages agent lifecycle and executes queries while
    maintaining state between queries within a session.

    Implementations should:
    - setup(): Initialize agent for a new session
    - run_query(): Execute a single query, maintaining state
    - get_reasoning_trace(): Return trace/logs from current session
    - cleanup(): Release agent resources
    """

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize the agent for a new session.

        Called once at the start of each session (task).
        The agent should start with fresh state.
        """
        pass

    @abstractmethod
    def run_query(self, query: str, data_sources: list = None) -> str:
        """
        Run a single query, maintaining session state.

        This method should NOT reset the agent state between calls.
        The agent should be able to reference prior queries and results
        within the same session.

        Args:
            query: The query/question to process
            data_sources: Optional list of file paths relevant to this query

        Returns:
            The agent's response as a string
        """
        pass

    @abstractmethod
    def get_reasoning_trace(self) -> Any:
        """
        Get the reasoning trace/logs from the current session.

        Returns:
            Implementation-specific trace data (list, dict, etc.)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup agent resources.

        Called at the end of each session to release resources.
        """
        pass

    def run_session(self, task: Dict[str, Any]) -> SessionResult:
        """
        Run a complete session (task with subtasks).

        This method orchestrates the execution of all subtasks followed
        by the main question, maintaining agent state throughout.

        Args:
            task: Task dictionary with keys:
                - id: Task identifier
                - query: Main question
                - answer: Expected final answer
                - answer_type: Type of answer (for evaluation)
                - subtasks: List of subtask dicts with query, answer, answer_type

        Returns:
            SessionResult with all subtask results and timing info
        """
        result = SessionResult(
            task_id=task["id"],
            main_query=task["query"],
            expected_final_answer=task["answer"],
            final_answer_type=task.get("answer_type", "string_exact"),
        )

        session_start = time.time()

        # Setup agent (fresh state for new session)
        try:
            self.setup()
        except Exception as e:
            result.session_error = f"Setup failed: {str(e)}"
            result.total_elapsed_seconds = time.time() - session_start
            return result

        try:
            # Run each subtask sequentially (maintaining state)
            subtasks = task.get("subtasks", [])
            result.subtasks_total = len(subtasks)

            for i, subtask in enumerate(subtasks):
                subtask_result = self._run_subtask(
                    subtask_id=f"step_{i + 1}",
                    subtask=subtask,
                )
                result.subtask_results.append(subtask_result)

            # Run final question (main task) with full data sources
            final_start = time.time()
            try:
                result.final_answer = self.run_query(
                    task["query"],
                    data_sources=task.get("data_sources", [])
                )
            except Exception as e:
                result.final_answer = None
                result.session_error = f"Final query failed: {str(e)}"

        except Exception as e:
            result.session_error = f"Session failed: {str(e)}"

        finally:
            # Always cleanup
            try:
                self.cleanup()
            except Exception:
                pass

        result.total_elapsed_seconds = time.time() - session_start
        return result

    def _run_subtask(
        self,
        subtask_id: str,
        subtask: Dict[str, Any],
    ) -> SubtaskResult:
        """
        Run a single subtask and return the result.

        Args:
            subtask_id: Identifier for this subtask (e.g., "step_1")
            subtask: Subtask dictionary with query, answer, answer_type

        Returns:
            SubtaskResult with query, expected/actual answers, timing
        """
        subtask_result = SubtaskResult(
            subtask_id=subtask_id,
            query=subtask["query"],
            expected_answer=subtask["answer"],
            answer_type=subtask.get("answer_type", "string_exact"),
        )

        start_time = time.time()

        try:
            subtask_result.actual_answer = self.run_query(
                subtask["query"],
                data_sources=subtask.get("data_sources", [])
            )
        except Exception as e:
            subtask_result.actual_answer = None
            subtask_result.error = str(e)

        subtask_result.elapsed_seconds = time.time() - start_time
        return subtask_result

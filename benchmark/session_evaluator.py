# -*- coding: utf-8 -*-
"""
Session evaluator for scoring session results.

This module evaluates session results using the existing KramaBench metrics,
computing both individual subtask scores and session-level aggregate metrics.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .session import SessionResult, SubtaskResult, SessionBenchmarkResult
from .metrics import metric_factory

# Map answer_type to primary metric for scoring
ANSWER_TYPE_TO_METRIC = {
    "numeric_exact": "success",
    "numeric_approximate": "rae_score",
    "string_exact": "success",
    "string_approximate": "llm_paraphrase",
    "list_exact": "f1",
    "list_approximate": "f1_approximate",
}

# Default success threshold
DEFAULT_SUCCESS_THRESHOLD = 0.9


def parse_answer(answer_str: Optional[str], answer_type: str) -> Any:
    """
    Parse an answer string into the appropriate type.

    Args:
        answer_str: Raw answer string from agent
        answer_type: Expected answer type

    Returns:
        Parsed answer value
    """
    if answer_str is None:
        return None

    answer_str = str(answer_str).strip()

    # Handle list types
    if answer_type.startswith("list"):
        try:
            # Try JSON parsing first
            return json.loads(answer_str)
        except json.JSONDecodeError:
            try:
                # Try eval for Python list syntax
                result = eval(answer_str)
                if isinstance(result, (list, tuple)):
                    return list(result)
            except:
                pass
        # Return as single-item list if parsing fails
        return [answer_str] if answer_str else []

    # Handle numeric types
    if answer_type.startswith("numeric"):
        try:
            # Remove percentage sign if present
            if answer_str.endswith("%"):
                return float(answer_str[:-1])
            return float(answer_str)
        except ValueError:
            return answer_str

    # String types - return as-is
    return answer_str


def evaluate_answer(
    actual: Any,
    expected: Any,
    answer_type: str,
) -> float:
    """
    Evaluate an answer against expected value using appropriate metric.

    Args:
        actual: Actual answer from agent
        expected: Expected answer from ground truth
        answer_type: Type of answer (determines metric)

    Returns:
        Score between 0.0 and 1.0
    """
    if actual is None:
        return 0.0

    # Get the appropriate metric
    metric_name = ANSWER_TYPE_TO_METRIC.get(answer_type, "success")

    try:
        metric = metric_factory(metric_name)
    except ValueError:
        # Fallback to success metric
        metric = metric_factory("success")

    try:
        # Parse actual answer
        parsed_actual = parse_answer(actual, answer_type)

        # Call metric - returns tuple (score, token_usage, ...)
        result = metric(parsed_actual, expected)

        # Extract score from result tuple
        if isinstance(result, tuple):
            score = result[0]
        else:
            score = result

        # Handle None scores
        if score is None:
            return 0.0

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, float(score)))

    except Exception as e:
        logging.warning(f"Evaluation error for {answer_type}: {e}")
        return 0.0


class SessionEvaluator:
    """
    Evaluates session results with multiple metrics.

    Computes scores for each subtask and aggregate session metrics.
    """

    def __init__(self, success_threshold: float = DEFAULT_SUCCESS_THRESHOLD):
        """
        Initialize the evaluator.

        Args:
            success_threshold: Minimum score to consider an answer correct
        """
        self.threshold = success_threshold

    def evaluate_session(
        self,
        result: SessionResult,
        task: Dict[str, Any],
    ) -> SessionResult:
        """
        Score all subtasks and compute session metrics.

        Args:
            result: SessionResult with actual answers (from runner)
            task: Original task with expected answers

        Returns:
            SessionResult with scores filled in
        """
        # Score each subtask
        correct_count = 0
        first_error = None

        for i, subtask_result in enumerate(result.subtask_results):
            # Skip if there was an error
            if subtask_result.error:
                if first_error is None:
                    first_error = i + 1
                continue

            # Evaluate the subtask answer
            score = evaluate_answer(
                actual=subtask_result.actual_answer,
                expected=subtask_result.expected_answer,
                answer_type=subtask_result.answer_type,
            )
            subtask_result.score = score

            if score >= self.threshold:
                correct_count += 1
            elif first_error is None:
                first_error = i + 1

        # Score final answer
        if result.final_answer is not None and result.session_error is None:
            result.final_score = evaluate_answer(
                actual=result.final_answer,
                expected=task["answer"],
                answer_type=task.get("answer_type", "string_exact"),
            )
        else:
            result.final_score = 0.0

        # Compute session metrics
        total_subtasks = len(result.subtask_results)
        result.subtasks_correct = correct_count
        result.subtasks_total = total_subtasks
        result.chain_accuracy = correct_count / total_subtasks if total_subtasks > 0 else 1.0
        result.first_error_step = first_error
        result.all_correct = (
            correct_count == total_subtasks and
            result.final_score >= self.threshold
        )

        return result

    def aggregate_results(
        self,
        session_results: List[SessionResult],
        system_name: str,
        workload: str,
    ) -> SessionBenchmarkResult:
        """
        Aggregate results from multiple sessions.

        Args:
            session_results: List of evaluated SessionResults
            system_name: Name of the system being evaluated
            workload: Name of the workload

        Returns:
            SessionBenchmarkResult with aggregate metrics
        """
        benchmark_result = SessionBenchmarkResult(
            system_name=system_name,
            workload=workload,
            total_sessions=len(session_results),
            session_results=session_results,
        )

        if not session_results:
            return benchmark_result

        # Aggregate metrics
        total_chain_accuracy = 0.0
        total_final_score = 0.0
        total_subtasks = 0
        subtasks_correct = 0
        total_time = 0.0

        for result in session_results:
            # Session-level
            if result.all_correct:
                benchmark_result.sessions_all_correct += 1
            if result.final_score >= self.threshold:
                benchmark_result.sessions_final_correct += 1

            total_chain_accuracy += result.chain_accuracy
            total_final_score += result.final_score
            total_time += result.total_elapsed_seconds

            # Subtask-level
            total_subtasks += result.subtasks_total
            subtasks_correct += result.subtasks_correct

        # Compute averages
        n = len(session_results)
        benchmark_result.avg_chain_accuracy = total_chain_accuracy / n
        benchmark_result.avg_final_score = total_final_score / n
        benchmark_result.total_subtasks = total_subtasks
        benchmark_result.subtasks_correct = subtasks_correct
        benchmark_result.subtask_accuracy = subtasks_correct / total_subtasks if total_subtasks > 0 else 0.0
        benchmark_result.total_elapsed_seconds = total_time

        return benchmark_result

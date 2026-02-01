# -*- coding: utf-8 -*-
"""
Session data structures for multi-query session-based benchmarking.

A session consists of a main task with multiple subtasks that are executed
sequentially while maintaining agent state between queries.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict


@dataclass
class SubtaskResult:
    """Result from a single subtask in a session."""

    subtask_id: str
    query: str
    expected_answer: Any
    answer_type: str = "string_exact"
    actual_answer: Optional[str] = None
    score: float = 0.0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subtask_id": self.subtask_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "answer_type": self.answer_type,
            "actual_answer": self.actual_answer,
            "score": self.score,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


@dataclass
class SessionResult:
    """Result from running a complete session (task with subtasks)."""

    task_id: str
    main_query: str = ""
    expected_final_answer: Any = None
    final_answer_type: str = "string_exact"

    # Subtask results
    subtask_results: List[SubtaskResult] = field(default_factory=list)

    # Final answer
    final_answer: Optional[str] = None
    final_score: float = 0.0

    # Session-level metrics
    all_correct: bool = False              # All subtasks + final correct
    chain_accuracy: float = 0.0            # Fraction of subtasks correct
    subtasks_correct: int = 0              # Count of correct subtasks
    subtasks_total: int = 0                # Total number of subtasks
    first_error_step: Optional[int] = None # Step where chain first broke (1-indexed)

    # Timing
    total_elapsed_seconds: float = 0.0

    # Error tracking
    session_error: Optional[str] = None

    # Consistency checks (optional)
    consistency_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "main_query": self.main_query,
            "expected_final_answer": self.expected_final_answer,
            "final_answer_type": self.final_answer_type,
            "subtask_results": [sr.to_dict() for sr in self.subtask_results],
            "final_answer": self.final_answer,
            "final_score": self.final_score,
            "all_correct": self.all_correct,
            "chain_accuracy": self.chain_accuracy,
            "subtasks_correct": self.subtasks_correct,
            "subtasks_total": self.subtasks_total,
            "first_error_step": self.first_error_step,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "session_error": self.session_error,
            "consistency_violations": self.consistency_violations,
        }


@dataclass
class SessionBenchmarkResult:
    """Aggregated results from running multiple sessions."""

    system_name: str
    workload: str
    total_sessions: int = 0

    # Session-level aggregates
    sessions_all_correct: int = 0          # Sessions where all subtasks + final correct
    sessions_final_correct: int = 0        # Sessions where final answer correct
    avg_chain_accuracy: float = 0.0        # Average chain accuracy across sessions
    avg_final_score: float = 0.0           # Average final score across sessions

    # Subtask-level aggregates
    total_subtasks: int = 0
    subtasks_correct: int = 0
    subtask_accuracy: float = 0.0

    # Timing
    total_elapsed_seconds: float = 0.0

    # Individual session results
    session_results: List[SessionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system_name": self.system_name,
            "workload": self.workload,
            "total_sessions": self.total_sessions,
            "sessions_all_correct": self.sessions_all_correct,
            "sessions_final_correct": self.sessions_final_correct,
            "avg_chain_accuracy": self.avg_chain_accuracy,
            "avg_final_score": self.avg_final_score,
            "total_subtasks": self.total_subtasks,
            "subtasks_correct": self.subtasks_correct,
            "subtask_accuracy": self.subtask_accuracy,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "session_results": [sr.to_dict() for sr in self.session_results],
        }

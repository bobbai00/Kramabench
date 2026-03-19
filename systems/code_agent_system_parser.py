# -*- coding: utf-8 -*-
"""
CodeAgentSystemParser - Parser for smolagents CodeAgent reasoning traces.

This module provides data structures and parsing utilities for the reasoning_trace.json
files produced by CodeAgentSystem. It leverages smolagents native data structures
where applicable and provides a clean interface for trace analysis.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def _count_sloc(code: str) -> int:
    """
    Count Source Lines of Code (SLOC) - non-blank, non-comment lines.

    Args:
        code: Python code string

    Returns:
        Number of SLOC
    """
    if not code:
        return 0

    sloc = 0
    in_multiline_string = False
    multiline_char = None

    for line in code.split('\n'):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Handle multiline strings (""" or ''')
        if not in_multiline_string:
            # Check if line starts a multiline string
            if '"""' in stripped or "'''" in stripped:
                # Count occurrences to determine if we're entering or it's self-contained
                triple_double = stripped.count('"""')
                triple_single = stripped.count("'''")

                if triple_double == 1:
                    in_multiline_string = True
                    multiline_char = '"""'
                elif triple_single == 1:
                    in_multiline_string = True
                    multiline_char = "'''"
                # If count is 2 or more, it's self-contained (e.g., """text""")

            # Skip single-line comments
            if stripped.startswith('#'):
                continue

            sloc += 1
        else:
            # Inside multiline string - check if it ends
            if multiline_char in stripped:
                in_multiline_string = False
                multiline_char = None
            # Don't count lines inside multiline strings (typically docstrings)
            continue

    return sloc


class StepStatus(Enum):
    """Status of a reasoning step."""
    SUCCESS = "success"
    PARSING_ERROR = "parsing_error"
    EXECUTION_ERROR = "execution_error"
    MAX_STEPS_ERROR = "max_steps_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorType(Enum):
    """Types of errors that can occur during agent execution."""
    AGENT_PARSING_ERROR = "AgentParsingError"
    AGENT_EXECUTION_ERROR = "AgentExecutionError"
    AGENT_MAX_STEPS_ERROR = "AgentMaxStepsError"
    INTERPRETER_ERROR = "InterpreterError"
    OTHER = "Other"

    @classmethod
    def from_error_string(cls, error_str: str) -> "ErrorType":
        """Parse error type from error string."""
        if not error_str:
            return cls.OTHER
        if "AgentParsingError" in error_str:
            return cls.AGENT_PARSING_ERROR
        if "AgentExecutionError" in error_str:
            return cls.AGENT_EXECUTION_ERROR
        if "AgentMaxStepsError" in error_str:
            return cls.AGENT_MAX_STEPS_ERROR
        if "InterpreterError" in error_str:
            return cls.INTERPRETER_ERROR
        return cls.OTHER


@dataclass
class CodeAgentError:
    """Represents an error in a code agent step."""
    error_type: ErrorType
    message: str
    raw: str

    @classmethod
    def from_dict(cls, error_data: str | dict) -> Optional["CodeAgentError"]:
        """Parse error from trace data."""
        if not error_data:
            return None

        raw = str(error_data)

        # Handle string format: "{'type': 'AgentParsingError', 'message': '...'}"
        if isinstance(error_data, str):
            error_type = ErrorType.from_error_string(error_data)
            # Extract message if possible
            if "'message':" in error_data:
                try:
                    # Parse the dict-like string
                    start = error_data.find("'message':") + len("'message':")
                    end = error_data.rfind("}")
                    message = error_data[start:end].strip().strip("'\"")
                except Exception:
                    message = error_data
            else:
                message = error_data
        else:
            error_type = ErrorType.from_error_string(error_data.get("type", ""))
            message = error_data.get("message", str(error_data))

        return cls(error_type=error_type, message=message, raw=raw)


@dataclass
class CodeAgentStep:
    """
    Represents a single step in the CodeAgent reasoning trace.

    This mirrors the smolagents ActionStep structure but is simplified for
    parsing from JSON traces.

    Attributes:
        step_number: Sequential step number (1-indexed)
        model_output: Raw output from the LLM
        code: Extracted Python code (if parsing succeeded)
        observations: Output from code execution
        output: Final output value (typically on last step)
        error: Error information if step failed
        is_final_answer: Whether this step produced the final answer
        status: Overall status of the step
    """
    step_number: int
    model_output: Optional[str] = None
    code: Optional[str] = None
    observations: Optional[str] = None
    output: Optional[str] = None
    error: Optional[CodeAgentError] = None
    is_final_answer: bool = False
    status: StepStatus = StepStatus.SUCCESS

    @classmethod
    def from_dict(cls, data: dict) -> "CodeAgentStep":
        """Parse a step from trace dictionary."""
        step_number = data.get("step", 0)
        model_output = data.get("model_output")
        code = data.get("code")
        observations = data.get("observations")
        output = data.get("output")
        is_final = data.get("is_final_answer", False)

        # Parse error
        error = CodeAgentError.from_dict(data.get("error"))

        # Determine status
        if error:
            if error.error_type == ErrorType.AGENT_PARSING_ERROR:
                status = StepStatus.PARSING_ERROR
            elif error.error_type == ErrorType.AGENT_MAX_STEPS_ERROR:
                status = StepStatus.MAX_STEPS_ERROR
            elif error.error_type in (ErrorType.AGENT_EXECUTION_ERROR, ErrorType.INTERPRETER_ERROR):
                status = StepStatus.EXECUTION_ERROR
            else:
                status = StepStatus.UNKNOWN_ERROR
        else:
            status = StepStatus.SUCCESS

        # Check if this is a final answer step
        if output is not None or is_final:
            is_final = True

        return cls(
            step_number=step_number,
            model_output=model_output,
            code=code,
            observations=observations,
            output=output,
            error=error,
            is_final_answer=is_final,
            status=status,
        )

    def has_code(self) -> bool:
        """Check if this step has extracted code."""
        return self.code is not None and len(self.code.strip()) > 0

    def has_observations(self) -> bool:
        """Check if this step has execution observations."""
        return self.observations is not None and len(self.observations.strip()) > 0

    def is_successful(self) -> bool:
        """Check if this step completed successfully."""
        return self.status == StepStatus.SUCCESS


@dataclass
class CodeAgentTrace:
    """
    Complete reasoning trace from a CodeAgent execution.

    Attributes:
        steps: List of reasoning steps
        query_id: Identifier for the query (if available)
        final_answer: The final answer produced
        total_steps: Total number of steps
        successful_steps: Number of successful steps
        error_steps: Number of steps with errors
    """
    steps: list[CodeAgentStep] = field(default_factory=list)
    query_id: Optional[str] = None
    final_answer: Optional[str] = None

    @property
    def total_steps(self) -> int:
        """Total number of steps in the trace."""
        return len(self.steps)

    @property
    def successful_steps(self) -> int:
        """Number of successful steps."""
        return sum(1 for s in self.steps if s.is_successful())

    @property
    def error_steps(self) -> int:
        """Number of steps with errors."""
        return sum(1 for s in self.steps if not s.is_successful())

    @property
    def parsing_errors(self) -> int:
        """Number of parsing errors."""
        return sum(1 for s in self.steps if s.status == StepStatus.PARSING_ERROR)

    @property
    def execution_errors(self) -> int:
        """Number of execution errors."""
        return sum(1 for s in self.steps if s.status == StepStatus.EXECUTION_ERROR)

    @property
    def reached_max_steps(self) -> bool:
        """Whether execution stopped due to max steps."""
        if self.steps:
            last_step = self.steps[-1]
            return last_step.status == StepStatus.MAX_STEPS_ERROR
        return False

    def get_final_step(self) -> Optional[CodeAgentStep]:
        """Get the final answer step if present."""
        for step in reversed(self.steps):
            if step.is_final_answer:
                return step
        return self.steps[-1] if self.steps else None

    def get_all_code(self) -> list[str]:
        """Get all code snippets from the trace."""
        return [s.code for s in self.steps if s.has_code()]

    def get_all_observations(self) -> list[str]:
        """Get all observations from the trace."""
        return [s.observations for s in self.steps if s.has_observations()]

    def get_error_summary(self) -> dict[str, int]:
        """Get summary of errors by type."""
        summary: dict[str, int] = {}
        for step in self.steps:
            if step.error:
                key = step.error.error_type.value
                summary[key] = summary.get(key, 0) + 1
        return summary

    def get_total_code_lines(self) -> int:
        """
        Get total SLOC (Source Lines of Code) from all valid code blocks.

        Counts non-blank, non-comment lines only.
        """
        total_lines = 0
        for code in self.get_all_code():
            if code:
                total_lines += _count_sloc(code)
        return total_lines

    def get_code_block_count(self) -> int:
        """Get number of valid code blocks in the trace."""
        return len(self.get_all_code())


class CodeAgentSystemParser:
    """
    Parser for CodeAgentSystem reasoning traces.

    This parser reads reasoning_trace.json files and converts them into
    structured CodeAgentTrace objects for analysis.

    Example:
        parser = CodeAgentSystemParser()
        trace = parser.parse_file("system_scratch/CodeAgentSystemGptO3/legal-easy-3/reasoning_trace.json")
        print(f"Total steps: {trace.total_steps}")
        print(f"Successful: {trace.successful_steps}")
        print(f"Final answer: {trace.final_answer}")
    """

    def parse_file(self, file_path: str | Path) -> CodeAgentTrace:
        """
        Parse a reasoning trace from a JSON file.

        Args:
            file_path: Path to the reasoning_trace.json file

        Returns:
            CodeAgentTrace object containing parsed steps
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trace = self.parse_data(data)

        # Try to extract query_id from path
        # Expected format: system_scratch/{system}/{query_id}/reasoning_trace.json
        try:
            trace.query_id = file_path.parent.name
        except Exception:
            pass

        return trace

    def parse_data(self, data: list[dict]) -> CodeAgentTrace:
        """
        Parse reasoning trace from JSON data.

        Args:
            data: List of step dictionaries

        Returns:
            CodeAgentTrace object containing parsed steps
        """
        steps = [CodeAgentStep.from_dict(step_data) for step_data in data]
        trace = CodeAgentTrace(steps=steps)

        # Extract final answer from the last step with output
        final_step = trace.get_final_step()
        if final_step:
            trace.final_answer = final_step.output

        return trace

    def parse_directory(self, directory: str | Path) -> dict[str, CodeAgentTrace]:
        """
        Parse all reasoning traces in a system output directory.

        Args:
            directory: Path to system output directory (e.g., system_scratch/CodeAgentSystemGptO3)

        Returns:
            Dictionary mapping query_id to CodeAgentTrace
        """
        directory = Path(directory)
        traces = {}

        for trace_file in directory.glob("*/reasoning_trace.json"):
            query_id = trace_file.parent.name
            try:
                traces[query_id] = self.parse_file(trace_file)
            except Exception as e:
                print(f"Error parsing {trace_file}: {e}")

        return traces


def parse_code_agent_trace(file_path: str | Path) -> CodeAgentTrace:
    """
    Convenience function to parse a reasoning trace file.

    Args:
        file_path: Path to reasoning_trace.json file

    Returns:
        Parsed CodeAgentTrace
    """
    parser = CodeAgentSystemParser()
    return parser.parse_file(file_path)


@dataclass
class CodeAgentAggregateStats:
    """Aggregate statistics across multiple traces."""
    total_queries: int = 0
    total_steps: int = 0
    total_successful_steps: int = 0
    total_error_steps: int = 0
    total_parsing_errors: int = 0
    total_execution_errors: int = 0
    queries_reached_max_steps: int = 0
    queries_with_final_answer: int = 0
    error_type_counts: dict[str, int] = field(default_factory=dict)

    # Per-query averages
    avg_steps_per_query: float = 0.0
    avg_successful_steps_per_query: float = 0.0
    avg_error_steps_per_query: float = 0.0

    # Success rates
    step_success_rate: float = 0.0
    query_completion_rate: float = 0.0

    # Valid code execution stats
    total_steps_with_code: int = 0
    successful_steps_with_code: int = 0
    valid_code_success_rate: float = 0.0

    # Code metrics (lines of code per query)
    code_lines_per_query: list[int] = field(default_factory=list)
    min_code_lines: int = 0
    max_code_lines: int = 0
    avg_code_lines: float = 0.0
    total_code_lines: int = 0
    total_code_blocks: int = 0

    # Average lines per code block (per query)
    avg_lines_per_block_per_query: list[float] = field(default_factory=list)
    min_avg_lines_per_block: float = 0.0
    max_avg_lines_per_block: float = 0.0
    avg_avg_lines_per_block: float = 0.0

    def compute_averages(self) -> None:
        """Compute average statistics."""
        if self.total_queries > 0:
            self.avg_steps_per_query = self.total_steps / self.total_queries
            self.avg_successful_steps_per_query = self.total_successful_steps / self.total_queries
            self.avg_error_steps_per_query = self.total_error_steps / self.total_queries
            self.query_completion_rate = self.queries_with_final_answer / self.total_queries

        if self.total_steps > 0:
            self.step_success_rate = self.total_successful_steps / self.total_steps

        if self.total_steps_with_code > 0:
            self.valid_code_success_rate = self.successful_steps_with_code / self.total_steps_with_code

        # Compute code line statistics
        if self.code_lines_per_query:
            self.min_code_lines = min(self.code_lines_per_query)
            self.max_code_lines = max(self.code_lines_per_query)
            self.avg_code_lines = sum(self.code_lines_per_query) / len(self.code_lines_per_query)
            self.total_code_lines = sum(self.code_lines_per_query)

        # Compute avg lines per block statistics (only for queries with code blocks)
        valid_avgs = [v for v in self.avg_lines_per_block_per_query if v > 0]
        if valid_avgs:
            self.min_avg_lines_per_block = min(valid_avgs)
            self.max_avg_lines_per_block = max(valid_avgs)
            self.avg_avg_lines_per_block = sum(valid_avgs) / len(valid_avgs)


def compute_aggregate_stats(traces: dict[str, CodeAgentTrace]) -> CodeAgentAggregateStats:
    """
    Compute aggregate statistics from multiple traces.

    Args:
        traces: Dictionary mapping query_id to CodeAgentTrace

    Returns:
        CodeAgentAggregateStats with computed statistics
    """
    stats = CodeAgentAggregateStats()
    stats.total_queries = len(traces)

    for query_id, trace in traces.items():
        stats.total_steps += trace.total_steps
        stats.total_successful_steps += trace.successful_steps
        stats.total_error_steps += trace.error_steps
        stats.total_parsing_errors += trace.parsing_errors
        stats.total_execution_errors += trace.execution_errors

        # Count steps with valid code and their success rate
        for step in trace.steps:
            if step.has_code():
                stats.total_steps_with_code += 1
                if step.is_successful():
                    stats.successful_steps_with_code += 1

        if trace.reached_max_steps:
            stats.queries_reached_max_steps += 1

        if trace.final_answer:
            stats.queries_with_final_answer += 1

        # Aggregate error types
        for error_type, count in trace.get_error_summary().items():
            stats.error_type_counts[error_type] = stats.error_type_counts.get(error_type, 0) + count

        # Collect code metrics
        code_lines = trace.get_total_code_lines()
        code_blocks = trace.get_code_block_count()
        stats.code_lines_per_query.append(code_lines)
        stats.total_code_blocks += code_blocks

        # Compute avg lines per block for this query
        if code_blocks > 0:
            avg_lines_per_block = code_lines / code_blocks
        else:
            avg_lines_per_block = 0.0
        stats.avg_lines_per_block_per_query.append(avg_lines_per_block)

    stats.compute_averages()
    return stats


def print_aggregate_stats(stats: CodeAgentAggregateStats, system_name: str) -> None:
    """Print formatted aggregate statistics."""
    print("=" * 70)
    print(f"CodeAgent System Statistics: {system_name}")
    print("=" * 70)
    print()
    print("QUERY SUMMARY")
    print("-" * 40)
    print(f"  Total queries:              {stats.total_queries}")
    print(f"  Queries with final answer:  {stats.queries_with_final_answer}")
    print(f"  Queries reached max steps:  {stats.queries_reached_max_steps}")
    print(f"  Query completion rate:      {stats.query_completion_rate:.1%}")
    print()
    print("STEP SUMMARY")
    print("-" * 40)
    print(f"  Total steps:                {stats.total_steps}")
    print(f"  Successful steps:           {stats.total_successful_steps}")
    print(f"  Error steps:                {stats.total_error_steps}")
    print(f"  Step success rate:          {stats.step_success_rate:.1%}")
    print()
    print("VALID CODE EXECUTION")
    print("-" * 40)
    print(f"  Steps with valid code:      {stats.total_steps_with_code}")
    print(f"  Successful executions:      {stats.successful_steps_with_code}")
    print(f"  Valid code success rate:    {stats.valid_code_success_rate:.1%}")
    print()
    print("AVERAGES PER QUERY")
    print("-" * 40)
    print(f"  Avg steps:                  {stats.avg_steps_per_query:.1f}")
    print(f"  Avg successful steps:       {stats.avg_successful_steps_per_query:.1f}")
    print(f"  Avg error steps:            {stats.avg_error_steps_per_query:.1f}")
    print()
    print("CODE METRICS (SLOC - non-blank, non-comment lines)")
    print("-" * 40)
    print(f"  Total code blocks:          {stats.total_code_blocks}")
    print(f"  Total SLOC:                 {stats.total_code_lines}")
    print(f"  SLOC per query (min/avg/max):     {stats.min_code_lines} / {stats.avg_code_lines:.1f} / {stats.max_code_lines}")
    print(f"  SLOC per block (min/avg/max):     {stats.min_avg_lines_per_block:.1f} / {stats.avg_avg_lines_per_block:.1f} / {stats.max_avg_lines_per_block:.1f}")
    print()
    print("ERROR BREAKDOWN")
    print("-" * 40)
    print(f"  Parsing errors:             {stats.total_parsing_errors}")
    print(f"  Execution errors:           {stats.total_execution_errors}")
    print()
    if stats.error_type_counts:
        print("ERROR TYPE COUNTS")
        print("-" * 40)
        for error_type, count in sorted(stats.error_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type:30s} {count}")
    print()


def main():
    """Main entry point for parsing CodeAgent traces."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and analyze CodeAgentSystem reasoning traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m systems.code_agent_system_parser CodeAgentSystemGptO3
  python -m systems.code_agent_system_parser CodeAgentSystemHaiku --verbose
  python -m systems.code_agent_system_parser CodeAgentSystemSonnet --output stats.json
        """
    )
    parser.add_argument(
        "system_name",
        help="Name of the system (directory name in system_scratch/)"
    )
    parser.add_argument(
        "--scratch-dir",
        default="system_scratch",
        help="Path to system scratch directory (default: system_scratch)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-query details"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON stats (optional)"
    )

    args = parser.parse_args()

    # Build path to system directory
    system_dir = Path(args.scratch_dir) / args.system_name

    if not system_dir.exists():
        print(f"Error: System directory not found: {system_dir}")
        return 1

    # Parse all traces
    trace_parser = CodeAgentSystemParser()
    traces = trace_parser.parse_directory(system_dir)

    if not traces:
        print(f"Error: No traces found in {system_dir}")
        return 1

    print(f"Parsed {len(traces)} traces from {system_dir}")
    print()

    # Compute and print aggregate stats
    stats = compute_aggregate_stats(traces)
    print_aggregate_stats(stats, args.system_name)

    # Print per-query details if verbose
    if args.verbose:
        print("PER-QUERY DETAILS")
        print("=" * 70)
        print(f"{'Query ID':<35} {'Steps':>6} {'Success':>8} {'Errors':>7} {'Answer':<15}")
        print("-" * 70)
        for query_id in sorted(traces.keys()):
            trace = traces[query_id]
            answer = trace.final_answer[:12] + "..." if trace.final_answer and len(trace.final_answer) > 15 else (trace.final_answer or "N/A")
            print(f"{query_id:<35} {trace.total_steps:>6} {trace.successful_steps:>8} {trace.error_steps:>7} {answer:<15}")
        print()

    # Output JSON if requested
    if args.output:
        import dataclasses
        output_data = {
            "system_name": args.system_name,
            "stats": dataclasses.asdict(stats),
            "traces": {
                qid: {
                    "total_steps": t.total_steps,
                    "successful_steps": t.successful_steps,
                    "error_steps": t.error_steps,
                    "parsing_errors": t.parsing_errors,
                    "execution_errors": t.execution_errors,
                    "reached_max_steps": t.reached_max_steps,
                    "final_answer": t.final_answer,
                    "error_summary": t.get_error_summary(),
                }
                for qid, t in traces.items()
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Stats written to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())

# -*- coding: utf-8 -*-
"""
DataflowSystemParser - Parser for Texera DataflowAgent message traces.

This module provides data structures and parsing utilities for the messages.json
files produced by DataflowSystem. It follows a similar design pattern to the
CodeAgentSystemParser for consistency.
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


class MessageRole(Enum):
    """Role of a message in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ContentType(Enum):
    """Type of content in a message."""
    TEXT = "text"
    TOOL_CALL = "tool-call"
    TOOL_RESULT = "tool-result"
    IMAGE = "image"


class ToolCallStatus(Enum):
    """Status of a tool call."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class ToolCallInput:
    """Input parameters for a tool call."""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def code(self) -> Optional[str]:
        """Get code parameter if present."""
        return self.raw.get("code")

    @property
    def operator_id(self) -> Optional[str]:
        """Get operatorId parameter if present."""
        return self.raw.get("operatorId")

    @property
    def summary(self) -> Optional[str]:
        """Get summary parameter if present."""
        return self.raw.get("summary")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter by key."""
        return self.raw.get(key, default)


@dataclass
class ToolCallOutput:
    """Output from a tool call execution."""
    output_type: str = "text"
    value: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallOutput":
        """Parse tool output from dictionary."""
        if isinstance(data, dict):
            output_type = data.get("type", "text")
            value = data.get("value", "")
            return cls(output_type=output_type, value=value, raw=data)
        return cls(value=str(data), raw={"value": str(data)})

    def is_error(self) -> bool:
        """Check if output indicates an error."""
        return "[ERROR]" in self.value or "error" in self.value.lower()


@dataclass
class ToolCall:
    """
    Represents a tool call made by the assistant.

    This is analogous to smolagents.ToolCall but adapted for the Texera dataflow agent.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the tool being called
        input: Input parameters for the tool
        output: Output from tool execution (if available)
        status: Status of the tool call
    """
    id: str
    name: str
    input: ToolCallInput
    output: Optional[ToolCallOutput] = None
    status: ToolCallStatus = ToolCallStatus.PENDING

    @classmethod
    def from_content(cls, content: dict) -> Optional["ToolCall"]:
        """Parse a tool call from message content."""
        if content.get("type") != "tool-call":
            return None

        return cls(
            id=content.get("toolCallId", ""),
            name=content.get("toolName", ""),
            input=ToolCallInput(raw=content.get("input", {})),
        )

    def is_operator_creation(self) -> bool:
        """Check if this is an operator creation call."""
        return self.name == "addOperator"

    def is_operator_modification(self) -> bool:
        """Check if this is an operator modification call."""
        return self.name == "modifyOperator"

    def is_workflow_execution(self) -> bool:
        """Check if this is a workflow execution call."""
        return self.name == "executeWorkflow"

    def is_link_operation(self) -> bool:
        """Check if this is a link operation."""
        return self.name in ("addLink", "removeLink")


@dataclass
class MessageContent:
    """
    A single content item within a message.

    Messages can contain multiple content items (text, tool calls, tool results).
    """
    content_type: ContentType
    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolCallOutput] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "MessageContent":
        """Parse content from dictionary."""
        content_type_str = data.get("type", "text")

        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            content_type = ContentType.TEXT

        text = data.get("text")
        tool_call = None
        tool_result = None
        tool_call_id = data.get("toolCallId")
        tool_name = data.get("toolName")

        if content_type == ContentType.TOOL_CALL:
            tool_call = ToolCall.from_content(data)
        elif content_type == ContentType.TOOL_RESULT:
            output_data = data.get("output", {})
            tool_result = ToolCallOutput.from_dict(output_data)

        return cls(
            content_type=content_type,
            text=text,
            tool_call=tool_call,
            tool_result=tool_result,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            raw=data,
        )


@dataclass
class Message:
    """
    A single message in the conversation trace.

    Attributes:
        role: Role of the message sender (user, assistant, tool)
        contents: List of content items in the message
        raw: Raw message dictionary
    """
    role: MessageRole
    contents: list[MessageContent] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Parse a message from dictionary."""
        role_str = data.get("role", "user")
        try:
            role = MessageRole(role_str)
        except ValueError:
            role = MessageRole.USER

        content_data = data.get("content", [])

        # Handle string content
        if isinstance(content_data, str):
            contents = [MessageContent(
                content_type=ContentType.TEXT,
                text=content_data,
                raw={"type": "text", "text": content_data},
            )]
        else:
            contents = [MessageContent.from_dict(c) for c in content_data]

        return cls(role=role, contents=contents, raw=data)

    def get_text(self) -> Optional[str]:
        """Get the first text content from the message."""
        for content in self.contents:
            if content.content_type == ContentType.TEXT and content.text:
                return content.text
        return None

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls from the message."""
        return [c.tool_call for c in self.contents if c.tool_call is not None]

    def get_tool_results(self) -> list[tuple[str, ToolCallOutput]]:
        """Get all tool results as (tool_call_id, output) tuples."""
        results = []
        for c in self.contents:
            if c.tool_result is not None:
                results.append((c.tool_call_id or "", c.tool_result))
        return results

    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role == MessageRole.USER

    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == MessageRole.ASSISTANT

    def is_tool_message(self) -> bool:
        """Check if this is a tool result message."""
        return self.role == MessageRole.TOOL

    def has_final_answer(self) -> bool:
        """Check if this message contains a final answer."""
        text = self.get_text()
        if text:
            return "Final Answer:" in text or "final answer:" in text.lower()
        return False


@dataclass
class DataflowStep:
    """
    Represents a single step in the dataflow agent execution.

    A step consists of:
    1. Assistant message with tool call(s)
    2. Tool message with result(s)

    This provides a view similar to CodeAgentStep for consistency.
    """
    step_number: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolCallOutput] = field(default_factory=list)
    assistant_text: Optional[str] = None
    has_error: bool = False
    is_final_answer: bool = False

    def get_code_snippets(self) -> list[str]:
        """Get all code snippets from tool calls in this step."""
        codes = []
        for tc in self.tool_calls:
            if tc.input.code:
                codes.append(tc.input.code)
        return codes

    def get_operator_ids(self) -> list[str]:
        """Get all operator IDs referenced in this step."""
        ids = []
        for tc in self.tool_calls:
            if tc.input.operator_id:
                ids.append(tc.input.operator_id)
        return ids


@dataclass
class DataflowTrace:
    """
    Complete message trace from a DataflowAgent execution.

    Attributes:
        messages: List of all messages in the conversation
        steps: Parsed execution steps (tool call + result pairs)
        query_id: Identifier for the query
        final_answer: The final answer produced
    """
    messages: list[Message] = field(default_factory=list)
    steps: list[DataflowStep] = field(default_factory=list)
    query_id: Optional[str] = None
    final_answer: Optional[str] = None

    @property
    def total_messages(self) -> int:
        """Total number of messages."""
        return len(self.messages)

    @property
    def total_steps(self) -> int:
        """Total number of execution steps."""
        return len(self.steps)

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool calls made."""
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def error_steps(self) -> int:
        """Number of steps with errors."""
        return sum(1 for s in self.steps if s.has_error)

    def get_user_prompt(self) -> Optional[str]:
        """Get the initial user prompt."""
        for msg in self.messages:
            if msg.is_user_message():
                return msg.get_text()
        return None

    def get_all_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls from the trace."""
        calls = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    def get_tool_call_summary(self) -> dict[str, int]:
        """Get summary of tool calls by name."""
        summary: dict[str, int] = {}
        for tc in self.get_all_tool_calls():
            summary[tc.name] = summary.get(tc.name, 0) + 1
        return summary

    def get_all_code(self) -> list[str]:
        """Get all code snippets from the trace."""
        codes = []
        for step in self.steps:
            codes.extend(step.get_code_snippets())
        return codes

    def get_operators_created(self) -> list[str]:
        """Get list of operator IDs created."""
        operators = []
        for tc in self.get_all_tool_calls():
            if tc.is_operator_creation() and tc.output:
                # Parse operator ID from output like "Added operator DataLoading-operator-1, ..."
                value = tc.output.value
                if "Added operator" in value:
                    parts = value.split(",")[0].replace("Added operator", "").strip()
                    operators.append(parts)
        return operators

    def get_total_code_lines(self) -> int:
        """
        Get total SLOC from addOperator and modifyOperator tool calls.

        Counts non-blank, non-comment lines from the 'code' parameter.
        """
        total_lines = 0
        for tc in self.get_all_tool_calls():
            if tc.is_operator_creation() or tc.is_operator_modification():
                code = tc.input.code
                if code:
                    total_lines += _count_sloc(code)
        return total_lines

    def get_code_block_count(self) -> int:
        """Get number of code blocks (addOperator + modifyOperator with code)."""
        count = 0
        for tc in self.get_all_tool_calls():
            if tc.is_operator_creation() or tc.is_operator_modification():
                if tc.input.code:
                    count += 1
        return count


class DataflowSystemParser:
    """
    Parser for DataflowSystem message traces.

    This parser reads messages.json files and converts them into structured
    DataflowTrace objects for analysis.

    Example:
        parser = DataflowSystemParser()
        trace = parser.parse_file("system_scratch/DataflowSystemGptO3/astronomy-easy-1/messages.json")
        print(f"Total steps: {trace.total_steps}")
        print(f"Tool calls: {trace.total_tool_calls}")
        print(f"Final answer: {trace.final_answer}")
    """

    def parse_file(self, file_path: str | Path) -> DataflowTrace:
        """
        Parse a message trace from a JSON file.

        Args:
            file_path: Path to the messages.json file

        Returns:
            DataflowTrace object containing parsed messages and steps
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trace = self.parse_data(data)

        # Try to extract query_id from path
        try:
            trace.query_id = file_path.parent.name
        except Exception:
            pass

        return trace

    def parse_data(self, data: list[dict]) -> DataflowTrace:
        """
        Parse message trace from JSON data.

        Args:
            data: List of message dictionaries

        Returns:
            DataflowTrace object containing parsed messages and steps
        """
        messages = [Message.from_dict(msg_data) for msg_data in data]
        trace = DataflowTrace(messages=messages)

        # Build steps from message pairs
        trace.steps = self._build_steps(messages)

        # Extract final answer
        trace.final_answer = self._extract_final_answer(messages)

        # Match tool results to tool calls
        self._match_tool_results(trace)

        return trace

    def _build_steps(self, messages: list[Message]) -> list[DataflowStep]:
        """Build execution steps from messages."""
        steps = []
        step_number = 0
        current_tool_calls: list[ToolCall] = []

        for msg in messages:
            if msg.is_assistant_message():
                tool_calls = msg.get_tool_calls()
                if tool_calls:
                    current_tool_calls = tool_calls
                else:
                    # Text-only assistant message might contain final answer
                    text = msg.get_text()
                    if text and msg.has_final_answer():
                        step_number += 1
                        steps.append(DataflowStep(
                            step_number=step_number,
                            assistant_text=text,
                            is_final_answer=True,
                        ))

            elif msg.is_tool_message() and current_tool_calls:
                step_number += 1
                results = [r[1] for r in msg.get_tool_results()]
                has_error = any(r.is_error() for r in results)

                step = DataflowStep(
                    step_number=step_number,
                    tool_calls=current_tool_calls.copy(),
                    tool_results=results,
                    has_error=has_error,
                )
                steps.append(step)
                current_tool_calls = []

        return steps

    def _extract_final_answer(self, messages: list[Message]) -> Optional[str]:
        """Extract final answer from messages."""
        for msg in reversed(messages):
            if msg.is_assistant_message():
                text = msg.get_text()
                if text and "Final Answer:" in text:
                    # Extract answer after "Final Answer:"
                    idx = text.find("Final Answer:")
                    answer = text[idx + len("Final Answer:"):].strip()
                    # Clean up common formatting
                    answer = answer.strip("*").strip()
                    return answer
        return None

    def _match_tool_results(self, trace: DataflowTrace) -> None:
        """Match tool results back to their tool calls."""
        # Build a map of tool call ID to results
        result_map: dict[str, ToolCallOutput] = {}

        for msg in trace.messages:
            if msg.is_tool_message():
                for call_id, result in msg.get_tool_results():
                    if call_id:
                        result_map[call_id] = result

        # Match results to tool calls
        for step in trace.steps:
            for tc in step.tool_calls:
                if tc.id in result_map:
                    tc.output = result_map[tc.id]
                    tc.status = (
                        ToolCallStatus.ERROR
                        if tc.output.is_error()
                        else ToolCallStatus.SUCCESS
                    )

    def parse_directory(self, directory: str | Path) -> dict[str, DataflowTrace]:
        """
        Parse all message traces in a system output directory.

        Args:
            directory: Path to system output directory (e.g., system_scratch/DataflowSystemGptO3)

        Returns:
            Dictionary mapping query_id to DataflowTrace
        """
        directory = Path(directory)
        traces = {}

        for trace_file in directory.glob("*/messages.json"):
            query_id = trace_file.parent.name
            try:
                traces[query_id] = self.parse_file(trace_file)
            except Exception as e:
                print(f"Error parsing {trace_file}: {e}")

        return traces


def parse_dataflow_trace(file_path: str | Path) -> DataflowTrace:
    """
    Convenience function to parse a message trace file.

    Args:
        file_path: Path to messages.json file

    Returns:
        Parsed DataflowTrace
    """
    parser = DataflowSystemParser()
    return parser.parse_file(file_path)


@dataclass
class DataflowAggregateStats:
    """Aggregate statistics across multiple traces."""
    total_queries: int = 0
    total_messages: int = 0
    total_steps: int = 0
    total_tool_calls: int = 0
    total_error_steps: int = 0
    queries_with_final_answer: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)

    # Per-query averages
    avg_messages_per_query: float = 0.0
    avg_steps_per_query: float = 0.0
    avg_tool_calls_per_query: float = 0.0

    # Success rates
    step_success_rate: float = 0.0
    query_completion_rate: float = 0.0

    # Valid code execution stats (steps with addOperator/modifyOperator code)
    total_steps_with_code: int = 0
    successful_steps_with_code: int = 0
    valid_code_success_rate: float = 0.0

    # Operator statistics
    total_operators_created: int = 0
    total_workflow_executions: int = 0

    # Code metrics (lines of code per query from addOperator/modifyOperator)
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
            self.avg_messages_per_query = self.total_messages / self.total_queries
            self.avg_steps_per_query = self.total_steps / self.total_queries
            self.avg_tool_calls_per_query = self.total_tool_calls / self.total_queries
            self.query_completion_rate = self.queries_with_final_answer / self.total_queries

        if self.total_steps > 0:
            successful_steps = self.total_steps - self.total_error_steps
            self.step_success_rate = successful_steps / self.total_steps

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


def compute_aggregate_stats(traces: dict[str, DataflowTrace]) -> DataflowAggregateStats:
    """
    Compute aggregate statistics from multiple traces.

    Args:
        traces: Dictionary mapping query_id to DataflowTrace

    Returns:
        DataflowAggregateStats with computed statistics
    """
    stats = DataflowAggregateStats()
    stats.total_queries = len(traces)

    for query_id, trace in traces.items():
        stats.total_messages += trace.total_messages
        stats.total_steps += trace.total_steps
        stats.total_tool_calls += trace.total_tool_calls
        stats.total_error_steps += trace.error_steps

        # Count steps with valid code (addOperator/modifyOperator with code)
        for step in trace.steps:
            has_code = any(
                (tc.is_operator_creation() or tc.is_operator_modification()) and tc.input.code
                for tc in step.tool_calls
            )
            if has_code:
                stats.total_steps_with_code += 1
                if not step.has_error:
                    stats.successful_steps_with_code += 1

        if trace.final_answer:
            stats.queries_with_final_answer += 1

        # Aggregate tool call counts
        for tool_name, count in trace.get_tool_call_summary().items():
            stats.tool_call_counts[tool_name] = stats.tool_call_counts.get(tool_name, 0) + count

        # Count operators and executions
        stats.total_operators_created += len(trace.get_operators_created())
        stats.total_workflow_executions += sum(
            1 for tc in trace.get_all_tool_calls() if tc.is_workflow_execution()
        )

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


def print_aggregate_stats(stats: DataflowAggregateStats, system_name: str) -> None:
    """Print formatted aggregate statistics."""
    print("=" * 70)
    print(f"Dataflow System Statistics: {system_name}")
    print("=" * 70)
    print()
    print("QUERY SUMMARY")
    print("-" * 40)
    print(f"  Total queries:              {stats.total_queries}")
    print(f"  Queries with final answer:  {stats.queries_with_final_answer}")
    print(f"  Query completion rate:      {stats.query_completion_rate:.1%}")
    print()
    print("MESSAGE & STEP SUMMARY")
    print("-" * 40)
    print(f"  Total messages:             {stats.total_messages}")
    print(f"  Total steps:                {stats.total_steps}")
    print(f"  Total tool calls:           {stats.total_tool_calls}")
    print(f"  Error steps:                {stats.total_error_steps}")
    print(f"  Step success rate:          {stats.step_success_rate:.1%}")
    print()
    print("VALID CODE EXECUTION (steps with addOperator/modifyOperator code)")
    print("-" * 40)
    print(f"  Steps with valid code:      {stats.total_steps_with_code}")
    print(f"  Successful executions:      {stats.successful_steps_with_code}")
    print(f"  Valid code success rate:    {stats.valid_code_success_rate:.1%}")
    print()
    print("AVERAGES PER QUERY")
    print("-" * 40)
    print(f"  Avg messages:               {stats.avg_messages_per_query:.1f}")
    print(f"  Avg steps:                  {stats.avg_steps_per_query:.1f}")
    print(f"  Avg tool calls:             {stats.avg_tool_calls_per_query:.1f}")
    print()
    print("CODE METRICS (SLOC - non-blank, non-comment lines)")
    print("-" * 40)
    print(f"  Total code blocks:          {stats.total_code_blocks}")
    print(f"  Total SLOC:                 {stats.total_code_lines}")
    print(f"  SLOC per query (min/avg/max):     {stats.min_code_lines} / {stats.avg_code_lines:.1f} / {stats.max_code_lines}")
    print(f"  SLOC per block (min/avg/max):     {stats.min_avg_lines_per_block:.1f} / {stats.avg_avg_lines_per_block:.1f} / {stats.max_avg_lines_per_block:.1f}")
    print()
    print("OPERATOR STATISTICS")
    print("-" * 40)
    print(f"  Total operators created:    {stats.total_operators_created}")
    print(f"  Total workflow executions:  {stats.total_workflow_executions}")
    print()
    if stats.tool_call_counts:
        print("TOOL CALL BREAKDOWN")
        print("-" * 40)
        for tool_name, count in sorted(stats.tool_call_counts.items(), key=lambda x: -x[1]):
            print(f"  {tool_name:30s} {count}")
    print()


def main():
    """Main entry point for parsing Dataflow traces."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and analyze DataflowSystem message traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m systems.dataflow_system_parser DataflowSystemGptO3
  python -m systems.dataflow_system_parser DataflowSystemSonnet --verbose
  python -m systems.dataflow_system_parser DataflowSystemHaiku --output stats.json
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
    trace_parser = DataflowSystemParser()
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
        print(f"{'Query ID':<35} {'Msgs':>5} {'Steps':>6} {'Tools':>6} {'Errs':>5} {'Answer':<15}")
        print("-" * 70)
        for query_id in sorted(traces.keys()):
            trace = traces[query_id]
            answer = trace.final_answer[:12] + "..." if trace.final_answer and len(trace.final_answer) > 15 else (trace.final_answer or "N/A")
            print(f"{query_id:<35} {trace.total_messages:>5} {trace.total_steps:>6} {trace.total_tool_calls:>6} {trace.error_steps:>5} {answer:<15}")
        print()

    # Output JSON if requested
    if args.output:
        import dataclasses
        output_data = {
            "system_name": args.system_name,
            "stats": dataclasses.asdict(stats),
            "traces": {
                qid: {
                    "total_messages": t.total_messages,
                    "total_steps": t.total_steps,
                    "total_tool_calls": t.total_tool_calls,
                    "error_steps": t.error_steps,
                    "final_answer": t.final_answer,
                    "tool_call_summary": t.get_tool_call_summary(),
                    "operators_created": t.get_operators_created(),
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

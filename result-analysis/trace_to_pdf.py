#!/usr/bin/env python3
"""
Convert agent trace JSON files to PDF documents.

Usage:
    python trace_to_pdf.py <trace.json> [--output <output.pdf>] [--format <codeagent|dataflow>]

Example:
    python trace_to_pdf.py traces/astronomy-easy-2_codeagent.json
    python trace_to_pdf.py traces/astronomy-easy-2_codeagent.json --output report.pdf
    python trace_to_pdf.py traces/biomedical-easy-2/dataflow_trace.json --format dataflow
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from fpdf import FPDF


def sanitize_text(text: str) -> str:
    """Replace Unicode characters that can't be encoded in latin-1."""
    replacements = {
        '\u2013': '-',  # en-dash
        '\u2014': '-',  # em-dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',  # non-breaking space
        '\u2022': '*',  # bullet
        '\u2212': '-',  # minus sign
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-latin-1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')


class TracePDF(FPDF):
    """Custom PDF class for trace documents."""

    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, self.title_text, align='C')
        self.ln(5)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_step(self, step_num: int, code: str, observations: str,
                 error: str = None, output: str = None, is_final: bool = False,
                 thought: str = None):
        """Add a step to the PDF."""
        # Sanitize all inputs
        code = sanitize_text(code) if code else ""
        observations = sanitize_text(observations) if observations else ""
        error = sanitize_text(error) if error else None
        output = sanitize_text(str(output)) if output else None
        thought = sanitize_text(thought) if thought else None

        # Step header
        self.set_font('Helvetica', 'B', 12)
        if is_final:
            self.set_text_color(0, 128, 0)  # Green
            header = f"Step {step_num} (Final Answer)"
        elif error:
            self.set_text_color(200, 0, 0)  # Red
            header = f"Step {step_num} (Error)"
        else:
            self.set_text_color(0, 0, 128)  # Blue
            header = f"Step {step_num}"

        self.cell(0, 8, header)
        self.ln(8)

        # Thought/Reasoning section
        if thought:
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(128, 0, 128)  # Purple for thoughts
            self.cell(0, 6, "Thought:")
            self.ln(5)

            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(80, 80, 80)
            self.set_fill_color(255, 250, 240)  # Light orange/cream background

            self._add_thought_block(thought)
            self.ln(3)

        # Code section
        if code:
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, "Code:")
            self.ln(5)

            self.set_font('Courier', '', 8)
            self.set_text_color(0, 0, 0)
            self.set_fill_color(245, 245, 245)

            # Add code block with background
            self._add_code_block(code)
            self.ln(3)

        # Execution logs section
        if observations:
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, "Execution Logs:")
            self.ln(5)

            self.set_font('Courier', '', 8)
            self.set_text_color(50, 50, 50)

            # Clean up observations
            obs_text = observations.strip()
            if obs_text.startswith("Execution logs:"):
                obs_text = obs_text[len("Execution logs:"):].strip()

            self._add_text_block(obs_text)
            self.ln(3)

        # Error section
        if error:
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(200, 0, 0)
            self.cell(0, 6, "Error:")
            self.ln(5)

            self.set_font('Courier', '', 8)
            self.set_text_color(180, 0, 0)
            self.set_fill_color(255, 240, 240)

            # Truncate very long error messages
            if len(error) > 500:
                error = error[:500] + "..."

            self._add_code_block(error, fill_color=(255, 240, 240))
            self.ln(3)

        # Final answer
        if is_final and output:
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(0, 128, 0)
            self.cell(0, 8, f"Final Answer: {sanitize_text(str(output))}")
            self.ln(8)

        # Separator line
        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def _add_thought_block(self, text: str):
        """Add a thought/reasoning block with background."""
        text = sanitize_text(text)
        self.set_fill_color(255, 250, 240)  # Light cream/orange

        lines = self._wrap_text(text, max_width=180)

        start_y = self.get_y()
        block_height = len(lines) * 4 + 4

        # Check if we need a new page
        if start_y + block_height > 280:
            self.add_page()
            start_y = self.get_y()

        # Draw background with left border
        self.set_draw_color(200, 150, 100)  # Brown border
        self.rect(10, start_y, 190, block_height, 'F')
        self.line(10, start_y, 10, start_y + block_height)  # Left accent line

        # Add text
        self.set_xy(14, start_y + 2)
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(60, 60, 60)
        for line in lines:
            self.cell(0, 4, line)
            self.ln(4)
            self.set_x(14)

        self.set_y(start_y + block_height + 2)

    def _add_code_block(self, text: str, fill_color=(245, 245, 245)):
        """Add a code block with background."""
        text = sanitize_text(text)
        self.set_fill_color(*fill_color)

        # Calculate block dimensions
        lines = self._wrap_text(text, max_width=180)

        # Draw background rectangle
        start_y = self.get_y()
        block_height = len(lines) * 4 + 4  # 4mm per line + padding

        # Check if we need a new page
        if start_y + block_height > 280:
            self.add_page()
            start_y = self.get_y()

        self.rect(10, start_y, 190, block_height, 'F')

        # Add text
        self.set_xy(12, start_y + 2)
        for line in lines:
            self.cell(0, 4, line)
            self.ln(4)
            self.set_x(12)

        self.set_y(start_y + block_height + 2)

    def _add_text_block(self, text: str):
        """Add a regular text block."""
        text = sanitize_text(text)
        lines = self._wrap_text(text, max_width=180)
        for line in lines:
            self.cell(0, 4, line)
            self.ln(4)

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width (in mm)."""
        result = []
        # Approximate: 1 character ≈ 1.8mm in Courier 8pt
        chars_per_line = int(max_width / 1.8)

        for line in text.split('\n'):
            # Handle empty lines
            if not line:
                result.append('')
                continue

            # Wrap long lines
            while len(line) > chars_per_line:
                # Try to break at space
                break_at = line.rfind(' ', 0, chars_per_line)
                if break_at == -1:
                    break_at = chars_per_line

                result.append(line[:break_at])
                line = line[break_at:].lstrip()

            result.append(line)

        return result


class DataflowTracePDF(FPDF):
    """Custom PDF class for dataflow trace documents with notebook-like layout."""

    # Maximum characters for tool output before truncation
    MAX_OUTPUT_LENGTH = 500

    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=15)
        self.cell_counter = 0

    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, self.title_text, align='C')
        self.ln(5)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_user_prompt(self, content: str):
        """Add the user prompt/question cell."""
        content = sanitize_text(content)
        self.cell_counter += 1

        # Cell header
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 100, 0)  # Dark green
        self.cell(0, 6, f"[User Prompt]")
        self.ln(6)

        # Content
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(240, 255, 240)  # Light green background

        # Truncate if too long
        if len(content) > 1500:
            # Find the question part
            question_start = content.find("Question:")
            if question_start != -1:
                # Show abbreviated version + question
                abbreviated = content[:200] + "\n\n... [instructions truncated] ...\n\n" + content[question_start:]
                content = abbreviated
            else:
                content = content[:1500] + "\n... [truncated]"

        self._add_text_block_with_bg(content, fill_color=(240, 255, 240))
        self.ln(5)

    def add_tool_call(self, tool_name: str, tool_call_id: str, inputs: Dict[str, Any]):
        """Add a tool call cell (like a code cell in a notebook)."""
        self.cell_counter += 1

        # Cell header with tool name
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 0, 180)  # Blue
        self.cell(0, 6, f"[{self.cell_counter}] Tool Call: {tool_name}")
        self.ln(6)

        # Tool call ID (smaller, gray)
        self.set_font('Courier', '', 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 4, f"call_id: {tool_call_id}")
        self.ln(5)

        # Parameters section
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, "Parameters:")
        self.ln(5)

        # Handle different tool types
        if tool_name in ('addOperator', 'modifyOperator', 'createOrModifyOperator'):
            self._add_operator_params(inputs)
        elif tool_name == 'executeWorkflow':
            self._add_execute_params(inputs)
        else:
            # Generic parameter display
            self._add_generic_params(inputs)

        # Separator
        self.set_draw_color(200, 200, 200)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)

    def _add_operator_params(self, inputs: Dict[str, Any]):
        """Add parameters for addOperator/modifyOperator calls."""
        # Operator ID
        if 'operatorId' in inputs:
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(100, 50, 0)  # Brown
            self.cell(25, 4, "operatorId:")
            self.set_font('Courier', '', 8)
            self.set_text_color(0, 0, 0)
            self.cell(0, 4, str(inputs['operatorId']))
            self.ln(5)

        # Summary
        if 'summary' in inputs:
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(100, 50, 0)
            self.cell(25, 4, "summary:")
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 4, str(inputs['summary']))
            self.ln(2)

        # Code (formatted as Python)
        if 'code' in inputs:
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(100, 50, 0)
            self.cell(0, 4, "code:")
            self.ln(4)

            self._add_python_code_block(inputs['code'])

    def _add_execute_params(self, inputs: Dict[str, Any]):
        """Add parameters for executeWorkflow calls."""
        if 'operatorId' in inputs:
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(100, 50, 0)
            self.cell(25, 4, "operatorId:")
            self.set_font('Courier', '', 8)
            self.set_text_color(0, 0, 0)
            self.cell(0, 4, str(inputs['operatorId']))
            self.ln(5)

    def _add_generic_params(self, inputs: Dict[str, Any]):
        """Add generic parameter display."""
        for key, value in inputs.items():
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(100, 50, 0)
            self.cell(30, 4, f"{key}:")
            self.set_font('Courier', '', 8)
            self.set_text_color(0, 0, 0)

            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            self.multi_cell(0, 4, value_str)
            self.ln(2)

    def _add_python_code_block(self, code: str):
        """Add a Python code block with syntax-like formatting, handling page breaks."""
        code = sanitize_text(code)
        lines = self._wrap_text(code, max_width=175)

        # Process lines in chunks that fit on pages
        line_height = 4
        page_bottom = 270  # Leave margin at bottom
        chunk_padding = 4  # Padding inside code block

        i = 0
        while i < len(lines):
            start_y = self.get_y()

            # Calculate how many lines fit on this page
            available_height = page_bottom - start_y - chunk_padding * 2
            lines_that_fit = max(1, int(available_height / line_height))

            # Get chunk of lines for this page
            chunk_end = min(i + lines_that_fit, len(lines))
            chunk_lines = lines[i:chunk_end]

            # Calculate block height for this chunk
            block_height = len(chunk_lines) * line_height + chunk_padding * 2

            # Draw background with border
            self.set_fill_color(248, 248, 248)
            self.set_draw_color(200, 200, 200)
            self.rect(12, start_y, 186, block_height, 'FD')

            # Add line numbers and code
            self.set_xy(14, start_y + chunk_padding)
            for j, line in enumerate(chunk_lines):
                line_num = i + j + 1
                # Line number
                self.set_font('Courier', '', 7)
                self.set_text_color(150, 150, 150)
                self.cell(8, line_height, f"{line_num:3d}")

                # Code line with basic syntax highlighting
                self.set_font('Courier', '', 8)
                self._add_highlighted_code_line(line)
                self.ln(line_height)
                self.set_x(14)

            self.set_y(start_y + block_height + 2)

            i = chunk_end

            # Add new page if there are more lines
            if i < len(lines):
                self.add_page()

    def _add_highlighted_code_line(self, line: str):
        """Add a code line with basic syntax coloring."""
        # Keywords to highlight
        keywords = {'def', 'return', 'import', 'from', 'if', 'else', 'elif', 'for', 'while',
                    'in', 'not', 'and', 'or', 'True', 'False', 'None', 'as', 'with', 'try',
                    'except', 'finally', 'class', 'lambda', 'yield', 'raise', 'pass', 'break',
                    'continue', 'global', 'nonlocal', 'assert', 'del'}

        # Simple approach: just color the whole line based on content
        stripped = line.strip()

        if stripped.startswith('#'):
            # Comment - green
            self.set_text_color(0, 128, 0)
        elif stripped.startswith(('def ', 'class ')):
            # Function/class definition - blue
            self.set_text_color(0, 0, 180)
        elif stripped.startswith(('import ', 'from ')):
            # Import - purple
            self.set_text_color(128, 0, 128)
        elif stripped.startswith(('return ', 'yield ')):
            # Return - dark red
            self.set_text_color(180, 0, 0)
        else:
            # Normal code - black
            self.set_text_color(0, 0, 0)

        self.cell(0, 4, line)

    def add_tool_result(self, tool_name: str, tool_call_id: str, output: Dict[str, Any]):
        """Add a tool result cell (like output cell in a notebook)."""
        # Result header
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(0, 128, 0)  # Green
        self.cell(0, 5, f"Output [{tool_name}]:")
        self.ln(5)

        # Extract output text
        output_text = ""
        if isinstance(output, dict):
            if output.get('type') == 'text':
                output_text = output.get('value', '')
            else:
                output_text = str(output)
        else:
            output_text = str(output)

        # Check for errors
        is_error = output_text.startswith('[ERROR]')

        # Truncate if too long
        truncated = False
        if len(output_text) > self.MAX_OUTPUT_LENGTH:
            output_text = output_text[:self.MAX_OUTPUT_LENGTH]
            truncated = True

        # Display output
        if is_error:
            self._add_text_block_with_bg(output_text, fill_color=(255, 235, 235))  # Light red
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(180, 0, 0)
        else:
            self._add_text_block_with_bg(output_text, fill_color=(245, 250, 255))  # Light blue

        if truncated:
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(128, 128, 128)
            self.cell(0, 4, "... [output truncated]")
            self.ln(4)

        self.ln(5)

    def add_assistant_text(self, text: str):
        """Add assistant's final text response."""
        text = sanitize_text(text)
        self.cell_counter += 1

        # Header
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(128, 0, 128)  # Purple
        self.cell(0, 6, f"[{self.cell_counter}] Assistant Response")
        self.ln(6)

        # Check for final answer
        final_answer = None
        if '**Final Answer:' in text:
            parts = text.split('**Final Answer:')
            explanation = parts[0].strip()
            final_answer = parts[1].replace('**', '').strip()
        else:
            explanation = text

        # Explanation text
        if explanation:
            self.set_font('Helvetica', '', 9)
            self.set_text_color(0, 0, 0)
            self._add_text_block_with_bg(explanation, fill_color=(250, 250, 255))
            self.ln(3)

        # Final answer box
        if final_answer:
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(0, 100, 0)
            self.set_fill_color(220, 255, 220)
            self.set_draw_color(0, 150, 0)

            # Draw answer box
            start_y = self.get_y()
            answer_text = f"Final Answer: {sanitize_text(final_answer)}"
            self.rect(10, start_y, 190, 10, 'FD')
            self.set_xy(15, start_y + 3)
            self.cell(0, 4, answer_text)
            self.ln(15)

    def _add_text_block_with_bg(self, text: str, fill_color=(245, 245, 245)):
        """Add a text block with background color."""
        text = sanitize_text(text)
        self.set_fill_color(*fill_color)

        lines = self._wrap_text(text, max_width=180)

        start_y = self.get_y()
        block_height = len(lines) * 4 + 4

        # Check if we need a new page
        if start_y + block_height > 275:
            self.add_page()
            start_y = self.get_y()

        # Draw background
        self.rect(10, start_y, 190, block_height, 'F')

        # Add text
        self.set_xy(12, start_y + 2)
        self.set_font('Courier', '', 8)
        self.set_text_color(0, 0, 0)

        for line in lines:
            self.cell(0, 4, line)
            self.ln(4)
            self.set_x(12)

        self.set_y(start_y + block_height + 2)

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width (in mm)."""
        result = []
        chars_per_line = int(max_width / 1.8)

        for line in text.split('\n'):
            if not line:
                result.append('')
                continue

            while len(line) > chars_per_line:
                break_at = line.rfind(' ', 0, chars_per_line)
                if break_at == -1:
                    break_at = chars_per_line

                result.append(line[:break_at])
                line = line[break_at:].lstrip()

            result.append(line)

        return result


def build_dataflow_pdf(trace: List[Dict[str, Any]], output_path: str, title: str) -> None:
    """Build PDF document from dataflow trace."""
    pdf = DataflowTracePDF(title)
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(0, 0, 100)
    pdf.cell(0, 10, title, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Dataflow Agent Trace", align='C')
    pdf.ln(12)

    # Track tool calls for matching with results
    pending_tool_calls: Dict[str, Dict] = {}

    for message in trace:
        role = message.get('role', '')
        content = message.get('content', '')

        if role == 'user':
            # User prompt
            if isinstance(content, str):
                pdf.add_user_prompt(content)

        elif role == 'assistant':
            # Assistant can have tool calls or text
            if isinstance(content, list):
                for item in content:
                    item_type = item.get('type', '')

                    if item_type == 'tool-call':
                        tool_call_id = item.get('toolCallId', '')
                        tool_name = item.get('toolName', '')
                        inputs = item.get('input', {})

                        pdf.add_tool_call(tool_name, tool_call_id, inputs)
                        pending_tool_calls[tool_call_id] = {
                            'toolName': tool_name,
                            'input': inputs
                        }

                    elif item_type == 'text':
                        text = item.get('text', '')
                        pdf.add_assistant_text(text)

            elif isinstance(content, str):
                pdf.add_assistant_text(content)

        elif role == 'tool':
            # Tool results
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'tool-result':
                        tool_call_id = item.get('toolCallId', '')
                        tool_name = item.get('toolName', '')
                        output = item.get('output', {})

                        # Skip executeWorkflow errors (they're noise)
                        if tool_name == 'executeWorkflow':
                            output_text = output.get('value', '') if isinstance(output, dict) else str(output)
                            if '[ERROR]' in output_text:
                                continue

                        pdf.add_tool_result(tool_name, tool_call_id, output)

    # Save PDF
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")


def detect_trace_format(trace: List[Dict[str, Any]]) -> str:
    """Auto-detect trace format based on structure."""
    if not trace:
        return 'codeagent'

    # Dataflow format has messages with 'role' field
    if isinstance(trace[0], dict) and 'role' in trace[0]:
        return 'dataflow'

    # Code agent format has 'step' field
    if isinstance(trace[0], dict) and 'step' in trace[0]:
        return 'codeagent'

    return 'codeagent'


def load_trace(path: str) -> List[Dict[str, Any]]:
    """Load trace JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_thought_from_model_output(model_output: str) -> str:
    """Extract the thought/reasoning from model_output before <code> tag."""
    if not model_output:
        return ""

    # Find the <code> tag
    code_start = model_output.find('<code>')
    if code_start == -1:
        # No code tag, the entire output is thought
        thought = model_output.strip()
    else:
        # Extract text before <code>
        thought = model_output[:code_start].strip()

    # Clean up common prefixes
    if thought.startswith('Thought:'):
        thought = thought[len('Thought:'):].strip()
    elif thought.startswith('Thought :'):
        thought = thought[len('Thought :'):].strip()

    return thought


def build_pdf(trace: List[Dict[str, Any]], output_path: str, title: str) -> None:
    """Build PDF document from CodeAgent trace."""
    pdf = TracePDF(title)
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(0, 0, 100)
    pdf.cell(0, 10, title, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "CodeAgent Reasoning Trace", align='C')
    pdf.ln(12)

    # Process each step
    for step_data in trace:
        step_num = step_data.get('step', '?')
        code = step_data.get('code', '')
        observations = step_data.get('observations', '')
        error = step_data.get('error', None)
        output = step_data.get('output', None)
        is_final = step_data.get('is_final_answer', False)
        model_output = step_data.get('model_output', '')

        # Extract thought from model_output
        thought = extract_thought_from_model_output(model_output)

        # Extract error message
        error_msg = None
        if error:
            if isinstance(error, dict):
                error_msg = error.get('message', str(error))
            else:
                error_msg = str(error)

        pdf.add_step(
            step_num=step_num,
            code=code,
            observations=observations,
            error=error_msg,
            output=output,
            is_final=is_final,
            thought=thought
        )

    # Save PDF
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert agent trace JSON to PDF'
    )
    parser.add_argument('trace_path', help='Path to trace JSON file')
    parser.add_argument('--output', '-o', help='Output PDF path (default: same name as input with .pdf)')
    parser.add_argument('--format', '-f', choices=['codeagent', 'dataflow', 'auto'],
                        default='auto', help='Trace format (default: auto-detect)')

    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        print(f"Error: File not found: {trace_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(trace_path.with_suffix('.pdf'))

    # Load trace
    trace = load_trace(args.trace_path)

    # Detect or use specified format
    if args.format == 'auto':
        trace_format = detect_trace_format(trace)
        print(f"Auto-detected format: {trace_format}")
    else:
        trace_format = args.format

    # Generate title from filename or parent directory
    if trace_path.stem == 'dataflow_trace':
        # Use parent directory name for dataflow traces
        title = trace_path.parent.name.replace('_', ' ').replace('-', ' ').title()
    else:
        title = trace_path.stem.replace('_', ' ').replace('-', ' ').title()

    # Build PDF based on format
    if trace_format == 'dataflow':
        build_dataflow_pdf(trace, output_path, title)
    else:
        build_pdf(trace, output_path, title)

    return 0


if __name__ == "__main__":
    exit(main())

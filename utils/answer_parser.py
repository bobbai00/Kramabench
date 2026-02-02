# -*- coding: utf-8 -*-
"""
Utility functions for parsing agent answers from responses.

This module provides shared logic for extracting final answers from
agent responses, used by both CodeAgentSystem and DataflowSystem.
"""

import json
import re
from typing import List, Optional


def parse_answer(response: str, messages: Optional[List[dict]] = None) -> str:
    """
    Extract the answer from agent response or messages.

    Args:
        response: Raw response from the agent
        messages: List of conversation messages (used if response is empty)

    Returns:
        Parsed answer string
    """
    # If we have a response, parse it
    if response:
        # First, look for explicit "Final Answer:" pattern (preferred)
        # NOTE: The colon is REQUIRED to avoid matching phrases like
        # "provide the final answer." which would incorrectly capture "."
        # Use findall to get ALL matches, then take the LAST one (most likely the actual answer)
        final_answer_pattern = r'\*?\*?Final Answer:\*?\*?\s*(.+?)(?:\n|$)'
        final_answer_matches = re.findall(final_answer_pattern, response, re.IGNORECASE)
        if final_answer_matches:
            # Take the last match (the final "Final Answer:" in the response)
            answer = final_answer_matches[-1].strip()
            # Clean markdown formatting
            answer = re.sub(r'^\*\*|\*\*$', '', answer).strip()
            answer = re.sub(r'^`|`$', '', answer).strip()
            # If it looks like a JSON array, return as-is
            if answer.startswith('[') and answer.endswith(']'):
                return answer
            # Remove trailing punctuation
            answer = answer.rstrip('.')
            return answer

        response_lower = response.lower()

        # Look for "final answer" section and extract the value after it
        markers = ["final answer", "answer:", "the answer is", "result:"]
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.find(marker)
                # Get text after the marker
                after_marker = response[idx + len(marker):]

                # Check for JSON array first (for list answers)
                json_array_match = re.search(r'(\[.*?\])', after_marker, re.DOTALL)
                if json_array_match:
                    try:
                        # Validate it's proper JSON
                        json.loads(json_array_match.group(1))
                        return json_array_match.group(1)
                    except json.JSONDecodeError:
                        pass

                # Look for a number (possibly with markdown formatting)
                # Pattern matches: **12964.8727**, 12964.8727, `12964.8727`, etc.
                number_pattern = r'[*_`:\s]*(-?[\d,]+\.?\d*)[*_`]*'
                match = re.search(number_pattern, after_marker)
                if match and match.group(1):
                    answer = match.group(1).replace(',', '')
                    if answer:  # Make sure it's not empty
                        return answer

                # If no number, take first non-empty content line
                lines = [l.strip() for l in after_marker.split('\n') if l.strip()]
                for line in lines:
                    # Skip markdown headers and empty-ish lines
                    if line.startswith('#') or line.startswith('-'):
                        continue
                    # Clean up markdown formatting
                    cleaned = re.sub(r'[*_`]', '', line).strip()
                    if cleaned:
                        return cleaned

        # If no marker found, look for numbers in the response
        # Often the answer is a standalone number
        number_match = re.search(r'\*\*(-?[\d,]+\.?\d*)\*\*', response)
        if number_match:
            return number_match.group(1).replace(',', '')

        # Return the last non-empty, non-header line
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for line in reversed(lines):
            if not line.startswith('#') and not line.startswith('-'):
                cleaned = re.sub(r'[*_`]', '', line).strip()
                if cleaned:
                    return cleaned

        return response.strip()

    # If response is empty, try to extract from messages
    if messages:
        # Look for the last assistant message with text content
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                return parse_answer(text, None)

        # If no assistant text, try to get data from last tool result
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool-result":
                            output = item.get("output", {})
                            if isinstance(output, dict):
                                value = output.get("value", "")
                                if value:
                                    # Return the tool output as the answer
                                    return f"(From tool result) {value[:500]}"

    return "No response from agent"

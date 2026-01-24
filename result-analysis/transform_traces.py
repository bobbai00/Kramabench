#!/usr/bin/env python3
"""Transform dataflow_trace.json files to match the haiku-4.5-example.json format."""

import json
import os
import glob

def extract_final_response(messages):
    """Extract the final response text from the last assistant message."""
    # Look for the last assistant message with text content
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            # Check if content is a list (tool calls or text)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
            # Check if content is a string directly
            elif isinstance(content, str):
                return content
    return ""

def transform_trace(input_path):
    """Transform a single dataflow_trace.json file to the new format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    # Extract the final response
    response = extract_final_response(messages)

    # Create the new format
    transformed = {
        "response": response,
        "messages": messages
    }

    return transformed

def main():
    base_dir = "/Users/baijiadong/Desktop/chenlab/KramaBench/result-analysis/o4mini"

    # Find all dataflow_trace.json files
    pattern = os.path.join(base_dir, "**/dataflow_trace.json")
    trace_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(trace_files)} dataflow_trace.json files")

    for trace_file in trace_files:
        print(f"Processing: {trace_file}")

        try:
            transformed = transform_trace(trace_file)

            # Write back to the same file
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(transformed, f, indent=2, ensure_ascii=False)

            response_preview = transformed["response"][:100] + "..." if len(transformed["response"]) > 100 else transformed["response"]
            print(f"  -> Response: {response_preview}")

        except Exception as e:
            print(f"  -> Error: {e}")

    print(f"\nTransformed {len(trace_files)} files")

if __name__ == "__main__":
    main()

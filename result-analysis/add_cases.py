#!/usr/bin/env python3
"""Add additional representative cases to result-analysis/o4mini/2_dataflow_succeed/"""

import os
import json
import shutil
from pathlib import Path

BASE_DIR = Path("/Users/baijiadong/Desktop/chenlab/KramaBench")
SCRATCH_DIR = BASE_DIR / "system_scratch"
OUTPUT_DIR = BASE_DIR / "result-analysis" / "o4mini" / "2_dataflow_succeed"
WORKLOAD_DIR = BASE_DIR / "workload"

CODE_AGENT = "CodeAgentSystemO4Mini"
DATAFLOW_SYSTEM = "DataflowSystemO4Mini"

# Cases to add (representative of recovery/debugging difficulties)
CASES_TO_ADD = [
    "legal-hard-22",      # 16+ steps retry exhaustion, CSV parsing nightmare
    "environment-hard-8", # Multi-step CSV header detection struggles
]

def get_domain(task_id: str) -> str:
    """Extract domain from task_id."""
    return task_id.rsplit('-', 2)[0]

def get_prompt(task_id: str) -> str:
    """Get task prompt from workload."""
    domain = get_domain(task_id)
    workload_file = WORKLOAD_DIR / f"{domain}.json"
    if workload_file.exists():
        with open(workload_file) as f:
            workload = json.load(f)
        for task in workload:
            if task.get("task_id") == task_id:
                return task.get("question", "")
    return ""

def get_answer(task_id: str) -> str:
    """Get expected answer from workload."""
    domain = get_domain(task_id)
    workload_file = WORKLOAD_DIR / f"{domain}.json"
    if workload_file.exists():
        with open(workload_file) as f:
            workload = json.load(f)
        for task in workload:
            if task.get("task_id") == task_id:
                return str(task.get("answer", ""))
    return ""

def transform_code_trace(trace_path: Path) -> list:
    """Transform code agent trace to standard format."""
    with open(trace_path) as f:
        return json.load(f)

def transform_dataflow_trace(messages_path: Path) -> dict:
    """Transform dataflow messages to standard format with response field."""
    with open(messages_path) as f:
        messages = json.load(f)

    # Extract final response from last assistant message
    response = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        response = item.get("text", "")
                        break
            elif isinstance(content, str):
                response = content
            if response:
                break

    return {
        "response": response,
        "messages": messages
    }

def create_scores_json(task_id: str) -> dict:
    """Create scores.json for the task."""
    return {
        "task_id": task_id,
        "code_agent": {
            "score": 0.0,
            "metric": "success"
        },
        "dataflow": {
            "score": 1.0,
            "metric": "success"
        }
    }

def add_case(task_id: str):
    """Add a single case to the output directory."""
    print(f"\nProcessing {task_id}...")

    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Get paths
    code_scratch = SCRATCH_DIR / CODE_AGENT / task_id
    df_scratch = SCRATCH_DIR / DATAFLOW_SYSTEM / task_id

    # 1. Write prompt.txt
    prompt = get_prompt(task_id)
    with open(task_dir / "prompt.txt", "w") as f:
        f.write(prompt)
    print(f"  - Wrote prompt.txt")

    # 2. Write answer.txt
    answer = get_answer(task_id)
    with open(task_dir / "answer.txt", "w") as f:
        f.write(answer)
    print(f"  - Wrote answer.txt ({answer[:50]}...)" if len(answer) > 50 else f"  - Wrote answer.txt ({answer})")

    # 3. Transform and write code_trace.json
    code_trace_path = code_scratch / "reasoning_trace.json"
    if code_trace_path.exists():
        code_trace = transform_code_trace(code_trace_path)
        with open(task_dir / "code_trace.json", "w") as f:
            json.dump(code_trace, f, indent=2, ensure_ascii=False)
        print(f"  - Wrote code_trace.json ({len(code_trace)} steps)")
    else:
        print(f"  - WARNING: No code trace found at {code_trace_path}")

    # 4. Transform and write dataflow_trace.json
    df_messages_path = df_scratch / "messages.json"
    if df_messages_path.exists():
        df_trace = transform_dataflow_trace(df_messages_path)
        with open(task_dir / "dataflow_trace.json", "w") as f:
            json.dump(df_trace, f, indent=2, ensure_ascii=False)
        print(f"  - Wrote dataflow_trace.json ({len(df_trace['messages'])} messages)")
    else:
        print(f"  - WARNING: No dataflow trace found at {df_messages_path}")

    # 5. Write scores.json
    scores = create_scores_json(task_id)
    with open(task_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  - Wrote scores.json")

def main():
    print("Adding representative cases to 2_dataflow_succeed...")

    for task_id in CASES_TO_ADD:
        add_case(task_id)

    print(f"\nDone! Added {len(CASES_TO_ADD)} cases.")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

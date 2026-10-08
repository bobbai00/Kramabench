# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""Run DSGym's ReAct agent on one task. Executed by DSGym's own interpreter:

    <DSGym>/.venv/bin/python systems/dsgym_runner.py request.json

The agent is upstream code throughout: `create_custom_task` builds the system
and user messages, `ReActDSAgent.solve_task` runs the loop against the executor
manager. This script adds what upstream does not keep: token usage and seconds
for every model call, seconds for every code execution, and the output limit
that `clean_jupyter_output` declares (`max_stdout_length`) but never applies.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import types

REQUEST_FIELDS = (
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "top_p",
    "reasoning_effort",
    "timeout",
)


def limit_output(text: str, limit: int) -> str:
    """Cut the way upstream cuts error tracebacks (`max_error_length`)."""
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def bind_packages(home: Path) -> None:
    """Make `dsgym.agents` and `dsgym.datasets.utils` importable on their own.

    `dsgym/__init__.py` and `dsgym/datasets/__init__.py` import every dataset
    loader (Kaggle, Selenium, ...). Binding the two packages to their
    directories skips those inits; the modules the agent uses load as they are.
    """
    sys.path.insert(0, str(home))
    for name in ("dsgym", "dsgym.datasets"):
        package = types.ModuleType(name)
        package.__path__ = [str(home / name.replace(".", "/"))]
        sys.modules[name] = package


def usage_totals(trace: Path) -> dict:
    """Tokens and seconds over the model calls a trace file records."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
        "model_calls": 0,
        "failed_model_calls": 0,
        "model_seconds": 0.0,
    }
    if not trace.exists():
        return totals
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        if event["phase"] not in ("response", "error"):
            continue
        totals["model_seconds"] += event["seconds"]
        if event["phase"] == "error":
            totals["failed_model_calls"] += 1
            continue
        usage = event["usage"]
        prompt_details = usage.get("prompt_tokens_details") or {}
        totals["model_calls"] += 1
        totals["input_tokens"] += usage.get("prompt_tokens") or 0
        totals["output_tokens"] += usage.get("completion_tokens") or 0
        totals["cached_tokens"] += prompt_details.get("cached_tokens") or 0
        totals["cache_creation_tokens"] += (
            prompt_details.get("cache_creation_tokens") or 0
        )
        totals["reasoning_tokens"] += (
            usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens") or 0
    return totals


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    out = Path(request["output_dir"])
    bind_packages(Path(request["dsgym_home"]))
    import httpx
    import litellm
    from dsgym.agents.environment.envs.allocated_code import env as environment
    from dsgym.agents.react_ds_agent import ReActDSAgent
    from dsgym.datasets.utils import create_custom_task

    httpx.get(request["manager_url"] + "/health", timeout=10).raise_for_status()
    trace, executions = out / "model-request-trace.jsonl", out / "executions.jsonl"

    def append(path, event):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with path.open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    completion, calls = litellm.completion, 0

    def recorded_completion(*args, **kwargs):
        nonlocal calls
        calls += 1
        number, started = calls, time.monotonic()
        # The gateway's model names carry no provider prefix; they speak the OpenAI API.
        kwargs.setdefault("custom_llm_provider", "openai")
        append(
            trace,
            {
                "phase": "request",
                "call": number,
                "request": {
                    key: kwargs[key] for key in REQUEST_FIELDS if key in kwargs
                },
            },
        )
        try:
            response = completion(*args, **kwargs)
        except Exception as error:
            append(
                trace,
                {
                    "phase": "error",
                    "call": number,
                    "exception": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "message": str(error)[:500],
                    "seconds": time.monotonic() - started,
                },
            )
            raise
        append(
            trace,
            {
                "phase": "response",
                "call": number,
                "response_model": response.model,
                "usage": response.usage.model_dump(),
                "choices": [choice.model_dump() for choice in response.choices],
                "seconds": time.monotonic() - started,
            },
        )
        return response

    litellm.completion = recorded_completion

    limit, clean, cleaned = (
        request["max_output_length"],
        environment.clean_jupyter_output,
        [],
    )

    def limited_output(raw_output, *args, **kwargs):
        text = clean(raw_output, *args, **kwargs)
        cleaned.append(len(text))
        return limit_output(text, limit)

    environment.clean_jupyter_output = limited_output

    tools = environment.AllocatedCodeToolGroup
    execute, allocate = tools.execute_code, tools.allocate_container

    def timed_execute(self, code):
        started, failure = time.monotonic(), None
        del cleaned[:]
        try:
            return execute(self, code)
        except Exception as error:
            failure = str(error)[:500]
            raise
        finally:
            size = cleaned[-1] if cleaned else None
            append(
                executions,
                {
                    "event": "execute",
                    "after_model_call": calls,
                    "seconds": time.monotonic() - started,
                    "code_chars": len(code),
                    "output_chars": size,
                    "truncated": size is not None and size > limit,
                    "error": failure,
                },
            )

    def timed_allocate(self):
        started = time.monotonic()
        try:
            return allocate(self)
        finally:
            append(
                executions, {"event": "allocate", "seconds": time.monotonic() - started}
            )

    tools.execute_code, tools.allocate_container = timed_execute, timed_allocate

    task = request["task"]
    sample = create_custom_task(
        query=task["query"], data_files=task["data_files"], context=task["context"]
    )
    (out / "prompt.json").write_text(
        json.dumps(sample["prompt"], indent=2, ensure_ascii=False) + "\n"
    )
    agent = ReActDSAgent(
        backend="litellm",
        model=request["model"],
        manager_url=request["manager_url"],
        max_turns=request["max_turns"],
        output_dir=str(out / "dsgym-outputs"),
        api_key=os.environ[request["api_key_env"]],
        base_url=request["api_base"],
    )
    started = time.time()
    result = agent.solve_task(sample)
    elapsed = time.time() - started
    events = (
        [json.loads(line) for line in executions.read_text().splitlines()]
        if executions.exists()
        else []
    )
    (out / "result.json").write_text(
        json.dumps(
            {
                "answer": result["solution"],
                "turns": result["turns"],
                "error": result["error"],
                "metadata": result["metadata"],
                "conversation": result["conversation"],
                "elapsed_seconds": elapsed,
                "execution_seconds": sum(
                    e["seconds"] for e in events if e["event"] == "execute"
                ),
                "executions": sum(e["event"] == "execute" for e in events),
                "truncated_outputs": sum(bool(e.get("truncated")) for e in events),
                **usage_totals(trace),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

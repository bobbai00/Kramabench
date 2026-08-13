# -*- coding: utf-8 -*-
"""
Python execution backends for ClaudeCodeSystem.

The agent runs with the Bash tool removed from its context entirely, so this
module is its only path to executing code. Two backends:

* StatelessExecutor  - fresh `python step_NN.py` subprocess per call. Nothing
                       carries over; the agent must re-load data every step.
* PersistentExecutor - one long-lived worker process (`_python_worker.py`)
                       holding a module namespace across calls, matching
                       smolagents CodeAgent semantics.

Both persist every snippet to `step_NN.py` in the task work dir, so a run is
replayable from disk regardless of backend.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_python_worker.py")

# Description text handed to the model. The two modes MUST advertise different
# state semantics: a stateless agent that assumes persistence burns turns on
# NameError, and a persistent agent that assumes it lacks state re-loads every
# CSV on every step and inflates token usage.
_DESC_STATELESS = (
    "Execute a Python snippet and return its stdout and stderr. "
    "Each call runs in a FRESH process: variables, imports, and loaded "
    "DataFrames from previous calls are GONE. Every snippet must be "
    "self-contained - re-import and re-load any data it needs. "
    "Print anything you want to see; nothing is returned implicitly."
)
_DESC_PERSISTENT = (
    "Execute a Python snippet and return its stdout and stderr. "
    "State PERSISTS across calls: variables, imports, and loaded DataFrames "
    "from previous calls are still available, like cells in a notebook. "
    "Print anything you want to see; nothing is returned implicitly."
)


def _truncate(text: str, limit: Optional[int]) -> str:
    if limit is None or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... [truncated, {omitted} more characters]"


def _truncate_tail(text: str, limit: Optional[int]) -> str:
    """Keep the END of `text`, not the start.

    Used for tracebacks: the exception type and message are the last line, so
    head-truncating a long traceback throws away the only part that says what
    actually went wrong.
    """
    if limit is None or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"... [truncated, {omitted} earlier characters]\n" + text[-limit:]


class _BaseExecutor:
    """Shared snippet-persistence and output-shaping logic."""

    def __init__(self, work_dir: str, python_bin: str, timeout: int, max_output_chars: Optional[int]):
        # Absolute: the snippet path is handed to a subprocess whose cwd is
        # this same directory, so a relative work_dir resolves twice and the
        # interpreter looks for work/work/step_NN.py.
        self.work_dir = os.path.abspath(work_dir)
        self.python_bin = python_bin
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.call_count = 0
        os.makedirs(work_dir, exist_ok=True)

    def _persist_snippet(self, code: str) -> str:
        self.call_count += 1
        path = os.path.join(self.work_dir, f"step_{self.call_count:02d}.py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def _format(self, output: str, error: str) -> Dict[str, Any]:
        body = _truncate(output if output.strip() else "(no output)", self.max_output_chars)
        if error:
            body = f"{body}\n--- error ---\n{_truncate_tail(error, self.max_output_chars)}"
        return {
            "content": [{"type": "text", "text": body}],
            "is_error": bool(error),
        }

    def close(self) -> None:
        pass


class StatelessExecutor(_BaseExecutor):
    """Fresh subprocess per call. No state carries over."""

    mode = "stateless"
    description = _DESC_STATELESS

    def run(self, code: str) -> Dict[str, Any]:
        path = self._persist_snippet(code)
        try:
            completed = subprocess.run(
                [self.python_bin, path],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._format("", f"Execution exceeded the {self.timeout}s timeout and was killed.")

        error = ""
        if completed.returncode != 0:
            # A bare sys.exit(N) writes nothing to stderr, so without this the
            # failure would be reported as a successful call with no output.
            error = completed.stderr or f"Process exited with status {completed.returncode} and no error output."
        return self._format(completed.stdout, error)


class PersistentExecutor(_BaseExecutor):
    """One long-lived worker holding a namespace across calls."""

    mode = "persistent"
    description = _DESC_PERSISTENT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [self.python_bin, WORKER_PATH],
            cwd=self.work_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _restart(self) -> None:
        self.close()
        self.restarts += 1
        self._start()

    def run(self, code: str) -> Dict[str, Any]:
        self._persist_snippet(code)
        if self.proc is None or self.proc.poll() is not None:
            self._start()

        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._restart()
            return self._format("", "The Python worker had died and was restarted. All previously "
                                    "defined variables are cleared - re-load any data you need.")

        line = _read_line_with_timeout(self.proc, self.timeout)
        if line is None:
            self._restart()
            return self._format("", f"Execution exceeded the {self.timeout}s timeout. The Python worker "
                                    "was restarted and ALL previously defined variables are cleared - "
                                    "re-load any data you need.")

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._restart()
            return self._format("", "The Python worker returned an unreadable response and was restarted. "
                                    "All previously defined variables are cleared.")

        return self._format(payload.get("output", ""), payload.get("error", ""))

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self.proc = None


def _read_line_with_timeout(proc: subprocess.Popen, timeout: int) -> Optional[str]:
    """Read one line from proc.stdout, or return None if it takes too long.

    `proc.stdout.readline()` has no timeout, so a snippet with an infinite loop
    would hang the whole benchmark run. Reading on a worker thread and joining
    with a deadline lets us give up and kill the process instead.
    """
    import threading

    result: list = []

    def _read():
        try:
            result.append(proc.stdout.readline())
        except Exception:
            result.append(None)

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive() or not result:
        return None
    return result[0]


def build_executor(exec_mode: str, work_dir: str, timeout: int,
                   max_output_chars: Optional[int], python_bin: Optional[str] = None) -> _BaseExecutor:
    """Construct the executor for `exec_mode` ("stateless" | "persistent")."""
    if exec_mode not in ("stateless", "persistent"):
        raise ValueError(f"exec_mode must be 'stateless' or 'persistent', got {exec_mode!r}")
    cls = StatelessExecutor if exec_mode == "stateless" else PersistentExecutor
    # sys.executable is the venv interpreter the harness itself runs under, so
    # the snippet sees the same pandas/pyarrow the benchmark was pinned to.
    return cls(work_dir=work_dir, python_bin=python_bin or sys.executable,
               timeout=timeout, max_output_chars=max_output_chars)

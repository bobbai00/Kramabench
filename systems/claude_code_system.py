# -*- coding: utf-8 -*-
"""
ClaudeCodeSystem - KramaBench System wrapper for the Claude Agent SDK.

Runs Claude Code as a library (`claude_agent_sdk`) rather than through the
LiteLLM gateway the other SUTs use. Authentication is inherited from the
`claude` CLI on this machine, so runs bill against the operator's own Claude
account, not an API key.

Three properties this wrapper guarantees:

1. **Python is the only tool.** `tools=[]` removes every built-in from the
   model's context - Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch,
   Task, TodoWrite. The sole tool is the in-process `run_python` MCP tool in
   `claude_code_executor`; the agent inspects files and computes with Python.

2. **Otherwise a stock Claude Code session.** The `claude_code` system-prompt
   preset, the CLI's own agent loop and tool implementations, and its default
   settings. One deliberate deviation: `setting_sources=["user"]` rather than
   the default user+project+local, because `cwd` lives under this repo and
   project scope would otherwise load KramaBench's `CLAUDE.md` - which
   documents the scoring pipeline and where ground truth is stored.

3. **One fresh session per task.** Each `serve_query` call is a one-off
   `query()` with no `resume`, no `continue_conversation`, and a wiped work
   dir, so no task can see another's history, files, or Python state.

Authentication is inherited from the `claude` CLI on this machine, so runs
bill against the operator's own Claude account rather than an API key.
"""

import asyncio
import contextlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from benchmark.benchmark_api import System
from systems.claude_code_executor import build_executor
from systems.data_source_utils import expand_data_sources
from utils.answer_parser import parse_answer

DEFAULT_MAX_STEPS = int(os.environ.get("CLAUDE_CODE_MAX_STEPS", 50))
DEFAULT_EXEC_TIMEOUT = int(os.environ.get("CLAUDE_CODE_EXEC_TIMEOUT", 300))
DEFAULT_MAX_OUTPUT_CHARS = int(os.environ.get("CLAUDE_CODE_MAX_OUTPUT_CHARS", 5000))

# Empty list = every built-in is removed from the model's context. The agent's
# ONLY tool is the in-process `run_python` MCP tool, so Bash, Read, Write,
# Edit, Glob, Grep, WebSearch, WebFetch, Task and TodoWrite are all absent.
BUILTIN_TOOLS: List[str] = []
PYTHON_TOOL = "mcp__py__run_python"

# Appended to Claude Code's own system prompt rather than replacing it. The
# preset describes a toolset this agent does not have, so without this note the
# model opens by trying to Read or Bash a file and burns turns discovering the
# tools are gone.
SYSTEM_PROMPT_APPEND = (
    "In this environment you have exactly one tool: `run_python`. "
    "There is no shell and no file-reading tool - inspect files, list "
    "directories, and compute results with Python code."
)


@contextlib.contextmanager
def subscription_auth():
    """Drop ANTHROPIC_API_KEY for the duration of an agent run.

    kb.py's load_env() sets ANTHROPIC_API_KEY=dummy because the litellm config
    references it, and the Claude CLI resolves an API key ahead of the OAuth
    profile - even an empty value wins that slot. Left in place, every agent
    call under kb.py would 401 while the same run works via evaluate.py.
    Restored on exit so the rest of the harness is unaffected.
    """
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def _usage_value(usage: Any, key: str) -> int:
    """Read a usage field whether the SDK hands back a dict or an object."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)


def _kb_usage(usage: Any) -> Dict[str, int]:
    """Per-step usage in the camelCase shape kb.py's step parsers read.

    `reasoningTokens` is always 0: the Anthropic API bills thinking inside
    output_tokens and never reports it as a separate figure, unlike the
    OpenAI-backed SUTs.
    """
    uncached = _usage_value(usage, "input_tokens")
    out = _usage_value(usage, "output_tokens")
    cache_read = _usage_value(usage, "cache_read_input_tokens")
    cache_write = _usage_value(usage, "cache_creation_input_tokens")
    # The Anthropic API reports input_tokens as the UNCACHED remainder, while
    # the litellm-backed SUTs report prompt tokens with cached ones included.
    # Report the inclusive figure so cross-arm comparisons - and kb.py's
    # cached/input cache-hit ratio - mean the same thing on both sides.
    return {
        "inputTokens": uncached + cache_read + cache_write,
        "uncachedInputTokens": uncached,
        "outputTokens": out,
        "cachedInputTokens": cache_read,
        "cacheCreationInputTokens": cache_write,
        "reasoningTokens": 0,
        "totalTokens": uncached + cache_read + cache_write + out,
    }


class ClaudeCodeSystem(System):
    """KramaBench System backed by the Claude Agent SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: Optional[str] = None,
        exec_mode: str = "stateless",
        max_steps: int = DEFAULT_MAX_STEPS,
        exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
        max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
        auth_mode: str = "subscription",
        verbose: bool = False,
        name: str = "ClaudeCodeSystem",
        *args, **kwargs
    ):
        super().__init__(name, verbose=verbose, *args, **kwargs)
        if exec_mode not in ("stateless", "persistent"):
            raise ValueError(f"exec_mode must be 'stateless' or 'persistent', got {exec_mode!r}")
        self.model = model
        self.effort = effort
        self.exec_mode = exec_mode
        self.max_steps = max_steps
        self.exec_timeout = exec_timeout
        self.max_output_chars = max_output_chars
        # "subscription" uses the `claude` CLI login and ignores any
        # ANTHROPIC_API_KEY in the environment; "api_key" leaves the
        # environment alone so a real key is used instead.
        self.auth_mode = auth_mode
        self.output_dir = f"./system_scratch/{name}"
        self.format_hints: Dict[str, str] = {}
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ setup

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        self.dataset_directory = dataset_directory
        self.dataset = {}
        for dirpath, _, filenames in os.walk(dataset_directory):
            for fname in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, fname), dataset_directory)
                self.dataset[rel_path] = None

        if self.verbose:
            print(f"[{self.name}] Found {len(self.dataset)} files in {dataset_directory}")

        self._load_format_hints(dataset_directory)

    def _load_format_hints(self, dataset_directory: str) -> None:
        """Load per-task answer-format hints for the domain, if present."""
        try:
            parts = str(dataset_directory).rstrip('/').split('/')
            if 'data' in parts:
                data_idx = parts.index('data')
                domain = parts[data_idx + 1]
                project_root = '/'.join(parts[:data_idx])
                hint_path = os.path.join(project_root, 'format_hint', f'{domain}.json')
                if os.path.exists(hint_path):
                    with open(hint_path, 'r') as f:
                        for hint in json.load(f):
                            self.format_hints[hint['id']] = hint.get('format_hint', '')
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] Could not load format hints: {e}")

    # ------------------------------------------------------------------ query

    def serve_query(
        self,
        query_text: str = "",
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        # The harness calls this with keyword `query=`; keep that name working
        # without shadowing the SDK's `query` function inside this module.
        query_text = kwargs.pop("query", query_text)
        if not self.dataset_directory:
            raise RuntimeError("Call process_dataset() first.")

        query_dir = os.path.join(self.output_dir, query_id)
        work_dir = os.path.join(query_dir, "work")
        # Wipe rather than reuse: step numbering restarts at 1 each run, so a
        # rerun that takes fewer steps would otherwise leave higher-numbered
        # step_NN.py files from the previous attempt sitting in the trace.
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)

        file_paths = self._resolve_file_paths(subset_files)
        prompt = self._build_prompt(query_text, query_id, file_paths)

        with open(os.path.join(query_dir, "prompt.txt"), "w") as f:
            f.write(prompt)
        with open(os.path.join(query_dir, "config.json"), "w") as f:
            json.dump({
                "model": self.model,
                # kb.py's cost command reads the model from config.json's
                # `model_type` (the key the other SUTs write) - mirror it so
                # this SUT doesn't report model "?".
                "model_type": self.model,
                "effort": self.effort,
                "exec_mode": self.exec_mode,
                "auth_mode": self.auth_mode,
                "max_steps": self.max_steps,
                "exec_timeout": self.exec_timeout,
                "max_output_chars": self.max_output_chars,
                "builtin_tools": BUILTIN_TOOLS,
            }, f, indent=2)

        executor = build_executor(
            exec_mode=self.exec_mode,
            work_dir=work_dir,
            timeout=self.exec_timeout,
            max_output_chars=self.max_output_chars,
        )

        auth_ctx = subscription_auth() if self.auth_mode == "subscription" else contextlib.nullcontext()
        started = time.time()
        try:
            with auth_ctx:
                response, steps, result_msg, agent_error = asyncio.run(
                    self._run_agent(prompt, work_dir, executor)
                )
        finally:
            executor.close()
        elapsed = time.time() - started

        stats = self._collect_stats(steps, result_msg, executor, elapsed, agent_error)
        answer = parse_answer(response)

        with open(os.path.join(query_dir, "response.txt"), "w") as f:
            f.write(response or "(empty response)")
        with open(os.path.join(query_dir, "react_steps.json"), "w") as f:
            # Wrapped in {"steps": [...]} rather than a bare list: that is the
            # shape kb.py's tokens/traces parsers expect.
            json.dump({"steps": steps}, f, indent=2, default=str)
        with open(os.path.join(query_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        with open(os.path.join(query_dir, "answer.json"), "w") as f:
            json.dump({"id": "main-task", "answer": answer}, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] {query_id} -> {answer!r} "
                  f"({stats['num_steps']} steps, {stats['total_tokens']} tok, {elapsed:.1f}s)")

        return {
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": "",
            "token_usage": stats["total_tokens"],
            "token_usage_input": stats["input_tokens"],
            "token_usage_output": stats["output_tokens"],
            "token_usage_cached": stats["cached_tokens"],
            "cost_usd": stats["cost_usd"],
        }

    def _resolve_file_paths(self, subset_files: Optional[List[str]]) -> List[str]:
        """Absolute paths to the task's data files.

        Absolute rather than relative because the agent's cwd is the per-task
        work dir, not the repo root.
        """
        if subset_files:
            paths = expand_data_sources(
                data_sources=subset_files,
                dataset_directory=self.dataset_directory,
                all_files=list(self.dataset.keys()),
                verbose=self.verbose,
            )
            return [os.path.abspath(p) for p in paths]
        return [os.path.join(os.path.abspath(self.dataset_directory), "**", "*")]

    def _build_prompt(self, query_text: str, query_id: str, file_paths: List[str]) -> str:
        format_hint = self.format_hints.get(query_id, "")
        return f"""Answer the following question using the data files below.

Data files (absolute paths):
{json.dumps(file_paths, indent=2)}

Some paths may contain wildcards (e.g. "folder/*" or "file-*.csv"). Use glob patterns to match those.

Question: {query_text}

Answer format: {format_hint}

Your last line MUST BE: **Final Answer: <value>**"""

    async def _run_agent(self, prompt: str, work_dir: str, executor) -> tuple:
        """Drive one agent run and collect its message stream."""

        @tool("run_python", executor.description, {"code": str})
        async def run_python(args: Dict[str, Any]) -> Dict[str, Any]:
            return executor.run(args["code"])

        python_server = create_sdk_mcp_server(name="py", version="1.0.0", tools=[run_python])

        options = ClaudeAgentOptions(
            model=self.model,
            # Stock Claude Code system prompt, plus a note about the tool
            # surface. Passing a bare string would REPLACE the preset.
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": SYSTEM_PROMPT_APPEND,
            },
            tools=BUILTIN_TOOLS,          # [] - run_python is the only tool
            mcp_servers={"py": python_server},
            allowed_tools=[PYTHON_TOOL],
            permission_mode="dontAsk",
            # "user" only: a terminal session loads user + project + local, but
            # cwd sits inside this repo, so project scope would pull in
            # KramaBench's own CLAUDE.md - which documents how scoring works
            # and where the ground truth lives. This is the one deliberate
            # deviation from a stock session.
            setting_sources=["user"],
            cwd=work_dir,
            max_turns=self.max_steps,
        )
        if self.effort:
            options.effort = self.effort

        steps: List[Dict[str, Any]] = []
        assistant_text: List[str] = []
        result_msg: Optional[ResultMessage] = None
        agent_error: Optional[str] = None

        # The SDK RAISES on some terminal conditions rather than yielding a
        # ResultMessage - exhausting max_turns is the common one. Uncaught, that
        # propagates out of serve_query and aborts the whole workload process,
        # losing every task after it. Contain it here: this task records the
        # error and scores zero, the run continues.
        try:
            result_msg = await self._consume(
                query(prompt=prompt, options=options), steps, assistant_text
            )
        except Exception as exc:  # noqa: BLE001 - any SDK failure must stay local
            agent_error = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"[{self.name}] agent run failed: {agent_error}")

        for i, step in enumerate(steps):  # reindex after coalescing
            step["index"] = i

        # ResultMessage.result is the agent's final text; fall back to the
        # assistant turns collected before a failure, then to the error itself.
        response = ""
        if result_msg is not None and isinstance(result_msg.result, str):
            response = result_msg.result
        if not response:
            response = "\n".join(assistant_text)
        if not response and agent_error:
            # The `Error:` prefix is what list_failed_tasks.py keys on to report
            # execution_error rather than a plain wrong answer.
            response = f"Error: {agent_error}"

        return response, steps, result_msg, agent_error

    async def _consume(self, stream, steps, assistant_text) -> Optional[ResultMessage]:
        """Drain the SDK message stream into `steps`; return the ResultMessage."""
        result_msg: Optional[ResultMessage] = None
        async for message in stream:
            if isinstance(message, AssistantMessage):
                blocks, tool_calls = [], []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        assistant_text.append(block.text)
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingBlock):
                        blocks.append({"type": "thinking", "thinking": block.thinking})
                    elif isinstance(block, ToolUseBlock):
                        blocks.append({"type": "tool_use", "id": block.id,
                                       "name": block.name, "input": block.input})
                        tool_calls.append({"toolCallId": block.id, "toolName": block.name,
                                           "input": block.input})

                # The SDK emits one AssistantMessage per content block, all
                # sharing a message_id and all carrying the SAME usage figures.
                # Merge them into one step: otherwise every consumer that sums
                # per-step usage (kb.py tokens/traces) multiplies one model
                # call's tokens by its block count.
                prior = steps[-1] if steps else None
                if (prior and prior["role"] == "agent" and message.message_id
                        and prior.get("message_id") == message.message_id):
                    prior["content"].extend(blocks)
                    prior["toolCalls"].extend(tool_calls)
                    continue

                steps.append({
                    "index": len(steps),
                    "role": "agent",
                    "model": message.model,
                    "message_id": message.message_id,
                    "content": blocks,
                    "toolCalls": tool_calls,
                    "usage": _kb_usage(message.usage),
                })
            elif isinstance(message, UserMessage):
                results = []
                content = message.content if isinstance(message.content, list) else []
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        results.append({"toolCallId": block.tool_use_id,
                                        "isError": bool(block.is_error),
                                        "result": block.content})
                if results:
                    steps.append({"index": len(steps), "role": "user", "toolResults": results})
            elif isinstance(message, ResultMessage):
                result_msg = message

        return result_msg

    def _collect_stats(self, steps, result_msg, executor, elapsed,
                       agent_error: Optional[str] = None) -> Dict[str, Any]:
        usage = getattr(result_msg, "usage", None)
        uncached_input = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        cached_tokens = _usage_value(usage, "cache_read_input_tokens")
        cache_creation = _usage_value(usage, "cache_creation_input_tokens")
        # Inclusive of cache reads/writes - see _kb_usage for why.
        input_tokens = uncached_input + cached_tokens + cache_creation

        tool_calls = sum(len(step.get("toolCalls", [])) for step in steps)
        tool_errors = sum(
            1
            for step in steps
            for result in step.get("toolResults", [])
            if result.get("isError")
        )

        return {
            "input_tokens": input_tokens,
            "uncached_input_tokens": uncached_input,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_creation_tokens": cache_creation,
            "total_tokens": input_tokens + output_tokens,
            # List-price accounting from the SDK. Under a Claude subscription
            # nothing is billed at this rate - treat token counts as the real
            # cost measure and this as notional.
            "cost_usd": getattr(result_msg, "total_cost_usd", None),
            # One step = one agent action, matching CodeAgentSystem's step
            # semantics (one step = one code execution). NOT the raw
            # AssistantMessage count, which is inflated by the SDK splitting
            # thinking/text/tool_use into separate messages.
            "num_steps": tool_calls,
            "num_agent_steps": len([s for s in steps if s["role"] == "agent"]),
            "num_turns": getattr(result_msg, "num_turns", None),
            "num_tool_calls": tool_calls,
            "num_tool_errors": tool_errors,
            "num_python_calls": executor.call_count,
            "worker_restarts": getattr(executor, "restarts", 0),
            "exec_mode": self.exec_mode,
            "model": self.model,
            "effort": self.effort,
            # Distinct per task - each serve_query is its own one-off session
            # with no resume/continue, so nothing carries between tasks.
            "session_id": getattr(result_msg, "session_id", None),
            # Set when the SDK raised instead of returning a ResultMessage -
            # most often max_turns exhaustion. The task is contained and scored
            # zero; the surrounding workload keeps going.
            "agent_error": agent_error,
            "terminal_reason": getattr(result_msg, "terminal_reason",
                                       "raised" if agent_error else None),
            "stop_reason": getattr(result_msg, "stop_reason", None),
            "is_error": getattr(result_msg, "is_error", bool(agent_error)),
            # Non-empty means the agent reached for something it was not granted;
            # with Bash removed from context this should stay empty.
            "permission_denials": getattr(result_msg, "permission_denials", None),
            "elapsed_seconds": round(elapsed, 2),
            "duration_api_ms": getattr(result_msg, "duration_api_ms", None),
        }


# --------------------------------------------------------------- arm variants
# One exported class per (model, exec_mode) pair: the analysis tooling keys off
# the SUT string for scratch dirs, measures CSVs, and aggregated_results.csv, so
# the two execution modes must not share a name.


class ClaudeCodeSystemHaiku45Stateless(ClaudeCodeSystem):
    """claude-haiku-4.5, fresh subprocess per run_python call."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model="claude-haiku-4-5",
            exec_mode="stateless",
            name="ClaudeCodeSystemHaiku45Stateless",
            verbose=verbose,
            *args, **kwargs
        )


class ClaudeCodeSystemHaiku45Persistent(ClaudeCodeSystem):
    """claude-haiku-4.5, persistent worker namespace across run_python calls."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model="claude-haiku-4-5",
            exec_mode="persistent",
            name="ClaudeCodeSystemHaiku45Persistent",
            verbose=verbose,
            *args, **kwargs
        )


class ClaudeCodeSystemHaiku45PersistentChars2k(ClaudeCodeSystem):
    """Persistent arm at a 2k run_python output cap.

    Everything else matches ClaudeCodeSystemHaiku45Persistent (the 5k default),
    so a flip between the two is attributable to the cap alone - the code-agent
    Chars2k/Chars5k pairs at the same knob.
    """

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model="claude-haiku-4-5",
            exec_mode="persistent",
            max_output_chars=2000,
            name="ClaudeCodeSystemHaiku45PersistentChars2k",
            verbose=verbose,
            *args, **kwargs
        )


# ===========================================================================
# SONNET-5 ARMS  (1k / 5k output cap) x replicate
# ===========================================================================
# Persistent exec_mode, matching ClaudeCodeSystemHaiku45Persistent* so a
# haiku-vs-sonnet read differs on the model alone. The 1k/5k pair mirrors the
# code-agent Chars1k/Chars5k arms at the same knob (run_python output cap), so
# the two SUT substrates stay comparable at equal budgets.
_SONNET_CC_BUDGETS = (("1k", 1000), ("5k", 5000))
_SONNET_CC_REPS = (0, 1, 2)


def _mk_sonnet_cc(chars_tag, chars, rep):
    name = f"ClaudeCodeSystemSonnetChars{chars_tag}Rep{rep}"
    if name in globals():
        raise RuntimeError(f"refusing to overwrite existing SUT {name}")

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super(cls, self).__init__(
            model="claude-sonnet-5",
            exec_mode="persistent",
            max_output_chars=chars,
            name=name,
            verbose=verbose,
            *args,
            **kwargs,
        )

    cls = type(name, (ClaudeCodeSystem,), {
        "__init__": __init__,
        "__doc__": f"claude-sonnet-5 Claude Code agent, {chars_tag} output cap, persistent (rep {rep}).",
    })
    return name, cls


CLAUDE_CODE_SONNET_NAMES = []
for _ctag, _chars in _SONNET_CC_BUDGETS:
    for _rep in _SONNET_CC_REPS:
        _n, _c = _mk_sonnet_cc(_ctag, _chars, _rep)
        globals()[_n] = _c
        CLAUDE_CODE_SONNET_NAMES.append(_n)

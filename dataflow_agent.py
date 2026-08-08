# -*- coding: utf-8 -*-
"""
Dataflow Agent - Python client for Texera Agent Service benchmarking.

This module provides a DataflowAgent class that communicates with the Texera Agent Service
to run benchmark tasks. Agent lifecycle (login, workflow create/delete, agent create/delete,
clear history, workflow & react-step retrieval) goes through REST. The chat turn itself —
`sendMessage` — is WebSocket-only after the agent-service refactor.
"""

import time
import os
import json
import requests
import websocket
from typing import Optional, Any
from dataclasses import dataclass, field

# ============================================================================
# Configuration Constants
# ============================================================================

# Texera Backend Configuration
TEXERA_API_ENDPOINT = "http://localhost:8080"
TEXERA_COMPUTING_UNIT_ENDPOINT = "http://localhost:8888"
TEXERA_AGENT_SERVICE_ENDPOINT = "http://localhost:3001"
TEXERA_WORKFLOW_EXECUTION_ENDPOINT = "http://localhost:8085"

# Authentication Configuration
TEXERA_USERNAME = "texera"
TEXERA_PASSWORD = "texera"

# Agent Settings (matches agent-service AgentSettingsApi)
# Available models: claude-haiku-4.5, claude-sonnet-4-5, gpt-5-mini, llama-local
AGENT_MODEL_TYPE = "claude-haiku-4.5"
AGENT_MAX_STEPS = 100
AGENT_MAX_OPERATOR_RESULT_CHAR_LIMIT = 2000
AGENT_MAX_OPERATOR_RESULT_CELL_CHAR_LIMIT = 2000
AGENT_OPERATOR_RESULT_SERIALIZATION_MODE = "tsv"  # only "tsv" is supported
AGENT_TOOL_TIMEOUT_SECONDS = 240
AGENT_EXECUTION_TIMEOUT_MINUTES = 4
AGENT_DISABLED_TOOLS: list[str] = []
AGENT_MODE = "code"  # "code" or "general"
AGENT_CONTEXT_MODE = "latest"  # "full", "delta", or "latest"
AGENT_PARALLEL_TOOL_CALLS = True
# Explicit LLM driver. None -> server auto-derives from modelType
# (local-llm -> "local-react", everything else -> "vercel-tool-use").
# Set to "local-react" or "vercel-tool-use" to force a driver, e.g. to
# benchmark a non-local model under the text-mode ReAct loop.
AGENT_DRIVER: Optional[str] = None
# Render each operator's `Properties:` line in the assembled snapshot
# (code operators' `code` blob is always stripped). None -> server default (true).
AGENT_INCLUDE_OPERATOR_PROPERTIES: Optional[bool] = None
# Tool-call dialect for the local text-mode driver ("local-react"). Selects BOTH
# the system prompt and the output parser the driver applies to the model's
# completion:
#   "qwen-xml"   — Qwen3-Coder `<tool_call><function=…><parameter=…>` XML.
#       Default, matching the agent-service's own default.
#   "react-text" — DeepSeek-style Thought / Action / Action Input (the format
#       used before the Qwen migration). Opt-in per system.
# Ignored by the "vercel-tool-use" driver (native tool calling).
AGENT_TOOL_DIALECT = "qwen-xml"

# Workflow Configuration
DEFAULT_WORKFLOW_NAME = "Benchmark Workflow"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AgentSettings:
    """Agent settings for API requests (matches agent-service AgentSettingsApi)."""

    max_steps: int = AGENT_MAX_STEPS
    # Convergence guard: max consecutive edits to the same operator before the
    # next same-operator modify is rejected. 0 = disabled (server no-op default).
    max_operator_edits: int = 0
    max_operator_result_char_limit: int = AGENT_MAX_OPERATOR_RESULT_CHAR_LIMIT
    max_operator_result_cell_char_limit: int = AGENT_MAX_OPERATOR_RESULT_CELL_CHAR_LIMIT
    operator_result_serialization_mode: str = AGENT_OPERATOR_RESULT_SERIALIZATION_MODE
    tool_timeout_seconds: int = AGENT_TOOL_TIMEOUT_SECONDS
    execution_timeout_minutes: int = AGENT_EXECUTION_TIMEOUT_MINUTES
    disabled_tools: list[str] = field(default_factory=list)
    agent_mode: str = AGENT_MODE
    context_mode: str = AGENT_CONTEXT_MODE  # "full", "delta", or "latest"
    parallel_tool_calls: bool = AGENT_PARALLEL_TOOL_CALLS
    allowed_operator_types: Optional[list[str]] = None  # None -> server default
    # Render each operator's `Properties:` line in the assembled snapshot.
    # None -> server default (true); code operators are stripped regardless.
    include_operator_properties: Optional[bool] = None
    # Render a compact `Schema:` line (col + type) per result — cheaper
    # substitute for full column-stats (lever E). False = no-op default.
    # Append a `Loader hint:` remediation line to a failed load-stage operator,
    # keyed on the exception class (plan4, lever D). False = no-op default.
    # Flag currency/percent/thousands string columns with cleaning hints
    # (value-format flags). False = no-op default.
    # Per-operator cardinality lineage line (rows in→out, kept %) + alarms on
    # empty-output / fan-out explosion (lever 1). False = no-op default.
    # Surface worker-flagged degenerate joins (near-zero key match w/ unmatched-key
    # sample, or many-to-many fan-out) as a `Join check` line (lever 2). No-op default.
    # Flag dead/disconnected DataLoading operators via typed-DAG reachability
    # (lever 3, `Graph check`). No-op default.
    # Surface coerce-to-NaN dataloss from to_numeric/to_datetime(errors="coerce")
    # with a sample of dropped values (lever 4). No-op default.
    thought_replay: bool = False
    thought_replay_k: int = 10
    agent_turns: bool = False
    # DELTA context-window budget (tokens; est. chars/4). 0 = unbounded (lossless
    # delta). >0 = render "original" raw delta (schema-only) while it fits, then
    # compact via compaction_strategy once it would overflow.
    context_window_tokens: int = 0
    # Static compaction rule (DELTA-only): auto-fold settled older events into the
    # compact deck once the trajectory is long enough, keeping the last K raw.
    static_compaction: bool = False
    # Optional base-preserving result-decay experiment. None keeps the baseline
    # context byte-identical apart from permanent renderer rules.
    frontier_decay_config: Optional[dict[str, Any]] = None
    role_policy_config: Optional[dict[str, Any]] = None
    source_provenance_hint: Optional[bool] = None
    # Multi-turn knobs. `user_task_placement="bottom"` moves the request under the
    # history so the history stays a byte-stable prefix across turns (prompt cache);
    # `turn_history="index"` collapses prior turns into a catalogue; and
    # `enable_recall_tool` exposes `recallState` so the model pulls back what the
    # catalogue elided. None/False keeps the render byte-identical to legacy.
    user_task_placement: Optional[str] = None
    turn_history: Optional[str] = None
    enable_recall_tool: bool = False
    # Expose `resumeFrom(turn)`: the model names the turn it builds on.
    enable_resume_tool: bool = False
    # Check that a final answer's numbers trace to materialized results or
    # recalled state; one feedback round on total failure.
    enable_answer_grounding: bool = False
    # Execution telemetry renders (worker computes both unconditionally):
    # coerce-to-NaN dataloss diagnostics, and the rows-in/rows-out Lineage fact.
    coercion_facts: bool = False
    row_lineage: bool = False
    # Versioned mode: one catalog over an immutable version DAG; strict
    # (operatorId, turn) refs on writes. See agent-service versionedMode.
    versioned_mode: bool = False
    # Turn-index cap: how many UPSTREAM operators of the current turn's work may
    # render sample rows, most-recently-touched first. None keeps the service
    # default (12). There is no code cap — the operators a turn touches always
    # render their code in full.
    index_rich_tables: Optional[int] = None
    # How many operators keep a full index entry (summary + schema) beyond the
    # ones this turn is touching, which are never evicted. None keeps the service
    # default (40). A COUNT, not an age: the age-based rule it replaced starved a
    # 31-operator task and flooded a 381-operator one with the same constant.
    index_detailed_operators: Optional[int] = None
    index_thin_observations: Optional[bool] = None
    # Rank-3 static rule (DELTA-only): fold resolved operator revisions into
    # one-line resolution facts. None keeps the baseline byte-identical.
    fold_resolved_revisions_config: Optional[dict[str, Any]] = None
    # Rank-4 static rule (LATEST): retire superseded isolated probes to a
    # compact extracted fact once their discovery is encoded downstream.
    probe_retirement_config: Optional[dict[str, Any]] = None
    # Expose the read-only inspectResult pull tool (demand-paging experiments).
    enable_inspect_tool: bool = False
    # Write-time render prefs: agent-declared outputSummary/showOutputStatistics
    # on createOrModifyOperator (view-only; backend always computes full data).
    enable_render_prefs: bool = False
    # Code-in-snapshot (LATEST-only): render each operator's current code in the
    # snapshot + tell the agent (tool-def/prompt/example) to write short summaries.
    enable_code_in_snapshot: bool = False
    # How DELTA compacts over budget: "sliding" (drop oldest events) or "compress"
    # (fold the prefix into a stats deck, resume raw delta). Default "compress".
    compaction_strategy: str = "compress"
    # compress-deck row sampling: fraction of each folded op's rows kept as a peek
    # alongside its stats, clamped to [3, max_result_rows||20]. Default 0.10.
    deck_sample_ratio: float = 0.10
    error_reflection: bool = False
    error_reflection_threshold: int = 3
    few_shot_prompt: bool = False
    flow_level: int = 0
    data_level: int = 0
    max_result_rows: int = 0
    # Inject an `Attempt reflection:` block on heavily-edited operators
    # (plan3, progress reflection). False = no-op default.
    attempt_reflection: bool = False
    column_stats: bool = False
    value_format: bool = False
    data_hints: bool = False
    # Tool-call dialect for the local-react driver ("qwen-xml" | "react-text").
    # Ignored by vercel-tool-use. Default is qwen-xml (the new format).
    tool_dialect: str = AGENT_TOOL_DIALECT
    # Deep-partial summarize() params patch applied on top of the preset selected
    # by context_mode (server validates + merges). Lets a SUT override per-op
    # render detail (e.g. defaults.result.latest.detail="shape" = no data rows).
    summarize_params: Optional[dict[str, Any]] = None

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to API request format."""
        payload: dict[str, Any] = {
            "maxSteps": self.max_steps,
            "maxOperatorEdits": self.max_operator_edits,
            "maxOperatorResultCharLimit": self.max_operator_result_char_limit,
            "maxOperatorResultCellCharLimit": self.max_operator_result_cell_char_limit,
            "operatorResultSerializationMode": self.operator_result_serialization_mode,
            "toolTimeoutSeconds": self.tool_timeout_seconds,
            "executionTimeoutMinutes": self.execution_timeout_minutes,
            "disabledTools": self.disabled_tools,
            "agentMode": self.agent_mode,
            "contextMode": self.context_mode,
            "parallelToolCalls": self.parallel_tool_calls,
            "thoughtReplay": self.thought_replay,
            "thoughtReplayK": self.thought_replay_k,
            "agentTurns": self.agent_turns,
            "contextWindowTokens": self.context_window_tokens,
            "staticCompaction": self.static_compaction,
            "compactionStrategy": self.compaction_strategy,
            "deckSampleRatio": self.deck_sample_ratio,
            "errorReflection": self.error_reflection,
            "errorReflectionThreshold": self.error_reflection_threshold,
            "fewShotPrompt": self.few_shot_prompt,
            "flowLevel": self.flow_level,
            "dataLevel": self.data_level,
            "maxResultRows": self.max_result_rows,
            "attemptReflection": self.attempt_reflection,
            "columnStats": self.column_stats,
            "valueFormat": self.value_format,
            "dataHints": self.data_hints,
            "toolDialect": self.tool_dialect,
        }
        if self.allowed_operator_types is not None:
            payload["allowedOperatorTypes"] = self.allowed_operator_types
        if self.include_operator_properties is not None:
            payload["includeOperatorProperties"] = self.include_operator_properties
        if self.frontier_decay_config is not None:
            payload["frontierDecayConfig"] = self.frontier_decay_config
        if self.role_policy_config is not None:
            payload["rolePolicyConfig"] = self.role_policy_config
        if self.source_provenance_hint is not None:
            payload["sourceProvenanceHint"] = self.source_provenance_hint
        if self.user_task_placement is not None:
            payload["userTaskPlacement"] = self.user_task_placement
        if self.turn_history is not None:
            payload["turnHistory"] = self.turn_history
        if self.enable_recall_tool:
            payload["enableRecallTool"] = True
        if self.enable_resume_tool:
            payload["enableResumeTool"] = True
        if self.enable_answer_grounding:
            payload["enableAnswerGrounding"] = True
        if self.coercion_facts:
            payload["coercionTelemetry"] = True
        if self.row_lineage:
            payload["rowLineage"] = True
        if self.versioned_mode:
            payload["versionedMode"] = True
        if self.index_rich_tables is not None:
            payload["indexRichTables"] = self.index_rich_tables
        if self.index_detailed_operators is not None:
            payload["indexDetailedOperators"] = self.index_detailed_operators
        if self.index_thin_observations is not None:
            payload["indexThinObservations"] = self.index_thin_observations
        if self.fold_resolved_revisions_config is not None:
            payload["foldResolvedRevisionsConfig"] = self.fold_resolved_revisions_config
        if self.probe_retirement_config is not None:
            payload["probeRetirementConfig"] = self.probe_retirement_config
        if self.enable_inspect_tool:
            payload["enableInspectTool"] = True
        if self.enable_render_prefs:
            payload["enableRenderPrefs"] = True
        if self.enable_code_in_snapshot:
            payload["enableCodeInSnapshot"] = True
        if self.summarize_params is not None:
            payload["summarizeParams"] = self.summarize_params
        return payload


@dataclass
class AgentInfo:
    """Agent information returned from API."""

    id: str
    name: str
    model_type: str
    state: str
    created_at: str
    settings: Optional[dict] = None
    delegate: Optional[dict] = None
    # Driver the server actually constructed the agent with
    # ("vercel-tool-use" | "local-react"), echoed back on create.
    driver: Optional[str] = None


@dataclass
class MessageResult:
    """Result from sending a message to the agent."""

    response: str
    messages: list[dict]  # Full conversation messages from this interaction
    usage: dict
    stats: dict
    stopped: bool
    error: Optional[str] = None


# ============================================================================
# Texera API Functions
# ============================================================================


def login(
        username: str = TEXERA_USERNAME,
        password: str = TEXERA_PASSWORD,
        api_endpoint: str = TEXERA_API_ENDPOINT,
) -> str:
    """
    Login to Texera and get an access token.

    Args:
        username: Texera username
        password: Texera password
        api_endpoint: Texera API endpoint URL

    Returns:
        JWT access token

    Raises:
        requests.HTTPError: If login fails
    """
    url = f"{api_endpoint}/api/auth/login"
    payload = {"username": username, "password": password}

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["accessToken"]


def create_workflow(
        token: str,
        name: str = DEFAULT_WORKFLOW_NAME,
        api_endpoint: str = TEXERA_API_ENDPOINT,
) -> int:
    """
    Create a new workflow in Texera.

    Args:
        token: JWT access token
        name: Workflow name
        api_endpoint: Texera API endpoint URL

    Returns:
        Workflow ID (wid)

    Raises:
        requests.HTTPError: If workflow creation fails
    """
    url = f"{api_endpoint}/api/workflow/create"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Empty workflow content structure
    empty_content = {
        "operators": [],
        "operatorPositions": {},
        "links": [],
        "commentBoxes": [],
        "settings": {},
    }

    payload = {
        "name": name,
        "content": json.dumps(empty_content),  # Content must be JSON stringified
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    # The response contains a workflow object with wid
    return data.get("workflow", {}).get("wid") or data.get("wid")


def delete_workflow(
        token: str, workflow_id: int, api_endpoint: str = TEXERA_API_ENDPOINT
) -> bool:
    """
    Delete a workflow from Texera.

    Args:
        token: JWT access token
        workflow_id: Workflow ID to delete
        api_endpoint: Texera API endpoint URL

    Returns:
        True if deletion was successful

    Raises:
        requests.HTTPError: If workflow deletion fails
    """
    url = f"{api_endpoint}/api/workflow/delete"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"wids": [workflow_id]}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return True


def get_or_create_computing_unit(
        token: str,
        computing_unit_endpoint: str = TEXERA_COMPUTING_UNIT_ENDPOINT,
        execution_endpoint: str = TEXERA_WORKFLOW_EXECUTION_ENDPOINT,
        name: str = "kramabench-cu",
) -> Optional[int]:
    """
    Get an existing computing unit or create a new one.

    The execution endpoint requires a real cuid for its FK constraint on
    workflow_executions; without one, every run logs a FK violation in
    cu-master even though the workflow still executes.

    Args:
        token: JWT access token
        computing_unit_endpoint: Computing unit service endpoint URL (default: port 8888)
        execution_endpoint: WORKFLOW_EXECUTION_SERVICE endpoint that the CU advertises
            (required for unitType=local; default: port 8085 / ComputingUnitMaster)
        name: Display name for the CU when one needs to be created

    Returns:
        Computing unit ID (cuid). Returns None only when the computing-unit
        service is unreachable.
    """
    headers = {"Authorization": f"Bearer {token}"}

    list_url = f"{computing_unit_endpoint}/api/computing-unit"
    try:
        response = requests.get(list_url, headers=headers, timeout=5)
        if response.status_code == 200:
            units = response.json()
            if units and len(units) > 0:
                return units[0].get("computingUnit", {}).get("cuid") or units[0].get(
                    "cuid"
                )
    except Exception as e:
        print(f"[DataflowAgent] Computing unit service not available: {e}")
        return None

    # No CU exists — create one. unitType=local means single-node (no k8s pod),
    # uri is the WORKFLOW_EXECUTION_SERVICE endpoint the engine reports back.
    create_url = f"{computing_unit_endpoint}/api/computing-unit/create"
    payload = {
        "name": name,
        "unitType": "local",
        "cpuLimit": "NaN",
        "memoryLimit": "NaN",
        "gpuLimit": "NaN",
        "jvmMemorySize": "NaN",
        "shmSize": "NaN",
        "uri": execution_endpoint,
    }
    try:
        response = requests.post(create_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("computingUnit", {}).get("cuid") or data.get("cuid")
    except Exception as e:
        print(f"[DataflowAgent] Failed to create computing unit: {e}")
        return None


# ============================================================================
# Agent Service Functions
# ============================================================================


def create_agent(
        model_type: str,
        token: str,
        workflow_id: int,
        computing_unit_id: int,
        settings: Optional[AgentSettings] = None,
        name: Optional[str] = None,
        driver: Optional[str] = None,
        agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
) -> AgentInfo:
    """
    Create a new agent in the agent service.

    Args:
        model_type: LLM model type to use
        token: JWT access token for delegate mode
        workflow_id: Workflow ID to associate with
        computing_unit_id: Computing unit ID for execution
        settings: Agent settings (optional)
        name: Custom agent name (optional)
        agent_endpoint: Agent service endpoint URL

    Returns:
        AgentInfo with created agent details

    Raises:
        requests.HTTPError: If agent creation fails
    """
    url = f"{agent_endpoint}/api/agents"

    payload: dict[str, Any] = {
        "modelType": model_type,
        "userToken": token,
        "workflowId": workflow_id,
    }
    # The new agent-service validates `computingUnitId` as a strict number when
    # present, so omit it entirely when no compute unit is available.
    if computing_unit_id is not None:
        payload["computingUnitId"] = computing_unit_id

    if name:
        payload["name"] = name

    # `driver` is a top-level create-request field, NOT part of the settings
    # object (it's bound at construction and not mutable via UpdateAgentSettings).
    # When omitted, the server auto-derives it from modelType.
    if driver is not None:
        payload["driver"] = driver

    if settings:
        payload["settings"] = settings.to_api_dict()

    # The merged agent-service requires the delegating user's JWT in the
    # Authorization header (not just the payload) on agent creation — it forwards
    # that identity to the LLM gateway. Without it, /api/agents returns 401.
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    return AgentInfo(
        id=data["id"],
        name=data["name"],
        model_type=data["modelType"],
        state=data["state"],
        created_at=data["createdAt"],
        settings=data.get("settings"),
        delegate=data.get("delegate"),
        driver=data.get("driver"),
    )


def delete_agent(
        agent_id: str, agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT
) -> bool:
    """
    Delete an agent from the agent service.

    Args:
        agent_id: Agent ID to delete
        agent_endpoint: Agent service endpoint URL

    Returns:
        True if deletion was successful

    Raises:
        requests.HTTPError: If agent deletion fails
    """
    url = f"{agent_endpoint}/api/agents/{agent_id}"
    response = requests.delete(url)
    response.raise_for_status()
    return True


def _http_to_ws_url(endpoint: str) -> str:
    """Convert an http(s):// endpoint URL into the matching ws(s):// scheme."""
    if endpoint.startswith("https://"):
        return "wss://" + endpoint[len("https://"):]
    if endpoint.startswith("http://"):
        return "ws://" + endpoint[len("http://"):]
    return endpoint


def send_message(
        agent_id: str,
        message: str,
        agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
        receive_timeout: int = 600,
        max_turn_seconds: Optional[float] = None,
) -> MessageResult:
    """
    Send a message to an agent via the WebSocket protocol and collect the response.

    The agent-service no longer exposes a REST `/message` endpoint; instead it
    publishes streaming step events over `ws://.../api/agents/:id/react`. This
    function opens the WS, sends `{type: "message", content: ...}`, and reads
    events until a `complete` or `error` arrives.

    Args:
        agent_id: Agent ID to send message to
        message: Message content
        agent_endpoint: Agent service HTTP endpoint URL (converted to ws://)
        receive_timeout: Per-recv socket timeout in seconds

    Returns:
        MessageResult with response, accumulated usage, stopped flag, and error.
        `messages` is left empty here — callers should fetch the ReAct trace via
        `get_agent_react_steps` if they need the per-step record.

    Raises:
        websocket.WebSocketException: If the WebSocket layer fails.
    """
    ws_url = f"{_http_to_ws_url(agent_endpoint)}/api/agents/{agent_id}/react"
    ws = websocket.create_connection(ws_url, timeout=receive_timeout)

    try:
        ws.send(json.dumps({"type": "message", "content": message, "messageSource": "chat"}))

        final_response = ""
        usage_total = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "reasoning_tokens": 0, "cached_input_tokens": 0,
        }
        step_count = 0
        stopped = False
        error: Optional[str] = None
        complete = False
        turn_start = time.time()

        while not complete:
            # Per-task wall-clock budget (opt-in via max_turn_seconds). On expiry
            # we gracefully `stop` the turn server-side (so the agent-service
            # finalizes the partial trace), then stop reading and return with a
            # timeout marker. The caller (serve_query) still fetches the partial
            # ReAct trace + workflow via REST and persists them for debugging,
            # then moves to the next task. This bounds slow-but-streaming tasks
            # (which emit steps periodically and so never trip the per-recv
            # `receive_timeout`) and was the cause of multi-hour eval stalls.
            if max_turn_seconds is not None:
                remaining = max_turn_seconds - (time.time() - turn_start)
                if remaining <= 0:
                    try:
                        ws.send(json.dumps({"type": "stop"}))
                    except Exception:
                        pass
                    error = (
                        f"turn budget exceeded ({max_turn_seconds:.0f}s); "
                        f"partial trace preserved, task abandoned"
                    )
                    stopped = True
                    break
                try:
                    ws.settimeout(min(receive_timeout, remaining))
                except Exception:
                    pass
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                # In budget mode a recv timeout just means "re-check the budget"
                # (tolerates long silent operator executions up to the budget);
                # without a budget, keep the original hard-timeout behavior.
                if max_turn_seconds is not None:
                    continue
                error = f"WebSocket recv timed out after {receive_timeout}s"
                break
            if not raw:
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ev_type = event.get("type")
            if ev_type == "step":
                step = event.get("step") or {}
                # Only count agent steps that arrived during this turn — the
                # WS `init` payload already enumerated prior steps separately,
                # so step events here are scoped to the current message.
                if step.get("role") == "agent":
                    step_count += 1
                    if step.get("isEnd") and step.get("content"):
                        final_response = step["content"]
                u = step.get("usage") or {}
                usage_total["input_tokens"] += int(u.get("inputTokens") or 0)
                usage_total["output_tokens"] += int(u.get("outputTokens") or 0)
                usage_total["total_tokens"] += int(u.get("totalTokens") or 0)
                usage_total["reasoning_tokens"] += int(u.get("reasoningTokens") or 0)
                usage_total["cached_input_tokens"] += int(u.get("cachedInputTokens") or 0)
            elif ev_type == "complete":
                complete = True
            elif ev_type == "error":
                error = event.get("error") or "unknown WebSocket error"
                break
            elif ev_type == "state":
                if event.get("state") == "STOPPING":
                    stopped = True
            # init / snapshot / nextContext / headChange are ignored — they
            # are UI affordances, not part of the turn's output.

        return MessageResult(
            response=final_response,
            messages=[],
            usage=usage_total,
            stats={"steps": step_count},
            stopped=stopped,
            error=error,
        )
    finally:
        try:
            ws.close()
        except Exception:
            pass


def get_agent_workflow(
        agent_id: str, agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT
) -> dict:
    """
    Get the workflow associated with an agent.

    Args:
        agent_id: Agent ID
        agent_endpoint: Agent service endpoint URL

    Returns:
        Workflow content directly (operators, links, operatorPositions, etc.)

    Raises:
        requests.HTTPError: If API call fails
    """
    url = f"{agent_endpoint}/api/agents/{agent_id}/workflow"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_agent_react_steps(
        agent_id: str, agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT
) -> dict:
    """
    Get the ReAct steps (reasoning trace) for an agent.

    Args:
        agent_id: Agent ID
        agent_endpoint: Agent service endpoint URL

    Returns:
        Dict with 'steps' (list of ReActStep) and 'state' (agent state string)

    Raises:
        requests.HTTPError: If API call fails
    """
    url = f"{agent_endpoint}/api/agents/{agent_id}/react-steps"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def clear_agent_history(
        agent_id: str, agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT
) -> bool:
    """
    Clear an agent's conversation history.

    Args:
        agent_id: Agent ID
        agent_endpoint: Agent service endpoint URL

    Returns:
        True if successful

    Raises:
        requests.HTTPError: If API call fails
    """
    url = f"{agent_endpoint}/api/agents/{agent_id}/clear"
    response = requests.post(url)
    response.raise_for_status()
    return True


def list_all_agents(
        agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
) -> list[dict]:
    """
    List all agents in the agent service.

    Args:
        agent_endpoint: Agent service endpoint URL

    Returns:
        List of agent dictionaries with id, name, state, etc.

    Raises:
        requests.HTTPError: If API call fails
    """
    url = f"{agent_endpoint}/api/agents/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # API returns {"agents": [...]} format
    return data.get("agents", []) if isinstance(data, dict) else data


def delete_all_agents(
        agent_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
) -> int:
    """
    Delete all agents in the agent service.

    Args:
        agent_endpoint: Agent service endpoint URL

    Returns:
        Number of agents deleted

    Raises:
        requests.HTTPError: If API call fails
    """
    agents = list_all_agents(agent_endpoint)
    deleted_count = 0

    for agent in agents:
        # Handle both dict format (with "id" key) and direct ID strings
        if isinstance(agent, dict):
            agent_id = agent.get("id")
        else:
            agent_id = str(agent)

        if agent_id:
            try:
                delete_agent(agent_id, agent_endpoint)
                deleted_count += 1
                print(f"[DataflowAgent] Deleted agent: {agent_id}")
            except Exception as e:
                print(f"[DataflowAgent] Failed to delete agent {agent_id}: {e}")

    return deleted_count


# ============================================================================
# DataflowAgent Class
# ============================================================================


class DataflowAgent:
    """
    A wrapper class for interacting with the Texera Agent Service.

    This class provides a similar interface to smolagents.CodeAgent for
    benchmarking purposes, allowing the DABstep benchmark to use Texera's
    dataflow-based agent instead of code-based agents.

    Example usage:
        agent = DataflowAgent(
            model_type="claude-sonnet-4-20250514",
            max_steps=10,
        )
        agent.setup()  # Login and create workflow

        answer = agent.run("What is the total revenue?")
        print(answer)

        # Inspect reasoning trace
        for step in agent.logs:
            print(step)

        agent.cleanup()  # Delete agent and workflow
    """

    def __init__(
            self,
            model_type: str = AGENT_MODEL_TYPE,
            driver: Optional[str] = AGENT_DRIVER,
            max_steps: int = AGENT_MAX_STEPS,
            max_operator_edits: int = 0,
            max_operator_result_char_limit: int = AGENT_MAX_OPERATOR_RESULT_CHAR_LIMIT,
            max_operator_result_cell_char_limit: int = AGENT_MAX_OPERATOR_RESULT_CELL_CHAR_LIMIT,
            operator_result_serialization_mode: str = AGENT_OPERATOR_RESULT_SERIALIZATION_MODE,
            tool_timeout_seconds: int = AGENT_TOOL_TIMEOUT_SECONDS,
            execution_timeout_minutes: int = AGENT_EXECUTION_TIMEOUT_MINUTES,
            disabled_tools: Optional[list[str]] = None,
            agent_mode: str = AGENT_MODE,
            context_mode: str = AGENT_CONTEXT_MODE,
            parallel_tool_calls: bool = AGENT_PARALLEL_TOOL_CALLS,
            allowed_operator_types: Optional[list[str]] = None,
            include_operator_properties: Optional[bool] = AGENT_INCLUDE_OPERATOR_PROPERTIES,
            thought_replay: bool = False,
            thought_replay_k: int = 10,
            agent_turns: bool = False,
            context_window_tokens: int = 0,
            static_compaction: bool = False,
            frontier_decay_config: Optional[dict[str, Any]] = None,
            role_policy_config: Optional[dict[str, Any]] = None,
            source_provenance_hint: Optional[bool] = None,
            user_task_placement: Optional[str] = None,
            turn_history: Optional[str] = None,
            enable_recall_tool: bool = False,
            enable_resume_tool: bool = False,
            enable_answer_grounding: bool = False,
            coercion_facts: bool = False,
            row_lineage: bool = False,
            versioned_mode: bool = False,
            index_rich_tables: Optional[int] = None,
            index_detailed_operators: Optional[int] = None,
            index_thin_observations: Optional[bool] = None,
            fold_resolved_revisions_config: Optional[dict[str, Any]] = None,
            probe_retirement_config: Optional[dict[str, Any]] = None,
            enable_inspect_tool: bool = False,
            enable_render_prefs: bool = False,
            enable_code_in_snapshot: bool = False,
            compaction_strategy: str = "compress",
            deck_sample_ratio: float = 0.10,
            error_reflection: bool = False,
            error_reflection_threshold: int = 3,
            few_shot_prompt: bool = False,
            flow_level: int = 0,
            data_level: int = 0,
            max_result_rows: int = 0,
            attempt_reflection: bool = False,
            column_stats: bool = False,
            value_format: bool = False,
            data_hints: bool = False,
            tool_dialect: str = AGENT_TOOL_DIALECT,
            summarize_params: Optional[dict[str, Any]] = None,
            texera_api_endpoint: str = TEXERA_API_ENDPOINT,
            computing_unit_endpoint: str = TEXERA_COMPUTING_UNIT_ENDPOINT,
            agent_service_endpoint: str = TEXERA_AGENT_SERVICE_ENDPOINT,
            username: str = TEXERA_USERNAME,
            password: str = TEXERA_PASSWORD,
            workflow_name: str = DEFAULT_WORKFLOW_NAME,
            agent_name: Optional[str] = None,
            verbosity_level: int = 1,
            max_turn_seconds: Optional[float] = None,
    ):
        """
        Initialize the DataflowAgent.

        Args:
            model_type: LLM model type to use
            driver: Explicit LLM driver ("vercel-tool-use" or "local-react");
                None lets the server auto-derive it from model_type
            max_steps: Maximum number of steps per message
            max_operator_result_char_limit: Max characters for operator results
            max_operator_result_cell_char_limit: Max characters per cell in results
            operator_result_serialization_mode: Result format (only "tsv" is supported)
            tool_timeout_seconds: Tool execution timeout in seconds
            execution_timeout_minutes: Workflow execution timeout in minutes
            disabled_tools: List of tool names to disable
            agent_mode: Agent mode ("code" or "general")
            context_mode: Snapshot-selection policy ("full", "delta", or "latest")
            parallel_tool_calls: Allow the model to emit multiple tool calls per turn
            allowed_operator_types: Optional whitelist of operator type names; None uses server default
            include_operator_properties: Render each operator's `Properties:` line in
                the assembled context; None uses server default (true)
            tool_dialect: Tool-call dialect for the local-react driver
                ("react-text" | "qwen-xml"); ignored by vercel-tool-use
            texera_api_endpoint: Texera backend API endpoint
            agent_service_endpoint: Agent service endpoint
            username: Texera username for authentication
            password: Texera password for authentication
            workflow_name: Name for the created workflow
            agent_name: Custom name for the agent (optional)
            verbosity_level: Logging verbosity (0=quiet, 1=normal, 2=verbose)
        """
        self.model_type = model_type
        # Explicit driver override; None lets the server auto-derive from model_type.
        self.driver = driver
        self.settings = AgentSettings(
            max_steps=max_steps,
            max_operator_edits=max_operator_edits,
            max_operator_result_char_limit=max_operator_result_char_limit,
            max_operator_result_cell_char_limit=max_operator_result_cell_char_limit,
            operator_result_serialization_mode=operator_result_serialization_mode,
            tool_timeout_seconds=tool_timeout_seconds,
            execution_timeout_minutes=execution_timeout_minutes,
            disabled_tools=disabled_tools or [],
            agent_mode=agent_mode,
            context_mode=context_mode,
            parallel_tool_calls=parallel_tool_calls,
            allowed_operator_types=allowed_operator_types,
            include_operator_properties=include_operator_properties,
            thought_replay=thought_replay,
            thought_replay_k=thought_replay_k,
            agent_turns=agent_turns,
            context_window_tokens=context_window_tokens,
            static_compaction=static_compaction,
            frontier_decay_config=frontier_decay_config,
            role_policy_config=role_policy_config,
            source_provenance_hint=source_provenance_hint,
            user_task_placement=user_task_placement,
            turn_history=turn_history,
            enable_recall_tool=enable_recall_tool,
            enable_resume_tool=enable_resume_tool,
            enable_answer_grounding=enable_answer_grounding,
            coercion_facts=coercion_facts,
            row_lineage=row_lineage,
            versioned_mode=versioned_mode,
            index_rich_tables=index_rich_tables,
            index_detailed_operators=index_detailed_operators,
            index_thin_observations=index_thin_observations,
            fold_resolved_revisions_config=fold_resolved_revisions_config,
            probe_retirement_config=probe_retirement_config,
            enable_inspect_tool=enable_inspect_tool,
            enable_render_prefs=enable_render_prefs,
            enable_code_in_snapshot=enable_code_in_snapshot,
            compaction_strategy=compaction_strategy,
            deck_sample_ratio=deck_sample_ratio,
            error_reflection=error_reflection,
            error_reflection_threshold=error_reflection_threshold,
            few_shot_prompt=few_shot_prompt,
            flow_level=flow_level,
            data_level=data_level,
            max_result_rows=max_result_rows,
            attempt_reflection=attempt_reflection,
            column_stats=column_stats,
            value_format=value_format,
            data_hints=data_hints,
            tool_dialect=tool_dialect,
            summarize_params=summarize_params,
        )
        self.texera_api_endpoint = texera_api_endpoint
        self.computing_unit_endpoint = computing_unit_endpoint
        self.agent_service_endpoint = agent_service_endpoint
        self.username = username
        self.password = password
        self.workflow_name = workflow_name
        self.agent_name = agent_name
        self.verbosity_level = verbosity_level
        # Per-task wall-clock budget (seconds). None/0 = disabled (no-op default,
        # so the baseline reproduces). Opt in via TEXERA_AGENT_MAX_TURN_SECONDS;
        # on expiry the turn is gracefully stopped, the partial trace is still
        # persisted by serve_query (via REST), and the run moves to the next
        # task — bounds pathological slow/looping tasks without losing the trace.
        if max_turn_seconds is None:
            _mts = os.environ.get("TEXERA_AGENT_MAX_TURN_SECONDS")
            try:
                max_turn_seconds = float(_mts) if _mts and float(_mts) > 0 else None
            except ValueError:
                max_turn_seconds = None
        self.max_turn_seconds = max_turn_seconds

        # State
        self._token: Optional[str] = None
        self._workflow_id: Optional[int] = None
        self._computing_unit_id: Optional[int] = None
        self._agent_info: Optional[AgentInfo] = None
        self._last_result: Optional[MessageResult] = None

    @property
    def last_result(self) -> Optional[MessageResult]:
        """Get the last message result."""
        return self._last_result

    @property
    def agent_id(self) -> Optional[str]:
        """Get the current agent ID."""
        return self._agent_info.id if self._agent_info else None

    def _log(self, message: str, level: int = 1):
        """Log a message if verbosity level is high enough."""
        if self.verbosity_level >= level:
            print(f"[DataflowAgent] {message}")

    def setup(self) -> "DataflowAgent":
        """
        Setup the agent by logging in and creating necessary resources.

        This method:
        1. Logs into Texera to get an access token
        2. Gets or creates a computing unit
        3. Creates a new workflow
        4. Creates an agent in the agent service

        Returns:
            self for method chaining

        Raises:
            requests.HTTPError: If any API call fails
        """
        self._log("Logging into Texera...")
        self._token = login(
            username=self.username,
            password=self.password,
            api_endpoint=self.texera_api_endpoint,
        )
        self._log("Login successful")

        self._log("Getting computing unit...")
        self._computing_unit_id = get_or_create_computing_unit(
            token=self._token,
            computing_unit_endpoint=self.computing_unit_endpoint,
        )
        self._log(f"Using computing unit: {self._computing_unit_id}")

        self._log("Creating workflow...")
        self._workflow_id = create_workflow(
            token=self._token,
            name=self.workflow_name,
            api_endpoint=self.texera_api_endpoint,
        )
        self._log(f"Created workflow: {self._workflow_id}")

        self._log("Creating agent...")
        self._agent_info = create_agent(
            model_type=self.model_type,
            token=self._token,
            workflow_id=self._workflow_id,
            computing_unit_id=self._computing_unit_id,
            settings=self.settings,
            name=self.agent_name,
            driver=self.driver,
            agent_endpoint=self.agent_service_endpoint,
        )
        self._log(
            f"Created agent: {self._agent_info.id} (name: {self._agent_info.name})"
        )

        return self

    def run(self, prompt: str) -> MessageResult:
        """
        Run the agent with a prompt and return the full message result.

        This is the main method for interacting with the agent, similar to
        smolagents.CodeAgent.run().

        Args:
            prompt: The prompt/question to send to the agent

        Returns:
            MessageResult containing response, usage, stats, stopped, and error

        Raises:
            RuntimeError: If agent is not set up
            requests.HTTPError: If API call fails
        """
        if not self._agent_info:
            raise RuntimeError("Agent not set up. Call setup() first.")

        self._log(f"Sending message: {prompt[:100]}...", level=2)

        result = send_message(
            agent_id=self._agent_info.id,
            message=prompt,
            agent_endpoint=self.agent_service_endpoint,
            max_turn_seconds=self.max_turn_seconds,
        )

        # Store the result for later access
        self._last_result = result

        self._log(f"Response received.", level=2)

        if result.error:
            self._log(f"Error: {result.error}", level=0)

        return result

    def clear_history(self):
        """Clear the agent's conversation history."""
        if self._agent_info:
            clear_agent_history(
                agent_id=self._agent_info.id,
                agent_endpoint=self.agent_service_endpoint,
            )
            self._last_result = None
            self._log("History cleared")

    def reset(self):
        """
        Reset the agent's conversation by clearing history.

        The agent-service no longer exposes a full reset endpoint (it was
        removed in the refactor); only `clear` remains. To get a truly fresh
        workflow alongside cleared history, callers should `cleanup()` and
        re-`setup()` instead.
        """
        if self._agent_info:
            clear_agent_history(
                agent_id=self._agent_info.id,
                agent_endpoint=self.agent_service_endpoint,
            )
            self._last_result = None
            self._log("Agent history cleared (reset)")

    def cleanup(self):
        """
        Cleanup resources by deleting the agent and workflow.

        Call this when done with the agent to free up resources.
        """
        if self._agent_info:
            self._log(f"Deleting agent: {self._agent_info.id}")
            try:
                delete_agent(
                    agent_id=self._agent_info.id,
                    agent_endpoint=self.agent_service_endpoint,
                )
            except Exception as e:
                self._log(f"Failed to delete agent: {e}", level=0)
            self._agent_info = None

        if self._workflow_id and self._token:
            self._log(f"Deleting workflow: {self._workflow_id}")
            try:
                delete_workflow(
                    token=self._token,
                    workflow_id=self._workflow_id,
                    api_endpoint=self.texera_api_endpoint,
                )
            except Exception as e:
                self._log(f"Failed to delete workflow: {e}", level=0)
            self._workflow_id = None

        self._token = None
        self._computing_unit_id = None
        self._last_result = None
        self._log("Cleanup complete")

    def __enter__(self) -> "DataflowAgent":
        """Context manager entry - setup the agent."""
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("DataflowAgent Example")
    print("=" * 50)

    # Create agent with context manager for automatic cleanup
    with DataflowAgent(
            model_type=AGENT_MODEL_TYPE,
            max_steps=AGENT_MAX_STEPS,
            verbosity_level=2,
    ) as agent:
        # Test with a simple question
        result = agent.run("What operators are available for data processing?")
        print(f"\nResponse: {result.response}")
        print(f"Usage: {result.usage}")
        print(f"Stats: {result.stats}")

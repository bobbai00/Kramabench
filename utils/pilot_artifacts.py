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

"""Durable first-attempt artifacts and explicitly incomplete token accounting."""

import json
import math
import os
import tempfile
import time
from pathlib import Path


class AttemptBundle:
    def __init__(self, path):
        self.path = Path(path)
        # An existing failed/empty directory still reserves the first attempt.
        self.path.mkdir(mode=0o700, parents=True, exist_ok=False)
        self._steps = {}

    def write(self, name, value, *, text=False):
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("artifact must be a single file name")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path, delete=False) as stream:
                temporary = Path(stream.name)
                if text:
                    stream.write(value)
                else:
                    json.dump(value, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path / name)
            directory_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def event(self, event):
        # No init payloads, delegates, tokens, or UI snapshots in the journal.
        # Real model input/output is intentionally preserved inside step events.
        if not isinstance(event, dict) or event.get("type") not in {"step", "state", "complete", "error"}:
            return
        selected = {"type": event["type"]}
        for key in {"step": ["step"], "state": ["state"], "error": ["error"]}.get(event["type"], []):
            if key in event:
                selected[key] = event[key]
        encoded = json.dumps({"received_at": time.time(), "event": selected}, allow_nan=False) + "\n"
        fd = os.open(self.path / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        step = event.get("step")
        if event["type"] == "step" and isinstance(step, dict) and isinstance(step.get("id"), str):
            self._steps[step["id"]] = json.loads(json.dumps(step))

    def stream_steps(self):
        return list(self._steps.values())


_TOKEN_KEYS = {
    "input_tokens": ("inputTokens", "input_tokens"),
    "output_tokens": ("outputTokens", "output_tokens"),
    "cached_tokens": ("cachedInputTokens", "cached_input_tokens", "cache_read_input_tokens"),
    "cache_creation_tokens": ("cacheCreationInputTokens", "cache_creation_input_tokens", "cache_creation_tokens"),
    "reasoning_tokens": ("reasoningTokens", "reasoning_tokens"),
}


def usage_report(steps, *, completed, pricing):
    """A complete estimate requires complete per-step usage and a closed turn.

    Observed counts/cost survive interruptions; they are not substituted for
    whole-attempt cost. Reasoning is already in output, never billed twice.
    This pilot uses an OpenAI model with no separate cache-creation charge.
    Missing cache-read usage is unknown, not silently priced as uncached.
    """
    # Import lazily: systems.__init__ registers the pilot SUT, which imports
    # this module. Read-only artifact consumers must not depend on import order.
    from systems.cost_utils import compute_cost

    unique = {}
    malformed = False
    for step in steps:
        if not isinstance(step, dict) or step.get("role") != "agent":
            continue
        identifier = step.get("id")
        if not isinstance(identifier, str) or not identifier:
            malformed = True
            continue
        unique[identifier] = step
    totals = dict.fromkeys(_TOKEN_KEYS, 0)
    known, missing, input_missing = 0, [], []
    for identifier, step in unique.items():
        usage = step.get("usage")
        if not isinstance(step.get("inputMessages"), list) or not step["inputMessages"]:
            input_missing.append(identifier)
        if not isinstance(usage, dict):
            missing.append(identifier)
            continue
        values = {}
        valid = True
        for target, aliases in _TOKEN_KEYS.items():
            raw = next((usage[key] for key in aliases if key in usage and usage[key] is not None), None)
            if raw is None and target in {"cache_creation_tokens", "reasoning_tokens"}:
                raw = 0
            if type(raw) is not int or raw < 0:
                valid = False
                break
            values[target] = raw
        if valid:
            valid = (
                values["cached_tokens"] + values["cache_creation_tokens"] <= values["input_tokens"]
                and values["reasoning_tokens"] <= values["output_tokens"]
            )
            total = usage.get("totalTokens", usage.get("total_tokens"))
            if total is not None:
                valid = valid and type(total) is int and total == values["input_tokens"] + values["output_tokens"]
        if not valid:
            missing.append(identifier)
            continue
        known += 1
        for key, value in values.items():
            totals[key] += value
    complete = completed is True and known > 0 and not missing and not malformed
    cost = None
    if pricing is not None and known:
        try:
            rates = pricing["rates"]
            valid_rates = all(
                type(rates[k]) in {int, float} and math.isfinite(rates[k]) and rates[k] >= 0
                for k in ("input", "output", "cache_read", "cache_creation")
            )
            if (
                valid_rates
                and pricing.get("input_includes_cached") is True
                and pricing.get("output_includes_reasoning") is True
            ):
                cost = compute_cost(
                    pricing["model"],
                    input_tokens=totals["input_tokens"],
                    output_tokens=totals["output_tokens"],
                    cached_tokens=totals["cached_tokens"],
                    cache_creation_tokens=totals["cache_creation_tokens"],
                    pricing=pricing,
                )
                if cost == 0 and totals["input_tokens"] + totals["output_tokens"] > 0:
                    cost = None  # zero/missing price or rounding: never a free win
        except (KeyError, TypeError, ValueError, OverflowError):
            cost = None
    return {
        **{key: value if known else None for key, value in totals.items()},
        "total_tokens": totals["input_tokens"] + totals["output_tokens"] if known else None,
        "num_steps": len(unique),
        "usage_steps": known,
        "missing_usage_step_ids": missing,
        "missing_input_messages_step_ids": input_missing,
        "captured_input_messages_complete": bool(unique) and not input_missing and not malformed,
        "input_trace_complete": completed is True and bool(unique) and not input_missing and not malformed,
        "usage_status": "complete" if complete else "partial" if known else "unknown",
        "cost_status": "complete" if complete and cost is not None else "partial" if cost is not None else "unknown",
        "cost_usd": cost if complete else None,
        "observed_cost_usd": cost,
    }

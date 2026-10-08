# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ordinary native benchmark runs preserve authoritative metrics and unknowns."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest

from dataflow_agent import MessageResult
from systems.dataflow_system import DataflowSystem


def ledger():
    return {
        "version": 1,
        "source": "transport",
        "usageStatus": "complete",
        "requests": [
            {
                "requestId": "r1",
                "requestedModel": "route",
                "status": "succeeded",
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "totalTokens": 120,
                    "cachedInputTokens": 40,
                    "cacheCreationInputTokens": 20,
                    "reasoningTokens": 10,
                },
                "rawUsage": {"cost": 0.0042},
            }
        ],
    }


def run_system(tmp_path, monkeypatch, *, failing=False):
    value = ledger()
    if failing:
        value["usageStatus"] = "partial"
        value["requests"].append(
            {
                "requestId": "r2",
                "status": "transport_error",
                "usage": None,
                "rawUsage": None,
            }
        )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                "model": "route",
                "rates": {
                    "input": 1e-6,
                    "output": 2e-6,
                    "cache_read": 0.1e-6,
                    "cache_creation": 1.25e-6,
                },
            }
        )
    )
    monkeypatch.setenv("TEXERA_BENCH_PRICING_SNAPSHOT", str(pricing))
    monkeypatch.delenv("TEXERA_BENCH_RESOURCE_SCOPE", raising=False)
    system = DataflowSystem(model_type="route", output_dir=str(tmp_path))
    system.dataset_directory, system.dataset = Path("data"), {}
    result = MessageResult(
        "Final Answer: 42",
        [],
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 40,
            "cache_creation_input_tokens": 20,
        },
        {"steps": 1},
        False,
        completed=True,
        model_accounting=value,
    )
    system.agent = SimpleNamespace(
        agent_id="a",
        agent_service_endpoint="http://mock",
        cleanup=Mock(),
        reset=Mock(),
        run=Mock(
            side_effect=RuntimeError("socket failed") if failing else None,
            return_value=result,
        ),
    )
    response = Mock()
    response.json.return_value = value
    with (
        patch.object(system, "_setup_agent"),
        patch(
            "systems.dataflow_system.get_agent_react_steps", return_value={"steps": []}
        ),
        patch("systems.dataflow_system.get_agent_workflow", return_value={}),
        patch(
            "utils.pilot_provenance.inspect_service_listener",
            return_value={"git_sha": "test", "source_dirty": False},
        ),
        patch("requests.get", return_value=response),
    ):
        system.serve_query("Count?", query_id="task")
    return json.loads((tmp_path / "task" / "stats.json").read_text()), tmp_path / "task"


def test_standard_wrapper_exports_complete_shared_accounting(tmp_path, monkeypatch):
    stats, folder = run_system(tmp_path, monkeypatch)
    assert stats["cache_creation_tokens"] == 20
    assert stats["cost_usd"] == pytest.approx(109e-6)
    metrics = json.loads((folder / stats["metrics_path"]).read_text())
    assert metrics["provider_reported_cost"]["cost_usd"] == 0.0042
    assert metrics["resources"]["status"] == "unavailable"
    assert (folder / stats["model_accounting_path"]).is_file()


def test_socket_failure_retains_paid_usage_and_unknown_cost(tmp_path, monkeypatch):
    stats, folder = run_system(tmp_path, monkeypatch, failing=True)
    assert stats["input_tokens"] == 100
    assert stats["cost_usd"] is None
    metrics = json.loads((folder / stats["metrics_path"]).read_text())
    assert metrics["model_calls"]["attempted"] == 2
    assert metrics["model_calls"]["failed"] == 1
    assert metrics["observed_usage"]["input_tokens"] == 100

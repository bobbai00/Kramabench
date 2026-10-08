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
"""Bridge ordinary KramaBench runs to the repository's shared native runtime."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

# The submodule's benchmark package differs from the parent benchmark runtime.
RUNTIME_ROOT = Path(__file__).resolve().parents[2]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))
from runtime.metrics import AttemptMetrics, write_json_exclusive  # noqa: E402
from runtime.native_accounting import (  # noqa: E402
    finish_native_metrics,
    load_pricing_snapshot,
    native_resource_window,
)


class NativeRunMetrics:
    def __init__(self, system, task, model):
        self.attempt = AttemptMetrics(
            uuid4().hex,
            benchmark="kramabench",
            system=system,
            task_id=task,
            manifest={"model": model},
        )
        self.pricing = load_pricing_snapshot(
            os.environ.get("TEXERA_BENCH_PRICING_SNAPSHOT"), model
        )
        scope_path = os.environ.get("TEXERA_BENCH_RESOURCE_SCOPE")
        self.scope = json.loads(Path(scope_path).read_text()) if scope_path else None
        self.resources = None

    @contextmanager
    def solve(self):
        window = None
        try:
            with native_resource_window(self.scope) as window:
                with self.attempt.lifecycle_clock("solve"):
                    yield
        finally:
            self.resources = (
                window.resources
                if window
                else {
                    "status": "unavailable",
                    "reason": "resource_collector_failed_before_start",
                }
            )
            self.resources["boundary"] = (
                "one synchronous solve request; setup and artifact collection excluded"
            )

    def finish(self, result, agent, folder):
        import requests

        accounting = getattr(result, "model_accounting", None)
        try:
            # The ordinary wrapper creates one agent per task. Its lifetime
            # ledger includes every transport retry and any legacy empty-turn
            # retries, even if only the last message's completion reached WS.
            response = requests.get(
                f"{agent.agent_service_endpoint}/api/agents/{agent.agent_id}/model-metrics",
                timeout=30,
            )
            response.raise_for_status()
            accounting = response.json()
        except (requests.RequestException, ValueError):
            pass
        metrics = finish_native_metrics(
            self.attempt,
            accounting,
            pricing=self.pricing,
            usage=result.usage,
            stopped=result.stopped,
            error=result.error,
            resources=self.resources,
        )
        receipt = Path(folder) / "attempts" / self.attempt.attempt_id
        self.attempt.write(receipt)
        accounting_path = None
        if accounting is not None:
            write_json_exclusive(receipt / "model-accounting.json", accounting)
            accounting_path = str(
                (receipt / "model-accounting.json").relative_to(folder)
            )
        return (
            metrics,
            str((receipt / "metrics.json").relative_to(folder)),
            accounting_path,
        )

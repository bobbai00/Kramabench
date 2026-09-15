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

"""Eight frozen full-workload arms, executed only through the native harness.

Reuse the pilot's single-dispatch capture, resource journal and partial usage
accounting. Service/CU lifecycle remains with the campaign supervisor.
"""

import json
import os
from pathlib import Path

from .native_pilot_system import NativePilotSystem
from .native_python_system import ALL_PILOT_ARMS, PilotArm
from utils.pilot_artifacts import AttemptBundle


ROUNDS = ("first", "recovery1", "recovery2")
CAMPAIGN_ID = "NativeCampaign20260915Rep1"


class CampaignArm(PilotArm):
    @property
    def system_name(self):
        model = {"gpt-5.6-luna": "Luna", "gpt-5.6-terra": "Terra"}[self.model_type]
        return f"DataflowSystem{model}NativeCampaign{self.key}20260915Rep1"


CAMPAIGN_ARMS = tuple(
    CampaignArm(**vars(arm)) for arm in ALL_PILOT_ARMS
    if arm.key in {"BatchParent", "DataOnly", "FlowOnly", "Combined"}
)


def frozen_pricing(model):
    scale = {"gpt-5.6-luna": 1, "gpt-5.6-terra": 10}[model]
    return {
        "version": 1, "kind": "token_estimate", "model": model,
        "source": {"table": CAMPAIGN_ID}, "currency": "USD", "unit": "per_token",
        "rates": {"input": scale * .2 / 1e6, "cache_read": scale * .02 / 1e6,
                  "output": scale * 1.2 / 1e6, "cache_creation": 0},
        "input_includes_cached": True, "output_includes_reasoning": True,
        "maximum_base_rate_input_tokens": 272000,
    }


class NativeCampaignSystem(NativePilotSystem):
    _campaign_guard = None

    def _qualify(self, stage, info):
        if self._campaign_guard is None:
            from utils.native_campaign import CampaignGuard
            self._campaign_guard = CampaignGuard()
        return self._campaign_guard(self, stage, info)

    def _attempt_inputs(self, query, query_id, subset_files):
        task = self.workload_data.get(query_id)
        if not task or query != task.get("query") or subset_files != task.get("data_sources"):
            raise ValueError("campaign task, query and oracle-file policy are frozen")
        if not query_id or Path(query_id).name != query_id or query_id in {".", ".."}:
            raise ValueError("invalid campaign task id")
        paths = sorted(self._expand_data_sources(subset_files))
        if self._campaign_guard is None and os.environ.get("NATIVE_CAMPAIGN_MANIFEST"):
            from utils.native_campaign import CampaignGuard
            self._campaign_guard = CampaignGuard()
        manifest = getattr(self._campaign_guard, "manifest", None)
        if isinstance(manifest, dict) and query_id in manifest.get("task_inputs", {}):
            paths = manifest["task_inputs"][query_id]
            if not isinstance(paths, list) or not paths or any(
                not isinstance(path, str) or Path(path).is_absolute() for path in paths
            ):
                raise ValueError("campaign task_inputs must contain relative prompt paths")
        self.campaign_query_id, self.campaign_prompt_paths = query_id, paths
        return task, self._build_prompt(query, paths, self.format_hints.get(query_id, ""))

    def _attempt_bundle(self, query_id):
        current = Path(self.output_dir) / query_id
        round_id = os.environ.get("NATIVE_CAMPAIGN_ROUND", "first")
        if round_id not in ROUNDS:
            raise ValueError("NATIVE_CAMPAIGN_ROUND must be first, recovery1 or recovery2")
        archives = Path(self.output_dir) / "_attempts" / query_id
        if (archives / round_id).exists():
            raise FileExistsError("campaign round already archived")
        if current.exists():
            if (current / "attempt.json").exists():
                metadata = json.loads((current / "attempt.json").read_text())
                old_round = metadata.get("round")
            else:
                # A killed process may have reserved the directory before its
                # first metadata write. Preserve that directory too.
                old_round = next((item for item in ROUNDS if not (archives / item).exists()), None)
            if old_round not in ROUNDS or ROUNDS.index(old_round) >= ROUNDS.index(round_id):
                raise FileExistsError("existing attempt requires a later explicit recovery round")
            target = archives / old_round
            if target.exists():
                raise FileExistsError("prior campaign attempt is already archived")
            archives.mkdir(parents=True, exist_ok=True)
            current.rename(target)  # Complete directory, including scorer output and raw events.
        self.campaign_round = round_id
        bundle = AttemptBundle(current)
        bundle.write("attempt.json", {"version": 1, "status": "reserved", **self._attempt_metadata()})
        return bundle

    def _attempt_metadata(self):
        return {"campaign": CAMPAIGN_ID, "round": self.campaign_round}

    def _attempt_pricing(self):
        return frozen_pricing(self.model_type)

    def _after_attempt(self, bundle, attempt):
        from utils.native_campaign import cleanup_campaign_resources
        cleanup = cleanup_campaign_resources(self, bundle, attempt)
        attempt["cleanup_status"] = cleanup["status"]
        attempt["backend_termination_verified"] = cleanup["backend_terminal_verified"]

    def _finalize_stats(self, bundle, attempt, stats, trace):
        long_steps = []
        for step in trace["steps"]:
            usage = step.get("usage") if isinstance(step, dict) else None
            inputs = usage.get("inputTokens", usage.get("input_tokens")) if isinstance(usage, dict) else None
            if type(inputs) is int and inputs > 272000:
                long_steps.append(step.get("id"))
        stats["long_context_step_ids"] = long_steps
        if long_steps:
            stats.update(base_rate_estimate_usd=stats["observed_cost_usd"], cost_usd=None,
                         observed_cost_usd=None, cost_status="unknown_long_context_rate")
        fields = ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens",
                  "usage_status", "cost_status", "cost_usd", "observed_cost_usd", "long_context_step_ids")
        current = {"round": attempt["round"], **{key: stats.get(key) for key in fields}}
        prior = []
        archive = Path(self.output_dir) / "_attempts" / bundle.path.name
        for round_id in ROUNDS[:ROUNDS.index(attempt["round"])]:
            folder = archive / round_id
            if folder.is_dir():
                saved = json.loads((folder / "stats.json").read_text()) if (folder / "stats.json").exists() else {}
                prior.append({"round": round_id, **{key: saved.get(key) for key in fields}})
        rows = prior + [current]
        complete = all(row.get("cost_usd") is not None for row in rows)
        observed = [row["observed_cost_usd"] for row in rows if row.get("observed_cost_usd") is not None]
        stats["first_pass"] = next((row for row in rows if row["round"] == "first"), None)
        stats["all_attempts"] = {
            "attempt_count": len(rows), "rounds": rows,
            "cost_usd": round(sum(row["cost_usd"] for row in rows), 6) if complete else None,
            "observed_cost_usd": round(sum(observed), 6) if observed else None,
            "cost_status": "complete" if complete else "partial" if observed else "unknown",
            "total_tokens": sum(row.get("total_tokens") or 0 for row in rows) if any(
                row.get("total_tokens") is not None for row in rows) else None,
        }
        # The existing cache-rebuild script consumes these legacy aliases.
        stats.update(token_usage=stats["total_tokens"], token_usage_input=stats["input_tokens"],
                     token_usage_output=stats["output_tokens"], runtime=stats["elapsed_seconds"])
        return stats


def _make_arm(spec):
    def __init__(self, verbose=False, *args, **kwargs):
        self.agent = None
        frozen = spec.settings()
        for key, value in frozen.items():
            if key in kwargs and (type(kwargs[key]) is not type(value) or kwargs[key] != value):
                raise ValueError(f"{spec.system_name}: {key} is frozen; use a new SUT name")
        unknown = kwargs.keys() - frozen.keys() - {"output_dir", "agent_service_endpoint", "computing_unit_id"}
        if unknown:
            raise ValueError(f"{spec.system_name}: frozen settings reject overrides {sorted(unknown)}")
        if "computing_unit_id" not in kwargs:
            raw = os.environ.get("NATIVE_CAMPAIGN_CUID", "")
            if not raw.isdecimal() or int(raw) <= 0:
                raise ValueError("NATIVE_CAMPAIGN_CUID must identify the campaign computing unit")
            kwargs["computing_unit_id"] = int(raw)
        if type(kwargs["computing_unit_id"]) is not int or kwargs["computing_unit_id"] <= 0:
            raise ValueError("computing_unit_id must be a positive integer")
        endpoint_env = "NATIVE_CAMPAIGN_BATCH_ENDPOINT" if spec.key == "BatchParent" else "NATIVE_CAMPAIGN_V2_ENDPOINT"
        kwargs.setdefault("agent_service_endpoint", os.environ.get(endpoint_env, f"http://localhost:{spec.port}"))
        kwargs.update(frozen)
        NativePilotSystem.__init__(self, *args, name=spec.system_name, verbose=verbose, **kwargs)
        self.pilot_spec = spec
        self._pilot_guard = self._qualify

    return type(spec.system_name, (NativeCampaignSystem,), {
        "__init__": __init__, "__module__": __name__,
        "__doc__": f"Frozen full KramaBench campaign: {spec.key}, {spec.model_type}, medium.",
    })


__all__ = [spec.system_name for spec in CAMPAIGN_ARMS]
for _spec in CAMPAIGN_ARMS:
    globals()[_spec.system_name] = _make_arm(_spec)

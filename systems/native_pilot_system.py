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

"""Single-attempt execution for the registered native pilot only.

The launch/qualification guard is supplied by the pilot runner. Legacy SUTs
retain their existing retry/setup policy. No destructor deletes live resources:
an interrupted response is not proof of backend termination.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

from .cost_utils import price_schedule
from .dataflow_system import DataflowSystem
from utils.answer_parser import parse_answer
from utils.pilot_artifacts import AttemptBundle, usage_report
from utils.resource_journal import ResourceJournal


PILOT_TASK = "environment-easy-3"
PILOT_TURN_SECONDS = 1800


def build_pilot_prompt(system):
    """One canonical file order for the pilot only; never include the gold answer."""
    task = system.workload_data[PILOT_TASK]
    paths = sorted(system._expand_data_sources(task["data_sources"]))
    return paths, system._build_prompt(task["query"], paths, system.format_hints.get(PILOT_TASK, ""))


def agent_json(agent, resource):
    url = f"{agent.agent_service_endpoint.rstrip('/')}/api/agents/{quote(agent.agent_id, safe='')}{resource}"
    token = getattr(agent, "_token", None)
    response = requests.get(
        url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=(3, 10), allow_redirects=False
    )
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"agent_read_http_{response.status_code}")
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("agent_read_nonobject")
    return value


def public_info(info):
    """Keep observed settings/identity, never the raw delegate/auth response."""
    if not isinstance(info, dict):
        return None
    selected = {key: info[key] for key in ("id", "modelType", "driver", "state", "settings") if key in info}
    delegate = info.get("delegate")
    if not isinstance(delegate, dict):
        delegate = {}
    selected["workflowId"] = delegate.get("workflowId")
    selected["computingUnitId"] = delegate.get("computingUnitId")
    return selected


class NativePilotSystem(DataflowSystem):
    _pilot_guard = None

    def bind_pilot_guard(self, guard):
        if not callable(guard):
            raise TypeError("pilot guard must be callable")
        self._pilot_guard = guard

    def _qualify(self, stage, info):
        evidence = self._pilot_guard(self, stage, info)
        if not isinstance(evidence, dict) or evidence.get("qualified") is not True:
            raise ValueError("pilot_guard_did_not_qualify")
        return evidence

    def process_dataset(self, dataset_directory):
        # Dataset preparation and cached scoring must not allocate a workflow.
        self._prepare_dataset(dataset_directory)

    def _setup_agent(self):
        # Each first attempt has an exclusive journal. Returned resource IDs
        # reach disk before the next creation request or any model dispatch.
        if not isinstance(getattr(self, "pilot_bundle", None), AttemptBundle):
            raise RuntimeError("native pilot setup requires a reserved attempt bundle")
        with ResourceJournal(self.pilot_bundle.path / "resource_allocations.jsonl") as journal:
            prefix = "native-python-" + journal.run_id
            super()._setup_agent(
                resource_callback=journal.record, workflow_name=prefix + "-workflow", agent_name=prefix + "-agent"
            )

    def cleanup(self):
        # The runner owns verified cleanup and keeps an explicit resource
        # journal. Legacy cleanup forgets IDs even when DELETE fails, and a
        # destructor must never abort an unknown in-flight shared execution.
        pass

    def _attempt_inputs(self, query, query_id, subset_files):
        task = self.workload_data.get(query_id)
        if (
            query_id != PILOT_TASK
            or not task
            or query != task.get("query")
            or subset_files != task.get("data_sources")
            or task.get("answer_type") != "numeric_exact"
        ):
            raise ValueError("pilot task, query and oracle-file policy are frozen")
        return task, build_pilot_prompt(self)[1]

    def _attempt_bundle(self, query_id):
        return AttemptBundle(Path(self.output_dir) / query_id)

    def _attempt_metadata(self):
        return {}

    def _attempt_pricing(self):
        return price_schedule(self.model_type)

    def _finalize_stats(self, bundle, attempt, stats, trace):
        return stats

    def _after_attempt(self, bundle, attempt):
        pass

    def serve_query(self, query, query_id="default-0", subset_files=None):
        if self._pilot_guard is None:
            raise RuntimeError("native pilot requires the launch/qualification guard")
        task, prompt = self._attempt_inputs(query, query_id, subset_files)
        bundle = self._attempt_bundle(query_id)
        self.pilot_bundle = bundle
        pricing = self._attempt_pricing()
        config = {
            "system_name": self.name,
            "query_id": query_id,
            "model_type": self.model_type,
            "agent_service_endpoint": self.agent_service_endpoint,
            "computing_unit_id": self.computing_unit_id,
            "dataset_directory": self.dataset_directory,
            "subset_files": subset_files,
            "agent_settings": self.pilot_spec.settings(),
            "empty_turn_retries": 0,
            "max_turn_seconds": PILOT_TURN_SECONDS,
            "pricing": pricing,
        }
        attempt = {
            "version": 1,
            "status": "not_started",
            "dispatched": False,
            "completed_event": False,
            "qualification": "pending",
            "backend_termination_verified": False,
            "resources": {},
            "errors": [],
            **self._attempt_metadata(),
        }
        bundle.write("prompt.txt", prompt, text=True)
        bundle.write("ground_truth.json", task)
        bundle.write("config.json", config)
        bundle.write("attempt.json", attempt)
        self.agent = None
        result = None
        exception = None
        stage = "before_setup"
        started = time.monotonic()
        try:
            config["preflight"] = self._qualify(stage, None)
            bundle.write("config.json", config)
            stage = "setup"
            self._setup_agent()
            self.agent.max_turn_seconds = PILOT_TURN_SECONDS
            stage = "before_dispatch"
            info = agent_json(self.agent, "")
            config["effective_agent_before"] = public_info(info)
            config["requested_wire_settings"] = self.agent.settings.to_api_dict()
            bundle.write("config.json", config)
            config["admission"] = self._qualify(stage, info)
            bundle.write("config.json", config)
            stage = "dispatch"
            attempt.update(status="running", dispatched=True, resources=self._resources())
            bundle.write("attempt.json", attempt)
            result = self.agent.run(prompt, empty_turn_retries=0, on_event=bundle.event)
        except BaseException as error:
            exception = error
            attempt["errors"].append({"stage": stage, "type": type(error).__name__})
        finally:
            # Capture after every dispatch/setup outcome, including EOF and
            # Ctrl-C. Exception text can contain credentials; retain only type.
            attempt["resources"] = self._resources()
            attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            attempt["completed_event"] = bool(result and result.completed is True)
            clean_completion = bool(result and result.completed is True and not result.error and not result.stopped)
            attempt["status"] = (
                "completed" if clean_completion else "interrupted" if attempt["dispatched"] else "not_started"
            )
            if result and result.error:
                attempt["errors"].append({"stage": "dispatch", "type": "AgentReportedError"})
            # Persist IDs before any potentially failing artifact request.
            bundle.write("attempt.json", attempt)
            capture, trace, snapshots, info_after, workflow = self._capture(bundle)
            agent_steps = [step for step in trace["steps"] if isinstance(step, dict) and step.get("role") == "agent"]
            final_step = agent_steps[-1] if agent_steps else {}
            model_final = final_step.get("isEnd") is True and (
                isinstance(final_step.get("usage"), dict)
                or isinstance(final_step.get("inputMessages"), list)
                or isinstance(final_step.get("toolCalls"), list)
            )
            if clean_completion and final_step.get("isEnd") and not model_final:
                # Synthetic error/stopped steps have no model input or usage.
                # Never parse numbers in error text as the task's final answer.
                attempt["status"] = "agent_error"
                clean_completion = False
            config["effective_agent_after"] = public_info(info_after)
            bundle.write("config.json", config)
            bundle.write("react_steps.json", trace)
            bundle.write("workflow.json", workflow)
            bundle.write("snapshots.json", snapshots)
            bundle.write("capture.json", capture)
            try:
                config["postflight"] = self._qualify("after_attempt", info_after)
                attempt["qualification"] = "passed" if stage == "dispatch" else "failed"
            except Exception as error:
                attempt["qualification"] = "failed"
                attempt["errors"].append({"stage": "after_attempt", "type": type(error).__name__})
            stats = usage_report(
                trace["steps"], completed=clean_completion and capture["trace_source"] == "service", pricing=pricing
            )
            stats["elapsed_seconds"] = attempt["elapsed_seconds"]
            stats["pricing"] = pricing
            # Never attribute unrelated preexisting trace usage to a rejected
            # preflight that made no model dispatch.
            if not attempt["dispatched"]:
                stats = {
                    **usage_report([], completed=False, pricing=pricing),
                    "elapsed_seconds": attempt["elapsed_seconds"],
                    "pricing": pricing,
                }
            stats = self._finalize_stats(bundle, attempt, stats, trace)
            bundle.write("stats.json", stats)
            # Score a real observed final answer even if the final completion
            # frame was lost. Protocol completion, usage completeness and answer
            # correctness are separate measurements; do not conflate them.
            response = (final_step.get("content") or "") if model_final and attempt["dispatched"] else ""
            answer = parse_answer(response, []) if response else ""
            explanation = {"id": "main-task", "answer": answer}
            bundle.write("response.txt", result.response if result else "", text=True)
            bundle.write("answer.json", explanation)
            attempt["capture_complete"] = capture["complete"]
            attempt["input_trace_complete"] = stats["input_trace_complete"]
            self._after_attempt(bundle, attempt)
            bundle.write("config.json", config)
            bundle.write("attempt.json", attempt)
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            raise exception
        return {
            "explanation": explanation,
            "pipeline_code": "",
            "token_usage": stats["total_tokens"],
            "token_usage_input": stats["input_tokens"],
            "token_usage_output": stats["output_tokens"],
            "token_usage_reasoning": stats["reasoning_tokens"],
            "token_usage_cached": stats["cached_tokens"],
            "cost_usd": stats["cost_usd"],
            "pilot_evaluation_eligible": attempt["dispatched"] and attempt["qualification"] == "passed",
        }

    def _resources(self):
        return {
            "agent_id": getattr(self.agent, "agent_id", None),
            "workflow_id": getattr(self.agent, "_workflow_id", None),
            "computing_unit_id": self.computing_unit_id,
        }

    def _capture(self, bundle):
        failures = []

        def read(resource):
            if not self.agent or not self.agent.agent_id:
                failures.append({"resource": resource, "type": "NoAgentIdentity"})
                return None
            try:
                value = agent_json(self.agent, resource)
                if resource.startswith("/snapshots/") and resource != "/snapshots/" + quote(
                    str(value.get("stepId", "")), safe=""
                ):
                    raise ValueError("snapshot_identity_mismatch")
                return value
            except Exception as error:
                failures.append({"resource": resource, "type": type(error).__name__})
                return None

        info = read("")
        trace = read("/react-steps")
        workflow = read("/workflow")
        source = "service"
        if (
            trace is None
            or not isinstance(trace.get("steps"), list)
            or any(not isinstance(step, dict) or not isinstance(step.get("id"), str) for step in trace["steps"])
        ):
            if trace is not None:
                failures.append({"resource": "/react-steps", "type": "InvalidTrace"})
            source = "stream_partial"
            trace = {"steps": bundle.stream_steps(), "state": None}
        ids = list(
            dict.fromkeys(
                step["id"] for step in trace["steps"] if isinstance(step, dict) and isinstance(step.get("id"), str)
            )
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            snapshots = list(pool.map(lambda identifier: read("/snapshots/" + quote(identifier, safe="")), ids))
        report = {
            "complete": not failures and source == "service",
            "trace_source": source,
            "errors": failures,
            "requested_snapshot_ids": ids,
        }
        return report, trace, {"snapshots": [value for value in snapshots if value is not None]}, info, workflow

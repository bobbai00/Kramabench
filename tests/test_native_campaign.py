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

import hashlib
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import systems
import systems.native_campaign_system as campaign
from dataflow_agent import AgentSettings, MessageResult
from systems.native_campaign_system import CAMPAIGN_ARMS, NativeCampaignSystem, frozen_pricing
from utils.native_campaign import CampaignGuard, cleanup_campaign_resources, file_binding
from utils.pilot_artifacts import AttemptBundle


Arm = systems.DataflowSystemLunaNativeCampaignCombined20260915Rep1
TASK = {
    "id": "legal-hard-2", "query": "Fixture query, not a benchmark answer.",
    "answer_type": "list_exact", "answer": ["a", "b"], "data_sources": ["fixture.csv"],
}
STEP = {
    "id": "s1", "role": "agent", "isEnd": True, "content": "Final Answer: [\"a\", \"b\"]",
    "toolCalls": [], "inputMessages": [{"role": "user", "content": TASK["query"]}],
    "usage": {"inputTokens": 100, "outputTokens": 20, "cachedInputTokens": 80, "totalTokens": 120},
}


class CampaignSettingsTest(unittest.TestCase):
    def test_eight_observe_only_arms_use_fresh_names_and_identical_settings(self):
        specs = campaign.OBSERVE_ONLY_CAMPAIGN_ARMS
        self.assertEqual(len(specs), 8)
        self.assertEqual(campaign.OBSERVE_ONLY_CAMPAIGN_ID, "NativeCampaignObserveOnly20260916Rep1")
        self.assertTrue({spec.system_name for spec in specs}.isdisjoint(
            spec.system_name for spec in CAMPAIGN_ARMS))
        with TemporaryDirectory() as directory:
            for spec in specs:
                with self.subTest(sut=spec.system_name):
                    prior = next(arm for arm in CAMPAIGN_ARMS
                                 if (arm.key, arm.model_type) == (spec.key, spec.model_type))
                    self.assertEqual(spec.settings(), prior.settings())
                    self.assertFalse(spec.settings()["enable_inspect_tool"])
                    model = "Luna" if spec.model_type == "gpt-5.6-luna" else "Terra"
                    self.assertEqual(spec.system_name,
                                     f"DataflowSystem{model}NativeCampaignObserveOnly{spec.key}20260916Rep1")
                    arm = getattr(systems, spec.system_name)(output_dir=directory, computing_unit_id=321)
                    self.assertEqual(arm.campaign_id, campaign.OBSERVE_ONLY_CAMPAIGN_ID)
                    self.assertEqual(arm.pilot_spec.reasoning_effort, "medium")
                    self.assertEqual(arm._attempt_pricing()["rates"], frozen_pricing(spec.model_type)["rates"])
                    self.assertEqual(arm._attempt_pricing()["source"]["table"], campaign.OBSERVE_ONLY_CAMPAIGN_ID)
                    arm.campaign_round = "first"
                    self.assertEqual(arm._attempt_metadata(), {
                        "campaign": campaign.OBSERVE_ONLY_CAMPAIGN_ID, "round": "first"})
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        getattr(systems, spec.system_name)(output_dir=directory, computing_unit_id=321,
                                                          enable_inspect_tool=True)

    def test_observe_only_cannot_archive_superseded_attempt_as_its_own(self):
        with TemporaryDirectory() as directory:
            old = Arm(output_dir=directory, computing_unit_id=321)
            with patch.dict(os.environ, {"NATIVE_CAMPAIGN_ROUND": "first"}):
                old._attempt_bundle(TASK["id"])
            before = (Path(directory) / TASK["id"] / "attempt.json").read_bytes()
            new = getattr(systems, campaign.OBSERVE_ONLY_CAMPAIGN_ARMS[0].system_name)(
                output_dir=directory, computing_unit_id=321)
            with patch.dict(os.environ, {"NATIVE_CAMPAIGN_ROUND": "recovery1"}):
                with self.assertRaisesRegex(ValueError, "campaign_namespace"):
                    new._attempt_bundle(TASK["id"])
            self.assertEqual((Path(directory) / TASK["id"] / "attempt.json").read_bytes(), before)
            self.assertFalse((Path(directory) / "_attempts").exists())

    def test_exact_eight_frozen_arms_preserve_pilot_settings(self):
        from systems.native_python_system import ALL_PILOT_ARMS

        self.assertEqual(len(CAMPAIGN_ARMS), 8)
        with TemporaryDirectory() as directory:
            for spec in CAMPAIGN_ARMS:
                pilot = next(p for p in ALL_PILOT_ARMS if (p.key, p.model_type) == (spec.key, spec.model_type))
                self.assertEqual(spec.settings(), pilot.settings())
                arm = getattr(systems, spec.system_name)(output_dir=directory, computing_unit_id=321)
                self.assertIsInstance(arm, NativeCampaignSystem)
                self.assertEqual(arm.model_type, spec.model_type)
                self.assertEqual(arm.agent_service_endpoint, f"http://localhost:{spec.port}")
                self.assertEqual(arm.pilot_spec.reasoning_effort, "medium")
                for override in ({"max_steps": 26}, {"schema_in_result": True}, {"model_type": "gpt-5.2"}):
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        getattr(systems, spec.system_name)(output_dir=directory, computing_unit_id=321, **override)

    def test_explicit_campaign_environment_and_no_implicit_cu(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "NATIVE_CAMPAIGN_CUID"):
                Arm(output_dir=directory)
            with patch.dict(os.environ, {"NATIVE_CAMPAIGN_CUID": "321", "NATIVE_CAMPAIGN_V2_ENDPOINT": "http://127.0.0.1:3011"}):
                arm = Arm(output_dir=directory)
                self.assertEqual(arm.computing_unit_id, 321)
                self.assertEqual(arm.agent_service_endpoint, "http://127.0.0.1:3011")

    def test_prices_are_frozen_per_model(self):
        for model, scale in (("gpt-5.6-luna", 1), ("gpt-5.6-terra", 10)):
            self.assertEqual(frozen_pricing(model)["rates"], {
                "input": scale * .2 / 1e6, "cache_read": scale * .02 / 1e6,
                "output": scale * 1.2 / 1e6, "cache_creation": 0,
            })

    def test_all_104_main_tasks_admit_under_unchanged_oracle_policy(self):
        root = Path(__file__).resolve().parents[1]
        domains = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")
        workloads = {domain: json.loads((root / "workload" / f"{domain}.json").read_text()) for domain in domains}
        tasks = [task for domain in domains for task in workloads[domain]]
        self.assertEqual(len(tasks), 104)
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            fixture = Path(directory)
            (fixture / "workload").symlink_to(root / "workload", target_is_directory=True)
            (fixture / "format_hint").symlink_to(root / "format_hint", target_is_directory=True)
            for domain in domains:
                (fixture / "data" / domain / "input").mkdir(parents=True)
            for spec in campaign.ALL_CAMPAIGN_ARMS:
                arm = getattr(systems, spec.system_name)(output_dir=directory, computing_unit_id=321)
                arm._setup_agent = Mock(side_effect=AssertionError("metadata loading must not allocate an agent"))
                arm._expand_data_sources = Mock(return_value=["data/fixture.csv"])
                for domain in domains:
                    # Exercise the real loader, including any *-tiny fixture
                    # that used to shadow canonical metadata during setup.
                    arm.process_dataset(fixture / "data" / domain / "input")
                    hints = {row["id"]: row["format_hint"] for row in
                             json.loads((root / "format_hint" / f"{domain}.json").read_text())}
                    for task in workloads[domain]:
                        with self.subTest(sut=spec.system_name, task=task["id"]):
                            observed, prompt = arm._attempt_inputs(task["query"], task["id"], task["data_sources"])
                            self.assertEqual(observed, task)
                            self.assertEqual(arm.format_hints[task["id"]], hints[task["id"]])
                            self.assertIn(task["query"], prompt)
                            self.assertIn(hints[task["id"]], prompt)
                            self.assertNotIn('"answer_type"', prompt)
                self.assertEqual(arm.workload_data, {task["id"]: task for task in tasks})
                arm._setup_agent.assert_not_called()

    def test_canonical_loader_preserves_format_hints_and_frozen_checks(self):
        from systems.dataflow_system import DataflowSystem

        root = Path(__file__).resolve().parents[1]
        canonical = json.loads((root / "workload/legal.json").read_text())[0]
        tiny = json.loads((root / "workload/legal-tiny.json").read_text())[0]
        self.assertEqual(canonical["id"], tiny["id"])
        self.assertNotEqual(canonical["data_sources"], tiny["data_sources"])
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            fixture = Path(directory)
            (fixture / "workload").symlink_to(root / "workload", target_is_directory=True)
            (fixture / "format_hint").symlink_to(root / "format_hint", target_is_directory=True)
            dataset = fixture / "data/legal/input"
            dataset.mkdir(parents=True)
            legacy = DataflowSystem(output_dir=directory)
            legacy._prepare_dataset(dataset)
            arm = Arm(output_dir=directory, computing_unit_id=321)
            arm.process_dataset(dataset)
            self.assertEqual(arm.workload_data[canonical["id"]], canonical)
            self.assertEqual(arm.format_hints, legacy.format_hints)
            # The fix is campaign-only; the shared/pilot loader is unchanged.
            self.assertEqual(legacy.workload_data[tiny["id"]], tiny)
            arm._expand_data_sources = Mock(return_value=["data/fixture.csv"])
            arm._attempt_bundle = Mock(side_effect=AssertionError("invalid inputs must not reserve an attempt"))
            arm._setup_agent = Mock(side_effect=AssertionError("invalid inputs must not allocate an agent"))
            for query, sources in ((canonical["query"] + " changed", canonical["data_sources"]),
                                   (canonical["query"], tiny["data_sources"]), (canonical["query"], [])):
                with self.assertRaisesRegex(ValueError, "query and oracle-file policy are frozen"):
                    arm.serve_query(query, canonical["id"], sources)
            arm._attempt_bundle.assert_not_called()
            arm._setup_agent.assert_not_called()


class CampaignAttemptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.environment = patch.dict(os.environ, {"NATIVE_CAMPAIGN_ROUND": "first"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.arm = Arm(output_dir=self.temporary.name, computing_unit_id=321)
        self.arm.dataset_directory = "data/legal/input"
        self.arm.dataset = {"fixture.csv": None}
        self.arm.workload_data = {TASK["id"]: TASK}
        self.arm._expand_data_sources = Mock(return_value=["data/legal/input/fixture.csv"])
        self.agent = SimpleNamespace(
            agent_id="fixture-agent", _workflow_id=456, _token="private-token",
            agent_service_endpoint="http://localhost:3011", settings=AgentSettings(),
            run=Mock(return_value=MessageResult(STEP["content"], [], {}, {}, False, completed=True)),
        )
        self.arm._setup_agent = Mock(side_effect=lambda: setattr(self.arm, "agent", self.agent))
        self.guard = Mock(return_value={"qualified": True, "manifest_sha256": "fixture"})
        self.arm._campaign_guard = self.guard
        self.info = {
            "id": "fixture-agent", "modelType": "gpt-5.6-luna", "driver": "vercel-tool-use",
            "state": "AVAILABLE", "settings": self.agent.settings.to_api_dict(),
            "delegate": {"workflowId": 456, "computingUnitId": 321, "userToken": "private-token"},
        }
        self.trace = {"steps": [STEP], "state": "AVAILABLE"}
        self.api = patch("systems.native_pilot_system.agent_json", side_effect=self.request)
        self.api.start()
        self.addCleanup(self.api.stop)
        cleanup = patch("utils.native_campaign.cleanup_campaign_resources", return_value={
            "status": "pending", "backend_terminal_verified": False,
        })
        cleanup.start()
        self.addCleanup(cleanup.stop)

    def request(self, agent, resource):
        if resource == "":
            return self.info
        if resource == "/react-steps":
            return self.trace
        if resource == "/workflow":
            return {"workflow": {"operators": [], "links": []}}
        if resource.startswith("/snapshots/"):
            return {"stepId": resource.rsplit("/", 1)[-1], "results": {"op": {"profile": {"version": "fixture"}}}}
        raise AssertionError(resource)

    def serve(self):
        return self.arm.serve_query(TASK["query"], TASK["id"], TASK["data_sources"])

    def artifact(self, name):
        return json.loads((Path(self.temporary.name) / TASK["id"] / name).read_text())

    def test_any_registered_main_task_uses_complete_capture_and_one_dispatch(self):
        result = self.serve()
        self.agent.run.assert_called_once()
        self.assertEqual(self.agent.run.call_args.kwargs["empty_turn_retries"], 0)
        self.assertEqual(result["explanation"]["answer"], '["a", "b"]')
        self.assertEqual(self.artifact("react_steps.json")["steps"], [STEP])
        self.assertEqual(len(self.artifact("snapshots.json")["snapshots"]), 1)
        self.assertEqual(self.artifact("attempt.json")["round"], "first")
        self.assertGreater(self.artifact("stats.json")["first_pass"]["cost_usd"], 0)
        self.assertEqual(self.artifact("stats.json")["all_attempts"]["attempt_count"], 1)
        self.assertNotIn("private-token", json.dumps(self.artifact("config.json")))

    def test_recovery_requires_round_and_archives_every_prior_artifact(self):
        self.serve()
        prior = Path(self.temporary.name) / TASK["id"]
        (prior / "evaluation.json").write_text('{"success":0}')
        original = {path.name: path.read_bytes() for path in prior.iterdir()}
        with self.assertRaises(FileExistsError):
            self.serve()
        with patch.dict(os.environ, {"NATIVE_CAMPAIGN_ROUND": "recovery1"}):
            self.serve()
            with self.assertRaises(FileExistsError):
                self.serve()
        archived = Path(self.temporary.name) / "_attempts" / TASK["id"] / "first"
        for name, value in original.items():
            self.assertEqual((archived / name).read_bytes(), value)
        stats = self.artifact("stats.json")
        self.assertEqual(stats["all_attempts"]["attempt_count"], 2)
        self.assertEqual(stats["all_attempts"]["cost_usd"], 2 * stats["first_pass"]["cost_usd"])
        self.assertFalse((prior / "evaluation.json").exists())

    def test_recovery_preserves_directory_killed_before_metadata(self):
        prior = Path(self.temporary.name) / TASK["id"]
        prior.mkdir()
        (prior / "partial.txt").write_text("before metadata")
        with patch.dict(os.environ, {"NATIVE_CAMPAIGN_ROUND": "recovery1"}):
            self.serve()
        self.assertEqual((Path(self.temporary.name) / "_attempts" / TASK["id"] / "first" / "partial.txt").read_text(), "before metadata")
        self.assertIsNone(self.artifact("stats.json")["all_attempts"]["cost_usd"])

    def test_explicit_manifest_input_map_changes_only_prompt_paths(self):
        self.guard.manifest = {"task_inputs": {TASK["id"]: ["data/legal/input/corrected.csv"]}}
        self.serve()
        prompt = (Path(self.temporary.name) / TASK["id"] / "prompt.txt").read_text()
        self.assertIn("corrected.csv", prompt)
        self.assertEqual(self.artifact("ground_truth.json"), TASK)
        self.assertEqual(self.artifact("config.json")["subset_files"], TASK["data_sources"])

    def test_failed_dispatch_keeps_observed_cost_and_partial_first_pass(self):
        def fail(prompt, *, on_event, **kwargs):
            on_event({"type": "step", "step": STEP})
            raise ConnectionError("private-token")

        self.agent.run.side_effect = fail
        def unavailable(agent, resource):
            if resource == "" and self.agent.run.call_count == 0:
                return self.info
            raise ConnectionError("private-token")
        with patch("systems.native_pilot_system.agent_json", side_effect=unavailable):
            result = self.serve()
        self.assertIsNone(result["cost_usd"])
        first = self.artifact("stats.json")["first_pass"]
        self.assertGreater(first["observed_cost_usd"], 0)
        self.assertEqual(first["usage_status"], "partial")
        self.agent.run.side_effect = None
        with patch.dict(os.environ, {"NATIVE_CAMPAIGN_ROUND": "recovery1"}):
            self.serve()
        self.assertIsNone(self.artifact("stats.json")["all_attempts"]["cost_usd"])
        self.assertGreater(self.artifact("stats.json")["all_attempts"]["observed_cost_usd"], first["observed_cost_usd"])

    def test_long_context_prices_are_flagged_unknown_not_underpriced(self):
        self.trace["steps"] = [{**STEP, "usage": {"inputTokens": 272001, "cachedInputTokens": 0, "outputTokens": 20}}]
        result = self.serve()
        stats = self.artifact("stats.json")
        self.assertEqual(stats["long_context_step_ids"], ["s1"])
        self.assertIsNone(result["cost_usd"])
        self.assertIsNone(stats["observed_cost_usd"])
        self.assertGreater(stats["base_rate_estimate_usd"], 0)

    def test_no_oracle_or_changed_task_rejected_before_setup(self):
        for query, subset in ((TASK["query"], []), ("changed", TASK["data_sources"])):
            with self.assertRaisesRegex(ValueError, "task|oracle"):
                self.arm.serve_query(query, TASK["id"], subset)
        self.arm._setup_agent.assert_not_called()

    def test_observe_only_inspector_rejection_never_dispatches_a_model(self):
        self.arm.campaign_id = campaign.OBSERVE_ONLY_CAMPAIGN_ID
        self.arm._campaign_guard = lambda system, stage, info: (
            {"qualified": True} if stage == "before_setup" else
            {"qualified": True, "tool_surface": CampaignGuard.check_tool_surface(system.agent)}
        )
        surface = {"systemPrompt": "Use dataflow.", "tools": [
            {"name": "inspectResult", "description": "Deprecated.", "inputSchema": {}, "enabled": False},
        ]}
        with patch("systems.native_pilot_system.agent_json", side_effect=lambda agent, resource:
                   surface if resource == "/system-info" else self.request(agent, resource)):
            self.serve()
        self.agent.run.assert_not_called()
        attempt = self.artifact("attempt.json")
        self.assertEqual(attempt["campaign"], campaign.OBSERVE_ONLY_CAMPAIGN_ID)
        self.assertFalse(attempt["dispatched"])
        self.assertEqual(attempt["errors"][0]["stage"], "before_dispatch")
        self.assertIsNone(self.artifact("stats.json")["cost_usd"])

    def test_observe_only_success_persists_exact_tool_surface_with_same_budget(self):
        self.arm.campaign_id = campaign.OBSERVE_ONLY_CAMPAIGN_ID
        self.arm._campaign_guard = lambda system, stage, info: (
            {"qualified": True} if stage == "before_setup" else
            {"qualified": True, "tool_surface": CampaignGuard.check_tool_surface(system.agent)}
        )
        surface = {"systemPrompt": "Use dataflow.", "tools": [
            {"name": "dataflow", "description": "Observe results.", "inputSchema": {}, "enabled": True},
        ]}
        with patch("systems.native_pilot_system.agent_json", side_effect=lambda agent, resource:
                   surface if resource == "/system-info" else self.request(agent, resource)):
            self.serve()
        self.agent.run.assert_called_once()
        self.assertEqual(self.agent.run.call_args.kwargs["empty_turn_retries"], 0)
        config = self.artifact("config.json")
        self.assertEqual(config["max_turn_seconds"], 1800)
        self.assertEqual(config["admission"]["tool_surface"]["tools"], surface["tools"])
        self.assertEqual(config["postflight"]["tool_surface"]["systemPrompt"], surface["systemPrompt"])
        self.assertEqual(config["pricing"]["source"]["table"], campaign.OBSERVE_ONLY_CAMPAIGN_ID)


class CampaignGuardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        for folder in ("data/legal/input", "workload", "format_hint", "worker"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "data/legal/input/fixture.csv").write_text("value\na\n")
        (self.root / "workload/legal.json").write_text(json.dumps([TASK]))
        (self.root / "format_hint/legal.json").write_text("[]")
        (self.root / "worker/data").symlink_to(self.root / "data", target_is_directory=True)
        self.bindings = [file_binding(self.root / name) for name in (
            "data/legal/input/fixture.csv", "workload/legal.json", "format_hint/legal.json")]
        self.manifest = {
            "version": 1, "campaign": "NativeCampaign20260915Rep1", "harness_sha": "a" * 40,
            "computing_unit_id": 321,
            "services": {"V2": {"endpoint": "http://localhost:3011", "launch_record": "fixture", "git_sha": "b" * 40}},
            "gateway": {"gpt-5.6-luna": {"name": "gpt-5.6-luna", "reasoning_effort": "medium"}},
            "runtime": {"endpoint": "http://localhost:8085", "pid": 1, "start_ticks": 2, "boot_id": "fixture"},
            "bindings": self.bindings,
        }
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))
        settings = AgentSettings()
        self.system = SimpleNamespace(
            campaign_query_id=TASK["id"], campaign_prompt_paths=["data/legal/input/*.csv"],
            workload_data={TASK["id"]: TASK}, pilot_spec=SimpleNamespace(key="Combined"),
            computing_unit_id=321, agent_service_endpoint="http://localhost:3011", model_type="gpt-5.6-luna",
            agent=SimpleNamespace(agent_id="fixture", _workflow_id=456, settings=settings),
        )
        self.info = {"id": "fixture", "modelType": "gpt-5.6-luna", "driver": "vercel-tool-use", "state": "AVAILABLE",
                     "settings": settings.to_api_dict(), "delegate": {"computingUnitId": 321, "workflowId": 456}}
        self.mocks = [
            patch.dict(os.environ, {"NATIVE_CAMPAIGN_MANIFEST": str(self.path)}),
            patch("utils.native_campaign.HARNESS_ROOT", self.root),
            patch("utils.native_campaign.subprocess.check_output", side_effect=lambda args, **kw: "a" * 40 if "HEAD" in args else ""),
            patch("utils.native_campaign.verify_service_launch", return_value={"service_endpoint": "http://localhost:3011", "git_sha": "b" * 40, "recorder_url": "http://localhost:8085"}),
            patch("utils.native_campaign.gateway_route", return_value=self.manifest["gateway"]["gpt-5.6-luna"]),
            patch("utils.native_campaign.runtime_identity", return_value=str(self.root / "worker")),
        ]
        for mock in self.mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def test_missing_manifest_blocks_before_model_dispatch(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "NATIVE_CAMPAIGN_MANIFEST"):
            CampaignGuard()

    def test_file_binding_detects_content_changes_even_if_size_is_same(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("before")
            binding = file_binding(path)
            CampaignGuard.check_binding(binding)
            path.write_text("after!")
            with self.assertRaisesRegex(ValueError, "binding"):
                CampaignGuard.check_binding(binding)

    def test_guard_accepts_bound_wildcards_and_effective_settings(self):
        guard = CampaignGuard()
        result = guard(self.system, "before_dispatch", self.info)
        self.assertIs(result["qualified"], True)
        self.assertEqual(result["manifest_sha256"], file_binding(self.path)["sha256"])

    def observe_only_guard(self):
        self.manifest["campaign"] = campaign.OBSERVE_ONLY_CAMPAIGN_ID
        self.path.write_text(json.dumps(self.manifest))
        self.system.campaign_id = campaign.OBSERVE_ONLY_CAMPAIGN_ID
        return CampaignGuard(campaign=campaign.OBSERVE_ONLY_CAMPAIGN_ID)

    def test_new_and_legacy_manifest_namespaces_cannot_mix(self):
        with self.assertRaisesRegex(ValueError, "campaign_manifest_invalid"):
            CampaignGuard(campaign=campaign.OBSERVE_ONLY_CAMPAIGN_ID)
        guard = self.observe_only_guard()
        with self.assertRaisesRegex(ValueError, "campaign_manifest_invalid"):
            CampaignGuard()
        self.system.campaign_id = campaign.CAMPAIGN_ID
        with self.assertRaisesRegex(ValueError, "campaign_namespace_mismatch"):
            guard(self.system, "before_setup", None)

    def test_all_eight_observe_only_launches_record_actual_prompt_and_tools(self):
        surface = {"systemPrompt": "Use dataflow to observe cached results.", "tools": [
            {"name": "dataflow", "description": "Edit, run and observe.",
             "inputSchema": {"type": "object", "properties": {}}, "enabled": True},
        ]}
        for spec in campaign.OBSERVE_ONLY_CAMPAIGN_ARMS:
            with self.subTest(sut=spec.system_name):
                key = "BatchParent" if spec.key == "BatchParent" else "V2"
                self.system.pilot_spec = spec
                self.system.model_type = spec.model_type
                self.system.agent_service_endpoint = f"http://localhost:{spec.port}"
                self.system.agent.settings = AgentSettings(**{
                    key: value for key, value in spec.settings().items() if key in AgentSettings.__dataclass_fields__})
                self.info["modelType"] = spec.model_type
                self.info["settings"] = self.system.agent.settings.to_api_dict()
                self.manifest["services"][key] = {
                    "endpoint": self.system.agent_service_endpoint, "launch_record": "fixture", "git_sha": "b" * 40}
                route = {"name": spec.model_type, "reasoning_effort": "medium"}
                self.manifest["gateway"][spec.model_type] = route
                guard = self.observe_only_guard()
                source = {"service_endpoint": self.system.agent_service_endpoint,
                          "git_sha": "b" * 40, "recorder_url": "http://localhost:8085"}
                with patch("utils.native_campaign.verify_service_launch", return_value=source), patch(
                    "utils.native_campaign.gateway_route", return_value=route), patch(
                    "systems.native_pilot_system.agent_json", return_value=surface) as api:
                    self.assertTrue(guard(self.system, "before_setup", None)["qualified"])
                    api.assert_not_called()
                    admitted = guard(self.system, "before_dispatch", self.info)
                    api.assert_called_once_with(self.system.agent, "/system-info")
                    evidence = admitted["tool_surface"]
                    self.assertEqual(evidence["systemPrompt"], surface["systemPrompt"])
                    self.assertEqual(evidence["tools"], surface["tools"])
                    self.assertEqual(evidence["tool_names"], ["dataflow"])
                    self.assertEqual(evidence["enabled_tool_names"], ["dataflow"])
                    encoded = json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
                    self.assertEqual(evidence["sha256"], hashlib.sha256(encoded).hexdigest())
                    self.assertEqual(guard(self.system, "after_attempt", self.info)["tool_surface"], evidence)
                    # Deprecated wire flag may disappear or remain inert false.
                    self.info["settings"]["enableInspectTool"] = False
                    self.assertTrue(guard(self.system, "before_dispatch", self.info)["qualified"])
                    self.info["settings"]["enableInspectTool"] = True
                    with self.assertRaisesRegex(ValueError, "enableInspectTool"):
                        guard(self.system, "before_dispatch", self.info)
                    self.info["settings"].pop("enableInspectTool")

    def test_observe_only_rejects_disabled_inspector_and_prompt_schema_leaks(self):
        guard = self.observe_only_guard()
        tool = {"name": "dataflow", "description": "Observe results.", "inputSchema": {}, "enabled": True}
        surfaces = [
            {"systemPrompt": "Use dataflow.", "tools": [tool, {**tool, "name": "inspectResult", "enabled": enabled}]}
            for enabled in (True, False)
        ] + [
            {"systemPrompt": "Use inspectResult on the output.", "tools": [tool]},
            {"systemPrompt": "Use dataflow.", "tools": [{**tool, "description": "Can call inspectResult."}]},
            {"systemPrompt": "Use dataflow.", "tools": [{**tool, "inputSchema": {"inspectResult": {}}}]},
        ]
        for surface in surfaces:
            with self.subTest(surface=surface), patch("systems.native_pilot_system.agent_json", return_value=surface):
                with self.assertRaisesRegex(ValueError, "campaign_inspect_result_forbidden"):
                    guard(self.system, "before_dispatch", self.info)
                with self.assertRaisesRegex(ValueError, "campaign_inspect_result_forbidden"):
                    guard(self.system, "after_attempt", self.info)

    def test_observe_only_rejects_missing_invalid_or_unavailable_tool_surface(self):
        guard = self.observe_only_guard()
        tool = {"name": "dataflow", "description": "Observe.", "inputSchema": {}, "enabled": True}
        for surface in ({}, {"systemPrompt": "", "tools": [tool]}, {"systemPrompt": "x", "tools": []},
                        {"systemPrompt": "x", "tools": [tool, tool]},
                        {"systemPrompt": "x", "tools": [{**tool, "enabled": False}]},
                        {"systemPrompt": "x", "tools": [{**tool, "name": "other"}]}):
            with self.subTest(surface=surface), patch("systems.native_pilot_system.agent_json", return_value=surface):
                with self.assertRaisesRegex(ValueError, "campaign_tool_surface"):
                    guard(self.system, "before_dispatch", self.info)
        with patch("systems.native_pilot_system.agent_json", side_effect=ConnectionError("unavailable")):
            with self.assertRaises(ConnectionError):
                guard(self.system, "before_dispatch", self.info)

    def test_only_official_paraphrase_cache_may_change_during_scoring(self):
        cache = self.root / "benchmark/fixtures/paraphrase_cache.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}")
        code = self.root / "benchmark/metrics.py"
        code.write_text("# frozen benchmark code\n")

        def git_output(arguments, **kwargs):
            result = subprocess.run(arguments, capture_output=True, text=True, check=True)
            return result.stdout

        def git(*arguments):
            return git_output(["git", "-C", str(self.root), *arguments]).strip()

        git("init", "--quiet")
        git("add", "benchmark")
        git("-c", "user.name=Campaign Test", "-c", "user.email=campaign-test@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture")
        self.manifest["harness_sha"] = git("rev-parse", "HEAD")
        self.path.write_text(json.dumps(self.manifest))
        guard = CampaignGuard()
        cache.write_text('{"official-scorer-cache-key":true}')
        with patch("utils.native_campaign.subprocess.check_output", side_effect=git_output):
            self.assertTrue(guard(self.system, "before_setup", None)["qualified"])
            for relative in ("benchmark/metrics.py", "benchmark/fixtures/other_cache.json",
                             "benchmark/fixtures/paraphrase_cache.json.extra"):
                with self.subTest(path=relative):
                    changed = self.root / relative
                    previous = changed.read_text() if changed.exists() else None
                    changed.write_text("unexpected runtime modification")
                    with self.assertRaisesRegex(ValueError, "campaign_harness_source_dirty"):
                        guard(self.system, "before_setup", None)
                    if previous is None:
                        changed.unlink()
                    else:
                        changed.write_text(previous)

    def test_guard_rejects_manifest_route_settings_and_input_drift(self):
        guard = CampaignGuard()
        with patch("utils.native_campaign.gateway_route", return_value={"reasoning_effort": "high"}):
            with self.assertRaisesRegex(ValueError, "gateway"):
                guard(self.system, "before_setup", None)
        with patch("utils.native_campaign.verify_service_launch", return_value={"service_endpoint": "http://localhost:3011", "git_sha": "changed"}):
            with self.assertRaisesRegex(ValueError, "service"):
                guard(self.system, "before_setup", None)
        self.info["settings"]["columnStats"] = True
        with self.assertRaisesRegex(ValueError, "columnStats"):
            guard(self.system, "before_dispatch", self.info)
        self.info["settings"]["columnStats"] = False
        (self.root / "data/legal/input/fixture.csv").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "binding"):
            guard(self.system, "before_setup", None)


    def test_added_wildcard_file_is_not_silently_unbound(self):
        guard = CampaignGuard()
        (self.root / "data/legal/input/additional.csv").write_text("new\n")
        with self.assertRaisesRegex(ValueError, "binding"):
            guard(self.system, "before_setup", None)


class CampaignCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = AttemptBundle(Path(self.temporary.name) / "task")
        self.resources = {"agent_id": "fixture", "workflow_id": 456, "computing_unit_id": 321}
        self.attempt = {"resources": self.resources, "completed_event": True, "capture_complete": True}
        self.system = SimpleNamespace(
            agent_service_endpoint="http://localhost:3011", computing_unit_id=321,
            agent=SimpleNamespace(_token="private-token"), pilot_spec=SimpleNamespace(key="Combined"),
            _campaign_guard=SimpleNamespace(manifest={
                "services": {"V2": {"launch_record": "fixture"}}, "runtime": {"endpoint": "http://localhost:8085"},
            }),
        )
        self.history = [{"status": 3, "eId": 789, "cuId": 321}]
        self.workflow = [{"isOwner": True, "workflow": {"wid": 456, "name": "owned-workflow"}}]
        self.agents = [{"id": "fixture"}]
        self.api = Mock()
        self.api.request.side_effect = self.request
        journal = {"allocations_complete": True, "references": {"computing_unit": 321}, "owned": {
            "agent": {"id": "fixture", "name": "owned-agent"}, "workflow": {"id": 456, "name": "owned-workflow"},
        }}
        for mock in (
            patch("utils.native_campaign.verify_service_launch", return_value={"service_endpoint": "http://localhost:3011"}),
            patch("utils.resource_journal.read_resource_journal", return_value=journal),
            patch("utils.pilot_cleanup.PilotResourceAPI", return_value=self.api),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def request(self, service, method, path, payload=None):
        if path == "/api/agents/fixture":
            if method == "DELETE":
                self.agents = []
                return None
            return {"id": "fixture", "name": "owned-agent", "state": "AVAILABLE",
                    "delegate": {"workflowId": 456, "computingUnitId": 321}}
        if path == "/api/agents/":
            return {"agents": self.agents}
        if path == "/api/workflow/list":
            return self.workflow
        if path == "/api/workflow/delete":
            self.assertEqual(payload, {"wids": [456]})
            self.workflow = []
            return None
        if path == "/api/executions/456":
            return self.history
        if path == "/api/computing-unit":
            return [{"isOwner": True, "computingUnit": {"cuid": 321, "type": "local", "uri": "http://localhost:8085"}}]
        raise AssertionError((service, method, path))

    def test_completed_turn_deletes_only_owned_agent_and_workflow(self):
        result = cleanup_campaign_resources(self.system, self.bundle, self.attempt)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["backend_terminal_verified"])
        mutations = [call.args[:3] for call in self.api.request.call_args_list if call.args[1] != "GET"]
        self.assertEqual(mutations, [("agent", "DELETE", "/api/agents/fixture"), ("texera", "POST", "/api/workflow/delete")])
        self.assertEqual(result["retained_computing_unit_id"], 321)

    def test_nonterminal_execution_retains_resources(self):
        self.history[0]["status"] = 1
        result = cleanup_campaign_resources(self.system, self.bundle, self.attempt)
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["backend_terminal_verified"])
        self.assertEqual(result["unresolved"], self.resources)
        self.assertTrue(all(call.args[1] == "GET" for call in self.api.request.call_args_list))

    def test_interrupted_turn_never_attempts_cleanup(self):
        self.attempt["completed_event"] = False
        self.assertEqual(cleanup_campaign_resources(self.system, self.bundle, self.attempt)["status"], "pending")
        self.api.request.assert_not_called()

    def test_unverified_deletion_retains_workflow_and_ids(self):
        def failed_delete(service, method, path, payload=None):
            if method == "DELETE":
                raise ConnectionError("private-token")
            return self.request(service, method, path, payload)
        self.api.request.side_effect = failed_delete
        result = cleanup_campaign_resources(self.system, self.bundle, self.attempt)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["unresolved"], self.resources)
        self.assertNotIn("private-token", json.dumps(result))

if __name__ == "__main__":
    unittest.main()

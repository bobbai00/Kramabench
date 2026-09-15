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

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from systems.cost_utils import compute_cost, price_schedule
from utils.pilot_artifacts import AttemptBundle, usage_report


def agent_step(identifier="s1", **changes):
    return {
        "id": identifier,
        "role": "agent",
        "isEnd": True,
        "content": "17",
        "inputMessages": [{"role": "user", "content": "fixture query"}],
        "usage": {
            "inputTokens": 100,
            "outputTokens": 20,
            "totalTokens": 120,
            "cachedInputTokens": 80,
            "reasoningTokens": 5,
        },
        **changes,
    }


class PilotArtifactsTest(unittest.TestCase):
    def test_artifact_helper_can_be_imported_without_importing_systems_first(self):
        result = subprocess.run([sys.executable, "-c", "import utils.pilot_artifacts"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_usage_uses_frozen_cache_prices_without_double_billing_reasoning(self):
        pricing = price_schedule("gpt-5.6-luna")
        report = usage_report([agent_step()], completed=True, pricing=pricing)
        self.assertEqual(report["usage_status"], "complete")
        self.assertEqual(report["input_tokens"], 100)
        self.assertEqual(report["cached_tokens"], 80)
        self.assertEqual(report["reasoning_tokens"], 5)
        self.assertEqual(report["cost_usd"], compute_cost("gpt-5.6-luna", 100, 20, 80))
        # Rates are captured before the attempt and must not be re-resolved.
        with patch("systems.cost_utils._prices", return_value=(9, 9, 9, 9)):
            again = usage_report([agent_step()], completed=True, pricing=pricing)
        self.assertEqual(report["cost_usd"], again["cost_usd"])
        self.assertEqual(pricing["kind"], "token_estimate")
        self.assertTrue(pricing["source"])

    def test_partial_trace_keeps_usage_but_never_reports_a_complete_cost(self):
        report = usage_report([agent_step()], completed=False, pricing=price_schedule("gpt-5.6-luna"))
        self.assertEqual(report["usage_status"], "partial")
        self.assertEqual(report["input_tokens"], 100)
        self.assertIsNone(report["cost_usd"])
        self.assertGreater(report["observed_cost_usd"], 0)

    def test_missing_usage_prices_and_impossible_counts_do_not_look_free(self):
        for steps, pricing in (
            ([], price_schedule("gpt-5.6-luna")),
            ([agent_step(usage={})], price_schedule("gpt-5.6-luna")),
            ([agent_step()], None),
            (
                [agent_step(usage={"inputTokens": 10, "outputTokens": 1, "cachedInputTokens": 11})],
                price_schedule("gpt-5.6-luna"),
            ),
        ):
            with self.subTest(steps=steps, priced=pricing is not None):
                report = usage_report(steps, completed=True, pricing=pricing)
                self.assertIsNone(report["cost_usd"])
                self.assertNotEqual(report["cost_status"], "complete")

    def test_step_ids_deduplicate_stream_updates_without_adding_usage_twice(self):
        report = usage_report(
            [agent_step(isEnd=False), agent_step()], completed=True, pricing=price_schedule("gpt-5.6-luna")
        )
        self.assertEqual(report["input_tokens"], 100)
        self.assertEqual(report["num_steps"], 1)

    def test_synthetic_error_and_missing_cache_counts_make_usage_incomplete(self):
        steps = [agent_step(), {"id": "err", "role": "agent", "isEnd": True, "content": "Error: stopped"}]
        report = usage_report(steps, completed=True, pricing=price_schedule("gpt-5.6-luna"))
        self.assertEqual(report["usage_status"], "partial")
        self.assertIsNone(report["cost_usd"])
        missing_cache = agent_step(usage={"inputTokens": 100, "outputTokens": 20})
        self.assertIsNone(
            usage_report([missing_cache], completed=True, pricing=price_schedule("gpt-5.6-luna"))["cost_usd"]
        )

    def test_bundle_refuses_existing_attempt_and_journals_before_finalization(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "task"
            bundle = AttemptBundle(target)
            bundle.event({"type": "step", "step": agent_step()})
            bundle.event({"type": "init", "delegate": {"userToken": "private"}})
            rows = [json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event"]["step"]["id"], "s1")
            self.assertEqual(bundle.stream_steps(), [agent_step()])
            with self.assertRaises(FileExistsError):
                AttemptBundle(target)
            self.assertEqual((target / "events.jsonl").stat().st_mode & 0o777, 0o600)
            self.assertNotIn("private", (target / "events.jsonl").read_text())

    def test_json_writes_are_complete_and_disallow_path_escape(self):
        with TemporaryDirectory() as directory:
            bundle = AttemptBundle(Path(directory) / "task")
            bundle.write("stats.json", {"cost_usd": None})
            self.assertEqual(json.loads((bundle.path / "stats.json").read_text()), {"cost_usd": None})
            for name in ("../outside.json", "/outside.json", "nested/stats.json"):
                with self.assertRaises(ValueError):
                    bundle.write(name, {})


if __name__ == "__main__":
    unittest.main()

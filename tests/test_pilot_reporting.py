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

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import kb
from benchmark.benchmark import Evaluator


class PilotReportingTest(unittest.TestCase):
    def test_official_numeric_scoring_does_not_construct_an_unused_llm_judge(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps([{"id": "fixture-1", "answer_type": "numeric_exact", "answer": 17}]))
            with patch("benchmark.benchmark.GPTInterface", side_effect=AssertionError("unexpected model setup")):
                evaluator = Evaluator(str(path), "benchmark/fixtures", directory, evaluate_pipeline=False)
                results = evaluator.evaluate_results([{"task_id": "fixture-1", "model_output": {"answer": "17"}}])
            self.assertEqual(results[0]["success"], 1)
            self.assertEqual(results[0]["token_usage_metrics"], 0)

    def test_unknown_invalid_and_partial_costs_remain_unknown_in_shared_helpers(self):
        with TemporaryDirectory() as directory, patch.object(kb, "KB_ROOT", Path(directory)):
            base = Path(directory) / "system_scratch" / "SUT"
            records = [None, "bad", -1, float("inf"), 0, 0.01]
            for index, cost in enumerate(records):
                target = base / f"fixture-easy-{index}"
                target.mkdir(parents=True)
                (target / "stats.json").write_text(json.dumps({"input_tokens": 100, "cost_usd": cost}))
            values = kb.load_cost_stats("SUT")
            self.assertEqual([value["cost"] for value in values], [None, None, None, None, None, 0.01])
            target = base / "fixture-easy-6"
            target.mkdir()
            (target / "stats.json").write_text(
                json.dumps({"input_tokens": 100, "cost_usd": 0.01, "cost_status": "partial"})
            )
            self.assertIsNone(kb.load_cost_stats("SUT")[-1]["cost"])

    def test_missing_cost_is_not_a_cheaper_both_pass_run(self):
        def costs(sut):
            return [
                {
                    "task_id": "fixture-easy-1",
                    "cost": None if sut == "A" else 1,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "num_steps": 1,
                }
            ]

        output = io.StringIO()
        with (
            patch.object(kb, "load_cost_stats", side_effect=costs),
            patch.object(kb, "load_task_success", return_value={"fixture-easy-1": 1}),
            contextlib.redirect_stdout(output),
        ):
            kb.cmd_compare(SimpleNamespace(sut=["A", "B"], top=5))
        self.assertIn("both pass 1", output.getvalue())
        self.assertIn("A cheaper 0", output.getvalue())
        self.assertIn("unknown", output.getvalue())

    def test_cost_command_handles_all_unknown_without_dividing_by_zero(self):
        records = [
            {
                "task_id": "fixture-easy-1",
                "workload": "fixture",
                "difficulty": "easy",
                "cost": None,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "num_steps": 1,
            }
        ]
        output = io.StringIO()
        with (
            patch.object(kb, "load_cost_stats", return_value=records),
            patch.object(kb, "_first_model", return_value="fixture"),
            patch.object(kb, "load_task_success", return_value={}),
            contextlib.redirect_stdout(output),
        ):
            for by in ("task", "workload"):
                kb.cmd_cost(SimpleNamespace(sut=["A"], trim_top=[10], by=by, top=5))
        self.assertIn("unknown", output.getvalue())


if __name__ == "__main__":
    unittest.main()

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

"""DSGym adapter accounting and request construction without model calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from systems.dsgym_runner import REQUEST_FIELDS, limit_output, usage_totals
from systems.dsgym_system import API_KEY_VARIABLE, DSGymSystem


class DSGymAccountingTests(unittest.TestCase):
    def test_responses_and_failures_are_counted_once(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {
                "cached_tokens": 800,
                "cache_creation_tokens": 20,
            },
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
        events = [
            {"phase": "request"},
            {"phase": "transport_request"},
            {"phase": "transport_headers", "status": 429, "seconds": 0.5},
            {"phase": "error", "seconds": 0.6},
            {"phase": "response", "usage": usage, "seconds": 4.0},
            {"phase": "response", "usage": usage, "seconds": 2.0},
        ]
        with TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            totals = usage_totals(trace)
            self.assertEqual(usage_totals(Path(temp) / "missing")["model_calls"], 0)
        self.assertEqual((totals["model_calls"], totals["failed_model_calls"]), (2, 1))
        self.assertEqual(
            (
                totals["input_tokens"],
                totals["output_tokens"],
                totals["cached_tokens"],
                totals["cache_creation_tokens"],
                totals["reasoning_tokens"],
            ),
            (2000, 100, 1600, 40, 60),
        )
        self.assertAlmostEqual(totals["model_seconds"], 6.6)

    def test_output_limit_retains_upstream_truncation_marker(self):
        self.assertEqual(limit_output("x" * 5000, 5000), "x" * 5000)
        self.assertEqual(
            limit_output("x" * 5001, 5000), "x" * 5000 + "\n... (truncated)"
        )


class DSGymRequestTests(unittest.TestCase):
    def test_request_uses_key_environment_and_preserves_explicit_answer(self):
        with TemporaryDirectory() as temp:
            with patch("systems.dsgym_system.os.makedirs"):
                system = DSGymSystem(
                    name="offline", api_key="offline-test-key", dsgym_home=temp
                )
            system.dataset_directory = Path(temp) / "input"
            system.output_dir = str(Path(temp) / "scratch")
            system.format_hints["offline-1"] = "A column name"
            task_dir = Path(system.output_dir) / "offline-1"

            def completed_runner(*args, **kwargs):
                self.assertEqual(kwargs["env"][API_KEY_VARIABLE], "offline-test-key")
                self.assertEqual(set(kwargs["env"]), {"PATH", API_KEY_VARIABLE})
                result = {
                    "answer": "flux_column",
                    "turns": 1,
                    "error": None,
                    "conversation": [],
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cached_tokens": 80,
                    "cache_creation_tokens": 0,
                    "reasoning_tokens": 5,
                    "model_calls": 1,
                    "failed_model_calls": 0,
                    "model_seconds": 0.2,
                }
                (task_dir / "result.json").write_text(json.dumps(result))
                return SimpleNamespace(wait=lambda: None, returncode=0)

            with patch(
                "systems.dsgym_system.subprocess.Popen", side_effect=completed_runner
            ):
                output = system.serve_query("Which column?", "offline-1")
            text = (task_dir / "request.json").read_text()
            self.assertNotIn("offline-test-key", text)
            request = json.loads(text)
            self.assertEqual(request["api_key_env"], API_KEY_VARIABLE)
            self.assertIn("Answer format: A column name", request["task"]["query"])
            self.assertEqual(output["explanation"]["answer"], "flux_column")
            self.assertEqual(
                json.loads((task_dir / "stats.json").read_text())["model_calls"], 1
            )
            self.assertNotIn("api_key", REQUEST_FIELDS)


if __name__ == "__main__":
    unittest.main()

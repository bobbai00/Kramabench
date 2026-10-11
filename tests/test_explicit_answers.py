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

"""Explicit tool answers and CodeAgent's completion protocol stay intact."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from code_agent import CodeAgentResult
from systems.code_agent_system import CodeAgentSystem
from utils.answer_parser import parse_answer, parse_explicit_answer


class ExplicitAnswerTests(unittest.TestCase):
    def test_labels_and_negative_values_survive_explicit_answers(self):
        listed = "['Proton_flux_>30_Mev', -0.193]"
        self.assertEqual(parse_explicit_answer(listed), listed)
        self.assertNotEqual(parse_answer(listed), listed)
        self.assertEqual(parse_explicit_answer(0), "0")
        self.assertEqual(parse_explicit_answer(-0.008003975), "-0.008003975")
        self.assertEqual(parse_explicit_answer("\n 42 \n"), "42")
        self.assertEqual(parse_explicit_answer("**Final Answer: 42**"), "42")
        self.assertEqual(parse_explicit_answer(""), "No response from agent")

    def test_a_repeated_label_is_not_part_of_the_answer(self):
        # Haiku 5.5 (2026-10-10) wrote the label twice on its last line.
        self.assertEqual(parse_answer("so 1.1413.\n\nFinal Answer: **Final Answer: 1.1413**"), "1.1413")
        self.assertEqual(parse_answer("**Final Answer: Final Answer: São Paulo**"), "São Paulo")
        self.assertEqual(parse_answer("Final Answer: **[1, 2]**"), "[1, 2]")
        self.assertEqual(parse_answer("**Final Answer: 42**"), "42")

    def test_successful_tool_answer_uses_the_code_completion_protocol(self):
        with TemporaryDirectory() as temp:
            with patch("systems.code_agent_system.os.makedirs"):
                system = CodeAgentSystem(name="offline", final_answer_tool_prompt=True)
            system.output_dir = str(Path(temp) / "scratch")
            system.dataset_directory = Path(temp) / "input"
            result = CodeAgentResult(
                response="flux_column",
                reasoning_trace=[],
                elapsed_seconds=0,
                state="success",
            )
            system.agent = SimpleNamespace(
                reset=lambda: None, run=lambda prompt: result
            )
            output = system.serve_query("Which column?", "offline-1")
            task = Path(system.output_dir) / "offline-1"
            self.assertEqual(output["explanation"]["answer"], "flux_column")
            prompt = (task / "prompt.txt").read_text()
            self.assertIn("final_answer(<value>)", prompt)
            self.assertNotIn("Your last line MUST BE", prompt)
            self.assertEqual(
                json.loads((task / "stats.json").read_text())["state"], "success"
            )

    def test_step_limit_still_parses_free_text(self):
        with TemporaryDirectory() as temp:
            with patch("systems.code_agent_system.os.makedirs"):
                system = CodeAgentSystem(name="offline", final_answer_tool_prompt=True)
            system.output_dir = str(Path(temp) / "scratch")
            system.dataset_directory = Path(temp) / "input"
            result = CodeAgentResult(
                response="I could not finish.\nFinal Answer: 42",
                reasoning_trace=[],
                elapsed_seconds=0,
                state="max_steps_error",
            )
            system.agent = SimpleNamespace(
                reset=lambda: None, run=lambda prompt: result
            )
            self.assertEqual(
                system.serve_query("q", "offline-1")["explanation"]["answer"], "42"
            )


if __name__ == "__main__":
    unittest.main()

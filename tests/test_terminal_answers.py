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
"""Terminal answers are explicit content, never heuristic text extraction."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from systems.dataflow_system import DataflowSystem
from systems.dataflow_system_parser import DataflowSystemParser
from systems.dataflow_agent_session import DataflowAgentSessionRunner
from utils import answer_parser


class TerminalAnswerTests(unittest.TestCase):
    def test_explicit_answers_preserve_multiline_values_and_identifiers(self):
        for answer in ["42", "['value_code', -0.193]", "first_row\nsecond_row", "| code | value |\n|---|---|\n| a_b | 2 |"]:
            with self.subTest(answer=answer):
                self.assertEqual(answer_parser.parse_explicit_answer(answer), answer)
        self.assertEqual(answer_parser.parse_explicit_answer("**Final Answer: 42**"), "42")
        self.assertEqual(answer_parser.parse_explicit_answer(0), "0")
        self.assertEqual(answer_parser.parse_explicit_answer("Final Answer: value_code"), "value_code")
        self.assertEqual(answer_parser.parse_explicit_answer("**Final Answer:** value_code"), "value_code")
        self.assertEqual(answer_parser.parse_explicit_answer("Final Answer: 42\nKeep this explanation."), "Final Answer: 42\nKeep this explanation.")
        self.assertEqual(answer_parser.parse_explicit_answer("The result is **42**."), "The result is **42**.")
        self.assertEqual(answer_parser.parse_explicit_answer(""), "No response from agent")

    def test_prompts_follow_task_format_without_inventing_a_marker(self):
        with TemporaryDirectory() as directory:
            system = DataflowSystem(output_dir=directory)
            prompt = system._build_prompt("How many?", ["data/example.csv"], "A single integer.")
            self.assertNotIn("Final Answer:", prompt)
            self.assertIn("A single integer.", prompt)
            requested = system._build_prompt("Use Final Answer: followed by the value.", [], "")
            self.assertEqual(requested.count("Final Answer:"), 1)
        session = object.__new__(DataflowAgentSessionRunner)
        session.dataset_directory = "data"
        for first, files in [(True, ["example.csv"]), (False, ["example.csv"]), (False, [])]:
            self.assertNotIn("Final Answer:", session._build_prompt("How many?", files, first))
        self.assertEqual(session._parse_answer("first_row\nsecond_row"), "first_row\nsecond_row")

    def test_wrapper_scores_the_whole_terminal_response(self):
        answer = "first_row\nsecond_row"
        with TemporaryDirectory() as directory:
            system = DataflowSystem(output_dir=directory)
            system.dataset_directory = directory
            system.dataset = {}
            system.agent = Mock(agent_id="test", agent_service_endpoint="http://unused")
            system.agent.run.return_value = SimpleNamespace(response=answer, messages=[], usage={}, stats={}, stopped=False, error=None)
            with patch.object(system, "_setup_agent"), patch("utils.pilot_provenance.inspect_service_listener", return_value={"git_sha": "fixture", "source_dirty": False}), patch("systems.dataflow_system.get_agent_react_steps", return_value={"steps": []}), patch("systems.dataflow_system.get_agent_workflow", return_value={}):
                result = system.serve_query("Return two labels", "fixture")
            self.assertEqual(result["explanation"]["answer"], answer)
            self.assertEqual((Path(directory) / "fixture" / "response.txt").read_text(), answer)
            self.assertNotIn("Final Answer:", (Path(directory) / "fixture" / "prompt.txt").read_text())

    def test_failed_wrapper_does_not_promote_a_tool_result_to_an_answer(self):
        with TemporaryDirectory() as directory:
            system = DataflowSystem(output_dir=directory)
            system.dataset_directory = directory
            system.dataset = {}
            system.agent = Mock(agent_id="test", agent_service_endpoint="http://unused")
            messages = [{"role": "tool", "content": [{"type": "tool-result", "output": {"value": "42"}}]}]
            system.agent.run.return_value = SimpleNamespace(response="", messages=messages, usage={}, stats={}, stopped=False, error="No answer produced")
            with patch.object(system, "_setup_agent"), patch("utils.pilot_provenance.inspect_service_listener", return_value={"git_sha": "fixture", "source_dirty": False}), patch("systems.dataflow_system.get_agent_react_steps", return_value={"steps": []}), patch("systems.dataflow_system.get_agent_workflow", return_value={}):
                result = system.serve_query("Return the total", "fixture")
            self.assertEqual(result["explanation"]["answer"], "No response from agent")

    def test_trace_parser_reads_accepted_structured_answers_only(self):
        answer = "first_row\nsecond_row"
        for status in ["ok", "error"]:
            action = {"type": "answer", "content": answer}
            result = {"kind": "dataflow-actions", "status": "completed" if status == "ok" else "error", "actions": [{"action": action, "status": status, "output": answer}]}
            trace = DataflowSystemParser().parse_data([
                {"role": "assistant", "content": [{"type": "tool-call", "toolName": "dataflow", "toolCallId": "c1", "input": {"actions": [action]}}]},
                {"role": "tool", "content": [{"type": "tool-result", "toolName": "dataflow", "toolCallId": "c1", "output": {"type": "json", "value": result}}]},
            ])
            with self.subTest(status=status):
                self.assertEqual(trace.final_answer, answer if status == "ok" else None)
                self.assertEqual(trace.steps[-1].is_final_answer, status == "ok")
                self.assertEqual(trace.steps[-1].has_error, status != "ok")
        pending = DataflowSystemParser().parse_data([
            {"role": "assistant", "content": [{"type": "tool-call", "toolName": "dataflow", "toolCallId": "c1", "input": {"actions": [{"type": "answer", "content": "42"}]}}]},
        ])
        self.assertIsNone(pending.final_answer)


if __name__ == "__main__":
    unittest.main()

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
import unittest
from tempfile import TemporaryDirectory
from dataflow_agent import MessageResult
from systems.dataflow_system import DataflowSystem, extract_native_answer

class NativeAnswerContractTest(unittest.TestCase):
    def test_native_wrapper_asks_for_answer_action(self):
        with TemporaryDirectory() as output:
            sut = DataflowSystem(agent_mode="native", output_dir=output)
            prompt = sut._build_prompt("Compute a ratio.", ["data/table.csv"], "A number")
            self.assertIn("answer.content", prompt)
            self.assertNotIn("Your last line MUST BE", prompt)
            sut.agent_mode = "code"
            self.assertIn("Your last line MUST BE", sut._build_prompt("Q", [], "A number"))

    def test_failure_text_cannot_be_graded_as_an_answer(self):
        result = MessageResult("No final answer was produced.", [], {}, {}, False, error="missing Answer")
        self.assertEqual(extract_native_answer(result, []), "")

    def test_successful_receipt_preserves_complete_content(self):
        for answer in ("10.87", '[2011, 2012, 2013]', "first line\nsecond line"):
            result = MessageResult("incorrect fallback", [], {}, {}, False, completed=True)
            receipt = {"kind":"dataflow-actions", "status":"completed", "answer":answer,
                       "actions":[{"action":{"type":"answer", "content":answer}, "status":"ok"}]}
            steps = [{"role":"agent", "toolResults":[{"output":receipt}]}]
            self.assertEqual(extract_native_answer(result, steps), answer)

    def test_failed_or_mixed_answer_receipts_are_not_submissions(self):
        result = MessageResult("unsubmitted reply", [], {}, {}, False, completed=False)
        receipt = {"kind":"dataflow-actions", "status":"error", "answer":"wrong", "actions":[]}
        self.assertEqual(extract_native_answer(result, [{"toolResults":[{"output":receipt}]}]), "")

    def test_completed_transport_is_not_proof_of_answer_submission(self):
        result = MessageResult('Error: model unavailable', [], {}, {}, False, completed=True)
        self.assertEqual(extract_native_answer(result, []), '')
        self.assertEqual(extract_native_answer(result, [{"role":"agent", "content":result.response}]), '')

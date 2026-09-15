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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import kb


class NativeTraceMetricsTest(unittest.TestCase):
    def metrics(self, steps, operators=None):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "react_steps.json").write_text(json.dumps({"steps": steps}))
            (root / "workflow.json").write_text(json.dumps({"workflow": {"operators": operators or []}}))
            return kb.react_metrics(root)

    def test_batches_count_requests_and_partial_errors_not_executions(self):
        steps = [{
            "role": "agent", "toolCalls": [{"toolName": "dataflow", "toolCallId": "b1", "input": {
                "operators": [
                    {"op": "scan", "id": "source"},
                    {"op": "derive", "id": "bad", "input": "source"},
                    {"op": "aggregate", "id": "total", "input": "bad"},
                ], "observe": ["total", "total", "missing"],
            }}], "toolResults": [{"toolCallId": "b1", "isError": False, "output": {
                "kind": "native-batch", "operators": [
                    {"id": "source", "op": "scan", "status": "added"},
                    {"id": "bad", "op": "derive", "status": "error", "diagnostic": {"stage": "compile"}},
                    {"id": "total", "op": "aggregate", "status": "blocked"},
                ], "observe": [], "observationErrors": ["missing total", "missing other"],
            }}], "usage": {"inputTokens": 1000, "cachedInputTokens": 800, "outputTokens": 100, "totalTokens": 1100},
        }]
        metrics = self.metrics(steps, [{"operatorType": "DataLoading", "nativeOp": {"type": "Scan"}}])
        self.assertEqual(metrics["native_batch_sizes"], [3])
        self.assertEqual(metrics["native_operator_attempts"], {"scan": 1, "derive": 1, "aggregate": 1})
        self.assertEqual(metrics["native_operator_outcomes"], {"added": 1, "error": 1, "blocked": 1})
        self.assertEqual(metrics["native_error_stages"], {"compile": 1})
        self.assertEqual(metrics["native_observe_requested"], 3)
        self.assertEqual(metrics["native_observe_accepted"], 0)
        self.assertEqual(metrics["native_observation_errors"], 2)
        self.assertEqual(metrics["tool_errors"], 0)  # a partial batch is not a whole-tool error
        self.assertEqual(metrics["edits"], {"source": 1, "bad": 1, "total": 1})
        self.assertEqual(metrics["native_wf_types"], {"Scan": 1})
        self.assertEqual(metrics["cached"], 800)
        self.assertNotIn("execution_pass_rate", metrics)  # not derivable from receipts

    def test_observe_only_and_error_pulls_are_not_operator_edits(self):
        metrics = self.metrics([
            {"role": "agent", "toolCalls": [
                {"toolName": "dataflow", "toolCallId": "b", "input": {"observe": ["answer"]}},
                {"toolName": "inspectError", "input": {"operatorId": "bad"}},
            ], "toolResults": [{"toolCallId": "b", "output": {
                "kind": "native-batch", "operators": [], "observe": ["answer"], "observationErrors": [],
            }}]},
        ])
        self.assertEqual(metrics["native_batch_sizes"], [0])
        self.assertEqual(metrics["native_observe_only_calls"], 1)
        self.assertEqual(metrics["native_error_inspection_calls"], 1)
        self.assertEqual(metrics["native_observe_accepted"], 1)
        self.assertEqual(metrics["edits"], {})

    def test_typed_native_and_legacy_code_edits_both_remain_counted(self):
        metrics = self.metrics([{"role": "agent", "toolCalls": [
            {"toolName": "scan", "input": {"id": "s"}},
            {"toolName": "compute", "input": {"id": "sql"}},
            {"toolName": "fill_null", "input": {"id": "filled"}},
            {"toolName": "createOrModifyOperator", "input": {"operatorId": "python"}},
            {"toolName": "deleteOperator", "input": {"operatorId": "python"}},
        ]}])
        self.assertEqual(metrics["edits"], {"s": 1, "sql": 1, "python": 1, "filled": 1})
        self.assertEqual(metrics["native_operator_attempts"], {"scan": 1, "compute": 1, "fill_null": 1})
        self.assertEqual(metrics["ops_deleted"], {"python"})
        self.assertEqual(metrics["tool_calls"], 5)

    def test_repair_counts_only_a_recorded_rejected_submission_then_acceptance(self):
        steps = []
        for i, status in enumerate(("modified", "error", "blocked", "modified", "modified")):
            steps.append({"role": "agent", "toolCalls": [{
                "toolName": "dataflow", "toolCallId": str(i),
                "input": {"operators": [{"op": "derive", "id": "x"}]},
            }], "toolResults": [{"toolCallId": str(i), "output": {
                "kind": "native-batch", "operators": [{"id": "x", "op": "derive", "status": status}],
                "observe": [], "observationErrors": [],
            }}]})
        metrics = self.metrics(steps)
        self.assertEqual(metrics["native_rejected_edit_repairs"], 1)
        self.assertEqual(metrics["max_edits"], 5)

    def test_missing_receipt_is_unreported_not_successful(self):
        metrics = self.metrics([{"role": "agent", "toolCalls": [{
            "toolName": "dataflow", "input": {"operators": [{"op": "filter", "id": "f"}]},
        }]}])
        self.assertEqual(metrics["native_operator_outcomes"], {"unreported": 1})
        self.assertEqual(metrics["native_rejected_edit_repairs"], 0)

    def test_native_deletion_uses_id_and_distinguishes_refusal(self):
        metrics = self.metrics([{"role": "agent", "toolCalls": [
            {"toolName": "deleteOperator", "toolCallId": "no", "input": {"id": "source"}},
            {"toolName": "deleteOperator", "toolCallId": "yes", "input": {"id": "leaf"}},
        ], "toolResults": [
            {"toolCallId": "no", "output": "[ERROR] `source` still feeds `leaf`"},
            {"toolCallId": "yes", "output": "deleted operator `leaf`"},
        ]}])
        self.assertEqual(metrics["native_delete_attempts"], {"source": 1, "leaf": 1})
        self.assertEqual(metrics["native_delete_outcomes"], {"error": 1, "deleted": 1})
        # Historical generic field counts attempted removals; actual final
        # membership still comes from workflow.json, not this estimate.
        self.assertEqual(metrics["ops_deleted"], {"source", "leaf"})

    def test_malformed_tool_input_is_reported_without_crashing_the_audit(self):
        metrics = self.metrics([{"role": "agent", "toolCalls": [
            {"toolName": "dataflow", "input": "broken JSON"},
            {"toolName": "dataflow", "input": {"operators": "not an array"}},
            {"toolName": "scan", "input": None},
        ]}])
        self.assertEqual(metrics["native_invalid_batch_calls"], 2)
        self.assertEqual(metrics["native_batch_sizes"], [])


if __name__ == "__main__":
    unittest.main()

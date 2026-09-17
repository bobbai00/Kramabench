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

import importlib.util
import json
from pathlib import Path
import unittest

REPORT = Path(__file__).resolve().parents[1] / 'judgment_runs/evidence_compaction_20260916'
spec = importlib.util.spec_from_file_location('compact_report', REPORT / 'analyze.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class CompactEvidenceReportTest(unittest.TestCase):
    def fixture(self):
        return [dict(configuration=key, task_id=f'legal-easy-{index}', round='first',
                     status='completed', dispatched=True, metric='success', score=score,
                     cost_usd=.02, input_tokens=100, cached_tokens=80, output_tokens=10,
                     reasoning_tokens=3)
                for key in report.ARMS for index, score in ((1, 1), (2, .5))]

    def analyze(self, rows):
        return report.summarize(rows, {'legal-easy-1', 'legal-easy-2'})

    def test_native_partial_credit_and_cost_denominator(self):
        arm = self.analyze(self.fixture())['arms'][0]
        self.assertEqual(arm['native_scores'][-1]['score'], 75)
        self.assertEqual(arm['exact_passes'], 1)
        self.assertEqual(arm['mean_cost_per_executed_task_usd'], .02)
        self.assertEqual(arm['known_model_cost_usd'], .04)

    def test_recovery_rows_are_rejected_not_selected(self):
        rows = self.fixture()
        rows[0]['round'] = 'recovery1'
        with self.assertRaisesRegex(ValueError, 'first'):
            self.analyze(rows)

    def test_duplicate_and_missing_tasks_rejected(self):
        rows = self.fixture()
        for invalid in (rows + [dict(rows[0])], rows[1:]):
            with self.assertRaises(ValueError):
                self.analyze(invalid)

    def test_unknown_model_cost_is_not_zero(self):
        rows = self.fixture()
        rows[0]['cost_usd'] = None
        arm = self.analyze(rows)['arms'][0]
        self.assertEqual(arm['model_attempts_with_unknown_cost'], 1)
        self.assertEqual(arm['known_model_cost_usd'], .02)
        self.assertIsNone(arm['mean_cost_per_executed_task_usd'])

    def test_blocked_task_stays_in_score_not_cost_mean(self):
        rows = self.fixture()
        rows[0].update(status='data_readiness_blocked', dispatched=False,
                       score=0, cost_usd=None)
        arm = self.analyze(rows)['arms'][0]
        self.assertEqual(arm['native_scores'][-1]['score'], 25)
        self.assertEqual(arm['tasks'], 2)
        self.assertEqual(arm['model_dispatches'], 1)
        self.assertEqual(arm['mean_cost_per_executed_task_usd'], .02)

    def test_saved_summary_reproduces_from_416_original_rows(self):
        rows = report.load_rows(REPORT / 'first_pass_tasks.csv')
        self.assertEqual(len(rows), 416)
        self.assertEqual(report.summarize(rows), json.loads((REPORT / 'summary.json').read_text()))


if __name__ == '__main__':
    unittest.main()

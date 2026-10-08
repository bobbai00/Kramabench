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

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kb


class ScoreLoaderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.results = self.root / "results" / "Example"
        self.results.mkdir(parents=True)
        self.root_patch = patch.object(kb, "KB_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.directory.cleanup)

    def write(self, timestamp, rows):
        with (self.results / f"legal_measures_{timestamp}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["task_id", "metric", "value"])
            writer.writerows(rows)

    def test_isolated_task_files_preserve_other_task_scores(self):
        self.write("001", [("legal-easy-1", "success", 1)])
        self.write("002", [("legal-easy-2", "success", 0)])
        self.assertEqual(kb.load_task_success("Example"), {"legal-easy-1": 1.0, "legal-easy-2": 0.0})

    def test_latest_task_result_replaces_an_earlier_pass(self):
        self.write("001", [("legal-easy-1", "success", 1), ("legal-easy-2", "success", 1)])
        self.write("002", [("legal-easy-1", "success", 0)])
        self.assertEqual(kb.load_task_success("Example"), {"legal-easy-1": 0.0, "legal-easy-2": 1.0})

    def test_latest_task_metrics_are_combined_without_previous_metrics(self):
        self.write("001", [("legal-easy-1", "success", 1)])
        self.write("002", [("legal-easy-1", "success", 0), ("legal-easy-1", "llm_paraphrase", 0.7)])
        self.assertEqual(kb.load_task_success("Example"), {"legal-easy-1": 0.7})

    def test_nonheadline_rows_do_not_replace_a_score(self):
        self.write("001", [("legal-easy-1", "success", 1)])
        self.write("002", [("legal-easy-1", "elapsed_seconds", 0)])
        self.assertEqual(kb.load_task_success("Example"), {"legal-easy-1": 1.0})


if __name__ == "__main__":
    unittest.main()

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

import copy
import json
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from systems.native_pilot_system import PILOT_TASK, build_pilot_prompt
from utils.pilot_inputs import freeze_pilot_inputs, verify_pilot_inputs


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PilotInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.harness, self.worker = self.root / "harness", self.root / "worker"
        self.harness.mkdir()
        self.worker.mkdir()
        self.dataset = self.harness / "data/environment/input"
        self.dataset.mkdir(parents=True)
        self.names = ["water-body-testing-2012.csv", "water-body-testing-2013.csv"]
        for name in self.names:
            (self.dataset / name).write_text("name,value\nfixture,1\n")
        (self.worker / "data").symlink_to(self.harness / "data", target_is_directory=True)
        self.task = {
            "id": PILOT_TASK,
            "query": "Count changed beaches.",
            "answer": 987654,
            "answer_type": "numeric_exact",
            "data_sources": self.names,
        }
        self.workload = self.harness / "workload.json"
        self.workload.write_text(json.dumps([self.task]))
        paths = ["data/environment/input/" + name for name in reversed(self.names)]
        self.system = SimpleNamespace(
            model_type="gpt-5.6-luna",
            dataset_directory=str(self.dataset),
            workload_data={PILOT_TASK: copy.deepcopy(self.task)},
            format_hints={PILOT_TASK: "Return one integer."},
            _expand_data_sources=lambda names: list(paths),
            _build_prompt=lambda query, files, hint: query + "\n" + json.dumps(files) + "\n" + hint,
        )

    def freeze(self):
        with working_directory(self.harness):
            return freeze_pilot_inputs(self.system, workload_path=self.workload, execution_directory=self.worker)

    def verify(self, manifest):
        with working_directory(self.harness):
            return verify_pilot_inputs(self.system, manifest)

    def test_freeze_binds_actual_harness_worker_bytes_and_prompt_not_gold(self):
        manifest = self.freeze()
        self.assertEqual([Path(item["prompt_path"]).name for item in manifest["files"]], self.names)
        self.assertEqual(len(manifest["files"]), 2)
        self.assertEqual(manifest["files"][0]["harness"]["sha256"], manifest["files"][0]["worker"]["sha256"])
        self.assertNotIn("987654", json.dumps(manifest))
        self.assertNotIn("fixture,1", json.dumps(manifest))
        self.assertTrue(self.verify(manifest)["inputs_verified"])
        self.assertFalse(self.verify(manifest)["runtime_qualified"])

    def test_prompt_order_is_stable_and_does_not_contain_gold(self):
        paths, prompt = build_pilot_prompt(self.system)
        self.assertEqual(paths, sorted(paths))
        self.assertNotIn("987654", prompt)
        self.assertLess(prompt.index("2012.csv"), prompt.index("2013.csv"))

    def test_terra_freezes_its_own_pricing_and_cannot_reuse_luna_manifest(self):
        luna = self.freeze()
        self.system.model_type = "gpt-5.6-terra"
        terra = self.freeze()
        self.assertEqual(terra["model"], "gpt-5.6-terra")
        self.assertEqual(terra["pricing"]["model"], "gpt-5.6-terra")
        self.assertEqual(terra["prompt_sha256"], luna["prompt_sha256"])
        self.assertNotEqual(terra["pricing"]["rates"], luna["pricing"]["rates"])
        with self.assertRaisesRegex(ValueError, "pilot_inputs_changed"):
            self.verify(luna)

    def test_unregistered_model_is_rejected(self):
        self.system.model_type = "gpt-5.6-tara"
        with self.assertRaisesRegex(ValueError, "pilot_model_changed"):
            self.freeze()

    def test_missing_worker_alias_is_not_a_valid_input_manifest(self):
        (self.worker / "data").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.freeze()

    def test_wrong_worker_data_is_rejected_before_any_dispatch(self):
        (self.worker / "data").unlink()
        target = self.worker / "data/environment/input"
        target.mkdir(parents=True)
        for name in self.names:
            (target / name).write_text("different data")
        with self.assertRaisesRegex(ValueError, "worker_data_mismatch"):
            self.freeze()

    def test_changed_file_or_alias_cannot_reuse_a_manifest(self):
        manifest = self.freeze()
        (self.dataset / self.names[0]).write_text("name,value\nfixture,2\n")
        with self.assertRaisesRegex(ValueError, "pilot_inputs_changed"):
            self.verify(manifest)

    def test_changed_prompt_hint_is_detected(self):
        manifest = self.freeze()
        self.system.format_hints[PILOT_TASK] = "Different format"
        with self.assertRaisesRegex(ValueError, "pilot_inputs_changed"):
            self.verify(manifest)

    def test_in_memory_task_must_match_the_executor_workload(self):
        self.system.workload_data[PILOT_TASK]["query"] = "another query"
        with self.assertRaisesRegex(ValueError, "pilot_task_mismatch"):
            self.freeze()

    def test_unknown_or_changed_pricing_is_not_free(self):
        with patch("utils.pilot_inputs.price_schedule", return_value=None):
            with self.assertRaisesRegex(ValueError, "pilot_price_unavailable"):
                self.freeze()
        manifest = self.freeze()
        changed = copy.deepcopy(manifest["pricing"])
        changed["rates"]["input"] *= 2
        with patch("utils.pilot_inputs.price_schedule", return_value=changed):
            with self.assertRaisesRegex(ValueError, "pilot_inputs_changed"):
                self.verify(manifest)

    def test_duplicate_or_different_task_file_policy_is_refused(self):
        self.task["data_sources"] = [self.names[0], self.names[0]]
        self.system.workload_data[PILOT_TASK] = self.task
        self.workload.write_text(json.dumps([self.task]))
        with self.assertRaisesRegex(ValueError, "pilot_source_policy_changed"):
            self.freeze()

    def test_manifest_cannot_be_verified_from_another_working_directory(self):
        manifest = self.freeze()
        with working_directory(self.worker), self.assertRaisesRegex(ValueError, "pilot_working_directory_changed"):
            verify_pilot_inputs(self.system, manifest)


if __name__ == "__main__":
    unittest.main()

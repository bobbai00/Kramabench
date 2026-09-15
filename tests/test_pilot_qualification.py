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
import unittest

from utils.pilot_qualification import check_effective_settings, check_default_python, check_worker_proof


class PilotQualificationTest(unittest.TestCase):
    def test_requested_settings_are_type_strict_and_required(self):
        requested = {"dataLevel": 1, "nativeProfileCollection": True, "enableRecallTool": False}
        check_effective_settings(requested, {"dataLevel": 1, "nativeProfileCollection": True})
        for actual in ({"dataLevel": True, "nativeProfileCollection": True}, {"dataLevel": 1}):
            with self.assertRaises(ValueError):
                check_effective_settings(requested, actual)

    def test_unrequested_additional_evidence_is_rejected(self):
        for extra in ("columnStats", "dataHints", "enableCodeInSnapshot", "enableInspectTool"):
            with self.assertRaises(ValueError):
                check_effective_settings({}, {extra: True})

    def test_named_python_environment_cannot_use_default_worker_proof(self):
        check_default_python({"operators": [{"operatorProperties": {"defaultEnv": True}}]})
        for properties in ({"defaultEnv": False, "envName": "other"}, {"envName": "other"}):
            with self.assertRaises(ValueError):
                check_default_python({"operators": [{"operatorProperties": properties}]})

    def test_worker_proof_requires_all_executed_fixtures_and_bound_runtime(self):
        binding, source = {"pid": 123, "build_bound": True}, "a" * 40
        reports = []
        for name, count in (
            ("python", 2),
            ("core", 2),
            ("evidence", 8),
            ("profile", 4),
            ("shapes", 2),
            ("diagnostics", 2),
        ):
            reports.append(
                {
                    "suite": name,
                    "command": ["bun", "run", f"benchmark/e2e/native-{name}-run.ts"],
                    "exit_code": 0,
                    "engine_before": binding,
                    "engine_after": binding,
                    "service_before": {"git_sha": source, "loaded_revision_verified": True},
                    "service_after": {"git_sha": source, "loaded_revision_verified": True},
                    "summary": {
                        "checks" if name == "python" else "reports": [
                            {"contextMode": "delta" if index % 2 == 0 else "latest"} for index in range(count)
                        ]
                    },
                }
            )
        proof = {"status": "passed", "source": source, "reports": reports}
        check_worker_proof(proof, binding, source)
        for changed in (
            {**proof, "reports": reports[:-1]},
            {**proof, "status": "failed"},
            {**proof, "source": "b" * 40},
        ):
            with self.assertRaises(ValueError):
                check_worker_proof(changed, binding, source)
        changed = copy.deepcopy(proof)
        changed["reports"][0]["engine_after"]["pid"] = 999
        with self.assertRaises(ValueError):
            check_worker_proof(changed, binding, source)

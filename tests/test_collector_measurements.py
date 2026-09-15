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

from copy import deepcopy
import unittest

from utils.collector_measurements import collector_report


def result(execution_id="101", *, version_id="a" * 64, producer_id="b" * 32, collection_ms=2):
    return {
        "resultVersion": {
            "version": 1,
            "id": version_id,
            "workflowId": 456,
            "operatorId": "source",
            "operatorRevision": "c" * 16,
            "definitionHash": "d" * 64,
            "subgraphHash": "e" * 64,
            "outputPort": 0,
            "executionId": execution_id,
            "complete": True,
        },
        "materialization": {"executionId": execution_id, "outputPort": 0, "cached": False},
        "executionContext": {"workflowId": 456, "executionId": execution_id},
        "tableProfile": {
            "version": 1,
            "resultVersion": version_id,
            "status": "available",
            "producerIds": [producer_id],
            "coverage": {
                "scope": "full",
                "complete": True,
                "expectedWorkers": 1,
                "receivedWorkers": 1,
                "rowsProcessed": 2,
            },
            "cost": {"collectionMs": collection_ms, "cellsProcessed": 4, "payloadBytes": 512},
            "columns": {"secret-column": {"frequencies": "private-values"}},
        },
        "sampleRecords": [{"secret-column": "private-row"}],
    }


def snapshots(*results):
    return {"snapshots": [{"stepId": f"s{index}", "results": {"source": value}} for index, value in enumerate(results)]}


class CollectorMeasurementsTest(unittest.TestCase):
    def report(self, bundle, **overrides):
        return collector_report(
            bundle,
            **{
                "workflow_id": 456,
                "execution_ids": ["101", "102"],
                "collection_requested": True,
                "snapshots_complete": True,
                **overrides,
            },
        )

    def test_cached_snapshots_count_the_producing_work_once(self):
        first = result()
        cached = deepcopy(first)
        cached["materialization"]["cached"] = True
        cached["executionContext"]["executionId"] = "999"  # retrieval is not production
        report = self.report(snapshots(first, cached, cached))
        self.assertEqual(report["status"], "observed")
        self.assertEqual(report["profile_occurrences"], 3)
        self.assertEqual(report["unique_productions"], 1)
        self.assertEqual(report["duplicates_ignored"], 2)
        self.assertEqual(report["observed_collection_ms"], 2)
        self.assertEqual(report["observed_payload_bytes"], 512)
        self.assertNotIn("private-values", str(report))
        self.assertNotIn("private-row", str(report))
        self.assertFalse(report["whole_attempt_overhead_complete"])

    def test_a_real_rerun_of_the_same_operator_is_not_a_cache_hit(self):
        report = self.report(
            snapshots(result(), result("102", version_id="f" * 64, producer_id="1" * 32, collection_ms=3))
        )
        self.assertEqual(report["unique_productions"], 2)
        self.assertEqual(report["observed_collection_ms"], 5)
        self.assertEqual(report["observed_cells_processed"], 8)

    def test_missing_or_stale_binding_is_not_measured_as_zero(self):
        for field, value in (("complete", False), ("workflowId", 999), ("operatorId", "other")):
            entry = result()
            entry["resultVersion"][field] = value
            report = self.report(snapshots(entry))
            self.assertIsNone(report["observed_collection_ms"])
            self.assertEqual(report["unbound_profile_occurrences"], 1)
        entry = result()
        entry["tableProfile"]["resultVersion"] = "0" * 64
        self.assertEqual(self.report(snapshots(entry))["unbound_profile_occurrences"], 1)

    def test_producing_execution_must_belong_to_the_attempt(self):
        report = self.report(snapshots(result("900")))
        self.assertEqual(report["unattributed_productions"], 1)
        self.assertIsNone(report["observed_collection_ms"])
        unknown = self.report(snapshots(result()), execution_ids=None)
        self.assertFalse(unknown["execution_scope_verified"])
        self.assertIsNone(unknown["observed_collection_ms"])

    def test_partial_worker_coverage_keeps_only_observed_cost(self):
        entry = result()
        entry["tableProfile"].update(status="partial")
        entry["tableProfile"]["coverage"].update(scope="partial", complete=False, expectedWorkers=2)
        report = self.report(snapshots(entry))
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["partial_productions"], 1)
        self.assertEqual(report["observed_collection_ms"], 2)
        self.assertFalse(report["whole_attempt_overhead_complete"])

    def test_conflicting_costs_are_not_added_or_arbitrarily_maximized(self):
        first = result()
        changed = deepcopy(first)
        changed["tableProfile"]["cost"]["collectionMs"] = 20
        report = self.report(snapshots(first, changed))
        self.assertEqual(report["conflicting_productions"], 1)
        self.assertIsNone(report["observed_collection_ms"])

    def test_worker_producer_reuse_under_a_different_result_is_not_double_counted(self):
        first = result()
        second = result("102", version_id="f" * 64)  # illegal reuse of the first producer
        report = self.report(snapshots(first, second))
        self.assertEqual(report["conflicting_productions"], 2)
        self.assertIsNone(report["observed_collection_ms"])

    def test_absent_profiles_and_collection_disabled_are_distinct(self):
        empty = {"snapshots": [{"stepId": "s1", "results": {}}]}
        self.assertEqual(self.report(empty)["status"], "unavailable")
        self.assertIsNone(self.report(empty)["observed_collection_ms"])
        disabled = self.report(empty, collection_requested=False)
        self.assertEqual(disabled["status"], "disabled_by_request")
        self.assertEqual(disabled["observed_collection_ms"], 0)
        wrong = self.report(snapshots(result()), collection_requested=False)
        self.assertIn("profile_present_when_disabled", wrong["issues"])

    def test_unavailable_profiles_and_bad_costs_remain_unknown(self):
        entry = result()
        entry["tableProfile"].update(status="unavailable", producerIds=[])
        report = self.report(snapshots(entry))
        self.assertEqual(report["unavailable_productions"], 1)
        self.assertIsNone(report["observed_collection_ms"])
        for cost in (-1, True, float("inf"), "2"):
            entry = result(collection_ms=cost)
            self.assertIsNone(self.report(snapshots(entry))["observed_collection_ms"])

    def test_capture_completeness_and_malformed_snapshots_are_explicit(self):
        report = self.report(snapshots(result()), snapshots_complete=False)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["observed_collection_ms"], 2)
        malformed = self.report({"snapshots": [None, {"results": []}]})
        self.assertIn("invalid_snapshot", malformed["issues"])
        self.assertIsNone(malformed["observed_collection_ms"])


if __name__ == "__main__":
    unittest.main()

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

from utils.execution_journal import execution_report


HEADER = {
    "version": 1,
    "kind": "recorder_started",
    "instanceId": "fixture-recorder",
    "computingUnitIds": [321, 999],
    "pending": 0,
    "completed": 0,
}


def request(
    identifier,
    *,
    compilation="passed",
    runtime=True,
    state="Completed",
    success=True,
    cuid=321,
):
    base = {
        "version": 1,
        "requestId": identifier,
        "workflowId": 456,
        "computingUnitId": cuid,
        "startedAt": "2026-09-15T00:00:00Z",
        "operatorCount": 2,
        "targetOperatorIds": ["answer"],
        "structuredResults": True,
        "requestSummaryTruncated": False,
    }
    start = {
        "version": 1,
        "instanceId": HEADER["instanceId"],
        "kind": "request_started",
        "record": base,
    }
    finish = {
        **start,
        "kind": "request_finished",
        "record": {
            **base,
            "elapsedMs": 10,
            "httpStatus": 200,
            "transport": "response",
            "executionId": str(100 + len(identifier)),
            "state": state,
            "success": success,
            "phaseEvidence": "recorded",
            "phaseMetrics": {
                "version": 1,
                "compilation": compilation,
                "compilationDurationMs": 2 if compilation in {"passed", "failed"} else None,
                "runtimeStartAttempted": runtime,
            },
        },
    }
    return start, finish


def footer(completed):
    return {
        **HEADER,
        "kind": "recorder_closed",
        "pending": 0,
        "completed": completed,
        "persistenceFailed": False,
    }


class ExecutionJournalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "execution_requests.jsonl"

    def report(self, events, *, tail=""):
        self.path.write_text("".join(json.dumps(event) + "\n" for event in events) + tail)
        return execution_report(
            self.path,
            workflow_id=456,
            computing_unit_id=321,
            expected_recorder_id=HEADER["instanceId"],
        )

    def test_measured_pass_rates_have_attempted_and_unknown_denominators(self):
        events = [
            HEADER,
            *request("ok"),
            *request(
                "compile-error",
                compilation="failed",
                runtime=False,
                state="Failed",
                success=False,
            ),
            *request(
                "not-run",
                compilation="not_attempted",
                runtime=False,
                state="Error",
                success=False,
            ),
            *request("runtime-error", state="Failed", success=False),
            footer(4),
        ]
        result = self.report(events)
        self.assertEqual(result["journal_status"], "closed")
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["compilation"]["passed"], 2)
        self.assertEqual(result["compilation"]["failed"], 1)
        self.assertEqual(result["compilation"]["not_attempted"], 1)
        self.assertEqual(result["compilation"]["known_attempted"], 3)
        self.assertEqual(result["compilation"]["pass_rate"], 2 / 3)
        self.assertEqual(result["runtime"]["pass_rate"], 0.5)
        self.assertEqual(result["compilation_duration_ms_observed"], 6)
        self.assertEqual(result["request_duration_ms_observed"], 40)

    def test_an_unfinished_start_survives_an_aborted_recorder(self):
        result = self.report([HEADER, *request("ok"), request("inflight")[0]])
        self.assertEqual(result["journal_status"], "open")
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["unfinished_requests"], 1)
        self.assertEqual(result["compilation"]["unknown"], 1)
        self.assertEqual(result["runtime"]["unknown"], 1)
        self.assertEqual(result["compilation"]["known_coverage"], 0.5)
        self.assertFalse(result["backend_termination_verified"])

    def test_footer_cannot_change_the_recorder_route_or_scope(self):
        for field, value in (("backend", "http://127.0.0.1:8086"), ("computingUnitIds", [321]), ("maxRequests", 1)):
            with self.subTest(field=field):
                result = self.report([HEADER, *request("ok"), {**footer(1), field: value}])
                self.assertEqual(result["journal_status"], "invalid")
                self.assertIn("inconsistent_footer", result["issues"])

    def test_scope_and_identical_replays_do_not_double_count(self):
        start, finish = request("ok")
        result = self.report(
            [
                HEADER,
                start,
                finish,
                start,
                finish,
                *request("other", cuid=999),
                footer(2),
            ]
        )
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["other_requests"], 1)
        self.assertEqual(result["duplicate_events"], 2)
        self.assertEqual(result["journal_status"], "closed")
        self.assertEqual(result["request_duration_ms_observed"], 10)

    def test_missing_and_empty_closed_journals_are_not_the_same(self):
        missing = execution_report(
            self.path,
            workflow_id=456,
            computing_unit_id=321,
            expected_recorder_id=HEADER["instanceId"],
        )
        self.assertEqual(missing["journal_status"], "missing")
        self.assertIsNone(missing["requests"])
        self.assertIsNone(missing["compilation"]["pass_rate"])
        empty = self.report([HEADER, footer(0)])
        self.assertEqual(empty["requests"], 0)
        self.assertIsNone(empty["runtime"]["pass_rate"])
        self.assertIsNone(empty["runtime"]["known_coverage"])

    def test_truncated_tail_or_orphan_finish_never_looks_complete(self):
        result = self.report([HEADER, request("inflight")[0]], tail='{"kind": "request_finished"')
        self.assertEqual(result["journal_status"], "invalid")
        self.assertEqual(result["requests"], 1)
        self.assertIn("malformed_event", result["issues"])
        orphan = self.report([HEADER, request("orphan")[1], footer(1)])
        self.assertEqual(orphan["runtime"]["passed"], 0)
        self.assertEqual(orphan["runtime"]["unknown"], 1)
        self.assertIn("finish_without_start", orphan["issues"])

    def test_conflicting_records_and_recorder_identity_are_rejected(self):
        start, finish = request("ok")
        changed = json.loads(json.dumps(finish))
        changed["record"]["success"] = False
        result = self.report([HEADER, start, finish, changed, footer(1)])
        self.assertEqual(result["runtime"]["unknown"], 1)
        self.assertIn("conflicting_event", result["issues"])
        wrong = self.report([{**HEADER, "instanceId": "not-our-recorder"}, start, finish])
        self.assertEqual(wrong["journal_status"], "invalid")
        self.assertIsNone(wrong["requests"])

    def test_missing_or_impossible_phases_do_not_become_successes(self):
        for phase in (
            None,
            {
                "version": 1,
                "compilation": "passed",
                "compilationDurationMs": -1,
                "runtimeStartAttempted": True,
            },
            {
                "version": 1,
                "compilation": "failed",
                "compilationDurationMs": 2,
                "runtimeStartAttempted": True,
            },
        ):
            start, finish = request("ok")
            finish["record"].update(phaseMetrics=phase, compilation="passed", execution="passed")
            result = self.report([HEADER, start, finish, footer(1)])
            self.assertEqual(result["compilation"]["unknown"], 1)
            self.assertEqual(result["runtime"]["unknown"], 1)

    def test_footer_is_not_proof_of_backend_termination(self):
        start, finish = request("aborted")
        finish["record"].update(transport="aborted", phaseMetrics=None, httpStatus=None)
        result = self.report([HEADER, start, finish, footer(1)])
        self.assertEqual(result["journal_status"], "closed")
        self.assertEqual(result["runtime"]["unknown"], 1)
        self.assertFalse(result["backend_termination_verified"])

    def test_malformed_field_types_remain_unknown_instead_of_crashing(self):
        start, finish = request("bad-state")
        finish["record"]["state"] = ["Completed"]
        result = self.report([HEADER, start, finish, footer(1)])
        self.assertEqual(result["runtime"]["unknown"], 1)
        invalid = self.report([HEADER, {"version": 1, "instanceId": HEADER["instanceId"], "kind": []}])
        self.assertEqual(invalid["journal_status"], "invalid")

    def test_changed_route_does_not_hide_a_conflicting_request(self):
        start, finish = request("route-change", cuid=999)
        finish["record"]["computingUnitId"] = 321
        result = self.report([HEADER, start, finish, footer(1)])
        self.assertEqual(result["journal_status"], "invalid")
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["runtime"]["unknown"], 1)


if __name__ == "__main__":
    unittest.main()

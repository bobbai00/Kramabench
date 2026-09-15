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

"""Observed collector work, deduplicated by production rather than retrieval.

collectionMs sums worker ProfileAccumulator.add() timers. It excludes profile
snapshot/encoding, transport and agent context rendering; it is neither total
execution latency nor collector CPU time. Failed/unreturned materializations
can be absent even from a complete snapshot capture. Never label these sums
whole-attempt overhead or manufacture zero cost for unavailable profiles.
"""

import math
import re


def _count(value, *, positive=False):
    return type(value) is int and (1 if positive else 0) <= value <= 9007199254740991


def _number(value):
    try:
        return type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _hex(value, length):
    return isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{" + str(length) + "}", value))


def _execution_id(value):
    return isinstance(value, str) and bool(re.fullmatch(r"\d{1,20}", value))


def _binding(operator_id, info, workflow_id):
    version, materialization, profile = info.get("resultVersion"), info.get("materialization"), info.get("tableProfile")
    if not all(isinstance(value, dict) for value in (version, materialization, profile)):
        return None
    if (
        not _count(workflow_id, positive=True)
        or type(version.get("version")) is not int
        or version["version"] != 1
        or not _count(version.get("workflowId"), positive=True)
        or version["workflowId"] != workflow_id
        or version.get("operatorId") != operator_id
        or not isinstance(operator_id, str)
        or not 0 < len(operator_id) <= 240
        or version.get("complete") is not True
        or not _hex(version.get("id"), 64)
        or not _hex(version.get("operatorRevision"), 16)
        or not _hex(version.get("definitionHash"), 64)
        or not _hex(version.get("subgraphHash"), 64)
        or not _count(version.get("outputPort"))
        or not _execution_id(version.get("executionId"))
        or materialization.get("executionId") != version["executionId"]
        or not _count(materialization.get("outputPort"))
        or materialization["outputPort"] != version["outputPort"]
        or type(materialization.get("cached")) is not bool
        or type(profile.get("version")) is not int
        or profile["version"] != 1
        or profile.get("resultVersion") != version["id"]
    ):
        return None
    return {
        "workflow_id": workflow_id,
        "operator_id": operator_id,
        "output_port": version["outputPort"],
        "execution_id": version["executionId"],
        "result_version": version["id"],
        "operator_revision": version["operatorRevision"],
        "definition_hash": version["definitionHash"],
        "subgraph_hash": version["subgraphHash"],
    }


def _measurement(profile):
    if profile.get("status") == "unavailable":
        return {"profile_status": "unavailable"}
    producers, cost, coverage = profile.get("producerIds"), profile.get("cost"), profile.get("coverage")
    if (
        profile.get("status") not in ("available", "partial")
        or not isinstance(producers, list)
        or not producers
        or any(not _hex(value, 32) for value in producers)
        or len(set(producers)) != len(producers)
        or not isinstance(cost, dict)
        or not isinstance(coverage, dict)
        or not _number(cost.get("collectionMs"))
        or not _count(cost.get("cellsProcessed"))
        or not _count(cost.get("payloadBytes"))
        or not _count(coverage.get("expectedWorkers"), positive=True)
        or not _count(coverage.get("receivedWorkers"), positive=True)
        or coverage["receivedWorkers"] != len(producers)
        or coverage["receivedWorkers"] > coverage["expectedWorkers"]
        or not _count(coverage.get("rowsProcessed"))
        or type(coverage.get("complete")) is not bool
        or coverage.get("scope") not in ("full", "partial")
    ):
        return None
    complete = coverage["complete"]
    if complete != (coverage["scope"] == "full") or complete != (profile["status"] == "available"):
        return None
    if complete and coverage["receivedWorkers"] != coverage["expectedWorkers"]:
        return None
    return {
        "profile_status": profile["status"],
        "producer_ids": sorted(producers),
        "coverage": {
            key: coverage[key] for key in ("scope", "complete", "expectedWorkers", "receivedWorkers", "rowsProcessed")
        },
        "collection_ms": cost["collectionMs"],
        "cells_processed": cost["cellsProcessed"],
        "payload_bytes": cost["payloadBytes"],
    }


def collector_report(snapshot_bundle, *, workflow_id, execution_ids, collection_requested, snapshots_complete):
    """Read decoded snapshot profiles; never read sample rows or column values.

    execution_ids must come from the caller's qualified, workflow/CU-scoped
    execution journal. None means unattributed, not an empty known set. The
    collection setting is explicitly the request; the outer guard must verify
    effective settings too. Observed component totals remain lower bounds.
    """
    scope_known = isinstance(execution_ids, (list, tuple, set, frozenset)) and all(
        _execution_id(value) for value in execution_ids
    )
    scope = set(execution_ids) if scope_known else set()
    report = {
        "version": 1,
        "status": "unavailable",
        "collection_requested": collection_requested,
        "snapshots_complete": snapshots_complete is True,
        "execution_scope_verified": scope_known,
        "whole_attempt_overhead_complete": False,
        "totals_scope": "observed_bound_productions_only",
        "profile_occurrences": 0,
        "unique_productions": 0,
        "duplicates_ignored": 0,
        "unbound_profile_occurrences": 0,
        "unavailable_productions": 0,
        "partial_productions": 0,
        "unattributed_productions": 0,
        "conflicting_productions": 0,
        "observed_collection_ms": None,
        "observed_cells_processed": None,
        "observed_payload_bytes": None,
        "productions": [],
        "issues": [],
    }
    issues, conflicts, entries, producer_owners = set(), set(), {}, {}
    if type(collection_requested) is not bool and collection_requested is not None:
        issues.add("invalid_collection_setting")
        report["collection_requested"] = None
    snapshots = snapshot_bundle.get("snapshots") if isinstance(snapshot_bundle, dict) else None
    if not isinstance(snapshots, list):
        return {**report, "issues": ["missing_snapshot_capture"]}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("results"), dict):
            issues.add("invalid_snapshot")
            continue
        for operator_id, info in snapshot["results"].items():
            if not isinstance(info, dict):
                issues.add("invalid_operator_result")
                continue
            if "tableProfile" not in info or info["tableProfile"] is None:
                continue
            report["profile_occurrences"] += 1
            binding = _binding(operator_id, info, workflow_id)
            if binding is None:
                report["unbound_profile_occurrences"] += 1
                issues.add("unbound_profile")
                continue
            key = tuple(binding[field] for field in ("workflow_id", "operator_id", "output_port", "execution_id"))
            measurement = _measurement(info["tableProfile"])
            if measurement is None:
                conflicts.add(key)
                issues.add("invalid_profile_measurement")
                continue
            entry = {**binding, **measurement}
            if key in entries:
                if entries[key] == entry:
                    report["duplicates_ignored"] += 1
                else:
                    conflicts.add(key)
                    issues.add("conflicting_production_measurement")
                continue
            entries[key] = entry
            for producer in entry.get("producer_ids", []):
                if producer in producer_owners and producer_owners[producer] != key:
                    conflicts.update((key, producer_owners[producer]))
                    issues.add("producer_identity_reused")
                producer_owners[producer] = key
    report["conflicting_productions"] = len(conflicts)
    for key, entry in entries.items():
        if key in conflicts:
            continue
        if entry["profile_status"] == "unavailable":
            report["unavailable_productions"] += 1
            issues.add("profile_unavailable")
            continue
        if not scope_known or entry["execution_id"] not in scope:
            report["unattributed_productions"] += 1
            issues.add("production_not_bound_to_attempt")
            continue
        report["productions"].append(entry)
        report["partial_productions"] += entry["profile_status"] == "partial"
    report["productions"].sort(key=lambda entry: (entry["execution_id"], entry["operator_id"], entry["output_port"]))
    report["unique_productions"] = len(report["productions"])
    if report["productions"]:
        for field in ("collection_ms", "cells_processed", "payload_bytes"):
            total = sum(entry[field] for entry in report["productions"])
            if _number(total) and (field == "collection_ms" or _count(total)):
                report["observed_" + field] = total
            else:
                issues.add("measurement_sum_overflow")
        report["status"] = (
            "partial" if issues or report["partial_productions"] or snapshots_complete is not True else "observed"
        )
    if collection_requested is False:
        if report["profile_occurrences"]:
            issues.add("profile_present_when_disabled")
            report["status"] = "invalid"
        elif not issues and snapshots_complete is True:
            report.update(
                status="disabled_by_request",
                observed_collection_ms=0,
                observed_cells_processed=0,
                observed_payload_bytes=0,
            )
    report["issues"] = sorted(issues)
    return report

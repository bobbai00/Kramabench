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

"""Native action counters, not execution/compilation outcome inference.

Batch receipts record accepted/rejected edits. They precede engine execution,
and observing may reuse a cached result without an RPC. Runtime repair and
compile/execution pass rates therefore need separate engine evidence.
"""

from collections import Counter


NATIVE_TOOLS = frozenset((
    "scan", "project", "compute", "cast", "fill_null", "transform", "derive", "filter",
    "aggregate", "join", "union", "distinct", "sort", "limit", "window", "split",
    "explode", "unpivot", "pivot", "udf",
))


def native_trace_metrics(steps, workflow):
    metrics = {
        "native_batch_sizes": [],
        "native_invalid_batch_calls": 0,
        "native_operator_attempts": Counter(),
        "native_operator_outcomes": Counter(),
        "native_error_stages": Counter(),
        "native_observe_requested": 0,
        "native_observe_accepted": 0,
        "native_observation_errors": 0,
        "native_observe_only_calls": 0,
        "native_error_inspection_calls": 0,
        "native_rejected_edit_repairs": 0,
        "native_delete_attempts": Counter(),
        "native_delete_outcomes": Counter(),
        "native_edits": Counter(),
        "native_ops_touched": set(),
        "native_wf_types": Counter(),
    }
    rejected = set()

    def touch(spec):
        if not isinstance(spec, dict):
            return
        opid = spec.get("id")
        kind = spec.get("op")
        if isinstance(kind, str):
            metrics["native_operator_attempts"][kind] += 1
        if isinstance(opid, str) and opid:
            metrics["native_ops_touched"].add(opid)
            metrics["native_edits"][opid] += 1

    for step in steps:
        if not isinstance(step, dict):
            continue
        results = {r.get("toolCallId"): r.get("output") for r in (step.get("toolResults") or [])
                   if isinstance(r, dict) and isinstance(r.get("toolCallId"), str)}
        for call in step.get("toolCalls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("toolName")
            raw = call.get("input")
            args = raw if isinstance(raw, dict) else {}
            if name in NATIVE_TOOLS:
                touch({**args, "op": name})
            elif name == "inspectError":
                metrics["native_error_inspection_calls"] += 1
            elif name == "deleteOperator" and isinstance(args.get("id"), str):
                opid = args["id"]
                metrics["native_delete_attempts"][opid] += 1
                output = results.get(call.get("toolCallId"))
                status = "unreported"
                if isinstance(output, str) and output.startswith("[ERROR]"):
                    status = "error"
                elif output == f"deleted operator `{opid}`":
                    status = "deleted"
                    rejected.discard(opid)
                metrics["native_delete_outcomes"][status] += 1
            elif name == "dataflow":
                specs = args.get("operators", [])
                observe = args.get("observe", [])
                if not isinstance(raw, dict) or not isinstance(specs, list) or not isinstance(observe, list):
                    metrics["native_invalid_batch_calls"] += 1
                    continue
                metrics["native_batch_sizes"].append(len(specs))
                metrics["native_observe_requested"] += len(observe)  # includes repeated/unavailable requests
                metrics["native_observe_only_calls"] += int(not specs and bool(observe))
                for spec in specs:
                    touch(spec)
                result = results.get(call.get("toolCallId"))
                if not isinstance(result, dict) or result.get("kind") != "native-batch":
                    metrics["native_operator_outcomes"]["unreported"] += len(specs)
                    continue
                accepted = result.get("observe")
                errors = result.get("observationErrors")
                metrics["native_observe_accepted"] += len(accepted) if isinstance(accepted, list) else 0
                metrics["native_observation_errors"] += len(errors) if isinstance(errors, list) else 0
                outcomes = result.get("operators")
                outcomes = outcomes if isinstance(outcomes, list) else []
                # Receipts retain input order. Mismatched/missing receipt slots
                # are unreported, never accepted based on another operator's ID.
                for i, spec in enumerate(specs):
                    receipt = outcomes[i] if i < len(outcomes) else None
                    valid = (isinstance(spec, dict) and isinstance(receipt, dict)
                             and receipt.get("id") == spec.get("id")
                             and receipt.get("op") == spec.get("op")
                             and receipt.get("status") in {"added", "modified", "error", "blocked"})
                    if not valid:
                        metrics["native_operator_outcomes"]["unreported"] += 1
                        continue
                    status = receipt["status"]
                    metrics["native_operator_outcomes"][status] += 1
                    opid = receipt.get("id")
                    diagnostic = receipt.get("diagnostic")
                    if status == "error" and isinstance(diagnostic, dict):
                        stage = diagnostic.get("stage")
                        if isinstance(stage, str):
                            metrics["native_error_stages"][stage] += 1
                    if not isinstance(opid, str):
                        continue
                    if status == "error":
                        rejected.add(opid)
                    elif status in {"added", "modified"} and opid in rejected:
                        metrics["native_rejected_edit_repairs"] += 1
                        rejected.remove(opid)
    for operator in workflow.get("operators", []):
        native = operator.get("nativeOp") or {}
        if isinstance(native, dict) and isinstance(native.get("type"), str):
            metrics["native_wf_types"][native["type"]] += 1
    return metrics

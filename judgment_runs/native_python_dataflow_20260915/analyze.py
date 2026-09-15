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

"""Local, read-only analysis of frozen attempts; never calls a model or scorer.

Run from the harness root. Only the two generated report files in this folder
are written. Raw attempts, verdicts, traces and first-attempt namespaces are
not changed. Missing attempts remain pending, not failures or zero-cost wins.
"""

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))

import kb
import pandas as pd
from systems.native_python_system import ALL_PILOT_ARMS
from utils.native_model_smoke import _input, _normalized


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def evidence_audit(steps, snapshots):
    frames = []
    for index, step in enumerate(steps):
        observed = {
            op
            for call in step.get("toolCalls", [])
            if call.get("toolName") == "dataflow"
            for op in call.get("input", {}).get("observe", [])
        }
        for op in sorted(observed):
            frame = snapshots.get(step["id"], {}).get("nativeObservations", {}).get(op)
            if not frame:
                continue
            witness = next(
                (
                    s
                    for s in steps[index + 1 :]
                    if _normalized(frame["text"]) in _normalized(_input(s))
                ),
                None,
            )
            frames.append(
                {
                    "operator": op,
                    "at_step": index + 1,
                    "step_id": step["id"],
                    "result_version": frame.get("resultVersion"),
                    "first_seen_step_id": witness["id"] if witness else None,
                    "selected": frame.get("selected", []),
                    "omitted": frame.get("omitted", []),
                    "chars": len(frame["text"]),
                    "facts": [
                        c
                        for c in frame.get("chunks", [])
                        if c["id"].startswith(("data.", "flow."))
                    ],
                }
            )
    return frames


def analyze_attempt(spec):
    directory = HARNESS / "system_scratch" / spec.system_name / "environment-easy-3"
    verdict, stats = read(directory / "verdict.json"), read(directory / "stats.json")
    metrics, attempt = (
        read(directory / "pilot_metrics.json"),
        read(directory / "attempt.json"),
    )
    steps = [
        s
        for s in read(directory / "react_steps.json").get("steps", [])
        if s.get("role") == "agent"
    ]
    snapshots = {
        s["stepId"]: s for s in read(directory / "snapshots.json").get("snapshots", [])
    }
    collector = metrics.get("collector_measurements", {})
    rates = stats.get("pricing", {}).get("rates", {})
    cold = None
    if stats.get("cost_status") == "complete":
        cold = (
            stats["input_tokens"] * rates["input"]
            + stats["output_tokens"] * rates["output"]
        )
    frames = evidence_audit(steps, snapshots)
    timeline = []
    for index, step in enumerate(steps):
        calls = []
        for call in step.get("toolCalls", []):
            args = call.get("input", {})
            specs = (
                args.get("operators", [])
                if call.get("toolName") == "dataflow"
                else [args]
            )
            calls.append(
                {
                    "tool": call.get("toolName"),
                    "observe": args.get("observe"),
                    "operators": [
                        {
                            key: op[key]
                            for key in (
                                "op",
                                "id",
                                "input",
                                "inputs",
                                "groupBy",
                                "on",
                                "preview",
                            )
                            if key in op
                        }
                        for op in specs
                        if isinstance(op, dict)
                    ],
                }
            )
        errors = [
            op.get("message")
            for result in step.get("toolResults", [])
            if isinstance(result.get("output"), dict)
            for op in result["output"].get("operators", [])
            if op.get("status") in ("error", "blocked")
        ]
        timeline.append(
            {"step": index + 1, "id": step["id"], "calls": calls, "edit_errors": errors}
        )
    return {
        "model": spec.model_type,
        "arm": spec.key,
        "sut": spec.system_name,
        "artifact_directory": str(directory.relative_to(HARNESS)),
        "artifact_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "answer.json",
                "attempt.json",
                "verdict.json",
                "stats.json",
                "pilot_metrics.json",
                "react_steps.json",
                "snapshots.json",
            )
            if (directory / name).exists()
        },
        "first_input_messages_sha256": hashlib.sha256(
            json.dumps(
                steps[0].get("inputMessages", []),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if steps
        else None,
        "answer": read(directory / "answer.json").get("answer"),
        "verdict": verdict,
        "attempt_status": attempt.get("status", "pending"),
        "stats": stats,
        "uncached_input_counterfactual_usd": cold,
        "execution": metrics.get("execution_measurements", {}),
        "collector": {
            key: value for key, value in collector.items() if key != "productions"
        },
        "trace_metrics": metrics.get("trace", {}),
        "step_tokens": kb.step_token_rows(directory),
        "evidence": frames,
        "seen_data_frames": sum(
            bool(f["first_seen_step_id"])
            and any(x.startswith("data.") for x in f["selected"])
            for f in frames
        ),
        "seen_flow_frames": sum(
            bool(f["first_seen_step_id"])
            and any(x.startswith("flow.") for x in f["selected"])
            for f in frames
        ),
        "timeline": timeline,
    }


def semantic_audit():
    frames = {
        year: pd.read_csv(
            HARNESS / f"data/environment/input/water-body-testing-{year}.csv", dtype=str
        )
        for year in (2012, 2013)
    }
    all_rows = pd.concat(frames.values(), ignore_index=True)
    locations = all_rows[
        ["Beach Name", "Community Code", "County Code"]
    ].drop_duplicates()
    gold_key = (
        all_rows["Community Code"]
        + all_rows["County Code"]
        + all_rows["Beach Name"].str.lower()
    )
    cases = []
    for keys in (["Beach Name"], ["Community Code", "Beach Name"]):
        grouped = []
        for data in frames.values():
            data = data.assign(
                v=data["Violation"].str.strip().str.lower().eq("yes").astype(int)
            )
            grouped.append(
                data.groupby(keys).agg(samples=("v", "size"), exceedances=("v", "sum"))
            )
        pair = grouped[0].join(
            grouped[1], how="inner", lsuffix="_2012", rsuffix="_2013"
        )
        count = int(
            (
                pair.exceedances_2013 / pair.samples_2013
                > pair.exceedances_2012 / pair.samples_2012
            ).sum()
        )
        cases.append({"group_and_join_keys": keys, "higher_rate_count": count})
    return {
        "scope": "offline explanation only; never provided to the evaluated model",
        "rows": {str(year): len(frame) for year, frame in frames.items()},
        "beach_names_in_multiple_locations": int(
            (locations.groupby("Beach Name").size() > 1).sum()
        ),
        "community_to_county_violations": int(
            (all_rows.groupby("Community Code")["County Code"].nunique() > 1).sum()
        ),
        "case_variant_groups_across_years": int(
            (
                all_rows.assign(lower=all_rows["Beach Name"].str.lower())
                .groupby(["Community Code", "lower"])["Beach Name"]
                .nunique()
                > 1
            ).sum()
        ),
        "missing_key_rows": int(
            all_rows[["Community Code", "County Code", "Beach Name"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "missing_violation_rows": int(all_rows["Violation"].isna().sum()),
        "gold_concatenation_collisions": int(len(locations) - gold_key.nunique()),
        "counterfactuals": cases,
    }


if __name__ == "__main__":
    rows = [analyze_attempt(spec) for spec in ALL_PILOT_ARMS]
    complete = all(
        r["verdict"].get("comparison_eligible") is True
        and r["verdict"].get("score") in (0, 1)
        and r["attempt_status"] == "completed"
        and r["stats"].get("cost_status") == "complete"
        and r["execution"].get("journal_status") == "closed"
        and r["execution"].get("unfinished_requests") == 0
        for r in rows
    )
    report = {
        "task": "environment-easy-3",
        "attempts_per_arm": 1,
        "rows": rows,
        "status": "complete" if complete else "in_progress",
        "input_hash_scope": "first agent step inputMessages only; sorted-key compact UTF-8 JSON; not full provider request",
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "semantic_audit.json").write_text(
        json.dumps(semantic_audit(), indent=2) + "\n"
    )
    for row in rows:
        print(
            json.dumps(
                {
                    "model": row["model"],
                    "arm": row["arm"],
                    "answer": row["answer"],
                    "score": row["verdict"].get("score"),
                    "cost_usd": row["stats"].get("cost_usd"),
                    "steps": row["stats"].get("num_steps"),
                    "data_frames": row["seen_data_frames"],
                    "flow_frames": row["seen_flow_frames"],
                }
            )
        )

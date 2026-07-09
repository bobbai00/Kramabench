#!/usr/bin/env python3
"""Analyze KramaBench A/B cases for accuracy and cost-compaction signals.

The script is read-only over benchmark artifacts. It compares the 12 GPT-5.2
dataflow matrix arms one dimension at a time:

  - context mode: Latest vs Delta
  - sample/result context: 3k vs 5k vs 7k
  - information level: StatsD2 vs SchemaOnly

It reports two tiers:

  - candidate: deterministic outcome/shape condition holds.
  - principle_match: the richer arm is the winner for accuracy, or the leaner
    arm is cheaper for a same-shape/same-answer cost case.

These are rule-based counts for triage. Manual trace audit is still required
before making paper-level causal claims.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PASS_THRESHOLD = 1.0
DEFAULT_MIN_COST_GAP = 0.005
DEFAULT_MIN_COST_RATIO = 0.10
DEFAULT_MIN_INPUT_GAP = 500


@dataclass(frozen=True)
class Arm:
    mode: str
    sample_k: int
    info: str
    sut: str


def load_kb(root: Path):
    spec = importlib.util.spec_from_file_location("kb_module", root / "kb.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {root / 'kb.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix_arms() -> list[Arm]:
    arms: list[Arm] = []
    for mode in ("Latest", "Delta"):
        for sample_k in (3, 5, 7):
            arms.append(
                Arm(
                    mode=mode,
                    sample_k=sample_k,
                    info="StatsD2",
                    sut=f"DataflowSystemGPT52{mode}Stats{sample_k}kD2",
                )
            )
            arms.append(
                Arm(
                    mode=mode,
                    sample_k=sample_k,
                    info="SchemaOnly",
                    sut=f"DataflowSystemGPT52{mode}{sample_k}kSchemaOnly",
                )
            )
    return arms


def pair_dimension(a: Arm, b: Arm) -> str | None:
    diffs = []
    if a.mode != b.mode:
        diffs.append("mode")
    if a.sample_k != b.sample_k:
        diffs.append("sample")
    if a.info != b.info:
        diffs.append("info")
    return diffs[0] if len(diffs) == 1 else None


def richer_arm(a: Arm, b: Arm, dim: str) -> Arm:
    if dim == "mode":
        return a if a.mode == "Delta" else b
    if dim == "sample":
        return a if a.sample_k > b.sample_k else b
    if dim == "info":
        return a if a.info == "StatsD2" else b
    raise ValueError(dim)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def answer_value(root: Path, sut: str, task_id: str) -> Any:
    obj = _load_json(root / "system_scratch" / sut / task_id / "answer.json")
    return obj.get("answer") if isinstance(obj, dict) else ""


def normalize_answer(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return ("number", round(float(value), 10))
    if isinstance(value, list):
        return ("list", tuple(normalize_answer(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), normalize_answer(v)) for k, v in value.items())))

    text = str(value).strip()
    if not text:
        return ("text", "")
    try:
        parsed = ast.literal_eval(text)
        if parsed is not text:
            return normalize_answer(parsed)
    except Exception:
        pass
    try:
        return ("number", round(float(text), 10))
    except Exception:
        pass
    return ("text", re.sub(r"\s+", " ", text.lower()))


def workflow(root: Path, sut: str, task_id: str) -> dict[str, Any]:
    obj = _load_json(root / "system_scratch" / sut / task_id / "workflow.json")
    if not isinstance(obj, dict):
        return {}
    inner = obj.get("workflow")
    return inner if isinstance(inner, dict) else {}


def operator_code(op: dict[str, Any]) -> str:
    props = op.get("operatorProperties") or {}
    return props.get("code") or props.get("pythonCode") or props.get("script") or ""


def workflow_features(root: Path, sut: str, task_id: str) -> dict[str, Any]:
    w = workflow(root, sut, task_id)
    ops = w.get("operators") or []
    links = w.get("links") or []
    types = Counter(op.get("operatorType") for op in ops)
    shape = sorted((op.get("operatorType"), len(op.get("inputPorts") or []), len(op.get("outputPorts") or [])) for op in ops)
    code_chunks = []
    code_hashes = []
    for op in sorted(ops, key=lambda item: item.get("operatorID", "")):
        code = re.sub(r"\s+", " ", operator_code(op)).strip()
        code_chunks.append(code)
        code_hashes.append((op.get("operatorType"), hashlib.sha1(code.encode()).hexdigest()[:12]))
    return {
        "ops": len(ops),
        "links": len(links),
        "types": dict(types),
        "shape": shape,
        "code_text": "\n".join(code_chunks),
        "code_hashes": sorted(code_hashes),
    }


def same_shape(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["ops"] == b["ops"]
        and a["links"] == b["links"]
        and Counter(a["types"]) == Counter(b["types"])
    )


def same_code(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["ops"] == b["ops"]
        and a["links"] == b["links"]
        and a["code_hashes"] == b["code_hashes"]
    )


def code_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = a["code_text"]
    right = b["code_text"]
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def react_metrics(kb, root: Path, sut: str, task_id: str) -> dict[str, Any]:
    task_dir = root / "system_scratch" / sut / task_id
    try:
        return kb.react_metrics(task_dir)
    except Exception:
        return {}


def cost_lookup(kb, sut: str) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in kb.load_cost_stats(sut)}


def config_settings(root: Path, sut: str, task_id: str) -> dict[str, Any]:
    cfg = _load_json(root / "system_scratch" / sut / task_id / "config.json")
    if not isinstance(cfg, dict):
        return {}
    settings = cfg.get("agent_settings")
    return settings if isinstance(settings, dict) else {}


def richer_setting_proven(rich: Arm, lean: Arm, dim: str, rich_cfg: dict[str, Any], lean_cfg: dict[str, Any]) -> bool:
    if dim == "mode":
        return rich.mode == "Delta" and lean.mode == "Latest" and rich_cfg.get("context_mode") == "delta"
    if dim == "sample":
        rich_limit = rich_cfg.get("max_operator_result_char_limit", 0) or 0
        lean_limit = lean_cfg.get("max_operator_result_char_limit", 0) or 0
        return rich.sample_k > lean.sample_k and rich_limit > lean_limit
    if dim == "info":
        rich_stats = bool(rich_cfg.get("column_stats"))
        lean_stats = bool(lean_cfg.get("column_stats"))
        rich_data_level = int(rich_cfg.get("data_level", 0) or 0)
        lean_data_level = int(lean_cfg.get("data_level", 0) or 0)
        return rich.info == "StatsD2" and lean.info == "SchemaOnly" and rich_stats and not lean_stats and rich_data_level > lean_data_level
    return False


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    kb = load_kb(root)
    arms = matrix_arms()
    success = {arm.sut: kb.load_task_success(arm.sut) for arm in arms}
    costs = {arm.sut: cost_lookup(kb, arm.sut) for arm in arms}

    summary: dict[str, Any] = {
        "pass_threshold": PASS_THRESHOLD,
        "min_cost_gap": args.min_cost_gap,
        "min_cost_ratio": args.min_cost_ratio,
        "min_input_gap_tokens": args.min_input_gap,
        "arms": [arm.__dict__ for arm in arms],
        "pairs": 0,
        "shared_task_pairs": 0,
        "accuracy": defaultdict(int),
        "cost": defaultdict(int),
    }
    accuracy_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    for i, first in enumerate(arms):
        for second in arms[i + 1 :]:
            dim = pair_dimension(first, second)
            if dim is None:
                continue
            rich = richer_arm(first, second, dim)
            lean = second if rich == first else first
            summary["pairs"] += 1

            rich_scores = success[rich.sut]
            lean_scores = success[lean.sut]
            rich_costs = costs[rich.sut]
            lean_costs = costs[lean.sut]
            common = sorted(set(rich_scores) & set(lean_scores) & set(rich_costs) & set(lean_costs))
            summary["shared_task_pairs"] += len(common)

            for task_id in common:
                rich_score = float(rich_scores.get(task_id, 0) or 0)
                lean_score = float(lean_scores.get(task_id, 0) or 0)
                rich_pass = rich_score >= PASS_THRESHOLD
                lean_pass = lean_score >= PASS_THRESHOLD
                rich_cost = float(rich_costs[task_id]["cost"])
                lean_cost = float(lean_costs[task_id]["cost"])
                rich_input = int(rich_costs[task_id].get("input_tokens", 0) or 0)
                lean_input = int(lean_costs[task_id].get("input_tokens", 0) or 0)
                rich_steps = int(rich_costs[task_id].get("num_steps", 0) or 0)
                lean_steps = int(lean_costs[task_id].get("num_steps", 0) or 0)

                if rich_pass != lean_pass:
                    summary["accuracy"][f"{dim}.candidate_flip"] += 1
                    if rich_pass:
                        rich_cfg = config_settings(root, rich.sut, task_id)
                        lean_cfg = config_settings(root, lean.sut, task_id)
                        rich_metrics = react_metrics(kb, root, rich.sut, task_id)
                        lean_metrics = react_metrics(kb, root, lean.sut, task_id)
                        setting_proven = richer_setting_proven(rich, lean, dim, rich_cfg, lean_cfg)
                        more_context_observed = rich_input - lean_input >= args.min_input_gap
                        more_trace_work = (
                            int(rich_metrics.get("tool_calls", 0) or 0) > int(lean_metrics.get("tool_calls", 0) or 0)
                            or int(rich_metrics.get("wf_ops", 0) or 0) > int(lean_metrics.get("wf_ops", 0) or 0)
                        )
                        principle_match = setting_proven and (more_context_observed or more_trace_work or dim == "info")
                        summary["accuracy"][f"{dim}.rich_wins"] += 1
                        if principle_match:
                            summary["accuracy"][f"{dim}.principle_match"] += 1
                        else:
                            summary["accuracy"][f"{dim}.needs_manual_review"] += 1
                        accuracy_rows.append(
                            {
                                "dimension": dim,
                                "task_id": task_id,
                                "principle_match": principle_match,
                                "rich_sut": rich.sut,
                                "lean_sut": lean.sut,
                                "rich_score": rich_score,
                                "lean_score": lean_score,
                                "rich_answer": answer_value(root, rich.sut, task_id),
                                "lean_answer": answer_value(root, lean.sut, task_id),
                                "rich_cost": rich_cost,
                                "lean_cost": lean_cost,
                                "rich_steps": rich_steps,
                                "lean_steps": lean_steps,
                                "rich_input_tokens": rich_input,
                                "lean_input_tokens": lean_input,
                                "input_gap": rich_input - lean_input,
                                "setting_proven": setting_proven,
                                "more_context_observed": more_context_observed,
                                "more_trace_work": more_trace_work,
                                "rich_tool_calls": rich_metrics.get("tool_calls", 0),
                                "lean_tool_calls": lean_metrics.get("tool_calls", 0),
                                "rich_wf_ops": rich_metrics.get("wf_ops", 0),
                                "lean_wf_ops": lean_metrics.get("wf_ops", 0),
                            }
                        )
                    else:
                        summary["accuracy"][f"{dim}.lean_wins_reverse"] += 1

                if rich_pass and lean_pass:
                    rich_answer = answer_value(root, rich.sut, task_id)
                    lean_answer = answer_value(root, lean.sut, task_id)
                    answers_same = normalize_answer(rich_answer) == normalize_answer(lean_answer)
                    if not answers_same:
                        continue
                    rich_features = workflow_features(root, rich.sut, task_id)
                    lean_features = workflow_features(root, lean.sut, task_id)
                    shape_same = same_shape(rich_features, lean_features)
                    if not shape_same:
                        continue
                    gap = rich_cost - lean_cost
                    denominator = max(lean_cost, 1e-9)
                    cost_ratio = gap / denominator
                    summary["cost"][f"{dim}.same_answer_same_shape_candidate"] += 1
                    if gap >= args.min_cost_gap and cost_ratio >= args.min_cost_ratio:
                        summary["cost"][f"{dim}.principle_match"] += 1
                        principle_match = True
                    else:
                        summary["cost"][f"{dim}.below_gap_threshold"] += 1
                        principle_match = False
                    code_same = same_code(rich_features, lean_features)
                    similarity = code_similarity(rich_features, lean_features)
                    if code_same:
                        summary["cost"][f"{dim}.same_code"] += 1
                    if rich_steps == lean_steps:
                        summary["cost"][f"{dim}.same_steps"] += 1
                    cost_rows.append(
                        {
                            "dimension": dim,
                            "task_id": task_id,
                            "principle_match": principle_match,
                            "rich_sut": rich.sut,
                            "lean_sut": lean.sut,
                            "answer": rich_answer,
                            "rich_cost": rich_cost,
                            "lean_cost": lean_cost,
                            "cost_gap": gap,
                            "cost_ratio": cost_ratio,
                            "rich_steps": rich_steps,
                            "lean_steps": lean_steps,
                            "rich_input_tokens": rich_input,
                            "lean_input_tokens": lean_input,
                            "input_gap": rich_input - lean_input,
                            "rich_total_tokens": rich_costs[task_id].get("total_tokens", 0),
                            "lean_total_tokens": lean_costs[task_id].get("total_tokens", 0),
                            "same_code": code_same,
                            "code_similarity": similarity,
                            "wf_ops": rich_features["ops"],
                            "wf_links": rich_features["links"],
                        }
                    )

    # Convert defaultdicts for JSON output.
    summary["accuracy"] = dict(sorted(summary["accuracy"].items()))
    summary["cost"] = dict(sorted(summary["cost"].items()))
    summary["accuracy_total_principle_match"] = sum(v for k, v in summary["accuracy"].items() if k.endswith(".principle_match"))
    summary["cost_total_principle_match"] = sum(v for k, v in summary["cost"].items() if k.endswith(".principle_match"))
    summary["accuracy_rich_wins_total"] = sum(v for k, v in summary["accuracy"].items() if k.endswith(".rich_wins"))
    summary["accuracy_candidate_flip_total"] = sum(v for k, v in summary["accuracy"].items() if k.endswith(".candidate_flip"))
    summary["cost_same_answer_same_shape_total"] = sum(
        v for k, v in summary["cost"].items() if k.endswith(".same_answer_same_shape_candidate")
    )
    strict_cost = defaultdict(int)
    for row in cost_rows:
        if not row["principle_match"]:
            continue
        dim = row["dimension"]
        if row["rich_steps"] == row["lean_steps"]:
            strict_cost[f"{dim}.principle_same_steps"] += 1
            if row["code_similarity"] >= 0.2:
                strict_cost[f"{dim}.principle_same_steps_code_sim_0_2"] += 1
            if row["code_similarity"] >= 0.5:
                strict_cost[f"{dim}.principle_same_steps_code_sim_0_5"] += 1
            if row["code_similarity"] >= 0.7:
                strict_cost[f"{dim}.principle_same_steps_code_sim_0_7"] += 1
        if row["code_similarity"] >= 0.5:
            strict_cost[f"{dim}.principle_code_sim_0_5"] += 1
    summary["strict_cost"] = dict(sorted(strict_cost.items()))
    summary["cost_principle_same_steps_total"] = sum(
        v for k, v in summary["strict_cost"].items() if k.endswith(".principle_same_steps")
    )
    summary["cost_principle_same_steps_code_sim_0_2_total"] = sum(
        v for k, v in summary["strict_cost"].items() if k.endswith(".principle_same_steps_code_sim_0_2")
    )
    summary["cost_principle_same_steps_code_sim_0_5_total"] = sum(
        v for k, v in summary["strict_cost"].items() if k.endswith(".principle_same_steps_code_sim_0_5")
    )
    summary["accuracy_rows"] = len(accuracy_rows)
    summary["cost_rows"] = len(cost_rows)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(out_dir / "accuracy_cases.csv", accuracy_rows)
    write_csv(out_dir / "cost_cases.csv", cost_rows)

    top_accuracy = sorted(accuracy_rows, key=lambda row: (not row["principle_match"], -abs(row["input_gap"]), -row["rich_cost"]))
    top_cost = sorted(cost_rows, key=lambda row: (not row["principle_match"], -row["cost_gap"], -row["code_similarity"]))
    write_csv(out_dir / "top_accuracy_cases.csv", top_accuracy[:100])
    write_csv(out_dir / "top_cost_cases.csv", top_cost[:100])

    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    print("Signal analyzer complete")
    print(f"  one-dimension pairs: {summary['pairs']}")
    print(f"  shared task-pair rows: {summary['shared_task_pairs']}")
    print()
    print("Accuracy flips")
    print(f"  candidates, any direction: {summary['accuracy_candidate_flip_total']}")
    print(f"  richer arm wins:          {summary['accuracy_rich_wins_total']}")
    print(f"  principle matches:        {summary['accuracy_total_principle_match']}")
    for dim in ("sample", "info", "mode"):
        print(
            f"    {dim:<6} candidates={summary['accuracy'].get(dim + '.candidate_flip', 0):>3} "
            f"rich_wins={summary['accuracy'].get(dim + '.rich_wins', 0):>3} "
            f"matches={summary['accuracy'].get(dim + '.principle_match', 0):>3} "
            f"reverse={summary['accuracy'].get(dim + '.lean_wins_reverse', 0):>3}"
        )
    print()
    print("Cost cases")
    print(f"  same answer + same shape candidates: {summary['cost_same_answer_same_shape_total']}")
    print(f"  principle matches:                   {summary['cost_total_principle_match']}")
    print(f"  principle + same steps:              {summary['cost_principle_same_steps_total']}")
    print(f"  principle + same steps + code>=.2:   {summary['cost_principle_same_steps_code_sim_0_2_total']}")
    print(f"  principle + same steps + code>=.5:   {summary['cost_principle_same_steps_code_sim_0_5_total']}")
    for dim in ("sample", "info", "mode"):
        print(
            f"    {dim:<6} candidates={summary['cost'].get(dim + '.same_answer_same_shape_candidate', 0):>3} "
            f"matches={summary['cost'].get(dim + '.principle_match', 0):>3} "
            f"same_steps={summary['cost'].get(dim + '.same_steps', 0):>3} "
            f"same_code={summary['cost'].get(dim + '.same_code', 0):>3} "
            f"strict>=.5={summary['strict_cost'].get(dim + '.principle_same_steps_code_sim_0_5', 0):>3}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path.cwd(), help="KramaBench root; default: cwd")
    parser.add_argument("--out-dir", default="judgment_runs/signal_analyzer", help="output directory")
    parser.add_argument("--min-cost-gap", type=float, default=DEFAULT_MIN_COST_GAP)
    parser.add_argument("--min-cost-ratio", type=float, default=DEFAULT_MIN_COST_RATIO)
    parser.add_argument("--min-input-gap", type=int, default=DEFAULT_MIN_INPUT_GAP)
    args = parser.parse_args()

    try:
        summary = analyze(args)
    except Exception as exc:
        print(f"analyzer failed: {exc}", file=sys.stderr)
        raise
    print_summary(summary)


if __name__ == "__main__":
    main()

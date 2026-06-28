#!/usr/bin/env python
"""
Bundle + feature extraction from a (delta-arm) trajectory.

A "candidate historical bundle" = one operator edit event (create/modify/delete)
in the trajectory. These are the pieces present in the full-delta context but
absent from latest-core. For each we compute structural features observable at
that point in the trajectory (no future leakage beyond liveness, which is a
retrospective property of the WHOLE trajectory used only to characterize the
bundle, not as a runtime feature). We also roll them up to task-level aggregates
used by the (pilot) contrastive model.

Usage:
    python bobflow_context_learning/features/extract.py   # over the manifest's delta runs
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "system_scratch")
MANIFEST = os.path.join(ROOT, "bobflow_context_learning", "data", "gate0_manifest.json")
OUT = os.path.join(ROOT, "bobflow_context_learning", "data", "features.json")
DELTA = "DataflowSystemGPT54Gate0Delta"

EDIT_TOOLS = {"createOrModifyOperator", "addOperator", "modifyOperator"}
DELETE_TOOLS = {"deleteOperator"}


def load_steps(arm, task_id):
    p = os.path.join(SCRATCH, arm, task_id, "react_steps.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if isinstance(d, list) else d.get("steps", d.get("reactSteps", []))


def extract_bundles(steps):
    """One bundle per operator edit/delete event, with structural features."""
    agent_steps = [s for s in steps if s.get("role") == "agent"]
    total = len(agent_steps)
    # index op -> list of (event_idx, verb, failed) for churn/supersede/liveness
    history = {}
    events = []
    known = set()
    for i, s in enumerate(agent_steps):
        calls = s.get("toolCalls") or []
        results = s.get("toolResults") or []
        for j, tc in enumerate(calls):
            tool = tc.get("toolName")
            inp = tc.get("input") or {}
            op = inp.get("operatorId")
            if not op:
                continue
            tr = results[j] if j < len(results) else {}
            # CAPTURE GAP: per-step operator execution RESULTS/ERRORS are rendered
            # into the live context, not persisted in react_steps.toolResults — the
            # edit's toolResult is only an "Added operator X" ack (isError stays
            # False even when the operator errors at runtime). So a direct failure
            # flag is mostly unobservable from the trace. We detect what we can
            # (tool-level error or an inline [ERROR] echo) and otherwise rely on the
            # OBSERVABLE proxy for "the agent hit a problem here": re-edits /
            # supersession (subsequent_edits, is_superseded), computed below.
            out = tr.get("output") or ""
            failed = bool(tr.get("isError")) or ("[ERROR]" in out if isinstance(out, str) else False)
            if tool in EDIT_TOOLS:
                verb = "modify" if op in known else "create"
                known.add(op)
            elif tool in DELETE_TOOLS:
                verb = "delete"
                known.discard(op)
            else:
                continue
            code = inp.get("code") or ""
            events.append({"event_idx": i, "op": op, "verb": verb, "failed": failed,
                           "code_len": len(code), "thought_len": len(s.get("content") or "")})
            history.setdefault(op, []).append((i, verb, failed))

    final_live = {op for op, h in history.items() if h[-1][1] != "delete"}
    bundles = []
    for e in events:
        op = e["op"]
        h = history[op]
        later = [x for x in h if x[0] > e["event_idx"]]
        bundles.append({
            "op": op,
            "event_idx": e["event_idx"],
            # ---- type ----
            "is_create": int(e["verb"] == "create"),
            "is_modify": int(e["verb"] == "modify"),
            "is_delete": int(e["verb"] == "delete"),
            # ---- temporal / recency ----
            "age_steps": (total - 1) - e["event_idx"],
            "normalized_age": ((total - 1) - e["event_idx"]) / max(total - 1, 1),
            # ---- execution ----
            "failed": int(e["failed"]),
            # ---- churn / version / liveness ----
            "edits_for_op": len(h),
            "subsequent_edits": len(later),
            "is_superseded": int(any(x[1] in ("create", "modify") for x in later)),
            "is_deleted_later": int(any(x[1] == "delete" for x in later)),
            "is_live_at_end": int(op in final_live),
            # ---- cost ----
            "code_len": e["code_len"],
            "token_cost_est": (e["code_len"] + e["thought_len"]) // 4,
        })
    return bundles


def task_features(bundles):
    """Aggregate bundle features → one feature vector characterizing the trajectory."""
    n = len(bundles)
    if n == 0:
        return {"n_bundles": 0, "n_failed": 0, "frac_failed": 0.0, "n_superseded": 0,
                "n_deleted": 0, "max_churn": 0, "mean_churn": 0.0, "n_distinct_ops": 0,
                "total_token_cost": 0}
    ops = {b["op"] for b in bundles}
    churn = {}
    for b in bundles:
        churn[b["op"]] = b["edits_for_op"]
    return {
        "n_bundles": n,
        "n_failed": sum(b["failed"] for b in bundles),
        "frac_failed": sum(b["failed"] for b in bundles) / n,
        "n_superseded": sum(b["is_superseded"] for b in bundles),
        "n_deleted": sum(b["is_delete"] for b in bundles),
        "max_churn": max(churn.values()),
        "mean_churn": sum(churn.values()) / len(churn),
        "n_distinct_ops": len(ops),
        "total_token_cost": sum(b["token_cost_est"] for b in bundles),
    }


def main():
    manifest = json.load(open(MANIFEST))
    out = []
    for item in manifest["sample"]:
        tid = item["task_id"]
        steps = load_steps(DELTA, tid)
        if steps is None:
            print(f"[features] {tid}: no delta trajectory yet, skip")
            continue
        bundles = extract_bundles(steps)
        feats = task_features(bundles)
        out.append({"task_id": tid, "workload": item["workload"],
                    "answer_type": item.get("answer_type"),
                    "task_features": feats, "bundles": bundles})
        print(f"[features] {tid:24s} {len(bundles):3d} bundles  "
              f"({feats['n_failed']} failed, {feats['n_superseded']} superseded, {feats['n_deleted']} deleted)")
    json.dump({"tasks": out}, open(OUT, "w"), indent=2)
    print(f"\n[features] wrote {OUT}  ({len(out)} tasks)")


if __name__ == "__main__":
    main()

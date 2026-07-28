#!/usr/bin/env python3
"""
step_results.json — per-react-step action + execution result, parsed from the
trace (no re-execution, no LLM, no derived fields).

Per acting step, exactly three fields:
  step             : index in react_steps.json steps[]
  action           : verbatim tool calls (operator id + submitted code / delete)
  execution_result : raw result text this step produced —
      delta  : the Observation block(s) of the NEW Agent Event(s) that appear in
               the next step's inputMessages (events present after this step
               that were not present before it)
      latest : the edited operator(s)' result block(s) parsed from the NEXT
               snapshot (next step's inputMessages)
      rejected tool calls (isError ACK, no event/render): the ACK string itself
      last step (no next input): null

Cache: system_scratch/<arm>/<task>/step_results.json
Run:  .venv/bin/python scripts/extract_step_results.py --arms A [--tasks ...]
      [--show TASK]   (print a task's records verbatim)
"""
import argparse, json, os, re, sys
from pathlib import Path

KB = Path(__file__).resolve().parent.parent

EVENT_RE = re.compile(r"## Agent Event (\d+)")


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def step_input_text(step):
    if not step.get("inputMessages"):
        return None
    return "\n".join(str(m.get("content", "")) for m in step["inputMessages"])


def events_in(txt):
    """event number -> full event block."""
    out = {}
    for chunk in re.split(r"(?=## Agent Event \d+)", txt)[1:]:
        m = EVENT_RE.match(chunk)
        if m:
            out[int(m.group(1))] = chunk
    return out


def latest_op_blocks(txt):
    """operatorId -> result block from a '# Current Dataflow' snapshot."""
    i = txt.rfind("# Current Dataflow")
    snap = txt[i:] if i >= 0 else txt
    out = {}
    for b in re.split(r"(?=### Operator )", snap)[1:]:
        m = re.match(r"### Operator `?([\w-]+)`?", b)
        if m:
            out[m.group(1)] = b.strip()
    return out


def action_text(step):
    parts = []
    res = {r.get("toolCallId"): r for r in (step.get("toolResults") or [])}
    for tc in (step.get("toolCalls") or []):
        nm = tc.get("toolName", "")
        inp = tc.get("input") or {}
        oid = inp.get("operatorId", "")
        if nm == "createOrModifyOperator":
            parts.append(f"[createOrModifyOperator] {oid}\n{inp.get('code', '')}")
        elif nm == "deleteOperator":
            parts.append(f"[deleteOperator] {oid}")
        else:
            parts.append(f"[{nm}] {json.dumps(inp, default=str)[:400]}")
    return "\n\n".join(parts)


def rejected_acks(step):
    res = {r.get("toolCallId"): r for r in (step.get("toolResults") or [])}
    acks = []
    for tc in (step.get("toolCalls") or []):
        r = res.get(tc.get("toolCallId"))
        if r and r.get("isError"):
            acks.append(str(r.get("output", "")))
    return acks


def accepted_opids(step):
    res = {r.get("toolCallId"): r for r in (step.get("toolResults") or [])}
    out = []
    for tc in (step.get("toolCalls") or []):
        r = res.get(tc.get("toolCallId"))
        inp = tc.get("input") or {}
        if tc.get("toolName") == "createOrModifyOperator" and inp.get("operatorId") \
                and r and not r.get("isError"):
            out.append(inp["operatorId"])
    return out


def extract(arm, task):
    d = jload(KB / "system_scratch" / arm / task / "react_steps.json")
    if not d:
        return None
    steps = d.get("steps", [])
    # mode from the last step that has input
    last_txt = next((step_input_text(s) for s in reversed(steps) if s.get("inputMessages")), "")
    mode = "delta" if "# Agent Events" in (last_txt or "") else "latest"

    records = []
    for i, s in enumerate(steps):
        if not (s.get("toolCalls")):
            continue
        # next step input text (what the agent saw after this action)
        nxt = next((step_input_text(steps[j]) for j in range(i + 1, len(steps))
                    if steps[j].get("inputMessages")), None)
        cur = step_input_text(s)
        result = None
        if nxt is not None:
            if mode == "delta":
                before = set(events_in(cur or "").keys())
                new = {n: blk for n, blk in events_in(nxt).items() if n not in before}
                if new:
                    obs = []
                    for n in sorted(new):
                        parts = new[n].split("Observation:", 1)
                        obs.append(parts[1].strip() if len(parts) > 1 else new[n].strip())
                    result = "\n\n".join(obs)
            else:  # latest
                blocks = latest_op_blocks(nxt)
                hits = [blocks[o] for o in accepted_opids(s) if o in blocks]
                if hits:
                    result = "\n\n".join(hits)
        acks = rejected_acks(s)
        if result is None and acks:
            result = "\n".join(acks)
        elif acks:
            result = result + "\n" + "\n".join(acks)
        records.append(dict(step=i, action=action_text(s), execution_result=result))
    return dict(mode=mode, records=records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--show", default=None, help="print this task's records verbatim")
    a = ap.parse_args()

    for arm in a.arms:
        base = KB / "system_scratch" / arm
        tasks = a.tasks or sorted(d.name for d in base.iterdir()
                                  if d.is_dir() and (d / "react_steps.json").exists())
        n_ok = 0
        for t in tasks:
            outp = base / t / "step_results.json"
            if outp.exists() and not a.force and a.show != t:
                n_ok += 1
                continue
            r = extract(arm, t)
            if r is None:
                continue
            json.dump(r, open(outp, "w"), indent=1)
            n_ok += 1
            if a.show == t:
                print(json.dumps(r, indent=2)[:6000])
        print(f"{arm}: {n_ok}/{len(tasks)} tasks extracted")


if __name__ == "__main__":
    main()

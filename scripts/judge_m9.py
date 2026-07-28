#!/usr/bin/env python3
"""
M9 — decision-process judge (decomposition / grounding / waste), cost-aware.

One judge call per task over the agent's FULL ordered trace (units = dataflow
operators or code cells, via judge_m5m6.extract_units). Three scores, each
anchored to an OBJECTIVE referent so the judge is general, not ad-hoc:

  decomposition : referent = the GOLD subtasks. Per subtask: does the agent's
                  structure contain an identifiable locus for it?
                  dedicated (own unit(s)) / fused (inside a multi-step unit) /
                  absent. score = mean(dedicated=1, fused=0.5, absent=0).
  grounding     : referent = the agent's OWN VISIBLE EVIDENCE. Per unit: was
                  this action justified by results visible BEFORE it (schemas,
                  tables, prior outputs), or did it ignore/contradict them or
                  guess at unseen structure? grounded / ungrounded /
                  unverifiable. score = grounded / (grounded + ungrounded).
  waste         : referent = the TRACE'S OWN HISTORY. Per unit: productive /
                  redundant_recompute (recomputes something already available) /
                  redundant_retry (repeats a failed approach without change) /
                  unused (output feeds neither later units nor the answer).
                  waste_frac (unit count) + waste_weighted (char-weighted) +
                  wasted_usd = cost_usd * waste_weighted.

The judge sees the task + gold subtasks + ordered trace. It does NOT see the
gold answer or whether the run passed — process is judged blind to outcome.

Cache: system_scratch/<arm>/<task>/judge_m9.json
Run:  .venv/bin/python scripts/judge_m9.py --arms A B [--tasks ...]
      [--model gpt-4o] [--api-base URL]   (api-base lets the judge run through
      the local litellm proxy when direct OpenAI quota is unavailable)
"""
import argparse, json, os, sys, importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("jm", KB / "scripts/judge_m5m6.py")
jm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jm)

UNIT_CAP = 4000  # chars per unit block in the trace doc


def extract_react_units(arm, task):
    """Dataflow trace at REACT-STEP grain (instead of surviving-operator grain):
    one unit per agent event (action + observation), PLUS pseudo-units for
    validation-REJECTED tool calls (isError ACKs never render as events).
    Deleted operators and superseded edits stay visible — symmetric with the
    code agent's cell trace, so waste is measured on equal footing.
    Falls back to None for non-dataflow arms (no react_steps.json)."""
    d = jm.jload(jm.KB / "system_scratch" / arm / task / "react_steps.json")
    if not d:
        return None
    ctx, trace = jm.last_context(arm, task)
    if ctx is None:
        return None
    import re
    units = []
    if "# Agent Events" in ctx:  # delta mode: events are the react steps
        for ev in re.split(r"(?=## Agent Event \d+)", ctx)[1:]:
            n = (re.match(r"## Agent Event (\d+)", ev) or [0, "?"])[1]
            parts = ev.split("Observation:", 1)
            action = parts[0]
            obs = parts[1] if len(parts) > 1 else ""
            flags = ["exec_error"] if "[ERROR]" in obs else []
            units.append(dict(id=f"event-{n}", type="ReactStep",
                              code=action.strip(), render=obs.strip() or None,
                              flags=flags))
    else:  # latest mode: unit per acting step; render = edited ops' fresh blocks
        steps = d.get("steps", [])
        snaps = []
        for s in steps:
            if s.get("inputMessages"):
                txt = "\n".join(str(m.get("content", "")) for m in s["inputMessages"])
                snaps.append(jm.renders_latest(txt))
            else:
                snaps.append(None)
        for k, s in enumerate(steps):
            tcs = s.get("toolCalls") or []
            if not tcs:
                continue
            acts, opids = [], []
            res = {r.get("toolCallId"): r for r in (s.get("toolResults") or [])}
            for tc in tcs:
                inp = tc.get("input") or {}
                oid = inp.get("operatorId")
                r = res.get(tc.get("toolCallId")) or {}
                acts.append(f"[{tc.get('toolName')}] {oid}\n{inp.get('code','')}\nACK: {str(r.get('output',''))[:300]}")
                if oid and not r.get("isError"):
                    opids.append(oid)
            nxt = next((snaps[j] for j in range(k + 1, len(steps)) if snaps[j]), {})
            rend = "\n".join(nxt.get(o, "") for o in opids if nxt.get(o)) or None
            flags = ["exec_error"] if any((res.get(tc.get("toolCallId")) or {}).get("isError") for tc in tcs) else []
            units.append(dict(id=f"step-{k}", type="ReactStep",
                              code="\n\n".join(acts), render=rend, flags=flags))
    # pseudo-units for REJECTED edits in delta mode (they produce no event)
    if "# Agent Events" in ctx:
        for k, s in enumerate(d.get("steps", [])):
            res = {r.get("toolCallId"): r for r in (s.get("toolResults") or [])}
            for tc in (s.get("toolCalls") or []):
                r = res.get(tc.get("toolCallId")) or {}
                if r.get("isError"):
                    inp = tc.get("input") or {}
                    units.append(dict(
                        id=f"step{k}-rejected-{inp.get('operatorId','?')}",
                        type="RejectedEdit",
                        code=inp.get("code", "") or str(inp)[:500],
                        render="[REJECTED] " + str(r.get("output", ""))[:400],
                        flags=["exec_error"]))
    return dict(mode="react", entries=units, edges=[])


M9_SYS = (
    "You audit the full execution trace of a data-analysis agent: the ordered "
    "sequence of units it built (each with its code and executed result). "
    "Judge the PROCESS only — you are not told whether the final answer was "
    "correct, and you must not guess at it. Make three orthogonal judgments, "
    "each against a stated referent:\n\n"
    "1. DECOMPOSITION — referent: the reference subtasks provided. For each "
    "subtask, is there an identifiable locus in the agent's trace that carries "
    "it out? 'dedicated' = one unit (or a clearly delimited part) is "
    "responsible for it; 'fused' = it happens inside a unit that also does "
    "other subtasks; 'absent' = no unit carries it out.\n\n"
    "2. GROUNDING — referent: the evidence visible to the agent BEFORE each "
    "unit (the results of earlier units: schemas, sample rows, computed "
    "tables). For each unit: 'grounded' = its action follows from and "
    "correctly uses that visible evidence (e.g. filters a column the schema "
    "shows, references upstream results consistently); 'ungrounded' = it "
    "ignores or contradicts visible evidence, or assumes structure/values it "
    "has not seen (e.g. hardcodes a column name never displayed, re-guesses "
    "after evidence answered the question); 'unverifiable' = nothing visible "
    "bears on the choice (e.g. the very first exploratory load).\n\n"
    "3. WASTE — referent: the trace's own history. For each unit: "
    "'productive' = adds new information or new computation toward the task; "
    "'redundant_recompute' = recomputes or re-displays something an earlier "
    "unit already produced; 'redundant_retry' = repeats an already-failed "
    "approach without a meaningful change; 'unused' = its output is used by "
    "no later unit and does not carry the task forward.\n\n"
    "Be strict but fair: exploration is NOT waste when the information was new "
    "at that point. Cover every listed id exactly once; keep notes under 20 "
    "words."
)


def m9_schema():
    return {"type": "json_schema", "json_schema": {
        "name": "m9_verdicts", "strict": True, "schema": {
            "type": "object",
            "properties": {
                "decomposition": {"type": "array", "items": {
                    "type": "object", "properties": {
                        "id": {"type": "string"},
                        "mapping": {"type": "string", "enum": ["dedicated", "fused", "absent"]},
                        "units": {"type": "array", "items": {"type": "string"}},
                        "note": {"type": "string"}},
                    "required": ["id", "mapping", "units", "note"],
                    "additionalProperties": False}},
                "decisions": {"type": "array", "items": {
                    "type": "object", "properties": {
                        "id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["grounded", "ungrounded", "unverifiable"]},
                        "note": {"type": "string"}},
                    "required": ["id", "verdict", "note"],
                    "additionalProperties": False}},
                "waste": {"type": "array", "items": {
                    "type": "object", "properties": {
                        "id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["productive", "redundant_recompute", "redundant_retry", "unused"]},
                        "note": {"type": "string"}},
                    "required": ["id", "kind", "note"],
                    "additionalProperties": False}},
            },
            "required": ["decomposition", "decisions", "waste"],
            "additionalProperties": False}}}


def trace_doc(df):
    parts = []
    if df.get("edges"):
        parts.append("Structure: " + ", ".join(df["edges"]))
    for e in df["entries"]:
        block = [f"### Unit `{e['id']}`" + (f" ({e['type']})" if e.get("type") else "")]
        block.append("Code:\n" + (jm.cap(e["code"], UNIT_CAP) if e["code"] else "[none]"))
        block.append("Result:\n" + (jm.cap(e["render"], UNIT_CAP) if e.get("render") else "[no rendered result]"))
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def judge_task(client, model, task_def, df):
    subs = [dict(id=s.get("id"), step=s.get("step", "")) for s in task_def.get("subtasks", [])]
    user = (
        f"Task: {task_def.get('query','')}\n\n"
        f"Reference subtasks:\n" +
        "\n".join(f"- id {s['id']}: {s['step'][:220]}" for s in subs) +
        f"\n\nAgent trace ({len(df['entries'])} units, in execution order):\n" +
        trace_doc(df) +
        "\n\nJudge: decomposition per subtask id; grounding and waste per unit id."
    )
    r = client.chat.completions.create(
        model=model, temperature=0,
        response_format=m9_schema(),
        messages=[{"role": "system", "content": M9_SYS},
                  {"role": "user", "content": user}],
    )
    out = json.loads(r.choices[0].message.content)
    usage = r.usage.total_tokens if r.usage else 0
    return out, usage


def score_task(arm, task, task_def, verdicts, df):
    subs = {s["id"] for s in task_def.get("subtasks", []) if s.get("id")}
    units = {e["id"]: e for e in df["entries"]}
    dmap = {d["id"]: d for d in verdicts.get("decomposition", []) if d.get("id") in subs}
    W = {"dedicated": 1.0, "fused": 0.5, "absent": 0.0}
    decomp = (sum(W.get(d["mapping"], 0) for d in dmap.values()) / len(subs)) if subs else None

    dec = {d["id"]: d["verdict"] for d in verdicts.get("decisions", []) if d.get("id") in units}
    g = sum(1 for v in dec.values() if v == "grounded")
    u = sum(1 for v in dec.values() if v == "ungrounded")
    grounding = g / (g + u) if (g + u) else None

    wk = {d["id"]: d["kind"] for d in verdicts.get("waste", []) if d.get("id") in units}
    wasted_ids = [i for i, k in wk.items() if k != "productive"]
    waste_frac = len(wasted_ids) / len(units) if units else None
    size = {i: len((units[i]["code"] or "")) + len(units[i].get("render") or "") for i in units}
    tot = sum(size.values()) or 1
    waste_weighted = sum(size[i] for i in wasted_ids) / tot

    stats = jm.jload(KB / "system_scratch" / arm / task / "stats.json") or {}
    cost = float(stats.get("cost_usd") or 0)
    return dict(
        decomposition=decomp,
        decomposition_detail={d["id"]: d["mapping"] for d in dmap.values()},
        grounding=grounding,
        grounding_detail=dec,
        waste_frac=waste_frac,
        waste_weighted=round(waste_weighted, 4),
        waste_detail=wk,
        wasted_usd=round(cost * waste_weighted, 6),
        cost_usd=cost,
        n_units=len(units), n_subtasks=len(subs),
    )


def run_task(client, model, W, arm, task, force=False, unit="final"):
    fname = "judge_m9.json" if unit == "final" else "judge_m9react.json"
    outp = KB / "system_scratch" / arm / task / fname
    if outp.exists() and not force:
        return jm.jload(outp)
    task_def = W.get(task)
    if not task_def or not task_def.get("subtasks"):
        return None
    if unit == "react":
        df = extract_react_units(arm, task) or jm.extract_units(arm, task)
    else:
        df = jm.extract_units(arm, task)
    if df is None or not df["entries"]:
        return None
    try:
        verdicts, usage = judge_task(client, model, task_def, df)
    except Exception as e:
        print(f"  [M9 ERR {arm}/{task}] {e}", file=sys.stderr)
        return None
    res = dict(arm=arm, task=task, judge_model=model, tokens=usage,
               **score_task(arm, task, task_def, verdicts, df))
    tmp = outp.with_suffix(".tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, outp)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--api-base", default=None,
                    help="override OpenAI base URL (e.g. http://localhost:4000 to "
                         "route the judge through the local litellm proxy)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unit", choices=["final", "react"], default="final",
                    help="dataflow unit grain: 'final' = surviving operators; "
                         "'react' = agent events incl. rejected edits (waste-symmetric "
                         "with code-agent cells). Cache: judge_m9react.json")
    a = ap.parse_args()

    jm.load_env()
    from openai import OpenAI
    kw = {}
    if a.api_base:
        kw["base_url"] = a.api_base
        kw["api_key"] = a.api_key or "dummy"
    else:
        kw["api_key"] = a.api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(**kw)
    W = jm.load_workload()

    jobs = []
    for arm in a.arms:
        tasks = a.tasks or sorted(
            d.name for d in (KB / "system_scratch" / arm).iterdir()
            if d.is_dir() and d.name in W
            and ((d / "react_steps.json").exists() or (d / "reasoning_trace.json").exists()))
        jobs += [(arm, t) for t in tasks]

    done = {}
    def work(j):
        arm, t = j
        try:
            return (arm, run_task(client, a.model, W, arm, t, force=a.force, unit=a.unit))
        except Exception as e:
            print(f"  [ERR {arm}/{t}] {e}", file=sys.stderr)
            return (arm, None)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (arm, r) in enumerate(ex.map(work, jobs)):
            if r:
                done.setdefault(arm, []).append(r)
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(jobs)}", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n{'arm':52s} {'n':>4s} {'decomp':>7s} {'ground':>7s} {'waste':>6s} {'wast$':>8s} {'$/task':>8s}")
    print("-" * 100)
    for arm in a.arms:
        rs = done.get(arm, [])
        if not rs:
            print(f"{arm:52s}    0")
            continue
        print(f"{arm:52s} {len(rs):4d} "
              f"{mean([r['decomposition'] for r in rs]):7.3f} "
              f"{mean([r['grounding'] for r in rs]):7.3f} "
              f"{mean([r['waste_frac'] for r in rs]):6.3f} "
              f"{mean([r['wasted_usd'] for r in rs]):8.5f} "
              f"{mean([r['cost_usd'] for r in rs]):8.5f}")


if __name__ == "__main__":
    main()

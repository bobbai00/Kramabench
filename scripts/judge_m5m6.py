#!/usr/bin/env python3
"""
M5 / M6 — final-dataflow judge metrics (single judge call per lens, no chunking).

Eval object (shared extractor, uniform across delta/latest context modes): the
agent's FINAL dataflow reconstructed from react_steps.json + workflow.json —
  * operators : workflow.json's surviving set, topological order
  * code      : last ACCEPTED createOrModifyOperator edit (toolResult present,
                isError=False); falls back to last submitted (flag code_unexecuted)
  * result    : the operator's LAST render in the trace —
                delta  -> last "- operator X added/updated ... result:" block in
                          the final cumulative context (downstream re-exec events
                          overwrite older renders, so this is current)
                latest -> the op's block in the final "# Current Dataflow" snapshot
  * flags     : result_missing (never rendered; e.g. step-cap truncation),
                exec_error (last render is an [ERROR] block)

Lenses (whole dataflow is the judge's SOLE input; verdicts id-keyed JSON with a
short explanation + operator citation):
  M5 value-extracted : is the subtask's ground-truth VALUE materialized in the
                       final dataflow's executed RESULTS? (file-name items: the
                       dataflow reading that file counts). The M5 lens returns a
                       3-way status per subtask: visible / computed_not_shown
                       (fused: computed internally, never output) / absent.
                       m5 = frac(visible); m7 = frac(visible|computed_not_shown)
                       — M7 credits fused intermediates, M5 does not.
  M6 step-done       : does the final dataflow PERFORM the subtask's step
                       (fusion/equivalent implementation OK; value correctness
                       irrelevant)?

Cache: system_scratch/<arm>/<task>/judge_m5m6.json (skip unless --force).
Run:  .venv/bin/python scripts/judge_m5m6.py --arms A B [--tasks ...] [--workers N]
"""
import argparse, json, re, glob, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
JUDGE_MODEL_DEFAULT = "gpt-4o-mini"
OP_CAP = 6000    # chars per operator block (head 4500 + tail 1500)
DOC_CAP = 90000  # chars for the whole dataflow doc


def load_env():
    for line in (KB / ".env").read_text().splitlines():
        line = re.sub(r"^export\s+", "", line.strip())
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_workload():
    W = {}
    for f in glob.glob(str(KB / "workload/*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")):
            continue
        for t in json.load(open(f)):
            if isinstance(t, dict) and t.get("id"):
                W[t["id"]] = t
    return W


def jload(p):
    p = Path(p)
    try:
        return json.load(open(p)) if p.exists() else None
    except Exception:
        return None


# ------------------------------------------------------- final-dataflow extractor

def final_codes(trace):
    """operatorId -> (code, accepted) from the tool-call log.
    accepted=True only if the edit's toolResult exists and isError is False.
    A later accepted edit wins; a later unexecuted submission does NOT override
    an earlier accepted one (the accepted code is what the render reflects)."""
    acc, sub = {}, {}
    for s in (trace or {}).get("steps", []):
        res = {r.get("toolCallId"): r for r in (s.get("toolResults") or [])}
        for tc in (s.get("toolCalls") or []):
            nm, inp = tc.get("toolName", ""), tc.get("input") or {}
            oid = inp.get("operatorId")
            r = res.get(tc.get("toolCallId"))
            ok = bool(r) and not r.get("isError", False)
            if nm == "createOrModifyOperator" and oid:
                sub[oid] = inp.get("code", "")
                if ok:
                    acc[oid] = inp.get("code", "")
            elif nm == "deleteOperator" and oid and ok:
                acc.pop(oid, None)
                sub.pop(oid, None)
    out = {}
    for oid in set(acc) | set(sub):
        if oid in acc:
            out[oid] = (acc[oid], True)
        else:
            out[oid] = (sub[oid], False)
    return out


def last_context(arm, task):
    d = jload(KB / "system_scratch" / arm / task / "react_steps.json")
    if not d:
        return None, None
    steps = [s for s in d.get("steps", []) if s.get("inputMessages")]
    if not steps:
        return None, d
    txt = "\n".join(str(m.get("content", "")) for m in steps[-1]["inputMessages"])
    return txt, d


OP_RENDER_RE = re.compile(r"- operator\s+`?([\w-]+)`?\s+(?:added|updated)")


def renders_delta(ctx):
    """operatorId -> last observation render block (later events overwrite)."""
    out = {}
    for ev in re.split(r"(?=## Agent Event \d+)", ctx)[1:]:
        obs = ev.split("Observation:", 1)
        if len(obs) < 2:
            continue
        obs = obs[1]
        ms = list(OP_RENDER_RE.finditer(obs))
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(obs)
            out[m.group(1)] = obs[m.start():end].strip()
    return out


def renders_latest(ctx):
    """operatorId -> block from the final '# Current Dataflow' snapshot."""
    i = ctx.rfind("# Current Dataflow")
    snap = ctx[i:] if i >= 0 else ctx
    out = {}
    blocks = re.split(r"(?=### Operator )", snap)
    for b in blocks[1:]:
        m = re.match(r"### Operator `?([\w-]+)`?", b)
        if m:
            out[m.group(1)] = b.strip()
    return out


def final_workflow(arm, task):
    w = jload(KB / "system_scratch" / arm / task / "workflow.json") or {}
    wf = w.get("workflow") or {}
    return wf.get("operators", []) or [], wf.get("links", []) or []


def topo_order(ops, links):
    ids = [o.get("operatorID") for o in ops]
    indeg = {i: 0 for i in ids}
    adj = {i: [] for i in ids}
    for l in links:
        s = (l.get("source") or {}).get("operatorID") or l.get("source")
        t = (l.get("target") or {}).get("operatorID") or l.get("target")
        if s in adj and t in indeg:
            adj[s].append(t)
            indeg[t] += 1
    out, q = [], [i for i in ids if indeg[i] == 0]
    while q:
        n = q.pop(0)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    return out + [i for i in ids if i not in out]


def cap(txt, n=OP_CAP):
    if len(txt) <= n:
        return txt
    return txt[: int(n * 0.75)] + "\n...[truncated]...\n" + txt[-int(n * 0.25):]


TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|^\w*Error:|Exception:", re.M)


def extract_codeagent(arm, task):
    """Code-agent trace (reasoning_trace.json): flat step list, each a code cell
    + its stdout. Uniform unit = one step; render = observations (printed
    output); edges = linear chain. Same downstream judge as the dataflow path."""
    d = jload(KB / "system_scratch" / arm / task / "reasoning_trace.json")
    if not d:
        return None
    entries, prev = [], None
    edges = []
    for e in d:
        sid = f"step-{e.get('step')}"
        code = e.get("code") or ""
        obs = e.get("observations") or ""
        # a step's delivered result is its stdout; the final step also carries
        # the answer in `output` (## Final Answer) — include it as a render line.
        out = e.get("output") or ""
        render = obs if not out else (obs + ("\n" + out if out not in obs else ""))
        flags = []
        if not code:
            flags.append("code_missing")
        if not render.strip():
            flags.append("result_missing")
        elif TRACEBACK_RE.search(render):
            flags.append("exec_error")
        if e.get("is_final_answer"):
            flags.append("final")
        entries.append(dict(id=sid, type="CodeStep", name="",
                            code=code, render=render or None, flags=flags))
        if prev is not None:
            edges.append(f"{prev}->{sid}")
        prev = sid
    return dict(mode="codeagent", entries=entries, edges=edges)


def extract_units(arm, task):
    """Dispatch on trace type. Code-agent has reasoning_trace.json; dataflow has
    react_steps.json."""
    if (KB / "system_scratch" / arm / task / "reasoning_trace.json").exists():
        return extract_codeagent(arm, task)
    return extract_dataflow(arm, task)


def extract_dataflow(arm, task):
    """The uniform eval object. Returns dict or None if no usable trace."""
    ctx, trace = last_context(arm, task)
    if ctx is None:
        return None
    mode = "delta" if "# Agent Events" in ctx else "latest"
    codes = final_codes(trace)
    rend = renders_delta(ctx) if mode == "delta" else renders_latest(ctx)
    ops, links = final_workflow(arm, task)
    if not ops:  # no surviving workflow -> fall back to ops we have code for
        ops = [{"operatorID": o} for o in codes]
        links = []
    order = topo_order(ops, links)
    meta = {o.get("operatorID"): o for o in ops}
    entries = []
    for oid in order:
        code, accepted = codes.get(oid, ("", True))
        r = rend.get(oid)
        flags = []
        if not code:
            flags.append("code_missing")
        elif not accepted:
            flags.append("code_unexecuted")
        if r is None:
            flags.append("result_missing")
        elif "[ERROR]" in r:
            flags.append("exec_error")
        o = meta.get(oid, {})
        entries.append(dict(id=oid, type=o.get("operatorType", ""),
                            name=(o.get("customDisplayName") or "").strip(),
                            code=code, render=r, flags=flags))
    edges = []
    for l in links:
        s = (l.get("source") or {}).get("operatorID") or l.get("source")
        t = (l.get("target") or {}).get("operatorID") or l.get("target")
        edges.append(f"{s}->{t}")
    return dict(mode=mode, entries=entries, edges=edges)


def dataflow_doc(df):
    parts = [f"Links: {', '.join(df['edges']) if df['edges'] else '(none)'}"]
    for e in df["entries"]:
        head = f"### Operator `{e['id']}`" + (f" ({e['type']}: {e['name'][:60]})" if e["type"] or e["name"] else "")
        body = [head, "Code:", cap(e["code"]) if e["code"] else "[NO CODE RECORDED]"]
        if e["render"] is not None:
            body += ["Latest executed result:", cap(e["render"])]
        else:
            body += ["Latest executed result:", "[RESULT MISSING — operator was never executed/rendered in the trace]"]
        parts.append("\n".join(body))
    doc = "\n\n".join(parts)
    if len(doc) > DOC_CAP:
        doc = doc[: int(DOC_CAP * 0.8)] + "\n...[dataflow doc truncated]...\n" + doc[-int(DOC_CAP * 0.2):]
    return doc


# ------------------------------------------------------------------------ judge

M5_SYS = (
    "You audit the FINAL dataflow (pipeline) built by a data-analysis agent. You "
    "are given every operator's code and its latest executed result. For each "
    "item, classify how the item's ground-truth VALUE(S) relate to this "
    "dataflow, as exactly one status:\n"
    '  "visible"             — the value appears in an executed result '
    "table/output. Accept pure FORMATTING differences (thousand separators, "
    "units, case, scientific notation, trailing zeros) and EXTRA PRECISION: "
    "when the result shows more digits than the ground truth, round the result "
    "to the ground truth's precision before comparing — 7.520031 MATCHES 7.52, "
    "1.3285239e-12 MATCHES 1.329e-12. After that rounding digits must agree "
    "exactly. If an item lists MULTIPLE values, ALL must appear. If the item's "
    "value is a file name/path, the dataflow actually reading that file in "
    "executed code counts as visible.\n"
    '  "computed_not_shown"  — the value appears in NO executed result, but the '
    "code, given the rendered inputs/outputs around it, unambiguously computes "
    "this exact quantity internally (e.g. a fused operator that computes the "
    "count and outputs only a percentage). Do NOT use this status when any "
    "executed result shows a DIFFERENT value for this same quantity — a "
    "different value means the pipeline computed something else.\n"
    '  "absent"              — neither: the value is not shown and the code does '
    "not clearly compute it, or a shown value for this quantity contradicts the "
    "ground truth (4.798 contradicts 4.796; 15383 contradicts 15388).\n"
    "Never base a status on the task statement or comments alone. "
    "CONTRADICTION PRIORITY: first apply the EXTRA PRECISION rounding rule; a "
    "value that matches after rounding is NEVER a contradiction (1.7666e-13 "
    "matches 1.767e-13). Only when a result value STILL differs after that "
    "rounding (4.798 vs 4.796) does it contradict, and then the status is "
    "absent even if the ground-truth number appears elsewhere as incidental "
    "metadata. "
    "Return one verdict per item, echoing the item's id exactly; keep each "
    "explanation to at most 25 words."
)
M6_SYS = (
    "You audit the FINAL dataflow (pipeline) built by a data-analysis agent, "
    "given every operator's code and its latest executed result. For each item "
    "(one processing step of a reference solution), decide whether this dataflow "
    "PERFORMS that step — implemented in its code, possibly fused with other "
    "steps or done in an equivalent way. Whether the resulting values are "
    "correct is irrelevant; judge only whether the step is done. Answer false "
    "if the step is absent from the final dataflow. "
    "Return one verdict per item, echoing the item's id exactly; keep each "
    "explanation to at most 25 words."
)


def lens_schema(lens):
    """Strict structured-output schema; array of verdicts (dynamic ids can't be
    object keys under strict mode), re-keyed by id after parsing."""
    verdict_props = {
        "id": {"type": "string"},
        "operator": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
    }
    if lens == "m5":
        verdict_props["status"] = {"type": "string",
                                   "enum": ["visible", "computed_not_shown", "absent"]}
    else:
        verdict_props["done"] = {"type": "boolean"}
    return {"type": "json_schema", "json_schema": {
        "name": f"{lens}_verdicts", "strict": True, "schema": {
            "type": "object",
            "properties": {"verdicts": {"type": "array", "items": {
                "type": "object", "properties": verdict_props,
                "required": sorted(verdict_props), "additionalProperties": False}}},
            "required": ["verdicts"], "additionalProperties": False}}}


def fmt_items(subs, lens):
    lines = []
    for s in subs:
        if lens == "m5":
            ans = json.dumps(s.get("answer"), ensure_ascii=False)
            if len(ans) > 300:
                ans = ans[:300] + "..."
            lines.append(f"- id {s['id']}: ground-truth value(s) = {ans}   (produced by step: {s.get('step','')[:160]})")
        else:
            lines.append(f"- id {s['id']}: step = {s.get('step','')[:220]}")
    return "\n".join(lines)


def judge_call(client, model, task_def, doc, subs, lens):
    sysmsg = M5_SYS if lens == "m5" else M6_SYS
    user = (
        f"Overall task: {task_def.get('query','')}\n\n"
        f"Final dataflow (operators in topological order):\n{doc}\n\n"
        f"Items:\n{fmt_items(subs, lens)}"
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format=lens_schema(lens),
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": user}],
            )
            lst = (json.loads(r.choices[0].message.content) or {}).get("verdicts", [])
            v = {str(item.get("id")): item for item in lst if isinstance(item, dict)}
            missing = [s["id"] for s in subs if s["id"] not in v]
            if missing and attempt < 2:
                continue  # schema guarantees shape but not id coverage; retry
            out = {}
            for s in subs:
                raw = v.get(s["id"], {})
                if lens == "m5":
                    out[s["id"]] = {"status": raw.get("status", "absent"),
                                    "operator": raw.get("operator"),
                                    "explanation": str(raw.get("explanation", ""))[:300]}
                else:
                    out[s["id"]] = {"done": bool(raw.get("done", False)),
                                    "operator": raw.get("operator"),
                                    "explanation": str(raw.get("explanation", ""))[:300]}
            usage = r.usage.total_tokens if r.usage else 0
            return out, usage
        except Exception as e:
            if attempt == 2:
                print(f"    [judge ERR {lens}]: {e}", file=sys.stderr)
                break
    fail = {"status": "absent", "operator": None, "explanation": "judge error"} if lens == "m5" \
        else {"done": False, "operator": None, "explanation": "judge error"}
    return {s["id"]: dict(fail) for s in subs}, 0


# --------------------------------------------------------------------- per-task

def run_task(client, model, W, arm, task, force=False, model_m5=None, skip_m6=False):
    outp = KB / "system_scratch" / arm / task / "judge_m5m6.json"
    if outp.exists() and not force:
        return jload(outp)
    subs = [dict(id=s.get("id"), step=s.get("step", ""), answer=s.get("answer"))
            for s in W[task].get("subtasks", [])]
    if not subs:
        return None
    df = extract_units(arm, task)
    if df is None or not df["entries"]:
        return None
    doc = dataflow_doc(df)
    v5, u5 = judge_call(client, model_m5 or model, W[task], doc, subs, "m5")
    if skip_m6:
        v6, u6 = {s["id"]: {"done": None, "operator": None, "explanation": ""} for s in subs}, 0
    else:
        v6, u6 = judge_call(client, model, W[task], doc, subs, "m6")
    per = {}
    for s in subs:
        st = v5[s["id"]]["status"]
        per[s["id"]] = {
            "m5_status": st,
            "m5_extracted": st == "visible",
            "m5_operator": v5[s["id"]]["operator"],
            "m5_explanation": v5[s["id"]]["explanation"],
            "m6_done": v6[s["id"]]["done"],
            "m6_operator": v6[s["id"]]["operator"],
            "m6_explanation": v6[s["id"]]["explanation"],
        }
    n = len(subs)
    res = dict(
        arm=arm, task=task, mode=df["mode"],
        n_operators=len(df["entries"]), n_subtasks=n,
        op_flags={e["id"]: e["flags"] for e in df["entries"] if e["flags"]},
        judge_model=model, judge_model_m5=model_m5 or model, tokens=u5 + u6,
        per_subtask=per,
        m5=sum(1 for s in subs if per[s["id"]]["m5_status"] == "visible") / n,
        m6=(sum(1 for s in subs if per[s["id"]]["m6_done"]) / n) if not skip_m6 else None,
        m7=sum(1 for s in subs if per[s["id"]]["m5_status"] in ("visible", "computed_not_shown")) / n,
    )
    tmp = outp.with_suffix(".tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, outp)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--skip-m6", action="store_true",
                    help="judge only the M5 value lens (M5/M7); m6 recorded as null")
    ap.add_argument("--model-m5", default=None,
                    help="judge model for the M5 (value) lens; defaults to --model. "
                         "M5 needs numeric care — a stronger judge (gpt-4o) is recommended.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry", action="store_true", help="print the dataflow doc for the first (arm, task) and exit")
    a = ap.parse_args()

    load_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    W = load_workload()

    jobs = []
    for arm in a.arms:
        tasks = a.tasks or sorted(
            d.name for d in (KB / "system_scratch" / arm).iterdir()
            if d.is_dir() and d.name in W
            and ((d / "react_steps.json").exists() or (d / "reasoning_trace.json").exists()))
        jobs += [(arm, t) for t in tasks]

    if a.dry:
        arm, t = jobs[0]
        df = extract_units(arm, t)
        print(f"# {arm} / {t}  mode={df['mode']}  ops={len(df['entries'])}")
        for e in df["entries"]:
            print(f"#   {e['id']} flags={e['flags']}")
        print(dataflow_doc(df))
        return

    done = {}
    def work(j):
        arm, t = j
        try:
            r = run_task(client, a.model, W, arm, t, force=a.force, model_m5=a.model_m5, skip_m6=a.skip_m6)
        except Exception as e:
            print(f"  [ERR {arm}/{t}] {e}", file=sys.stderr)
            r = None
        return (arm, t, r)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (arm, t, r) in enumerate(ex.map(work, jobs)):
            if r:
                done.setdefault(arm, []).append(r)
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(jobs)}", flush=True)

    print(f"\n{'arm':52s} {'n':>4s} {'M5':>7s} {'M6':>7s} {'M7':>7s} {'tok/task':>9s}")
    print("-" * 92)
    for arm in a.arms:
        rs = done.get(arm, [])
        if not rs:
            print(f"{arm:52s} {'0':>4s} {'—':>7s} {'—':>7s} {'—':>7s}")
            continue
        m5 = sum(r["m5"] for r in rs) / len(rs)
        m6v=[r["m6"] for r in rs if r.get("m6") is not None]
        m6 = sum(m6v)/len(m6v) if m6v else float('nan')
        m7 = sum(r.get("m7", r["m5"]) for r in rs) / len(rs)
        tok = sum(r["tokens"] for r in rs) / len(rs)
        print(f"{arm:52s} {len(rs):4d} {m5:7.3f} {m6:7.3f} {m7:7.3f} {tok:9.0f}")


if __name__ == "__main__":
    main()

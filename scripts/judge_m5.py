#!/usr/bin/env python3
"""
M5 — task-completion judge over the agent's FINAL dataflow (one call per task).

Replaces the old M5/M6/M7 trio with a single question: for each ground-truth
step of a reference solution, does the agent's final dataflow PERFORM that step?

  computed : the dataflow performs the step — either its result is present in an
             operator's executed result, or the operator code carries it out
             internally (fused/equivalent implementations count).
  absent   : the dataflow does not perform the step at all.

Value correctness is deliberately NOT judged. A step done with the wrong column
or wrong filter is still "computed"; the expected value is supplied only as
evidence that the step ran. (The old M5 penalised value mismatches and M7
credited fused intermediates; both distinctions are dropped.)

Eval object (shared extractor, uniform across delta/latest context modes): the
agent's FINAL dataflow reconstructed from react_steps.json + workflow.json —
  * operators : workflow.json's surviving set, topological order
  * code      : last ACCEPTED createOrModifyOperator edit; falls back to last
                submitted (flag code_unexecuted)
  * result    : the operator's LAST render in the trace (delta: last render in
                the final cumulative context; latest: its block in the final
                "# Current Dataflow" snapshot)
  * flags     : result_missing, exec_error

Cache: system_scratch/<arm>/<task>/judge_m5.json (version 1). Deliberately a
DIFFERENT file from the old judge_m5m6.json — never mix caches across judge
versions.

Run: .venv/bin/python scripts/judge_m5.py --arms A B [--tasks ...] [--workers N]
"""
import argparse, json, re, glob, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
JUDGE_MODEL_DEFAULT = "gpt-5.4-mini"
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

VERSION = 1

M5_SYS = (
    "You are evaluating how completely a data-analysis agent solved a task.\n\n"
    "The agent answers a data-science question by building a DATAFLOW: a pipeline "
    "of operators, each with source code and an executed result. A reference "
    "solution for the same question decomposes it into GROUND-TRUTH STEPS — the "
    "steps that are necessary to arrive at the final answer.\n\n"
    "Your job: for each ground-truth step, decide whether the agent's final "
    "dataflow PERFORMS that step.\n\n"
    "You are given:\n"
    "  1. TASK — the question the agent was asked.\n"
    "  2. FINAL DATAFLOW — every operator in the agent's finished pipeline, in "
    "topological order, with its source code and its latest executed result.\n"
    "  3. GROUND-TRUTH STEPS — the necessary steps of a reference solution. Each "
    "has a description and, where applicable, the value that step should produce. "
    "The value is EVIDENCE that the step ran, not a correctness test.\n\n"
    "Classify each ground-truth step as exactly one of:\n\n"
    '  "computed" — the dataflow performs this step. Either\n'
    "      (a) the step's result is present in an operator's executed result, or\n"
    "      (b) the operator code carries out this step internally, even when the "
    "result is never displayed — for example a fused operator that computes a "
    "count and outputs only a percentage derived from it.\n"
    "    An equivalent or fused implementation counts. The step does not need to "
    "be its own operator.\n\n"
    '  "absent" — the dataflow does not perform this step at all. Nothing in the '
    "code carries it out and no result reflects it.\n\n"
    "Rules:\n"
    "  - Judge only whether the step is PERFORMED. Whether the agent's number is "
    "right is irrelevant — a step done with the wrong column, wrong filter, or "
    'wrong value is still "computed".\n'
    "  - Judge from the code and the executed results only. Never infer that a "
    "step happened from the task statement, an operator summary, or a code "
    "comment.\n"
    "  - When matching a step's expected value against a result, ignore "
    "formatting (thousand separators, units, case, scientific notation, trailing "
    "zeros) and extra precision.\n"
    "  - If a step reads a named file, an operator whose executed code actually "
    "reads that file performs it.\n\n"
    "Return one verdict per ground-truth step, echoing its id exactly. Keep each "
    "explanation to at most 25 words, and cite the operator that performs the step."
)

SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "m5_verdicts", "strict": True, "schema": {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {"type": "string", "enum": ["computed", "absent"]},
                "operator": {"type": ["string", "null"]},
                "explanation": {"type": "string"},
            },
            "required": ["id", "status", "operator", "explanation"],
            "additionalProperties": False}}},
        "required": ["verdicts"], "additionalProperties": False}}}


def fmt_steps(subs):
    lines = []
    for s in subs:
        line = f"- id {s['id']}: {s.get('step','')[:220]}"
        if s.get("answer") is not None:
            v = json.dumps(s["answer"], ensure_ascii=False)
            if len(v) > 300:
                v = v[:300] + "..."
            line += f"   (expected value: {v})"
        lines.append(line)
    return "\n".join(lines)


def judge_call(client, model, task_def, doc, subs):
    user = (
        f"TASK:\n{task_def.get('query','')}\n\n"
        f"FINAL DATAFLOW (operators in topological order):\n{doc}\n\n"
        f"GROUND-TRUTH STEPS:\n{fmt_steps(subs)}"
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, response_format=SCHEMA,
                messages=[{"role": "system", "content": M5_SYS},
                          {"role": "user", "content": user}],
            )
            lst = (json.loads(r.choices[0].message.content) or {}).get("verdicts", [])
            v = {str(x.get("id")): x for x in lst if isinstance(x, dict)}
            # the schema fixes the shape but not id coverage — retry on gaps
            if [s["id"] for s in subs if s["id"] not in v] and attempt < 2:
                continue
            out = {s["id"]: {
                "status": v.get(s["id"], {}).get("status", "absent"),
                "operator": v.get(s["id"], {}).get("operator"),
                "explanation": str(v.get(s["id"], {}).get("explanation", ""))[:300],
            } for s in subs}
            return out, (r.usage.total_tokens if r.usage else 0)
        except Exception as e:
            if attempt == 2:
                print(f"    [judge ERR]: {e}", file=sys.stderr)
    return ({s["id"]: {"status": "absent", "operator": None,
                       "explanation": "judge error"} for s in subs}, 0)


# --------------------------------------------------------------------- per-task

def run_task(client, model, W, arm, task, force=False):
    outp = KB / "system_scratch" / arm / task / "judge_m5.json"
    if outp.exists() and not force:
        cached = jload(outp)
        if cached and cached.get("version") == VERSION:
            return cached
    subs = [dict(id=s.get("id"), step=s.get("step", ""), answer=s.get("answer"))
            for s in W[task].get("subtasks", [])]
    if not subs:
        return None
    df = extract_units(arm, task)
    if df is None or not df["entries"]:
        return None
    v, tok = judge_call(client, model, W[task], dataflow_doc(df), subs)
    n = len(subs)
    res = dict(
        arm=arm, task=task, version=VERSION, judge_model=model, mode=df["mode"],
        n_operators=len(df["entries"]), n_steps=n, tokens=tok,
        op_flags={e["id"]: e["flags"] for e in df["entries"] if e["flags"]},
        per_step={s["id"]: v[s["id"]] for s in subs},
        m5=sum(1 for s in subs if v[s["id"]]["status"] == "computed") / n,
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry", action="store_true",
                    help="print the rendered prompt for the first (arm, task) and exit")
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
        subs = [dict(id=s.get("id"), step=s.get("step", ""), answer=s.get("answer"))
                for s in W[t].get("subtasks", [])]
        print("=" * 30, "SYSTEM", "=" * 30)
        print(M5_SYS)
        print("=" * 30, "USER", "=" * 32)
        print(f"TASK:\n{W[t].get('query','')}\n\n"
              f"FINAL DATAFLOW (operators in topological order):\n{dataflow_doc(df)}\n\n"
              f"GROUND-TRUTH STEPS:\n{fmt_steps(subs)}")
        return

    done = {}
    def work(j):
        arm, t = j
        try:
            return (arm, t, run_task(client, a.model, W, arm, t, force=a.force))
        except Exception as e:
            print(f"  [ERR {arm}/{t}] {e}", file=sys.stderr)
            return (arm, t, None)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (arm, t, r) in enumerate(ex.map(work, jobs)):
            if r:
                done.setdefault(arm, []).append(r)
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(jobs)}", flush=True)

    print(f"\n{'arm':52s} {'n':>4s} {'M5':>7s} {'tok/task':>9s}")
    print("-" * 76)
    for arm in a.arms:
        rs = done.get(arm, [])
        if not rs:
            print(f"{arm:52s} {0:4d} {'—':>7s}")
            continue
        print(f"{arm:52s} {len(rs):4d} "
              f"{sum(r['m5'] for r in rs)/len(rs):7.3f} "
              f"{sum(r['tokens'] for r in rs)/len(rs):9.0f}")


if __name__ == "__main__":
    main()

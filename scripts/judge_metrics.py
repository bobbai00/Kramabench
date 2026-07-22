#!/usr/bin/env python3
"""
M3 / M4 — chunked LLM-judge metrics over the agent's own rendered context.

Design (finalized 2026-07-21):
  * Source: the LAST react step's `inputMessages` (byte-exact record of what the
    agent saw; in DELTA mode it contains the whole event history — verified
    38/38 identical to the union over all steps).
  * Chunking:
      - DELTA  : one chunk per `## Agent Event N` (carries summary + code +
                 result tables). The pre-event preamble (# User Task) is
                 DROPPED so values quoted in the question never count.
      - LATEST : one chunk per `### Operator` block of the final snapshot, in
                 topological order, with the operator's real code attached from
                 `toolCalls[].input.code` (LATEST renders no code).
  * Judge: ONE call per chunk per lens; all gold subtasks listed; verdicts
    returned as JSON keyed by subtask-ID (kills list-misalignment); binary
    yes/no only (judges are reliable on yes/no, wobbly on grades); temperature 0.
      - M3 lens: does THIS excerpt contain the subtask's evidence VALUE(s)?
        (code + result tables; format-tolerant; lists strict: ALL values.)
      - M4 lens: does THIS excerpt show the agent PERFORMING the step?
        (action/code/summary; correctness of values irrelevant.)
  * Aggregation: subtask = yes iff ANY chunk yes; task score = fraction of
    subtasks yes (the spectrum comes from counting, not from graded judgments).
    M4 has two flavors: process (any event) and deliverable (only chunks whose
    operator survives in the final workflow).
  * Failure modes (Bob's taxonomy), per failed task:
      mode1 = step missing (M4<1) ; mode2 = steps done, value absent
      (M4=1, M3<1) ; mode3 = everything present, still failed (M4=1, M3=1).

Results cache: system_scratch/<arm>/<task>/judge_m3m4.json  (skip unless --force).

Run directly:
  .venv/bin/python scripts/judge_metrics.py --arms A B --tasks-file F [--lens both]
or via kb.py:
  ./kb.py judge --sut A B --tasks-file F
"""
import argparse, json, re, glob, os, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
SK = ["f1", "success", "rae_score", "llm_paraphrase", "f1_approximate"]
JUDGE_MODEL_DEFAULT = "gpt-4o-mini"
CHUNK_CAP = 12000  # chars per chunk fed to the judge (head 9k + tail 3k)


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


# ---------------------------------------------------------------- trace parsing

def last_context(arm, task):
    """Concatenated content of the last react step's inputMessages."""
    d = jload(KB / "system_scratch" / arm / task / "react_steps.json")
    if not d:
        return None, None
    steps = [s for s in d.get("steps", []) if s.get("inputMessages")]
    if not steps:
        return None, d
    txt = "\n".join(str(m.get("content", "")) for m in steps[-1]["inputMessages"])
    return txt, d


def op_codes(trace):
    """operatorId -> last submitted code (from the tool-call log)."""
    codes = {}
    for s in (trace or {}).get("steps", []):
        for tc in (s.get("toolCalls") or []):
            if tc.get("toolName") == "createOrModifyOperator":
                inp = tc.get("input") or {}
                if inp.get("operatorId"):
                    codes[inp["operatorId"]] = inp.get("code", "")
            elif tc.get("toolName") == "deleteOperator":
                codes.pop((tc.get("input") or {}).get("operatorId"), None)
    return codes


def final_workflow_ops(arm, task):
    w = jload(KB / "system_scratch" / arm / task / "workflow.json") or {}
    wf = w.get("workflow") or {}
    ops = wf.get("operators", []) or []
    links = wf.get("links", []) or []
    return ops, links


def topo_order(ops, links):
    ids = [o.get("operatorID") for o in ops]
    indeg = {i: 0 for i in ids}
    adj = {i: [] for i in ids}
    for l in links:
        s = (l.get("source") or {}).get("operatorID") or l.get("source")
        t = (l.get("target") or {}).get("operatorID") or l.get("target")
        if s in adj and t in indeg:
            adj[s].append(t); indeg[t] += 1
    out, q = [], [i for i in ids if indeg[i] == 0]
    while q:
        n = q.pop(0); out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    return out + [i for i in ids if i not in out]  # cycles/orphans last


def dag_overview(ops, links):
    parts = []
    for o in ops:
        nm = (o.get("customDisplayName") or "").strip()
        parts.append(f"{o.get('operatorID')}({o.get('operatorType')}: {nm[:60]})")
    edges = []
    for l in links:
        s = (l.get("source") or {}).get("operatorID") or l.get("source")
        t = (l.get("target") or {}).get("operatorID") or l.get("target")
        edges.append(f"{s}->{t}")
    return "Operators: " + "; ".join(parts) + ("\nLinks: " + ", ".join(edges) if edges else "")


def cap(txt):
    if len(txt) <= CHUNK_CAP:
        return txt
    return txt[:9000] + "\n...[truncated]...\n" + txt[-3000:]


def chunk_delta(ctx):
    """One chunk per agent event; preamble (# User Task) dropped."""
    parts = re.split(r"(?=## Agent Event \d+)", ctx)
    chunks = []
    for p in parts[1:]:  # parts[0] = preamble -> dropped by design
        if "Action" not in p and "result:" not in p and "Observation" not in p:
            continue  # pure-thought event: nothing to audit
        opids = set(re.findall(r"operatorId:\s*`?([\w-]+)`?", p))
        opids |= set(re.findall(r"operator\s+`?([\w-]+)`?\s+(?:added|updated)", p))
        label = (re.match(r"## Agent Event (\d+)", p) or [None, "?"])[1]
        chunks.append(dict(label=f"event-{label}", opids=sorted(opids), text=cap(p)))
    return chunks


def chunk_latest(ctx, arm, task, trace):
    """One chunk per operator block of the final snapshot, topo order, code attached."""
    i = ctx.find("# Current Dataflow")
    snap = ctx[i:] if i >= 0 else ctx
    blocks = re.split(r"(?=### Operator )", snap)
    per_op = {}
    for b in blocks[1:]:
        m = re.match(r"### Operator `?([\w-]+)`?", b)
        if m:
            per_op[m.group(1)] = b
    codes = op_codes(trace)
    ops, links = final_workflow_ops(arm, task)
    chunks = []
    for oid in topo_order(ops, links):
        if oid not in per_op:
            continue
        body = per_op[oid]
        code = codes.get(oid, "")
        if code:
            body += f"\nCode of `{oid}`:\n{code}"
        chunks.append(dict(label=f"op-{oid}", opids=[oid], text=cap(body)))
    return chunks


# ---------------------------------------------------------------- judge

M3_SYS = (
    "You audit one excerpt of a data-analysis agent's context. For each item, decide "
    "whether THIS EXCERPT contains the item's evidence VALUE(S) — visible in a result "
    "table, schema line, sample, or computed output, or produced by code shown here. "
    "Accept formatting differences (rounding to ~3+ significant figures, thousand "
    "separators, units, case). If an item lists MULTIPLE values, answer true only if "
    "ALL of them appear. Answer false when the value itself is absent, even if the "
    "excerpt looks related. Reply ONLY with JSON: "
    '{"verdicts": {"<item-id>": true|false, ...}} covering every item.'
)
M4_SYS = (
    "You audit one excerpt of a data-analysis agent's context. For each item (a "
    "processing step), decide whether THIS EXCERPT shows the agent PERFORMING that "
    "step — via its action code, operator summary, or an output that clearly results "
    "from doing the step. Judge only whether the step was performed here; whether the "
    "values are correct is irrelevant. Answer false if this excerpt does not itself "
    "evidence the step. Reply ONLY with JSON: "
    '{"verdicts": {"<item-id>": true|false, ...}} covering every item.'
)


def fmt_items(subs, lens):
    lines = []
    for s in subs:
        if lens == "m3":
            ans = json.dumps(s.get("answer"), ensure_ascii=False)
            if len(ans) > 300:
                ans = ans[:300] + "..."
            lines.append(f"- id {s['id']}: evidence value(s) = {ans}   (produced by step: {s.get('step','')[:160]})")
        else:
            lines.append(f"- id {s['id']}: step = {s.get('step','')[:220]}")
    return "\n".join(lines)


def judge_chunk(client, model, task_def, dag, chunk, subs, lens):
    sysmsg = M3_SYS if lens == "m3" else M4_SYS
    user = (
        f"Overall task: {task_def.get('query','')}\n\n"
        f"Pipeline overview:\n{dag}\n\n"
        f"Excerpt ({chunk['label']}):\n---\n{chunk['text']}\n---\n\n"
        f"Items:\n{fmt_items(subs, lens)}"
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": user}],
            )
            v = (json.loads(r.choices[0].message.content) or {}).get("verdicts", {})
            out = {}
            for s in subs:
                raw = v.get(s["id"], v.get(str(s["id"]), False))
                out[s["id"]] = str(raw).strip().lower() in ("true", "yes", "1")
            usage = r.usage.total_tokens if r.usage else 0
            return out, usage
        except Exception as e:
            if attempt == 2:
                print(f"    [judge ERR {lens} {chunk['label']}]: {e}", file=sys.stderr)
                return {s["id"]: False for s in subs}, 0
    return {s["id"]: False for s in subs}, 0


# ---------------------------------------------------------------- per-task

def run_task(client, model, W, arm, task, lenses, force=False, verbose=False):
    outp = KB / "system_scratch" / arm / task / "judge_m3m4.json"
    if outp.exists() and not force:
        return jload(outp)
    ctx, trace = last_context(arm, task)
    subs = [dict(id=s.get("id"), step=s.get("step", ""), answer=s.get("answer"))
            for s in W[task].get("subtasks", [])]
    if ctx is None or not subs:
        return None
    mode = "delta" if "# Agent Events" in ctx else "latest"
    chunks = chunk_delta(ctx) if mode == "delta" else chunk_latest(ctx, arm, task, trace)
    if not chunks:
        return None
    ops, links = final_workflow_ops(arm, task)
    surviving = {o.get("operatorID") for o in ops}
    dag = dag_overview(ops, links)
    per = {s["id"]: {"m3": False, "m4_process": False, "m4_deliverable": False,
                     "m3_chunks": [], "m4_chunks": []} for s in subs}
    tokens = 0
    for ch in chunks:
        ch_deliv = (not ch["opids"]) or any(o in surviving for o in ch["opids"])
        for lens in lenses:
            v, u = judge_chunk(client, model, W[task], dag, ch, subs, lens)
            tokens += u
            for sid, yes in v.items():
                if not yes:
                    continue
                if lens == "m3":
                    per[sid]["m3"] = True; per[sid]["m3_chunks"].append(ch["label"])
                else:
                    per[sid]["m4_process"] = True; per[sid]["m4_chunks"].append(ch["label"])
                    if ch_deliv:
                        per[sid]["m4_deliverable"] = True
            if verbose:
                yes = [k for k, x in v.items() if x]
                print(f"    {task} {ch['label']} [{lens}] yes: {[y.split('-')[-1] for y in yes]}")
    n = len(subs)
    res = dict(
        arm=arm, task=task, mode=mode, n_chunks=len(chunks), n_subtasks=n,
        judge_model=model, tokens=tokens, lenses=lenses, per_subtask=per,
        m3=sum(per[s]["m3"] for s in per) / n,
        m4_process=sum(per[s]["m4_process"] for s in per) / n,
        m4_deliverable=sum(per[s]["m4_deliverable"] for s in per) / n,
    )
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w"), indent=1)
    return res


# ---------------------------------------------------------------- aggregation

def answer_score(arm, task):
    ev = jload(KB / "system_scratch" / arm / task / "evaluation.json")
    if not ev:
        return None
    for k in SK:
        if isinstance(ev.get(k), (int, float)):
            return float(ev[k])
    return None


def failure_mode(r):
    """Bob's taxonomy for a FAILED task, from its M3/M4 coverage."""
    if r["m4_process"] < 0.999:
        return "mode1-step-missing"
    if r["m3"] < 0.999:
        return "mode2-value-absent"
    return "mode3-had-all-still-failed"


def summarize(arms, results, tasks):
    print(f"\n{'='*74}\nSUMMARY  (binary judge verdicts; task score = % of subtasks covered)\n{'='*74}")
    header = f"{'arm':44s} {'n':>3s} {'M3':>6s} {'M4proc':>7s} {'M4deliv':>8s}"
    print(header); print("-" * len(header))
    for arm in arms:
        rs = [results[(arm, t)] for t in tasks if results.get((arm, t))]
        if not rs:
            continue
        print(f"{arm:44s} {len(rs):3d} {st.mean([r['m3'] for r in rs]):6.3f} "
              f"{st.mean([r['m4_process'] for r in rs]):7.3f} "
              f"{st.mean([r['m4_deliverable'] for r in rs]):8.3f}")
    if len(arms) == 2:
        a, b = arms
        m = [t for t in tasks if results.get((a, t)) and results.get((b, t))]
        for key, nm in [("m3", "M3"), ("m4_process", "M4proc"), ("m4_deliverable", "M4deliv")]:
            d = [results[(b, t)][key] - results[(a, t)][key] for t in m]
            up = sum(1 for x in d if x > 0.05); dn = sum(1 for x in d if x < -0.05)
            print(f"  Δ {nm:8s} (matched {len(m)}): {st.mean(d):+.3f}   {up}up/{dn}dn/{len(m)-up-dn}flat")
    # failure-mode split per arm
    print(f"\nFailure modes (failed tasks only; pass = answer score >= 0.9):")
    for arm in arms:
        modes = {}
        for t in tasks:
            r = results.get((arm, t))
            if not r:
                continue
            a = answer_score(arm, t)
            if a is None or a >= 0.9:
                continue
            modes[failure_mode(r)] = modes.get(failure_mode(r), 0) + 1
        tot = sum(modes.values()) or 1
        parts = "  ".join(f"{k}={v} ({v/tot:.0%})" for k, v in sorted(modes.items()))
        print(f"  {arm}: n_failed={tot}  {parts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks-file")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--lens", choices=["m3", "m4", "both"], default="both")
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="re-judge even if cached")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    W = load_workload()
    tasks = a.tasks or [t for t in open(a.tasks_file).read().split() if t]
    tasks = [t for t in tasks if t in W][: a.limit or None]
    lenses = ["m3", "m4"] if a.lens == "both" else [a.lens]
    jobs = [(arm, t) for arm in a.arms for t in tasks]
    results = {}
    print(f"[judge] {len(jobs)} (arm,task) jobs, lenses={lenses}, model={a.judge_model}")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_task, client, a.judge_model, W, arm, t, lenses,
                          a.force, a.verbose): (arm, t) for arm, t in jobs}
        done = 0
        for f in list(futs):
            r = f.result()
            arm, t = futs[f]
            done += 1
            if r:
                results[(arm, t)] = r
            if done % 10 == 0:
                print(f"[judge] {done}/{len(jobs)} done")
    summarize(a.arms, results, tasks)


if __name__ == "__main__":
    main()

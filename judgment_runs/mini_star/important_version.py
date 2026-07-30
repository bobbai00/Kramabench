#!/usr/bin/env python3
"""What deserves to be called an IMPORTANT version?

Rather than assume (errors? peaks?), let the traces vote. A version is important
if losing it demonstrably cost the agent something. Three objective signatures,
all measurable from existing traces:

  S1 REBUILD  — the agent submits code that closely matches a version it had
                already written earlier (same op or a different one). It rebuilt
                what it lost -> that earlier version was important.
  S2 DESTROY  — an operator's result regressed materially (non-empty -> empty, or
                a >90% row collapse that is NOT an aggregation-to-1) and the
                agent then had to edit it again. The pre-regression DATA version
                was important.
  S3 ORPHAN   — an operator produced a healthy result and was then deleted or
                abandoned, and a later operator's code references the same file
                or column set. Its result was a probe whose finding got evicted.

For each signature we report frequency and its association with failure, so the
rule can be aimed at whatever actually predicts trouble.
"""
import json, glob, re, sys, difflib, statistics as st
from collections import defaultdict

ARMS = sys.argv[1:] or (
    [f"DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate{r}" for r in range(5)]
    + [f"DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate{r}" for r in range(5)]
)
OP_BLOCK = re.compile(r"### Operator `([^`]+)` \([^)]*\)(.*?)(?=\n### |\Z)", re.S)
SHAPE = re.compile(r"Output Table:\s*([\d,]+)\s*rows?,\s*(\d+)\s*cols?")
ATM = {"numeric_exact": "success", "string_exact": "success", "list_exact": "f1",
       "numeric_approximate": "rae_score", "list_approximate": "f1_approximate",
       "string_approximate": "llm_paraphrase"}


def norm_code(c: str) -> str:
    c = re.sub(r"#.*", "", c or "")
    c = re.sub(r"\s+", " ", c)
    return c.strip()


def score(task_dir):
    try:
        ev = json.load(open(f"{task_dir}/evaluation.json"))
        gt = json.load(open(f"{task_dir}/ground_truth.json"))
    except Exception:
        return None
    k = ATM.get(gt.get("answer_type") or "")
    v = ev.get(k) if k and isinstance(ev.get(k), (int, float)) else None
    if v is None:
        c = [float(ev[m]) for m in ATM.values() if isinstance(ev.get(m), (int, float))]
        v = max(c) if c else None
    return v


def analyse(path):
    doc = json.load(open(path))
    steps = doc.get("steps", [])
    # code submissions in order: (step_idx, opId, code)
    subs = []
    deletes = []
    for i, s in enumerate(steps):
        if s.get("role") != "agent":
            continue
        for tc in s.get("toolCalls") or []:
            inp = tc.get("input") or {}
            op = inp.get("operatorId")
            if not op:
                continue
            if tc.get("toolName") == "deleteOperator":
                deletes.append((i, op))
            elif inp.get("code"):
                subs.append((i, op, norm_code(inp["code"])))
    # rendered per-step shapes: step -> {op: (rows, cols, errored)}
    shapes = []
    for s in steps:
        ims = s.get("inputMessages")
        if not ims:
            continue
        ctx = "\n".join(str(m.get("content") or "") for m in ims)
        cur = {}
        for m in OP_BLOCK.finditer(ctx):
            op, body = m.group(1), m.group(2)
            res = body.split("Operator code:")[0]
            err = "[ERROR]" in res
            sm = SHAPE.search(res)
            rows = int(sm.group(1).replace(",", "")) if sm else None
            cols = int(sm.group(2)) if sm else None
            cur[op] = (rows, cols, err)
        shapes.append(cur)

    # ---- S1 REBUILD: later code closely matches an earlier, superseded version
    rebuild = 0
    rebuild_cross_op = 0
    for j in range(len(subs)):
        sj, opj, cj = subs[j]
        if len(cj) < 60:
            continue
        for i in range(j):
            si, opi, ci = subs[i]
            if len(ci) < 60 or sj - si < 2:
                continue
            # superseded: op i was edited again after i, or deleted
            later_edit = any(op == opi and s > si for s, op, _ in subs[i + 1:])
            deleted = any(op == opi and s > si for s, op in deletes)
            if not (later_edit or deleted):
                continue
            if difflib.SequenceMatcher(None, ci, cj).ratio() >= 0.85:
                rebuild += 1
                if opi != opj:
                    rebuild_cross_op += 1
                break

    # ---- S2 DESTROY: material result regression followed by another edit
    destroy = 0
    for op in {o for cur in shapes for o in cur}:
        seq = [cur.get(op) for cur in shapes]
        prev = None
        for k, v in enumerate(seq):
            if v is None or v[0] is None:
                continue
            if prev is not None:
                pr, _, perr = prev
                cr, _, cerr = v
                if pr and pr > 0 and not perr:
                    collapsed = (cr == 0) or (cr is not None and cr <= max(1, pr * 0.1) and cr > 1)
                    if collapsed or (cerr and not perr):
                        # did the agent then edit it again?
                        if any(o == op and s >= k for s, o, _ in subs):
                            destroy += 1
                            break
            prev = v

    # ---- S3 ORPHAN: healthy op deleted, its file referenced again later
    orphan = 0
    for di, dop in deletes:
        # was it healthy before deletion?
        healthy = any(dop in cur and cur[dop][2] is False and (cur[dop][0] or 0) > 0
                      for cur in shapes[:max(di, 1)])
        if not healthy:
            continue
        # find the file literals it used
        files = set()
        for s, op, c in subs:
            if op == dop:
                files |= set(re.findall(r"['\"]([^'\"]+\.(?:csv|xlsx|txt|json|cdf|npz|dat|tle|lst))['\"]", c))
        if not files:
            continue
        if any(s > di and (files & set(re.findall(r"['\"]([^'\"]+\.\w+)['\"]", c))) for s, op, c in subs):
            orphan += 1
    return dict(rebuild=rebuild, rebuild_cross_op=rebuild_cross_op, destroy=destroy,
                orphan=orphan, n_subs=len(subs), n_steps=len(shapes))


rows = []
for arm in ARMS:
    for f in glob.glob(f"system_scratch/{arm}/*/react_steps.json"):
        d = "/".join(f.split("/")[:-1])
        sc = score(d)
        if sc is None:
            continue
        try:
            r = analyse(f)
        except Exception:
            continue
        r["pass"] = 1 if sc >= 0.9 else 0
        r["task"] = f.split("/")[-2]
        rows.append(r)

n = len(rows)
print(f"task-runs analysed: {n}\n")
print(f"{'signature':<34}{'runs w/ it':>11}{'rate':>8}{'pass|has':>10}{'pass|not':>10}{'delta':>8}")
print("-" * 82)
for key, label in (("rebuild", "S1 rebuilt a lost code version"),
                   ("destroy", "S2 destroyed a good data version"),
                   ("orphan", "S3 orphaned a healthy probe")):
    has = [r for r in rows if r[key]]
    non = [r for r in rows if not r[key]]
    ph = st.mean([r["pass"] for r in has]) * 100 if has else float("nan")
    pn = st.mean([r["pass"] for r in non]) * 100 if non else float("nan")
    print(f"{label:<34}{len(has):>11}{len(has)/n*100:>7.1f}%{ph:>9.1f}%{pn:>9.1f}%{ph-pn:>+8.1f}")

xo = sum(r["rebuild_cross_op"] for r in rows)
print(f"\n  S1 detail: {sum(r['rebuild'] for r in rows)} rebuild events, {xo} of them onto a DIFFERENT operator")
print(f"  mean code submissions/run: {st.mean([r['n_subs'] for r in rows]):.1f}")

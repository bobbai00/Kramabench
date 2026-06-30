#!/usr/bin/env python3
"""
kb.py — KramaBench CLI for the Texera DataflowAgent benchmark.

One entry point for the whole loop we run by hand: launch systems, run/re-run
specific tasks, re-evaluate, find failures, and read scores + traces. It wraps
the existing tools (evaluate.py, evaluate_dataflow_system.sh,
scripts/list_failed_tasks.py, compute_scores.py) and adds the things that bit
us in practice:

  * auto-exports OPENAI_API_KEY from ./.env  — the llm_paraphrase / f1_approximate
    metrics call gpt-4o-mini, so a missing/dummy key silently corrupts scores.
  * ORACLE mode (--use_truth_subset) on by default — matches the recorded configs.
  * --parallel  — fan workloads/groups out concurrently.
  * a STALL WATCHDOG — gpt-5.x can hang in GENERATING with no client-side
    timeout (e.g. astronomy-hard-8/-11). Any group whose log goes quiet for
    --watchdog-min minutes (and isn't done) is killed so the run can finish; a
    killed task simply keeps its prior score (monotonic, never regresses).
  * --isolate — run each task as its own watchdogged process so one hang can't
    take out the rest of its workload group (the group-kill limitation of
    run_dataflow_tasks.sh).

Commands (run `kb.py <cmd> -h` for details):
  systems                 list available SUT classes
  run        --sut S      run a system over full workloads
  tasks      --sut S      run specific task ids (auto re-evaluates after)
  reeval     --sut S      rebuild bulk cache + rescore (evaluate_dataflow_system.sh)
  failed     --sut S      list failed / score-0 task ids
  rerun-failed --sut S    detect score-0 tasks -> rerun -> reeval -> scores
  scores     [--sut S]    leaderboard / per-SUT scores (compute_scores.py)
  cost       --sut S      LLM cost + token usage totals (by workload/difficulty/task)
  tokens     --sut S      per-step token breakdown + aggregated stats + input-growth curve
  traces     --sut S      per-task traces + react-step/workflow metrics (system_scratch/)

Examples:
  ./kb.py run --sut DataflowSystemGPT54DeltaSchemaConverge --parallel
  ./kb.py tasks --sut DataflowSystemGPT54LatestSchemaConverge --ids "legal-hard-2 astronomy-easy-1" --parallel
  ./kb.py rerun-failed --sut DataflowSystemGPT54DeltaSchemaConverge --parallel
  ./kb.py failed --sut DataflowSystemGPT54DeltaSchemaConverge --zero-only --ids-only
  ./kb.py scores --sut DataflowSystemGPT54DeltaSchemaConverge
  ./kb.py cost --sut DataflowSystemGPT54DeltaSchemaConverge --by workload
  ./kb.py tokens --sut DataflowSystemGPT54DeltaSchemaConverge --task legal-hard-2
  ./kb.py tokens --sut DataflowSystemGPT54LatestSchemaConverge DataflowSystemGPT54DeltaSchemaConverge
  ./kb.py traces --sut DataflowSystemGPT54DeltaSchemaConverge --task legal-hard-2
"""
import argparse, csv, json, os, re, signal, statistics, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent
VENV_PY = KB_ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
WORKLOADS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]

_PROCS = []  # track children for clean Ctrl+C


def load_env():
    """Export OPENAI_API_KEY (and friends) from ./.env so scoring metrics work."""
    envf = KB_ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")  # litellm config refs it; gpt-* don't need it


def _cleanup(*_):
    for p in _PROCS:
        if p.poll() is None:
            p.terminate()
    sys.exit(130)


signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


# ----------------------------- helpers -----------------------------
def workload_of(task_id):
    m = re.match(r"^(.*)-(easy|hard)-\d+$", task_id)
    return m.group(1) if m else task_id.split("-")[0]


def group_by_workload(task_ids):
    g = {}
    for t in task_ids:
        g.setdefault(workload_of(t), []).append(t)
    return g


def workload_task_ids(w):
    p = KB_ROOT / "workload" / f"{w}.json"
    return [t["id"] for t in json.load(open(p))] if p.exists() else []


def list_suts():
    """SUT class names exported from the `systems` package."""
    try:
        out = subprocess.run(
            [PY, "-c", "import systems,inspect; "
                       "print('\\n'.join(n for n in dir(systems) "
                       "if n.endswith('System') or 'System' in n))"],
            cwd=KB_ROOT, capture_output=True, text=True)
        names = [n for n in out.stdout.split() if n and n[0].isupper()]
        return sorted(set(names))
    except Exception:
        return []


def _score(t):
    """Score as float; None / unscored (missing_eval) -> 0.0."""
    s = t.get("score")
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def read_failed(sut, zero_only=False, workload=None):
    """Return list of failed-task dicts via scripts/list_failed_tasks.py --json."""
    r = subprocess.run([PY, "scripts/list_failed_tasks.py", "--sut", sut, "--json"],
                       cwd=KB_ROOT, capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:
        sys.stderr.write(r.stdout + r.stderr)
        return []
    rows = [t for wl in d.values() for t in wl]
    if zero_only:
        rows = [t for t in rows if _score(t) == 0.0]
    if workload:
        rows = [t for t in rows if t.get("workload") == workload]
    return rows


def scores_per_workload(logtext):
    return [float(x) for x in re.findall(r"Total score is: *([0-9.]+)", logtext)]


# ----------------------------- the runner (with watchdog) -----------------------------
def _spawn(sut, workload, task_ids, oracle, logpath):
    cmd = [PY, "evaluate.py", "--sut", sut, "--workload", workload,
           "--no_pipeline_eval", "--verbose"]
    if task_ids:
        cmd += ["--task_id", *task_ids]
    if oracle:
        cmd.append("--use_truth_subset")
    p = subprocess.Popen(cmd, cwd=KB_ROOT, stdout=open(logpath, "w"),
                         stderr=subprocess.STDOUT)
    _PROCS.append(p)
    return p


def _watch_and_wait(jobs, watchdog_min, poll=15):
    """jobs: list of dicts {name, proc, log, start}. Kill any that stall."""
    while any(j["proc"].poll() is None for j in jobs):
        now = time.time()
        for j in jobs:
            p = j["proc"]
            if p.poll() is not None:
                continue
            try:
                done = "Total score is" in Path(j["log"]).read_text()
            except Exception:
                done = False
            if done:
                continue
            try:
                stale = (now - os.path.getmtime(j["log"])) / 60
            except OSError:
                stale = 0
            if stale >= watchdog_min:
                print(f"[kb][watchdog] {j['name']} stalled {stale:.0f}min "
                      f"(no progress, gpt-5.x hang) -> killing; it keeps its prior score")
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
        time.sleep(poll)
    for j in jobs:
        j["exit"] = j["proc"].poll()


def run_groups(sut, groups, oracle=True, parallel=False, watchdog_min=8,
               isolate=False, label="run"):
    """groups: {workload: [task_ids]}  ([] = whole workload).  Returns logdir."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    logdir = KB_ROOT / "logs" / f"kb-{label}-{ts}"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"[kb] {sut} | oracle={oracle} parallel={parallel} isolate={isolate} "
          f"watchdog={watchdog_min}min")
    print(f"[kb] logs -> {logdir.relative_to(KB_ROOT)}")

    # Build the unit list. isolate -> one job per task; else one job per workload.
    units = []  # (name, workload, task_ids)
    for w, tids in groups.items():
        if isolate and tids:
            for t in tids:
                units.append((t, w, [t]))
        else:
            units.append((w, w, tids))

    def launch(name, w, tids):
        lp = logdir / f"{name}.log"
        n = len(tids) if tids else "all"
        print(f"[kb] start {name} ({n} task(s))")
        return {"name": name, "proc": _spawn(sut, w, tids, oracle, lp),
                "log": str(lp), "start": time.time()}

    if parallel:
        jobs = [launch(*u) for u in units]
        _watch_and_wait(jobs, watchdog_min)
    else:
        jobs = []
        for u in units:
            j = launch(*u)
            _watch_and_wait([j], watchdog_min)
            jobs.append(j)

    ok = sum(1 for j in jobs if j.get("exit") == 0)
    print(f"[kb] done: {ok}/{len(jobs)} units exit 0 "
          f"({sum(1 for j in jobs if j.get('exit') not in (0, None))} failed/killed)")
    return logdir


# ----------------------------- commands -----------------------------
def cmd_systems(a):
    suts = list_suts()
    if not suts:
        print("(could not import systems package)")
        return
    print(f"{len(suts)} SUT classes:")
    for s in suts:
        print(f"  {s}")


def cmd_run(a):
    load_env()
    wls = a.workloads.split() if a.workloads else WORKLOADS
    if a.limit:
        groups = {w: workload_task_ids(w)[:a.limit] for w in wls}
        print(f"[kb] --limit {a.limit}: first {a.limit} task(s) per workload (quick smoke)")
    else:
        groups = {w: [] for w in wls}
    run_groups(a.sut, groups, oracle=not a.no_oracle, parallel=a.parallel,
               watchdog_min=a.watchdog_min, isolate=a.isolate, label="run")
    if not a.no_reeval:
        print("\n[kb] rebuild cache + score")
        reeval(a.sut, wls)
        show_scores(a.sut)


def cmd_tasks(a):
    load_env()
    ids = a.ids.split() if a.ids else []
    if not ids:
        sys.exit("provide --ids \"task-1 task-2 ...\"")
    groups = group_by_workload(ids)
    run_groups(a.sut, groups, oracle=not a.no_oracle, parallel=a.parallel,
               watchdog_min=a.watchdog_min, isolate=a.isolate, label="tasks")
    if not a.no_reeval:
        print("\n[kb] re-evaluating affected workloads (partial runs don't write the bulk cache)")
        reeval(a.sut, list(groups.keys()))
        show_scores(a.sut)


def cmd_reeval(a):
    load_env()
    wls = a.workloads.split() if a.workloads else None
    reeval(a.sut, wls)
    show_scores(a.sut)


def cmd_failed(a):
    rows = read_failed(a.sut, zero_only=a.zero_only, workload=a.workload)
    if a.ids_only:
        print(" ".join(t["task_id"] for t in rows))
        return
    for t in sorted(rows, key=lambda x: (x["workload"], x["task_id"])):
        print(f"  {t['task_id']:<30} score={_score(t):.3f} "
              f"reason={t.get('reason',''):<16} ({t.get('answer_type','')})")
    from collections import Counter
    by = Counter(t["workload"] for t in rows)
    print(f"\n{len(rows)} {'score-0' if a.zero_only else 'failed'} tasks: {dict(by)}")


def cmd_rerun_failed(a):
    load_env()
    rows = read_failed(a.sut, zero_only=not a.all_failed)
    ids = [t["task_id"] for t in rows]
    if a.limit:
        ids = ids[:a.limit]
    if not ids:
        print("[kb] nothing to rerun")
        return
    print(f"[kb] rerunning {len(ids)} {'failed' if a.all_failed else 'score-0'} tasks")
    groups = group_by_workload(ids)
    run_groups(a.sut, groups, oracle=not a.no_oracle, parallel=a.parallel,
               watchdog_min=a.watchdog_min, isolate=a.isolate, label="rerunfail")
    print("\n[kb] re-evaluating + scoring")
    reeval(a.sut, list(groups.keys()))
    show_scores(a.sut)


def cmd_scores(a):
    load_env()
    show_scores(a.sut)


# ----------------------------- cost (from stats.json cost_usd) -----------------------------
SCORE_METRICS = {"success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"}


def load_cost_stats(sut):
    """Per-task cost+token records from each system_scratch/<sut>/<task>/stats.json.
    cost = the harness-computed `cost_usd` (litellm pricing, incl. cache-read discounts —
    so gpt-5.4 etc. price correctly, unlike a hardcoded table)."""
    base = KB_ROOT / "system_scratch" / sut
    recs = []
    if not base.is_dir():
        return recs
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        st = _load(d / "stats.json")
        if not st:
            continue
        parts = d.name.rsplit("-", 2)
        recs.append({
            "task_id": d.name,
            "workload": parts[0],
            "difficulty": parts[1] if len(parts) >= 3 else "?",
            "input_tokens": int(st.get("input_tokens", 0) or 0),
            "output_tokens": int(st.get("output_tokens", 0) or 0),
            "total_tokens": int(st.get("total_tokens", 0) or 0),
            "num_steps": int(st.get("num_steps", 0) or 0),
            "cost": float(st.get("cost_usd", 0) or 0),
        })
    return recs


def _first_model(sut):
    base = KB_ROOT / "system_scratch" / sut
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        cfg = _load(d / "config.json")
        if cfg.get("model_type"):
            return cfg["model_type"]
    return "?"


def load_task_success(sut):
    """{task_id: max score across SCORE_METRICS} from the latest measures CSV per workload."""
    rdir = KB_ROOT / "results" / sut
    if not rdir.is_dir():
        return {}
    latest = {}
    for f in sorted(rdir.glob("*_measures_*.csv")):
        latest[f.name.split("_measures_")[0]] = f
    succ = {}
    for f in latest.values():
        try:
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if row.get("metric") in SCORE_METRICS:
                        tid = row.get("task_id")
                        try:
                            v = float(row.get("value", 0) or 0)
                        except (TypeError, ValueError):
                            v = 0.0
                        if tid:
                            succ[tid] = max(succ.get(tid, 0.0), v)
        except Exception:
            pass
    return succ


def cmd_cost(a):
    data = {s: load_cost_stats(s) for s in a.sut}
    data = {s: r for s, r in data.items() if r}
    if not data:
        sys.exit("no stats.json found under system_scratch/<SUT>/ for the given SUT(s)")

    # totals (one row per SUT — works as a comparison when several --sut given)
    print(f"{'SUT':<44}{'tasks':>6}{'total $':>11}{'$/task':>9}{'in tok':>13}{'out tok':>13}{'steps':>8}")
    print("-" * 104)
    for s, recs in data.items():
        n = len(recs); tc = sum(r["cost"] for r in recs)
        ti = sum(r["input_tokens"] for r in recs); to = sum(r["output_tokens"] for r in recs)
        ts = sum(r["num_steps"] for r in recs)
        print(f"{s:<44}{n:>6}{('$%.2f' % tc):>11}{('$%.4f' % (tc / n)):>9}{ti:>13,}{to:>13,}{ts:>8,}")
    print("\n(cost = sum of stats.json cost_usd — litellm pricing, includes cache-read discounts)")

    for s, recs in data.items():
        miss = sum(1 for r in recs if not r["cost"])
        tag = f"  [{miss} task(s) missing cost_usd]" if miss else ""
        print(f"\n=== {s}  (model={_first_model(s)}){tag} ===")
        if a.by == "task":
            top = sorted(recs, key=lambda r: -r["cost"])[:a.top]
            print(f"  top {len(top)} tasks by cost:")
            for r in top:
                print(f"    {r['task_id']:<30} ${r['cost']:.4f}  "
                      f"({r['input_tokens']:,} in / {r['output_tokens']:,} out, {r['num_steps']} steps)")
            continue
        groups = defaultdict(lambda: {"cost": 0.0, "in": 0, "out": 0, "n": 0})
        for r in recs:
            g = groups[r[a.by]]
            g["cost"] += r["cost"]; g["in"] += r["input_tokens"]; g["out"] += r["output_tokens"]; g["n"] += 1
        print(f"  by {a.by}:")
        print(f"    {a.by:<13}{'tasks':>6}{'cost':>11}{'$/task':>9}{'in tok':>13}{'out tok':>13}")
        for k, g in sorted(groups.items()):
            print(f"    {k:<13}{g['n']:>6}{('$%.4f' % g['cost']):>11}{('$%.4f' % (g['cost'] / g['n'])):>9}"
                  f"{g['in']:>13,}{g['out']:>13,}")
        succ = load_task_success(s)
        if succ:
            cp = [r["cost"] for r in recs if succ.get(r["task_id"], 0) >= 1.0]
            cf = [r["cost"] for r in recs if r["task_id"] in succ and succ.get(r["task_id"], 0) < 1.0]
            print("  by outcome (scored tasks):")
            for label, lst in [("passed", cp), ("failed", cf)]:
                if lst:
                    print(f"    {label:<8}{len(lst):>4} tasks  ${sum(lst):>8.4f} total  ${sum(lst) / len(lst):.4f}/task")


def react_metrics(task_dir):
    """Aggregate trace metrics, grounded in the trace folder's react_steps.json
    (final-workflow shape comes from workflow.json in the same folder)."""
    rs = _load(task_dir / "react_steps.json")
    steps = rs.get("steps", []) if isinstance(rs, dict) else []
    m = {"steps": len(steps), "agent_steps": 0, "user_steps": 0, "steps_with_tools": 0,
         "tool_calls": 0, "tool_errors": 0, "tools": Counter(),
         "ops_touched": set(), "ops_deleted": set(), "edits": Counter(),
         "in": 0, "out": 0, "total": 0, "reasoning": 0, "cached": 0}
    for s in steps:
        role = s.get("role")
        if role == "agent":
            m["agent_steps"] += 1
        elif role == "user":
            m["user_steps"] += 1
        tcs = s.get("toolCalls") or []
        if tcs:
            m["steps_with_tools"] += 1
        for tc in tcs:
            m["tool_calls"] += 1
            m["tools"][tc.get("toolName")] += 1
            opid = (tc.get("input") or {}).get("operatorId")
            if tc.get("toolName") == "deleteOperator" and opid:
                m["ops_deleted"].add(opid)
            elif tc.get("toolName") == "createOrModifyOperator" and opid:
                m["ops_touched"].add(opid)
                m["edits"][opid] += 1
        for tr in (s.get("toolResults") or []):
            if tr.get("isError"):
                m["tool_errors"] += 1
        u = s.get("usage") or {}
        m["in"] += u.get("inputTokens", 0) or 0
        m["out"] += u.get("outputTokens", 0) or 0
        m["total"] += u.get("totalTokens", 0) or 0
        m["reasoning"] += u.get("reasoningTokens", 0) or 0
        m["cached"] += u.get("cachedInputTokens", 0) or 0
    m["final_ops_react"] = len(m["ops_touched"] - m["ops_deleted"])  # final ops per the trace
    m["max_edits"] = max(m["edits"].values()) if m["edits"] else 0
    m["max_edits_op"] = m["edits"].most_common(1)[0][0] if m["edits"] else None
    wf = _load(task_dir / "workflow.json")
    w = (wf.get("workflow") or {}) if isinstance(wf, dict) else {}
    m["wf_ops"] = len(w.get("operators", []))
    m["wf_links"] = len(w.get("links", []))
    m["wf_types"] = Counter(o.get("operatorType") for o in w.get("operators", []))
    return m


def cmd_traces(a):
    scratch = KB_ROOT / "system_scratch" / a.sut
    if not scratch.is_dir():
        sys.exit(f"no traces at {scratch}")
    if a.task:
        _show_one_trace(scratch / a.task)
        return
    dirs = sorted(d for d in scratch.iterdir() if d.is_dir())
    if a.workload:
        dirs = [d for d in dirs if workload_of(d.name) == a.workload]
    print(f"{len(dirs)} traces under {scratch.relative_to(KB_ROOT)}  (✓/✗ | steps | tools | final ops | answer)")
    mets = []
    for d in dirs:
        ev = _load(d / "evaluation.json")
        ans = _load(d / "answer.json")
        succ = ev.get("success")
        succ = "?" if succ is None else ("✓" if succ else "✗")
        m = react_metrics(d)
        mets.append(m)
        a_str = str((ans or {}).get("answer", ""))[:40].replace("\n", " ")
        print(f"  {succ}  {d.name:<30} steps={m['steps']:>2} tools={m['tool_calls']:>3} ops={m['wf_ops']:>2}  {a_str}")
    if not mets:
        return
    print(f"\n=== aggregate over {len(mets)} traces (from react_steps.json) ===")

    def stat(key):
        vals = [mm[key] for mm in mets]
        return sum(vals) / len(vals), statistics.median(vals), max(vals)

    for label, key in [("react steps", "steps"), ("agent steps", "agent_steps"),
                       ("tool calls", "tool_calls"), ("tool errors", "tool_errors"),
                       ("final ops (workflow)", "wf_ops"), ("final links", "wf_links"),
                       ("max edits / op", "max_edits"), ("total tokens", "total"),
                       ("reasoning tokens", "reasoning"), ("cached tokens", "cached")]:
        avg, med, mx = stat(key)
        print(f"  {label:<22} avg {avg:>9.1f}   median {med:>9.1f}   max {mx:>10,}")
    tools, types = Counter(), Counter()
    for mm in mets:
        tools += mm["tools"]; types += mm["wf_types"]
    print(f"  tool-call totals       : {dict(tools)}")
    print(f"  final operator types   : {dict(types)}")


# ----------------------------- per-step token usage -----------------------------
def step_token_rows(task_dir):
    """Per-AGENT-step token usage from react_steps.json (in order). Each row:
    in/out/cached/reasoning/total + the tools called that step."""
    rs = _load(task_dir / "react_steps.json")
    steps = rs.get("steps", []) if isinstance(rs, dict) else []
    rows = []
    for s in steps:
        if s.get("role") != "agent":
            continue
        u = s.get("usage") or {}
        rows.append({
            "in": int(u.get("inputTokens", 0) or 0),
            "out": int(u.get("outputTokens", 0) or 0),
            "cached": int(u.get("cachedInputTokens", 0) or 0),
            "reasoning": int(u.get("reasoningTokens", 0) or 0),
            "total": int(u.get("totalTokens", 0) or 0),
            "tools": [tc.get("toolName") for tc in (s.get("toolCalls") or [])],
        })
    return rows


def _pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


def cmd_tokens(a):
    # ---- single task: full per-step table ----
    if a.task:
        d = KB_ROOT / "system_scratch" / a.sut[0] / a.task
        rows = step_token_rows(d)
        if not rows:
            sys.exit(f"no per-step usage in {d}/react_steps.json")
        print(f"=== {a.sut[0]} / {a.task} — per-step token usage ({len(rows)} agent steps) ===")
        print(f"  {'step':>4} {'in':>10} {'out':>7} {'cached':>10} {'reason':>7} {'total':>10} {'cum_total':>11}  tools")
        cum = 0
        for i, r in enumerate(rows, 1):
            cum += r["total"]
            tc = Counter(t for t in r["tools"] if t)
            ts = " ".join(f"{k}×{v}" if v > 1 else k for k, v in tc.items())
            print(f"  {i:>4} {r['in']:>10,} {r['out']:>7,} {r['cached']:>10,} {r['reasoning']:>7,} "
                  f"{r['total']:>10,} {cum:>11,}  {ts}")
        tin = sum(r["in"] for r in rows); tout = sum(r["out"] for r in rows)
        tca = sum(r["cached"] for r in rows); tto = sum(r["total"] for r in rows)
        print(f"  {'sum':>4} {tin:>10,} {tout:>7,} {tca:>10,} {'':>7} {tto:>10,}")
        print(f"  per-step avg : in {tin // len(rows):,}  out {tout // len(rows):,}  total {tto // len(rows):,}"
              f"   |  cache-hit {_pct(tca, tin):.0f}% of input")
        print(f"  input growth : step1 {rows[0]['in']:,} -> step{len(rows)} {rows[-1]['in']:,}  "
              f"(Δ {rows[-1]['in'] - rows[0]['in']:+,})")
        return

    # ---- aggregate over all tasks, per SUT (several --sut compare) ----
    def mean(key, src):
        return sum(x[key] for x in src) / len(src) if src else 0.0

    for sut in a.sut:
        base = KB_ROOT / "system_scratch" / sut
        if not base.is_dir():
            print(f"\n=== {sut}: no traces ==="); continue
        task_tot, all_steps = [], []
        by_index = defaultdict(list)   # step-index -> per-step rows (the growth curve)
        for dd in sorted(base.iterdir()):
            if not dd.is_dir() or dd.name.startswith("_"):
                continue
            rows = step_token_rows(dd)
            if not rows:
                continue
            all_steps.extend(rows)
            for i, r in enumerate(rows, 1):
                by_index[i].append(r)
            task_tot.append({k: sum(r[k] for r in rows) for k in ("in", "out", "cached", "reasoning", "total")}
                            | {"steps": len(rows)})
        if not task_tot:
            print(f"\n=== {sut}: no per-step usage in react_steps.json ==="); continue
        n, ns = len(task_tot), len(all_steps)
        ti, tc = mean("in", task_tot), mean("cached", task_tot)
        print(f"\n=== {sut}  ({_first_model(sut)}) — token breakdown: {n} tasks, {ns} agent-steps ===")
        print(f"  per-task mean : in {ti:>9,.0f}  out {mean('out', task_tot):>7,.0f}  cached {tc:>9,.0f}  "
              f"reasoning {mean('reasoning', task_tot):>7,.0f}  total {mean('total', task_tot):>9,.0f}  "
              f"steps {mean('steps', task_tot):.1f}  cache-hit {_pct(tc, ti):.0f}%")
        si, sc = mean("in", all_steps), mean("cached", all_steps)
        med_in = statistics.median([r["in"] for r in all_steps]); mx_in = max(r["in"] for r in all_steps)
        print(f"  per-step in   : mean {si:>9,.0f}  median {med_in:>9,.0f}  max {mx_in:>9,.0f}  cache-hit {_pct(sc, si):.0f}%")
        print(f"  per-step out  : mean {mean('out', all_steps):>9,.0f}  reasoning {mean('reasoning', all_steps):>7,.0f}")
        kmax = min(max(by_index), a.max_steps)
        print(f"  input-growth curve (mean input tokens @ step k; n = tasks reaching k):")
        for k in range(1, kmax + 1):
            rs = by_index.get(k, [])
            if rs:
                print(f"    step {k:>2}: in {mean('in', rs):>9,.0f}  cached {mean('cached', rs):>9,.0f}  "
                      f"out {mean('out', rs):>7,.0f}  (n={len(rs)})")


# ----------------------------- score / trace helpers -----------------------------
def reeval(sut, workloads=None):
    env = dict(os.environ)
    if workloads:
        env["WORKLOADS_OVERRIDE"] = " ".join(workloads)
    env["SUT"] = sut
    p = subprocess.Popen(["bash", "evaluate_dataflow_system.sh"], cwd=KB_ROOT, env=env)
    _PROCS.append(p)
    p.wait()


def show_scores(sut=None):
    """Run compute_scores.py and print the leaderboard (optionally one SUT block)."""
    r = subprocess.run([PY, "compute_scores.py"], cwd=KB_ROOT, capture_output=True, text=True)
    txt = r.stdout
    if not sut:
        print(txt)
        return
    block = _extract_block(txt, sut)
    print(block if block else f"(no scores found for {sut} — run `reeval --sut {sut}` first)")
    # also the one-line leaderboard entry if present
    for line in txt.splitlines():
        if line.strip().startswith(sut) and "%" in line:
            print("leaderboard:", line.strip())
            break


def _extract_block(txt, sut):
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == sut:  # header line between two ==== rules
            start = i - 1
            end = i + 1
            while end < len(lines) and not lines[end].startswith("====="):
                end += 1
            # include the OVERALL block: scan to the line after OVERALL
            j = i + 1
            last = j
            while j < len(lines) and "OVERALL" not in lines[j]:
                last = j; j += 1
            return "\n".join(lines[start:j + 1])
    return None


def _report_scores_from_logs(logdir, wls):
    print("\n[kb] per-workload Total score (from this run):")
    for w in wls:
        lp = Path(logdir) / f"{w}.log"
        sc = scores_per_workload(lp.read_text()) if lp.exists() else []
        print(f"  {w:<13} {sc[-1] if sc else 'n/a'}")


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def _show_one_trace(d):
    if not d.is_dir():
        sys.exit(f"no trace dir {d}")
    gt = _load(d / "ground_truth.json")
    ans = _load(d / "answer.json")
    ev = _load(d / "evaluation.json")
    st = _load(d / "stats.json")
    print(f"=== {d.name} ===")
    print(f"  query   : {gt.get('query','')[:200]}")
    print(f"  gold    : {gt.get('answer')}  ({gt.get('answer_type')})")
    print(f"  answer  : {(ans or {}).get('answer')}")
    print(f"  success : {ev.get('success')}")
    metrics = {k: v for k, v in ev.items()
               if k not in ('success', 'model_output', 'code', 'id', 'task_id')
               and not k.startswith('token_usage')}
    if metrics:
        print(f"  metrics : {metrics}")
    if st:
        print(f"  stats   : steps={st.get('num_steps')} tokens={st.get('total_tokens')} "
              f"cost=${st.get('cost_usd')} elapsed={st.get('elapsed_seconds')}s")
    m = react_metrics(d)
    print("  --- react steps (react_steps.json) ---")
    print(f"  steps        : {m['steps']} ({m['agent_steps']} agent / {m['user_steps']} user); "
          f"{m['steps_with_tools']} with tool calls")
    print(f"  tool calls   : {m['tool_calls']} — {dict(m['tools'])}")
    print(f"  tool errors  : {m['tool_errors']}")
    edit_note = f" (most: {m['max_edits']}x on {m['max_edits_op']})" if m['max_edits_op'] else ""
    print(f"  operators    : {len(m['ops_touched'])} created/modified, {len(m['ops_deleted'])} deleted{edit_note}")
    print(f"  step tokens  : in={m['in']:,} out={m['out']:,} total={m['total']:,} "
          f"reasoning={m['reasoning']:,} cached={m['cached']:,}")
    if m['agent_steps']:
        print(f"  per-step avg : in={m['in'] // m['agent_steps']:,} out={m['out'] // m['agent_steps']:,} "
              f"(over {m['agent_steps']} agent steps)")
    print("  --- final workflow (workflow.json) ---")
    print(f"  operators    : {m['wf_ops']} (types: {dict(m['wf_types'])}); links: {m['wf_links']}")
    if m['wf_ops'] != m['final_ops_react']:
        print(f"  note         : react steps imply {m['final_ops_react']} final ops (vs {m['wf_ops']} in workflow.json)")
    print(f"  files        : {', '.join(sorted(p.name for p in d.iterdir()))}")


# ----------------------------- argparse -----------------------------
def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)  # interleave [kb] logs with subprocess output
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="kb.py", description="KramaBench DataflowAgent CLI",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    def P(name, desc):
        """add a subparser whose own -h shows the full `desc` (first line = parent help)."""
        return sub.add_parser(name, help=desc.strip().splitlines()[0], description=desc,
                              formatter_class=argparse.RawDescriptionHelpFormatter)

    def add_run_opts(p):
        p.add_argument("--parallel", action="store_true",
                       help="fan workloads/tasks out concurrently (one OS process each)")
        p.add_argument("--no-oracle", action="store_true",
                       help="use the full data lake instead of --use_truth_subset (oracle is the default)")
        p.add_argument("--watchdog-min", type=int, default=8, metavar="N",
                       help="kill any group whose log is silent N min — gpt-5.x hang guard (default 8)")

    p = P("systems",
          "list the SUT class names available from the systems package\n"
          "(these are the values you pass to --sut).\n\nexample:\n  kb.py systems")
    p.set_defaults(fn=cmd_systems)

    p = P("run",
          "run a SUT over FULL workloads (every task in each workload).\n"
          "oracle mode (--use_truth_subset) is ON by default; a stall watchdog kills any\n"
          "workload group that hangs; afterwards the bulk cache is rebuilt and scores printed.\n\n"
          "examples:\n"
          "  kb.py run --sut DataflowSystemGPT54DeltaSchemaConverge --parallel\n"
          "  kb.py run --sut <S> --workloads \"legal environment\"\n"
          "  kb.py run --sut <S> --limit 3 --parallel        # quick smoke: first 3 tasks/workload")
    p.add_argument("--sut", required=True, help="SUT class name (see `kb.py systems`)")
    p.add_argument("--workloads", metavar='"a b"', help='subset of the 6 workloads, e.g. "legal environment" (default: all)')
    p.add_argument("--limit", type=int, default=0, metavar="N", help="only first N tasks per workload (quick smoke; 0=all)")
    p.add_argument("--isolate", action="store_true", help="one process per task (a hang can't take out siblings)")
    p.add_argument("--no-reeval", action="store_true", help="skip the post-run rebuild+rescore")
    add_run_opts(p); p.set_defaults(fn=cmd_run)

    p = P("tasks",
          "run SPECIFIC task ids (auto-grouped by workload), then auto re-evaluate and print\n"
          "scores — partial runs don't write the bulk cache, so the reeval is implied.\n\n"
          "example:\n  kb.py tasks --sut <S> --ids \"legal-hard-2 astronomy-easy-1\" --parallel")
    p.add_argument("--sut", required=True, help="SUT class name")
    p.add_argument("--ids", required=True, metavar='"id ..."', help='space-separated task ids, e.g. "legal-hard-2 wildfire-easy-1"')
    p.add_argument("--isolate", action="store_true", help="one process per task (a hang can't kill siblings)")
    p.add_argument("--no-reeval", action="store_true", help="skip the post-run rebuild+rescore")
    add_run_opts(p); p.set_defaults(fn=cmd_tasks)

    p = P("reeval",
          "rebuild the bulk response_cache from system_scratch (answer.json = source of truth)\n"
          "and rescore via --use_system_cache. run this after partial task reruns to sync\n"
          "results/aggregated_results.csv. no agent runs — cheap (wraps evaluate_dataflow_system.sh).\n\n"
          "example:\n  kb.py reeval --sut <S> --workloads \"legal\"")
    p.add_argument("--sut", required=True, help="SUT class name")
    p.add_argument("--workloads", metavar='"a b"', help="subset of workloads (default: all 6)")
    p.set_defaults(fn=cmd_reeval)

    p = P("failed",
          "list non-passing task ids for a SUT (wraps scripts/list_failed_tasks.py).\n"
          "--zero-only keeps only score==0 (and unscored) tasks; --ids-only prints a\n"
          "space-separated list you can pipe into `tasks --ids`.\n\n"
          "example:\n  kb.py failed --sut <S> --zero-only --ids-only")
    p.add_argument("--sut", required=True, help="SUT class name")
    p.add_argument("--workload", help="filter to one workload")
    p.add_argument("--zero-only", action="store_true", help="only score==0 / unscored tasks")
    p.add_argument("--ids-only", action="store_true", help="print space-separated ids (for piping)")
    p.set_defaults(fn=cmd_failed)

    p = P("rerun-failed",
          "composite recovery: detect score-0 tasks -> rerun them (watchdogged) -> reeval ->\n"
          "print scores. monotonic: a 0 task can only improve, and a killed/hung rerun keeps\n"
          "its prior 0 (never regresses good results). --limit caps how many to attempt.\n\n"
          "example:\n  kb.py rerun-failed --sut <S> --parallel")
    p.add_argument("--sut", required=True, help="SUT class name")
    p.add_argument("--all-failed", action="store_true", help="rerun ALL failed (score<1), not just score==0")
    p.add_argument("--isolate", action="store_true", help="one process per task")
    p.add_argument("--limit", type=int, default=0, metavar="N", help="rerun at most N tasks (0=all; handy for smokes)")
    add_run_opts(p); p.set_defaults(fn=cmd_rerun_failed)

    p = P("scores",
          "print scores from compute_scores.py (the repo's leaderboard aggregator):\n"
          "per-workload Score/Correct/Total + OVERALL. reads existing results — no re-run.\n"
          "--sut shows just that system's block; omit it for the full leaderboard.\n\n"
          "example:\n  kb.py scores --sut <S>")
    p.add_argument("--sut", help="show just this SUT's block (omit for the full leaderboard)")
    p.set_defaults(fn=cmd_scores)

    p = P("cost",
          "show LLM cost + token usage for one or more SUTs, from each task's stats.json\n"
          "cost_usd (litellm pricing — includes cache-read discounts, so gpt-5.4 etc. price\n"
          "correctly). breakdown via --by {workload,difficulty,task}; pass several --sut to compare.\n\n"
          "examples:\n"
          "  kb.py cost --sut DataflowSystemGPT54DeltaSchemaConverge\n"
          "  kb.py cost --sut <S> --by difficulty\n"
          "  kb.py cost --sut <S1> <S2>                 # compare totals\n"
          "  kb.py cost --sut <S> --by task --top 10")
    p.add_argument("--sut", required=True, nargs="+", metavar="SUT", help="one or more SUT class names")
    p.add_argument("--by", choices=["workload", "difficulty", "task"], default="workload",
                   help="breakdown dimension (default: workload)")
    p.add_argument("--top", type=int, default=20, metavar="N",
                   help="for --by task: show top N tasks by cost (default 20)")
    p.set_defaults(fn=cmd_cost)

    p = P("tokens",
          "per-step token-usage breakdown from react_steps.json. with --task: a full\n"
          "step-by-step table (in/out/cached/reasoning/total + cumulative + tools per step).\n"
          "without --task: aggregated stats over all tasks — per-task means, per-step\n"
          "distribution, cache-hit %, and the input-GROWTH curve (mean input tokens at step k,\n"
          "which exposes the context-accumulation dynamics). pass several --sut to compare arms.\n\n"
          "examples:\n"
          "  kb.py tokens --sut DataflowSystemGPT54DeltaSchemaConverge --task legal-hard-2\n"
          "  kb.py tokens --sut <S1> <S2>                 # compare per-step token profiles\n"
          "  kb.py tokens --sut <S> --max-steps 8")
    p.add_argument("--sut", required=True, nargs="+", metavar="SUT", help="one or more SUT class names")
    p.add_argument("--task", help="show the full per-step table for one task id (uses the first --sut)")
    p.add_argument("--max-steps", type=int, default=12, metavar="N",
                   help="how many step-indices to show in the input-growth curve (default 12)")
    p.set_defaults(fn=cmd_tokens)

    p = P("traces",
          "query per-task artifacts under system_scratch/<SUT>/. without --task: list every\n"
          "task with its success flag + parsed answer. with --task: full detail (query, gold,\n"
          "answer, metrics, token/step stats, react-step count, files on disk).\n\n"
          "example:\n  kb.py traces --sut <S> --task legal-hard-2")
    p.add_argument("--sut", required=True, help="SUT class name")
    p.add_argument("--task", help="show one task id in full detail")
    p.add_argument("--workload", help="filter the listing to one workload")
    p.set_defaults(fn=cmd_traces)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

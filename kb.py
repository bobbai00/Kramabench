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
  compare    --sut A B    A-vs-B: outcome matrix + both-pass cost split + cost dominators
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
  ./kb.py compare --sut DataflowSystemGPT54LatestSchemaConverge DataflowSystemGPT54DeltaSchemaConverge
"""
import argparse, csv, json, math, os, re, signal, statistics, subprocess, sys, time
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
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    p = subprocess.Popen(cmd, cwd=KB_ROOT, stdout=open(logpath, "w"),
                         stderr=subprocess.STDOUT, env=env)
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
    safe_sut = re.sub(r"[^A-Za-z0-9_.-]+", "_", sut)
    logdir = KB_ROOT / "logs" / f"kb-{label}-{safe_sut}-{ts}-{os.getpid()}"
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
        max_parallel = int(os.environ.get("KB_MAX_PARALLEL", "6") or "0")
        if max_parallel > 0 and len(units) > max_parallel:
            print(f"[kb] parallel limit: {max_parallel} concurrent unit(s)")
            jobs = []
            for i in range(0, len(units), max_parallel):
                batch = [launch(*u) for u in units[i:i + max_parallel]]
                _watch_and_wait(batch, watchdog_min)
                jobs.extend(batch)
        else:
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
    if a.trim_top:
        print("\ntrimmed $/task (drops the highest-cost PCT of tasks within each SUT):")
        header = f"{'SUT':<44}{'trim':>7}{'kept':>9}{'dropped':>9}{'trim $':>11}{'$/task':>9}"
        print(header)
        print("-" * len(header))
        for s, recs in data.items():
            ordered = sorted(recs, key=lambda r: r["cost"], reverse=True)
            for pct in a.trim_top:
                if pct < 0 or pct >= 100:
                    sys.exit("--trim-top values must be in [0, 100)")
                drop = math.floor(len(ordered) * pct / 100)
                kept = ordered[drop:] if drop else ordered
                tc = sum(r["cost"] for r in kept)
                print(f"{s:<44}{(str(pct) + '%'):>7}{len(kept):>9}{drop:>9}"
                      f"{('$%.2f' % tc):>11}{('$%.4f' % (tc / len(kept))):>9}")
    print("\n(cost = sum of stats.json cost_usd — litellm pricing, includes cache-read discounts)")

    for s, recs in data.items():
        miss = sum(1 for r in recs if not r["cost"])
        tag = f"  [{miss} task(s) missing cost_usd]" if miss else ""
        if a.by == "task" and a.top <= 0:
            continue
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


def cmd_compare(a):
    """Pairwise A-vs-B: outcome matrix, both-pass cost split, and cost dominators."""
    A, B = a.sut
    sa, sb = load_task_success(A), load_task_success(B)
    ca = {r["task_id"]: r for r in load_cost_stats(A)}
    cb = {r["task_id"]: r for r in load_cost_stats(B)}
    if not ca or not cb:
        sys.exit("missing stats for one of the SUTs (run them first)")
    pa = lambda t: sa.get(t, 0) >= 1.0
    pb = lambda t: sb.get(t, 0) >= 1.0
    short = lambda s: s.replace("DataflowSystem", "").replace("CodeAgentSystem", "CA:")

    print(f"{'SUT':<40}{'tasks':>6}{'pass':>6}{'pass%':>7}{'total$':>9}{'tokens':>13}")
    for s, succ, cost in [(A, sa, ca), (B, sb, cb)]:
        n = len(cost); pw = sum(1 for t in cost if succ.get(t, 0) >= 1.0)
        tc = sum(r["cost"] for r in cost.values()); tk = sum(r["total_tokens"] for r in cost.values())
        print(f"{short(s):<40}{n:>6}{pw:>6}{(100*pw/n if n else 0):>6.0f}%{('$%.2f' % tc):>9}{tk:>13,}")
    print(f"  A = {A}\n  B = {B}")

    common = sorted(set(ca) & set(cb))
    both = [t for t in common if pa(t) and pb(t)]
    onlyA = [t for t in common if pa(t) and not pb(t)]
    onlyB = [t for t in common if pb(t) and not pa(t)]
    neither = [t for t in common if not pa(t) and not pb(t)]
    print(f"\noutcome over {len(common)} shared tasks: "
          f"both pass {len(both)} | A-only {len(onlyA)} | B-only {len(onlyB)} | both fail {len(neither)}")

    a_ch = [t for t in both if ca[t]["cost"] < cb[t]["cost"]]
    b_ch = [t for t in both if cb[t]["cost"] < ca[t]["cost"]]
    print(f"both-pass cost ({len(both)} tasks): "
          f"A cheaper {len(a_ch)} (−${sum(cb[t]['cost']-ca[t]['cost'] for t in a_ch):.3f}) | "
          f"B cheaper {len(b_ch)} (−${sum(ca[t]['cost']-cb[t]['cost'] for t in b_ch):.3f})")

    net = sum(cb[t]["cost"] - ca[t]["cost"] for t in common)  # +ve = B pricier
    dom = sorted(common, key=lambda t: abs(cb[t]["cost"] - ca[t]["cost"]), reverse=True)
    print(f"\ncost gap (Δ = B−A) net over shared = ${net:+.3f}  (+ = B pricier).  top {a.top} dominators:")
    print(f"  {'task':<22}{'Δ(B-A)':>10}   A($/steps/PF)      B($/steps/PF)")
    cum = 0.0
    for t in dom[:a.top]:
        d = cb[t]["cost"] - ca[t]["cost"]; cum += d
        print(f"  {t:<22}{('%+.4f' % d):>10}   ${ca[t]['cost']:.4f}/{ca[t]['num_steps']:>2}/{'P' if pa(t) else 'F'}"
              f"     ${cb[t]['cost']:.4f}/{cb[t]['num_steps']:>2}/{'P' if pb(t) else 'F'}")
    if net:
        print(f"  -> top {a.top} = ${cum:+.3f}  ({100*cum/net:.0f}% of the net gap)")
    extra = sorted(set(ca) ^ set(cb))
    if extra:
        print(f"\n(note: {len(extra)} task(s) only in one SUT, excluded from the shared comparison: {extra})")


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


# ----------------------------- venn (A vs B category analysis) -----------------------------
ANSWER_TYPE_METRIC = {"numeric_exact": "success", "string_exact": "success",
                      "list_exact": "f1", "numeric_approximate": "rae_score",
                      "list_approximate": "f1_approximate", "string_approximate": "llm_paraphrase"}


def answer_scores(sut):
    """{task_id: answer-type-aware score} from system_scratch evaluation.json +
    ground_truth.json (the levers-report convention; robust to metric mixes)."""
    base = KB_ROOT / "system_scratch" / sut
    out = {}
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if not d.is_dir():
            continue
        ev, gt = _load(d / "evaluation.json"), _load(d / "ground_truth.json")
        if not ev:
            continue
        k = ANSWER_TYPE_METRIC.get(gt.get("answer_type") or "")
        v = ev.get(k) if k and isinstance(ev.get(k), (int, float)) else None
        if v is None:
            vals = [float(ev[x]) for x in SCORE_METRICS if isinstance(ev.get(x), (int, float))]
            v = max(vals) if vals else 0.0
        out[d.name] = float(v)
    return out


def judge_scores(sut, metric="m3"):
    """{task_id: score} from the cached chunked-LLM-judge metrics
    (scripts/judge_metrics.py -> system_scratch/<sut>/<task>/judge_m3m4.json).
    metric: m3 (evidence-in-context) | m4_process | m4_deliverable."""
    base = KB_ROOT / "system_scratch" / sut
    out = {}
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if not d.is_dir():
            continue
        j = _load(d / "judge_m3m4.json")
        if j and isinstance(j.get(metric), (int, float)):
            out[d.name] = float(j[metric])
    return out


def task_op_features(sut, task):
    """Per-operator structural features for one task: role, depth, LOC, edits,
    source files (+ext/size). Pure function of workflow.json + react_steps.json."""
    d = KB_ROOT / "system_scratch" / sut / task
    wf = (_load(d / "workflow.json") or {}).get("workflow") or {}
    doc = _load(d / "react_steps.json")
    edits, code_by = Counter(), {}
    for st in (doc.get("steps", []) if isinstance(doc, dict) else []):
        if st.get("role") != "agent":
            continue
        for tc in st.get("toolCalls") or []:
            inp = tc.get("input") or {}
            if inp.get("operatorId") and inp.get("code"):
                edits[inp["operatorId"]] += 1
                code_by[inp["operatorId"]] = inp["code"]
    fin, fout, parents = Counter(), Counter(), defaultdict(list)
    for l in wf.get("links", []):
        s_, t_ = l["source"]["operatorID"], l["target"]["operatorID"]
        fout[s_] += 1; fin[t_] += 1; parents[t_].append(s_)
    depth = {}

    def dep(o, seen=()):
        if o in depth:
            return depth[o]
        if o in seen or not parents[o]:
            depth[o] = 0
            return 0
        depth[o] = 1 + max(dep(p, seen + (o,)) for p in parents[o])
        return depth[o]

    feats = []
    for o in wf.get("operators", []):
        op = o["operatorID"]
        code = code_by.get(op, str(o.get("operatorProperties", {}).get("code", "")))
        files = sorted(set(re.findall(r"data/[\w\-./]+\.\w+", code)))
        f0 = files[0] if files else None
        feats.append(dict(op=op,
                          role="source" if fin[op] == 0 else ("sink" if fout[op] == 0 else "interior"),
                          depth=dep(op), fanin=fin[op], fanout=fout[op], parents=parents[op],
                          loc=code.count("\n") + 1, edits=edits.get(op, 1),
                          file=f0, ext=os.path.splitext(f0)[1] if f0 else None,
                          fsize_kb=(os.path.getsize(KB_ROOT / f0) // 1024
                                    if f0 and (KB_ROOT / f0).exists() else None)))
    return feats


def _cat_profile(sut, tasks, label):
    """Aggregate operator/file characteristics over a category's tasks."""
    ops = [f for t in tasks for f in task_op_features(sut, t)]
    if not ops:
        print(f"  {label}: (no ops)")
        return
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    roles = Counter(o["role"] for o in ops)
    exts = Counter(o["ext"] for o in ops if o["ext"])
    sizes = [o["fsize_kb"] for o in ops if o["fsize_kb"] is not None]
    locs = [o["loc"] for o in ops]
    multi = sum(1 for o in ops if o["edits"] >= 2)
    print(f"  {label}: {len(tasks)} task(s), {len(ops)} ops | roles {dict(roles)} | "
          f"multi-edit {multi} ({100*multi/len(ops):.0f}%) | medLOC {med(locs)} | "
          f"medFile {med(sizes)}KB | exts {dict(exts.most_common(5))}")


def cmd_venn(a):
    """A-vs-B Venn + both-pass cost split + per-category operator/file stats."""
    A, B = a.sut
    sa, sb = answer_scores(A), answer_scores(B)
    ca = {r["task_id"]: r for r in load_cost_stats(A)}
    cb = {r["task_id"]: r for r in load_cost_stats(B)}
    th = a.th
    common = sorted(set(sa) & set(sb))
    both = [t for t in common if sa[t] >= th and sb[t] >= th]
    onlyA = [t for t in common if sa[t] >= th and sb[t] < th]
    onlyB = [t for t in common if sb[t] >= th and sa[t] < th]
    neither = [t for t in common if sa[t] < th and sb[t] < th]
    chron = set()
    cf = Path(a.chronic) if a.chronic else (KB_ROOT / "judgment_runs/levers_report/chronic_flippers.json")
    if cf.exists():
        chron = set(json.load(open(cf)))
    tagged = lambda ts: [t + ("*" if t in chron else "") for t in ts]

    aw = len(onlyA); bw = len(onlyB); bp = len(both)
    cheapA = [t for t in both if t in ca and t in cb and ca[t]["cost"] < cb[t]["cost"]]
    cheapB = [t for t in both if t in ca and t in cb and cb[t]["cost"] < ca[t]["cost"]]
    gA = sum(cb[t]["cost"] - ca[t]["cost"] for t in cheapA)
    gB = sum(ca[t]["cost"] - cb[t]["cost"] for t in cheapB)
    short = lambda s: s.replace("DataflowSystem", "").replace("CodeAgentSystem", "CA:")

    print(f"A = {A}\nB = {B}\npass threshold = {th} (answer-type metric); * = chronic flipper\n")
    print(f"          .───────────────.      .───────────────.")
    print(f"        /   A-only         \\   /        B-only    \\")
    print(f"       |                    \\ /                     |")
    print(f"       |     {aw:>3}         |{bp:^7}|          {bw:>3}      |")
    print(f"       |                    / \\    both pass        |")
    print(f"        \\                 /   \\                    /")
    print(f"          '───────────────'      '───────────────'")
    print(f"                     both fail: {len(neither)}   (shared tasks: {len(common)})")
    print(f"\nboth-pass cost split ({bp} tasks): "
          f"A cheaper on {len(cheapA)} (saves ${gA:.3f}) | B cheaper on {len(cheapB)} (saves ${gB:.3f})")
    print(f"\nA-only: {tagged(onlyA)}")
    print(f"B-only: {tagged(onlyB)}")

    print(f"\n=== operator/file characteristics per category ===")
    _cat_profile(A, onlyA, f"A-only wins ({short(A)} ops)")
    _cat_profile(B, onlyB, f"B-only wins ({short(B)} ops)")
    topA = sorted(cheapA, key=lambda t: -(cb[t]["cost"] - ca[t]["cost"]))[:a.top]
    topB = sorted(cheapB, key=lambda t: -(ca[t]["cost"] - cb[t]["cost"]))[:a.top]
    _cat_profile(A, topA, f"both-pass, A much cheaper (top {len(topA)})")
    _cat_profile(B, topB, f"both-pass, B much cheaper (top {len(topB)})")


# ----------------------------- case-metrics (per-Venn-case op/cardinality/file/data-issue metrics) -----------------------------
STATS_ARM = "DataflowSystemGPT52DeltaStats3kD2"  # data-issue source: engine-computed full-data Output Table profiles
_TBL = re.compile(r"Output Table: (\d+) rows?, (\d+) cols")
_ELISION = re.compile(r"\n\s*\.\.\.(\t\.\.\.)+")
_TRUNC = ("…(truncated", "...[truncated]...")
_DELTA_OP = re.compile(r"^- operator (\S+) (?:added|updated)\s*$", re.M)
_LATEST_OP = re.compile(r"^#{2,4} (?:Operator )?`(\S+)` \(\w+\)\s*$", re.M)
_SECTION = re.compile(r"^#{1,6} ", re.M)


def _final_context(sut, task):
    doc = _load(KB_ROOT / "system_scratch" / sut / task / "react_steps.json")
    steps = [s for s in (doc.get("steps", []) if isinstance(doc, dict) else [])
             if s.get("role") == "agent" and s.get("inputMessages")]
    return "\n".join(str(m.get("content", "")) for m in steps[-1]["inputMessages"]) if steps else ""


def _op_blocks(ctx):
    """opId -> LAST rendered block for that op. Handles both grammars (DELTA
    '- operator X added/updated' event lines; LATEST '### Operator `x` (Type)'
    sections). Block ends at the next op header or markdown section header,
    ignoring header-lookalikes inside ``` fences (col-0 python comments)."""
    fences, pos = [], 0
    while True:
        i = ctx.find("```", pos)
        if i < 0:
            break
        j = ctx.find("```", i + 3)
        fences.append((i, len(ctx) if j < 0 else j + 3))
        pos = len(ctx) if j < 0 else j + 3
    outside = lambda i: all(not (a <= i < b) for a, b in fences)
    heads = sorted([(m.start(), m.group(1)) for p in (_DELTA_OP, _LATEST_OP)
                    for m in p.finditer(ctx) if outside(m.start())])
    stops = sorted([m.start() for m in _SECTION.finditer(ctx) if outside(m.start())]
                   + [h[0] for h in heads])
    out = {}
    for start, op in heads:
        nl = ctx.find("\n", start)
        begin = len(ctx) if nl < 0 else nl + 1
        end = min((s for s in stops if s > start), default=len(ctx))
        out[op] = ctx[begin:min(end, begin + 12000)]
    return out


_FROWS = {}


def _file_line_rows(path):
    """Raw row count for line-oriented formats (data-row count for headered csv/tsv)."""
    if path in _FROWS:
        return _FROWS[path]
    p, r = KB_ROOT / path, None
    if p.exists() and p.suffix.lower() in (".csv", ".tsv", ".txt", ".jsonl", ".tle", ".text"):
        try:
            with open(p, "rb") as f:
                r = sum(ch.count(b"\n") for ch in iter(lambda: f.read(1 << 20), b""))
            if p.suffix.lower() in (".csv", ".tsv"):
                r = max(0, r - 1)
        except OSError:
            r = None
    _FROWS[path] = r
    return r


def _block_issues(b):
    """Data-issue facts from one rendered stats block (engine full-data profile)."""
    iss, rows = {}, None
    m = _TBL.search(b)
    if m:
        rows = int(m.group(1))
    m = re.search(r"duplicate rows: (\d+) of (\d+)", b)
    if m:
        iss["dup_pct"] = 100 * int(m.group(1)) / max(1, int(m.group(2)))
    m = re.search(r"empty rows: (\d+) of (\d+)", b)
    if m:
        iss["empty_rows_pct"] = 100 * int(m.group(1)) / max(1, int(m.group(2)))
    m = re.search(r"empty columns: \[([^\]]*)\]", b)
    if m:
        iss["empty_cols"] = m.group(1).count('"') // 2
    m = re.search(r"headers: (\d+) of (\d+) columns are unnamed", b)
    if m:
        iss["unnamed_cols"] = int(m.group(1))
    nulls = [int(x) for x in re.findall(r"null=(\d+)", b)]
    if rows and nulls:
        iss["maxnull_pct"] = 100 * max(nulls) / rows
    cols = re.findall(r'^\s*- "[^"]+" \((\w+)\)', b, re.M)
    if cols:
        iss["str_share"] = 100 * sum(1 for c in cols if c == "str") / len(cols)
    return iss, rows


_LAKE = None


def lake_data_issues():
    """file -> engine-measured facts (rows_loaded + dirtiness), parsed from the
    stats arm's rendered profiles of the source op(s) that load it. Dirtiness is
    the MAX over sightings (rawest load); rows_loaded likewise."""
    global _LAKE
    if _LAKE is not None:
        return _LAKE
    _LAKE = {}
    base = KB_ROOT / "system_scratch" / STATS_ARM
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if not d.is_dir():
            continue
        blocks = _op_blocks(_final_context(STATS_ARM, d.name))
        for f in task_op_features(STATS_ARM, d.name):
            if f["role"] != "source" or not f["file"] or f["op"] not in blocks:
                continue
            iss, rows = _block_issues(blocks[f["op"]])
            rec = _LAKE.setdefault(f["file"], {"rows_loaded": None, "sightings": 0})
            rec["sightings"] += 1
            if rows is not None:
                rec["rows_loaded"] = max(rec["rows_loaded"] or 0, rows)
            for k, v in iss.items():
                rec[k] = max(rec.get(k) or 0, v)
    return _LAKE


def _cap_of(sut):
    m = re.search(r"(\d+)k", sut)
    return int(m.group(1)) * 1000 if m else 3000


def task_op_metrics(sut, task):
    """task_op_features + render-derived cardinality per op: out_rows/out_cols
    (the always-rendered 'Output Table: N rows, M cols'), in_rows (sources: the
    engine-loaded file rows from the stats arm, else raw line count; interiors:
    sum of parents' out_rows), shown_rows (table body lines actually rendered),
    capped (render hit the char cap: elision/truncation marks or near-cap block)."""
    feats = task_op_features(sut, task)
    blocks = _op_blocks(_final_context(sut, task))
    lake, cap = lake_data_issues(), _cap_of(sut)
    by = {f["op"]: f for f in feats}
    for f in feats:
        b = blocks.get(f["op"], "")
        m = None
        for m in _TBL.finditer(b):
            pass
        f["out_rows"] = int(m.group(1)) if m else None
        f["out_cols"] = int(m.group(2)) if m else None
        f["cells"] = f["out_rows"] * f["out_cols"] if m else None
        f["shown_rows"] = sum(1 for ln in b.splitlines() if ln.count("\t") >= 2)
        f["capped"] = bool(_ELISION.search(b)) or any(t in b for t in _TRUNC) or len(b) >= 0.88 * cap
    for f in feats:
        if f["role"] == "source":
            fi = lake.get(f["file"] or "", {})
            f["in_rows"] = fi.get("rows_loaded") if fi.get("rows_loaded") is not None \
                else (_file_line_rows(f["file"]) if f["file"] else None)
            for k in ("dup_pct", "empty_rows_pct", "empty_cols", "unnamed_cols", "maxnull_pct", "str_share"):
                f[k] = fi.get(k)
        else:
            pr = [by[p]["out_rows"] for p in f["parents"] if p in by and by[p]["out_rows"] is not None]
            f["in_rows"] = sum(pr) if pr else None
    return feats


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _p90(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))] if xs else None


def _fmtn(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}" if v < 100 else f"{v:,.0f}"
    return f"{v:,}"


def _arm_agg(ops, tasks):
    """Aggregate op metrics for one arm over one category's tasks."""
    if not ops:
        return {}
    src = [o for o in ops if o["role"] == "source"]
    inter = [o for o in ops if o["role"] == "interior"]
    sink = [o for o in ops if o["role"] == "sink"]
    reduce_n = [o for o in inter if o["in_rows"] and o["out_rows"] is not None]
    return dict(
        n_ops=len(ops), ops_per_task=round(len(ops) / max(1, len(tasks)), 1),
        depth_mean=round(sum(o["depth"] for o in ops) / len(ops), 2),
        depth_med=_med([o["depth"] for o in ops]),
        depth_max=max(o["depth"] for o in ops),
        roles={"source": len(src), "interior": len(inter), "sink": len(sink)},
        multi_edit_pct=round(100 * sum(1 for o in ops if o["edits"] >= 2) / len(ops)),
        loc_med=_med([o["loc"] for o in ops]),
        capped_pct=round(100 * sum(1 for o in ops if o["capped"]) / len(ops)),
        src_out_rows_med=_med([o["out_rows"] for o in src]),
        src_out_rows_p90=_p90([o["out_rows"] for o in src]),
        src_in_rows_med=_med([o["in_rows"] for o in src]),
        inter_out_rows_med=_med([o["out_rows"] for o in inter]),
        inter_out_rows_p90=_p90([o["out_rows"] for o in inter]),
        sink_out_rows_med=_med([o["out_rows"] for o in sink]),
        cells_med=_med([o["cells"] for o in ops]),
        cells_p90=_p90([o["cells"] for o in ops]),
        rows_by_depth={d: _med([o["out_rows"] for o in ops
                                if (o["depth"] if o["depth"] < 2 else 2) == d])
                       for d in (0, 1, 2)},
        reduce_share_pct=(round(100 * sum(1 for o in reduce_n if o["out_rows"] < 0.95 * o["in_rows"])
                                / len(reduce_n)) if reduce_n else None),
        shown_vs_out_med=_med([100 * o["shown_rows"] / o["out_rows"]
                               for o in ops if o["out_rows"] and o["shown_rows"]]),
    )


def _files_agg(ops):
    """Distinct source files across a category's ops (one record per file)."""
    files = {}
    for o in ops:
        if o["role"] == "source" and o["file"]:
            files[o["file"]] = o
    recs = list(files.values())
    if not recs:
        return {}
    dirty = [r for r in recs if (r.get("dup_pct") or 0) >= 5 or (r.get("empty_rows_pct") or 0) > 0
             or (r.get("unnamed_cols") or 0) > 0 or (r.get("maxnull_pct") or 0) >= 20]
    return dict(
        n_files=len(recs),
        formats=dict(Counter(r["ext"] for r in recs if r["ext"]).most_common()),
        size_kb_med=_med([r["fsize_kb"] for r in recs]),
        size_kb_p90=_p90([r["fsize_kb"] for r in recs]),
        size_kb_max=max((r["fsize_kb"] or 0) for r in recs),
        rows_med=_med([r["in_rows"] for r in recs]),
        rows_max=max((r["in_rows"] or 0) for r in recs),
        dirty_files=len(dirty), dirty_pct=round(100 * len(dirty) / len(recs)),
        dup5_files=sum(1 for r in recs if (r.get("dup_pct") or 0) >= 5),
        emptyrow_files=sum(1 for r in recs if (r.get("empty_rows_pct") or 0) > 0),
        unnamed_files=sum(1 for r in recs if (r.get("unnamed_cols") or 0) > 0),
        highnull_files=sum(1 for r in recs if (r.get("maxnull_pct") or 0) >= 20),
        str_share_med=_med([r.get("str_share") for r in recs]),
        files={r["file"]: dict(ext=r["ext"], kb=r["fsize_kb"], rows=r["in_rows"],
                               dup_pct=round(r.get("dup_pct") or 0, 1),
                               empty_rows_pct=round(r.get("empty_rows_pct") or 0, 1),
                               unnamed=r.get("unnamed_cols") or 0,
                               maxnull_pct=round(r.get("maxnull_pct") or 0, 1),
                               str_share=round(r.get("str_share") or 0)) for r in recs},
    )


def cmd_case_metrics(a):
    """Per-Venn-case metrics for an A-vs-B pair: operator depth + cardinality
    (in/out rows, incl. source input tables), source file size/format/rows,
    engine-measured data issues, render pressure. Prints per-category detail +
    a cross-category matrix; dumps everything to JSON."""
    A, B = a.sut
    sa, sb = answer_scores(A), answer_scores(B)
    ca = {r["task_id"]: r for r in load_cost_stats(A)}
    cb = {r["task_id"]: r for r in load_cost_stats(B)}
    th, common = a.th, sorted(set(sa) & set(sb))
    both = [t for t in common if sa[t] >= th and sb[t] >= th]
    onlyA = [t for t in common if sa[t] >= th and sb[t] < th]
    onlyB = [t for t in common if sb[t] >= th and sa[t] < th]
    gap = lambda t: (cb[t]["cost"] - ca[t]["cost"]) / max(ca[t]["cost"], cb[t]["cost"], 1e-9)
    costed = [t for t in both if t in ca and t in cb]
    cheapA = [t for t in costed if gap(t) >= a.gap]
    cheapB = [t for t in costed if -gap(t) >= a.gap]
    chron = set()
    cf = Path(a.chronic) if a.chronic else (KB_ROOT / "judgment_runs/levers_report/chronic_flippers.json")
    if cf.exists():
        chron = set(json.load(open(cf)))
    short = lambda s: s.replace("DataflowSystemGPT52", "")
    tag = lambda t: t + ("*" if t in chron else "")

    print(f"A = {A}\nB = {B}")
    print(f"pass th = {th}; cost-gap floor = {a.gap:.0%} of the dearer arm (twin noise); * = chronic flipper")
    print(f"venn: A-only {len(onlyA)} | both {len(both)} | B-only {len(onlyB)} | "
          f"neither {len(common) - len(onlyA) - len(onlyB) - len(both)}")
    print(f"both-pass cost: A cheaper on {sum(1 for t in costed if gap(t) > 0)} "
          f"(material ≥{a.gap:.0%}: {len(cheapA)}) | B cheaper on {sum(1 for t in costed if gap(t) < 0)} "
          f"(material: {len(cheapB)})")

    cats = [(f"{short(A)} wins, {short(B)} fails", onlyA, "A"),
            (f"{short(B)} wins, {short(A)} fails", onlyB, "B"),
            (f"both pass, {short(A)} materially cheaper", cheapA, "A"),
            (f"both pass, {short(B)} materially cheaper", cheapB, "B")]
    dump = {"A": A, "B": B, "th": th, "gap": a.gap, "categories": {}}
    matrix = []

    for label, tasks, side in cats:
        print(f"\n{'=' * 100}\nCASE: {label}  ({len(tasks)} tasks)")
        if not tasks:
            matrix.append((label, {}, {}, {}))
            continue
        opsA = {t: task_op_metrics(A, t) for t in tasks}
        opsB = {t: task_op_metrics(B, t) for t in tasks}
        print(f"{'task':28s} {'cost A/B $':>13s} {'steps A/B':>10s} {'ops A/B':>8s} "
              f"{'maxD A/B':>9s} {'srcRows':>9s} issues")
        per_task = {}
        for t in tasks:
            fa, fb = opsA[t], opsB[t]
            srcrows = sum(o["in_rows"] or 0 for o in fa if o["role"] == "source") or \
                      sum(o["in_rows"] or 0 for o in fb if o["role"] == "source")
            src_ops = [o for o in fa + fb if o["role"] == "source"]
            iss = sorted({nm for o in src_ops for nm, key, floor in
                          (("dup", "dup_pct", 5), ("empty", "empty_rows_pct", 0),
                           ("unnamed", "unnamed_cols", 0), ("null", "maxnull_pct", 20))
                          if (o.get(key) or 0) > floor})
            row = dict(cost_a=ca.get(t, {}).get("cost"), cost_b=cb.get(t, {}).get("cost"),
                       steps_a=ca.get(t, {}).get("num_steps"), steps_b=cb.get(t, {}).get("num_steps"),
                       ops_a=len(fa), ops_b=len(fb),
                       maxd_a=max([o["depth"] for o in fa], default=0),
                       maxd_b=max([o["depth"] for o in fb], default=0),
                       src_rows=srcrows, issues=iss)
            per_task[t] = row
            print(f"{tag(t):28s} {(row['cost_a'] or 0):>6.2f}/{(row['cost_b'] or 0):<6.2f} "
                  f"{(row['steps_a'] or 0):>4d}/{(row['steps_b'] or 0):<5d} {row['ops_a']:>3d}/{row['ops_b']:<4d} "
                  f"{row['maxd_a']:>4d}/{row['maxd_b']:<4d} {srcrows:>9,d} {','.join(iss) or '-'}")
        aggA = _arm_agg([o for t in tasks for o in opsA[t]], tasks)
        aggB = _arm_agg([o for t in tasks for o in opsB[t]], tasks)
        fils = _files_agg([o for t in tasks for o in (opsA[t] + opsB[t])])
        for nm, g in ((short(A), aggA), (short(B), aggB)):
            if not g:
                continue
            print(f"  [{nm}] ops={g['n_ops']} ({g['ops_per_task']}/task) roles={g['roles']} "
                  f"multi-edit={g['multi_edit_pct']}% capped={g['capped_pct']}% LOCmed={g['loc_med']}")
            print(f"         depth mean/med/max={g['depth_mean']}/{g['depth_med']}/{g['depth_max']}  "
                  f"rows@depth0/1/2+={_fmtn(g['rows_by_depth'][0])}/{_fmtn(g['rows_by_depth'][1])}/{_fmtn(g['rows_by_depth'][2])}")
            print(f"         cardinality: src in/out med={_fmtn(g['src_in_rows_med'])}/{_fmtn(g['src_out_rows_med'])} "
                  f"(p90 {_fmtn(g['src_out_rows_p90'])})  interior out med={_fmtn(g['inter_out_rows_med'])} "
                  f"(p90 {_fmtn(g['inter_out_rows_p90'])})  sink={_fmtn(g['sink_out_rows_med'])}  "
                  f"cells med/p90={_fmtn(g['cells_med'])}/{_fmtn(g['cells_p90'])}")
            print(f"         reduce-share={g['reduce_share_pct']}%  rendered/actual rows med="
                  f"{_fmtn(g['shown_vs_out_med'])}%")
        if fils:
            print(f"  [files] n={fils['n_files']} formats={fils['formats']} "
                  f"sizeKB med/p90/max={_fmtn(fils['size_kb_med'])}/{_fmtn(fils['size_kb_p90'])}/{_fmtn(fils['size_kb_max'])} "
                  f"rows med/max={_fmtn(fils['rows_med'])}/{_fmtn(fils['rows_max'])}")
            print(f"          issues: dirty {fils['dirty_files']}/{fils['n_files']} ({fils['dirty_pct']}%) | "
                  f"dup≥5% {fils['dup5_files']} | empty-rows {fils['emptyrow_files']} | "
                  f"unnamed {fils['unnamed_files']} | null≥20% {fils['highnull_files']} | "
                  f"str-share med {_fmtn(fils['str_share_med'])}%")
            for f, r in sorted(fils["files"].items(), key=lambda kv: -(kv[1]["kb"] or 0))[:a.top]:
                print(f"          {f}  {r['ext']} {_fmtn(r['kb'])}KB rows={_fmtn(r['rows'])} "
                      f"dup={r['dup_pct']}% empty={r['empty_rows_pct']}% unnamed={r['unnamed']} "
                      f"maxnull={r['maxnull_pct']}% str={r['str_share']}%")
        dump["categories"][label] = dict(tasks=[tag(t) for t in tasks], per_task=per_task,
                                         agg_A=aggA, agg_B=aggB, files=fils,
                                         ops_A={t: opsA[t] for t in tasks},
                                         ops_B={t: opsB[t] for t in tasks})
        matrix.append((label, aggA, aggB, fils))

    print(f"\n{'=' * 100}\nCROSS-CASE MATRIX")
    print(f"{'metric':34s} | " + " | ".join(f"{lbl[:30]:>30s}" for lbl, *_ in matrix))
    rows_spec = [
        ("tasks", lambda gA, gB, f, ts: str(ts)),
        ("ops/task (A|B)", lambda gA, gB, f, ts: f"{gA.get('ops_per_task','-')}|{gB.get('ops_per_task','-')}"),
        ("depth med (A|B)", lambda gA, gB, f, ts: f"{gA.get('depth_med','-')}|{gB.get('depth_med','-')}"),
        ("src in-rows med", lambda gA, gB, f, ts: _fmtn(gA.get('src_in_rows_med'))),
        ("interior out-rows med (A|B)", lambda gA, gB, f, ts: f"{_fmtn(gA.get('inter_out_rows_med'))}|{_fmtn(gB.get('inter_out_rows_med'))}"),
        ("rows@depth2+ med (A)", lambda gA, gB, f, ts: _fmtn((gA.get('rows_by_depth') or {}).get(2))),
        ("capped% (A|B)", lambda gA, gB, f, ts: f"{gA.get('capped_pct','-')}|{gB.get('capped_pct','-')}"),
        ("multi-edit% (A|B)", lambda gA, gB, f, ts: f"{gA.get('multi_edit_pct','-')}|{gB.get('multi_edit_pct','-')}"),
        ("file KB med", lambda gA, gB, f, ts: _fmtn(f.get('size_kb_med'))),
        ("file rows med", lambda gA, gB, f, ts: _fmtn(f.get('rows_med'))),
        ("dirty-file %", lambda gA, gB, f, ts: f"{f.get('dirty_pct','-')}"),
        ("formats", lambda gA, gB, f, ts: ",".join(f"{k}:{v}" for k, v in list((f.get('formats') or {}).items())[:3])),
    ]
    for name, fn in rows_spec:
        cells = []
        for lbl, gA, gB, f in matrix:
            ts = len(dump["categories"].get(lbl, {}).get("tasks", []))
            cells.append(fn(gA or {}, gB or {}, f or {}, ts) if (gA or gB or f) else "-")
        print(f"{name:34s} | " + " | ".join(f"{c:>30s}" for c in cells))

    out = Path(a.json) if a.json else (KB_ROOT / "judgment_runs/levers_report/case_metrics" /
                                       f"{short(A)}_vs_{short(B)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dump, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out.relative_to(KB_ROOT)}")


def cmd_judge(a):
    """Run the chunked-LLM-judge metrics (M3 evidence-in-context / M4 step-performed)
    over one or more SUTs. Thin wrapper around scripts/judge_metrics.py; results are
    cached per task in system_scratch/<sut>/<task>/judge_m3m4.json and readable via
    judge_scores()."""
    load_env()
    cmd = [PY, str(KB_ROOT / "scripts/judge_metrics.py"), "--arms", *a.sut,
           "--lens", a.lens, "--judge-model", a.judge_model, "--workers", str(a.workers)]
    if a.tasks_file:
        cmd += ["--tasks-file", a.tasks_file]
    if a.ids:
        cmd += ["--tasks", *a.ids.split()]
    if a.force:
        cmd.append("--force")
    if a.verbose:
        cmd.append("--verbose")
    p = subprocess.Popen(cmd, cwd=KB_ROOT)
    _PROCS.append(p)
    p.wait()


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
          "  kb.py cost --sut <S1> <S2> --trim-top 5 10 # drop top-cost outliers\n"
          "  kb.py cost --sut <S1> <S2>                 # compare totals\n"
          "  kb.py cost --sut <S> --by task --top 10")
    p.add_argument("--sut", required=True, nargs="+", metavar="SUT", help="one or more SUT class names")
    p.add_argument("--by", choices=["workload", "difficulty", "task"], default="workload",
                   help="breakdown dimension (default: workload)")
    p.add_argument("--top", type=int, default=20, metavar="N",
                   help="for --by task: show top N tasks by cost; 0 suppresses task details (default 20)")
    p.add_argument("--trim-top", type=float, nargs="*", default=[], metavar="PCT",
                   help="also print $/task after dropping the highest-cost PCT of tasks per SUT")
    p.set_defaults(fn=cmd_cost)

    p = P("compare",
          "pairwise A-vs-B comparison of two SUTs from existing results (read-only):\n"
          "per-SUT pass-rate + total cost/tokens, the outcome matrix (both pass / A-only /\n"
          "B-only / both fail), the both-pass cost split (who's cheaper when both succeed),\n"
          "and the cost DOMINATORS — the few tasks driving the total-cost gap, each with\n"
          "steps + pass/fail, plus what %% of the gap the top-N account for.\n\n"
          "example:\n  kb.py compare --sut DataflowSystemGPT54LatestSchemaConverge DataflowSystemGPT54DeltaSchemaConverge")
    p.add_argument("--sut", required=True, nargs=2, metavar="SUT", help="exactly two SUT class names (A B)")
    p.add_argument("--top", type=int, default=10, metavar="N", help="how many cost dominators to show (default 10)")
    p.set_defaults(fn=cmd_compare)

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

    p = P("venn",
          "A-vs-B Venn analysis: exclusive wins per side, both-pass intersection, the\n"
          "cost-efficiency split inside the intersection, and operator/file characteristics\n"
          "of the tasks in each category (roles, edits, LOC, file ext/size). Uses the\n"
          "answer-type-aware score (levers-report convention); marks chronic flippers.\n\n"
          "example:\n  kb.py venn --sut DataflowSystemGPT52Delta3kSchemaOnly DataflowSystemGPT52Delta5kSchemaOnly")
    p.add_argument("--sut", required=True, nargs=2, metavar="SUT", help="exactly two SUT class names (A B)")
    p.add_argument("--th", type=float, default=0.9, help="pass threshold on the answer-type metric (default 0.9)")
    p.add_argument("--top", type=int, default=8, metavar="N", help="top-N cost-gap tasks per side to profile (default 8)")
    p.add_argument("--chronic", metavar="JSON", help="chronic-flipper task list (default: judgment_runs/levers_report/chronic_flippers.json)")
    p.set_defaults(fn=cmd_venn)

    p = P("case-metrics",
          "per-Venn-case operator/cardinality/file/data-issue metrics for an A-vs-B pair.\n"
          "Cases: A-only wins, B-only wins, both-pass with a material cost gap each way.\n"
          "Per case and per arm: operator depth, input/output cardinality (sources use the\n"
          "engine-loaded input-table rows from the stats arm's full-data profiles), render\n"
          "pressure (capped share, rendered-vs-actual rows), source file size/format/rows,\n"
          "and engine-measured data issues (duplicate/empty rows, unnamed headers, nulls).\n"
          "Prints per-task detail + per-case aggregates + a cross-case matrix; dumps JSON to\n"
          "judgment_runs/levers_report/case_metrics/.\n\n"
          "example:\n  kb.py case-metrics --sut DataflowSystemGPT52Delta3kSchemaOnly DataflowSystemGPT52Delta5kSchemaOnly")
    p.add_argument("--sut", required=True, nargs=2, metavar="SUT", help="exactly two SUT class names (A B)")
    p.add_argument("--th", type=float, default=0.9, help="pass threshold on the answer-type metric (default 0.9)")
    p.add_argument("--gap", type=float, default=0.10,
                   help="material cost-gap floor as a fraction of the dearer arm (default 0.10 = the twin-noise band)")
    p.add_argument("--top", type=int, default=10, metavar="N", help="max per-case file rows to print (default 10)")
    p.add_argument("--chronic", metavar="JSON", help="chronic-flipper task list (default: judgment_runs/levers_report/chronic_flippers.json)")
    p.add_argument("--json", metavar="OUT", help="JSON output path (default: judgment_runs/levers_report/case_metrics/<A>_vs_<B>.json)")
    p.set_defaults(fn=cmd_case_metrics)

    p = P("judge",
          "chunked LLM-judge metrics over the agent's own rendered context:\n"
          "  M3 = evidence-in-context (per gold subtask: did the agent SEE the value?)\n"
          "  M4 = step-performed (per gold subtask: did the agent DO the step?)\n"
          "Source = last react step's inputMessages; chunk by event (delta) or by\n"
          "operator+code (latest); one judge call per chunk, binary verdicts keyed by\n"
          "subtask id; task score = %% of subtasks covered. Cached per task in\n"
          "system_scratch/<sut>/<task>/judge_m3m4.json (re-run with --force).\n\n"
          "example:\n  kb.py judge --sut DataflowSystemGPT5MiniDelta1kSchemaOnly "
          "DataflowSystemGPT5MiniDelta5kSchemaOnly --tasks-file tasks.txt")
    p.add_argument("--sut", required=True, nargs="+", metavar="SUT", help="one or more SUT class names")
    p.add_argument("--tasks-file", metavar="FILE", help="file with whitespace-separated task ids")
    p.add_argument("--ids", metavar="'T1 T2'", help="explicit task ids (quoted, space-separated)")
    p.add_argument("--lens", choices=["m3", "m4", "both"], default="both")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true", help="re-judge even if cached")
    p.add_argument("--verbose", action="store_true", help="print per-chunk verdicts")
    p.set_defaults(fn=cmd_judge)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

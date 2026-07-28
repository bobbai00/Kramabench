"""Per-knob flip structure; cost+accuracy oracle router; divergence-step localization."""
import json, os, sys, itertools, statistics, hashlib
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
from collections import Counter, defaultdict

SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
M = json.load(open(f'{SC}/score_matrix.json'))
classes = json.load(open(f'{SC}/classes.json'))
tasks = sorted(M)
KNOBS = list(next(iter(M.values())).keys())
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'
def mean(xs): return sum(xs)/len(xs)

# ---- 1. per-knob flip rate (task flips within that knob's reps)
print("===== per-knob flip structure =====")
for k in KNOBS:
    flip = sum(1 for t in tasks if len(set(1 if x>=0.5 else 0 for x in M[t][k]))>1)
    print(f"  {k:12s} within-knob coin tasks: {flip}/98")

# is the flip set shared? pairwise jaccard of coin sets
sets = {k: {t for t in tasks if len(set(1 if x>=0.5 else 0 for x in M[t][k]))>1} for k in KNOBS}
pairs = [(a,b,len(sets[a]&sets[b])/len(sets[a]|sets[b])) for a,b in itertools.combinations(KNOBS,2)]
print(f"  pairwise coin-set jaccard: median={statistics.median([p[2] for p in pairs]):.2f} min={min(p[2] for p in pairs):.2f} max={max(p[2] for p in pairs):.2f}")

# ---- 2. oracle cost+accuracy router
cost = {}
for k, b in BASES.items():
    per = []
    for i in range(6):
        a = f'{b}Replicate{i}'
        if os.path.isdir(f'{ROOT}/{a}'):
            recs = {r['task_id']: r['cost'] for r in kb.load_cost_stats(a)}
            if len(recs) >= 90: per.append(recs)
    cost[k] = {t: mean([p[t] for p in per if t in p]) for t in tasks if any(t in p for p in per)}

acc = {k: {t: mean(M[t][k]) for t in tasks} for k in KNOBS}
static_k = 'C5_DS2k'
static_acc = mean([acc[static_k][t] for t in tasks])
static_cost = mean([cost[static_k][t] for t in tasks])
# router: cheapest knob with rep-mean >= 0.7 (confident pass); else best-acc knob
r_acc, r_cost = [], []
for t in tasks:
    ok = [k for k in KNOBS if acc[k][t] >= 0.7]
    pick = min(ok, key=lambda k: cost[k][t]) if ok else max(KNOBS, key=lambda k: acc[k][t])
    r_acc.append(acc[pick][t]); r_cost.append(cost[pick][t])
print(f"\n===== oracle cost+accuracy router (pick cheapest knob with rep-mean>=0.7) =====")
print(f"  static {static_k}: acc={static_acc:.3f} cost=${static_cost:.4f}")
print(f"  oracle router  : acc={mean(r_acc):.3f} cost=${mean(r_cost):.4f}  (acc +{mean(r_acc)-static_acc:.3f}, cost {100*(mean(r_cost)-static_cost)/static_cost:+.0f}%)")

# ---- 3. divergence-step localization for same-config pass/fail pairs (coins)
print("\n===== divergence localization (same knob, pass rep vs fail rep) =====")
def op_codes(arm, t):
    """ordered list of executed agent code cells from react_steps.json"""
    p = f'{ROOT}/{arm}/{t}/react_steps.json'
    try:
        doc = json.load(open(p))
    except Exception:
        return None
    steps = doc if isinstance(doc, list) else doc.get('steps') or doc.get('react_steps') or []
    out = []
    for s in steps:
        if s.get('role') != 'agent': continue
        for tc in s.get('toolCalls') or []:
            args = tc.get('args') or tc.get('arguments') or {}
            code = args.get('code') or args.get('pythonCode') or ''
            if code: out.append(code)
    return out

loc = Counter(); details = []
for t in tasks:
    if classes[t] != 'coin': continue
    for k in KNOBS:
        b = BASES[k]
        reps = [(i, M[t][k][idx]) for idx, i in enumerate(
                [i for i in range(6) if os.path.isdir(f'{ROOT}/{b}Replicate{i}') and len(kb.answer_scores(f'{b}Replicate{i}'))>=90])]
        ps = [i for i,s in reps if s>=0.5]; fs = [i for i,s in reps if s<0.5]
        if not ps or not fs: continue
        a1, a2 = f'{b}Replicate{ps[0]}', f'{b}Replicate{fs[0]}'
        c1, c2 = op_codes(a1, t), op_codes(a2, t)
        if not c1 or not c2: continue
        n = min(len(c1), len(c2)); div = n
        for j in range(n):
            if hashlib.md5(c1[j].encode()).hexdigest() != hashlib.md5(c2[j].encode()).hexdigest():
                div = j; break
        frac = div / max(len(c1), len(c2), 1)
        loc['step0' if div==0 else ('early(<1/3)' if frac<1/3 else 'mid' if frac<2/3 else 'late(>2/3)')] += 1
        details.append((t, k, div, len(c1), len(c2)))
        break  # one pair per task (first knob with both) to avoid overweighting
print("  first-divergence position (one pair per coin task):", dict(loc))
zero = [d for d in details if d[2]==0]
print(f"  diverge at the VERY FIRST code cell: {len(zero)}/{len(details)}")
for t,k,div,l1,l2 in details[:40]:
    print(f"    {t:26s} {k:10s} div@{div} lens {l1}/{l2}")

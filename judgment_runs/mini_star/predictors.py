"""Do process metrics (M5 materialization, M10 step verdicts) predict pass WITHIN coin tasks?
If yes -> a runtime controller could detect a doomed trace and intervene."""
import json, os, sys, statistics
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
from collections import Counter

SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
classes = json.load(open(f'{SC}/classes.json'))
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'

rows = []
for k, b in BASES.items():
    for i in range(6):
        arm = f'{b}Replicate{i}'
        if not os.path.isdir(f'{ROOT}/{arm}'):
            continue
        sc = kb.answer_scores(arm)
        if len(sc) < 90:
            continue
        for t, s in sc.items():
            d = f'{ROOT}/{arm}/{t}'
            rec = dict(task=t, knob=k, rep=i, score=s, cls=classes.get(t))
            j5 = f'{d}/judge_m5m6.json'
            if os.path.exists(j5):
                try:
                    j = json.load(open(j5))
                    per = j.get('per_subtask') or {}
                    st = [v.get('m5_status') for v in per.values()]
                    m6 = [v.get('m6_done') for v in per.values()]
                    if st:
                        rec['m5'] = sum(1 for x in st if x == 'visible') / len(st)
                        rec['m7'] = sum(1 for x in st if x in ('visible', 'computed_not_shown')) / len(st)
                    md = [x for x in m6 if x is not None]
                    if md:
                        rec['m6'] = sum(1 for x in md if x) / len(md)
                except Exception:
                    pass
            j10 = f'{d}/judge_m10.json'
            if os.path.exists(j10):
                try:
                    j = json.load(open(j10))
                    for v in ['useful', 'wrong_param', 'thwarted', 'off_task', 'redundant']:
                        rec[v] = j.get(f'{v}_frac')
                    rec['n_steps'] = j.get('n_steps')
                except Exception:
                    pass
            rows.append(rec)

print(f"cells={len(rows)}  with M5={sum(1 for r in rows if 'm5' in r)}  with M10={sum(1 for r in rows if 'useful' in r)}")
coins = [r for r in rows if r['cls'] == 'coin']
print(f"coin cells={len(coins)}  M5={sum(1 for r in coins if 'm5' in r)}  M10={sum(1 for r in coins if 'useful' in r)}")

def rate(rs): return sum(1 for r in rs if r['score'] >= 0.5) / len(rs) if rs else float('nan')

for metric in ['m5', 'm7', 'm6', 'useful', 'wrong_param', 'thwarted', 'redundant']:
    have = [r for r in coins if r.get(metric) is not None]
    if len(have) < 30:
        print(f"  {metric}: only {len(have)} coin cells — skip"); continue
    vals = sorted(r[metric] for r in have)
    lo_t, hi_t = vals[len(vals)//3], vals[2*len(vals)//3]
    lo = [r for r in have if r[metric] <= lo_t]
    hi = [r for r in have if r[metric] >= hi_t]
    print(f"  {metric:12s} n={len(have):4d}  low-tercile(<= {lo_t:.2f}) pass={rate(lo):.2f}  high-tercile(>= {hi_t:.2f}) pass={rate(hi):.2f}  spread={rate(hi)-rate(lo):+.2f}")

# also global (all tasks) for reference
print("\n--- all cells (not just coins) ---")
for metric in ['m5', 'm7', 'useful', 'wrong_param', 'thwarted', 'redundant']:
    have = [r for r in rows if r.get(metric) is not None]
    if len(have) < 30: continue
    vals = sorted(r[metric] for r in have)
    lo_t, hi_t = vals[len(vals)//3], vals[2*len(vals)//3]
    lo = [r for r in have if r[metric] <= lo_t]; hi = [r for r in have if r[metric] >= hi_t]
    print(f"  {metric:12s} n={len(have):4d}  low pass={rate(lo):.2f}  high pass={rate(hi):.2f}  spread={rate(hi)-rate(lo):+.2f}")

json.dump(rows, open(f'{SC}/cells.json', 'w'))

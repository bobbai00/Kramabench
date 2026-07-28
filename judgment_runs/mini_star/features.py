"""Task features vs outcome classes; majority-vote realizable gains; randomness correlates."""
import json, statistics, itertools, sys, os, glob
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
from collections import Counter, defaultdict

SCRATCH = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
M = json.load(open(f'{SCRATCH}/score_matrix.json'))
classes = json.load(open(f'{SCRATCH}/classes.json'))
tasks = sorted(M)
KNOBS = list(next(iter(M.values())).keys())
PASS = 0.5
def mean(xs): return sum(xs)/len(xs)

# ============ majority-vote (self-consistency proxy): P(>=2 of 3 reps pass) ============
print("===== realizable resampling: majority-of-3 pass prob (vs single-shot) =====")
for k in KNOBS:
    reps_by_task = {t: M[t][k] for t in tasks}
    n = len(next(iter(reps_by_task.values())))
    single = mean([mean([1 if x>=PASS else 0 for x in v]) for v in reps_by_task.values()])
    maj = []
    for t, v in reps_by_task.items():
        ps = [1 if x>=PASS else 0 for x in v]
        combos = list(itertools.combinations(ps, 3)) if len(ps)>=3 else [tuple(ps)]
        maj.append(mean([1 if sum(c)>=2 else 0 for c in combos]))
    print(f"  {k:12s} reps={n} single={single:.3f} maj3={mean(maj):.3f} (+{mean(maj)-single:.3f})")

# ============ task features ============
import json as J
feat = {}
for wl in ['archeology','astronomy','biomedical','environment','legal','wildfire']:
    w = J.load(open(f'/home/bob/Desktop/bobflow/Kramabench/workload/{wl}.json'))
    for t in w:
        tid = t.get('id')
        if tid is None: continue
        feat[tid] = {
            'workload': wl,
            'difficulty': 'hard' if '-hard-' in tid else 'easy',
            'answer_type': t.get('answer_type'),
            'n_sources': len(t.get('data_sources') or []),
            'n_subtasks': len(t.get('subtasks') or []),
        }

def xtab(key):
    tab = defaultdict(Counter)
    for t in tasks:
        f = feat.get(t)
        if not f: continue
        cls = classes[t]
        g = 'PASS' if cls in ('stable_pass','mostly_pass') else ('FAIL' if cls in ('never_pass','mostly_fail','stable_low') else 'COIN')
        tab[f[key] if key!='n_sources' else ('1' if f[key]==1 else '2-3' if f[key]<=3 else '4+')][g]+=1
    print(f"\n--- {key} ---")
    for kk in sorted(tab, key=str):
        c = tab[kk]; n = sum(c.values())
        print(f"  {str(kk):22s} n={n:3d}  PASS {c['PASS']/n:5.1%}  COIN {c['COIN']/n:5.1%}  FAIL {c['FAIL']/n:5.1%}")

print("\n===== outcome class by task feature (PASS=stable+mostly_pass, FAIL=never+mostly_fail+stable_low) =====")
for key in ['workload','difficulty','answer_type','n_sources']:
    xtab(key)
# n_subtasks binned
tab = defaultdict(Counter)
for t in tasks:
    f = feat.get(t)
    if not f: continue
    cls = classes[t]
    g = 'PASS' if cls in ('stable_pass','mostly_pass') else ('FAIL' if cls in ('never_pass','mostly_fail','stable_low') else 'COIN')
    b = '<=4' if f['n_subtasks']<=4 else '5-7' if f['n_subtasks']<=7 else '8+'
    tab[b][g]+=1
print("\n--- n_subtasks ---")
for kk in ['<=4','5-7','8+']:
    c = tab[kk]; n = sum(c.values()) or 1
    print(f"  {kk:22s} n={n:3d}  PASS {c['PASS']/n:5.1%}  COIN {c['COIN']/n:5.1%}  FAIL {c['FAIL']/n:5.1%}")

# ============ randomness correlates within coin tasks ============
print("\n===== within COIN tasks: pass-rep vs fail-rep run characteristics =====")
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
arm_stats = {}
for k, b in BASES.items():
    for i in range(6):
        arm = f'{b}Replicate{i}'
        d = f'/home/bob/Desktop/bobflow/Kramabench/system_scratch/{arm}'
        if not os.path.isdir(d): continue
        recs = {r['task_id']: r for r in kb.load_cost_stats(arm)}
        if len(recs) >= 90:
            arm_stats[(k, arm)] = recs

coins = [t for t in tasks if classes[t]=='coin']
rows_pass, rows_fail = [], []
for t in coins:
    for (k, arm), recs in arm_stats.items():
        if t not in recs: continue
        sc_map = kb.answer_scores(arm) if False else None
# faster: rebuild score lookup once
score_lookup = {}
for (k, arm) in arm_stats:
    score_lookup[(k,arm)] = kb.answer_scores(arm)
for t in coins:
    for (k, arm), recs in arm_stats.items():
        s = score_lookup[(k,arm)].get(t)
        if s is None or t not in recs: continue
        r = recs[t]
        (rows_pass if s>=PASS else rows_fail).append(r)
def agg(rows, f): return statistics.median([r[f] for r in rows])
for f in ['num_steps','cost','input_tokens','output_tokens']:
    print(f"  {f:14s} pass-med={agg(rows_pass,f):10.4f}  fail-med={agg(rows_fail,f):10.4f}")
print(f"  n pass-runs={len(rows_pass)} fail-runs={len(rows_fail)}")

# flip-rate by workload for coins
print("\ncoin tasks by workload:", Counter(feat[t]['workload'] for t in coins if t in feat).most_common())
print("coin tasks:", coins)

# chronic flipper overlap with old (gpt-5.2 era) list
old = json.load(open('/home/bob/Desktop/bobflow/Kramabench/judgment_runs/levers_report/chronic_flippers.json'))
old_set = set(old if isinstance(old, list) else old.keys())
print(f"\noverlap with gpt-5.2-era chronic_flippers ({len(old_set)}): {len(set(coins)&old_set)} shared")
print(sorted(set(coins)&old_set))

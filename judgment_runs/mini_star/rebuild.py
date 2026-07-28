import json, os, sys
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'
SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
def scores_fix(a):
    base = os.path.join(ROOT, a); out = {}
    for d in sorted(os.listdir(base)):
        p = f'{base}/{d}/evaluation.json'
        if not os.path.exists(p): continue
        try: ev = json.load(open(p))
        except Exception: continue
        gt = kb._load(f'{base}/{d}/ground_truth.json')
        k = kb.ANSWER_TYPE_METRIC.get(gt.get('answer_type') or '')
        if k and k in ev and ev[k] is None:   # judge never ran -> not a zero
            continue
        v = ev.get(k) if k and isinstance(ev.get(k), (int, float)) else None
        if v is None:
            vals = [float(ev[x]) for x in kb.SCORE_METRICS if isinstance(ev.get(x), (int, float))]
            v = max(vals) if vals else 0.0
        out[d] = float(v)
    return out
BASES = {'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
matrix = {}
for k, b in BASES.items():
    reps = {}
    for i in range(6):
        a = f'{b}Replicate{i}'
        if os.path.isdir(f'{ROOT}/{a}'):
            s = scores_fix(a)
            if len(s) >= 90: reps[a] = s
    matrix[k] = reps
    print(k, len(reps), 'reps')
tasks = sorted(set.intersection(*[set(s) for k in matrix for s in matrix[k].values()]))
M = {t: {k: [matrix[k][a][t] for a in matrix[k]] for k in matrix} for t in tasks}
json.dump(M, open(f'{SC}/score_matrix.json','w'))
print('matched tasks:', len(tasks))
def mean(xs): return sum(xs)/len(xs)
# classes
classes = {}
for t in tasks:
    v = [s for k in matrix for s in M[t][k]]
    if all(x >= .5 for x in v): c='stable_pass'
    elif all(x < .5 for x in v): c='never_pass' if all(x==0 for x in v) else 'stable_low'
    else:
        pr = mean([1 if x>=.5 else 0 for x in v])
        c = 'mostly_pass' if pr>=.8 else ('mostly_fail' if pr<=.2 else 'coin')
    classes[t]=c
json.dump(classes, open(f'{SC}/classes.json','w'))
from collections import Counter
print(Counter(classes.values()).most_common())
# variance decomposition
cells=[(t,k,s) for t in tasks for k in matrix for s in M[t][k]]
grand=mean([c[2] for c in cells])
tm={t:mean([s for k in matrix for s in M[t][k]]) for t in tasks}
tkm={(t,k):mean(M[t][k]) for t in tasks for k in matrix}
sst=sum((s-grand)**2 for _,_,s in cells)
print(f"SS task {sum((tm[t]-grand)**2 for t,_,_ in cells)/sst:.1%} | knob {sum((tkm[(t,k)]-tm[t])**2 for t,k,_ in cells)/sst:.1%} | rep {sum((s-tkm[(t,k)])**2 for t,k,s in cells)/sst:.1%}")
print('knob means:', {k: round(mean([mean(M[t][k]) for t in tasks]),3) for k in matrix})

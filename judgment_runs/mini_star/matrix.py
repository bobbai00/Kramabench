"""Build task x knob x rep score matrix + headroom decomposition."""
import json, statistics, itertools, sys
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb

KNOBS = {
    'anchor_D1k':  'DataflowSystemGPT5MiniDelta1kSchemaOnly',
    'C1_D5k':      'DataflowSystemGPT5MiniDelta5kSchemaOnly',
    'C2_DS1k':     'DataflowSystemGPT5MiniDeltaStats1kD2',
    'C5_DS2k':     'DataflowSystemGPT5MiniDeltaStats2kD2',
    'C4_DS5k':     'DataflowSystemGPT5MiniDeltaStats5kD2',
    'C3_L1kCode':  'DataflowSystemGPT5MiniLatest1kCodeInSnap',
    'C6_LS1k':     'DataflowSystemGPT5MiniLatestStats1kD2',
}

# collect reps: only complete ones (>= 90 scored tasks)
matrix = {}   # knob -> rep_arm -> {task: score}
for k, base in KNOBS.items():
    reps = {}
    for i in range(6):
        arm = f'{base}Replicate{i}'
        s = kb.answer_scores(arm)
        if len(s) >= 90:
            reps[arm] = s
    matrix[k] = reps

tasks = sorted(set.intersection(*[set(s) for k in matrix for s in matrix[k].values()]))
print(f"knobs={len(matrix)} total_runs={sum(len(v) for v in matrix.values())} matched_tasks={len(tasks)}")
for k, v in matrix.items():
    print(f"  {k}: {len(v)} reps")

# --- flatten: per task, per knob, list of rep scores
M = {t: {k: [matrix[k][a][t] for a in matrix[k]] for k in matrix} for t in tasks}
json.dump(M, open('/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad/score_matrix.json', 'w'))

PASS = 0.5  # treat score>=0.5 as pass for pass-rate views; keep continuous elsewhere

def mean(xs): return sum(xs) / len(xs)

# ============ 1. HEADROOM DECOMPOSITION ============
print("\n===== headroom (continuous score, mean over tasks) =====")
knob_means = {k: mean([mean(M[t][k]) for t in tasks]) for k in matrix}
best_static = max(knob_means, key=knob_means.get)
print("per-knob expected (rep-mean):", {k: round(v,3) for k,v in sorted(knob_means.items(), key=lambda x:-x[1])})

# oracle over EVERYTHING (any knob, any rep) = union ceiling incl. lottery
oracle_all = mean([max(max(M[t][k]) for k in matrix) for t in tasks])
# oracle knob-router judged on rep-mean (realizable by routing, net of luck):
oracle_knob = mean([max(mean(M[t][k]) for k in matrix) for t in tasks])
# resampling ceiling: best single knob, max over its reps (luck within one config)
resample_best = {k: mean([max(M[t][k]) for t in tasks]) for k in matrix}
rb = max(resample_best.values())
rbk = max(resample_best, key=resample_best.get)
print(f"best static knob         : {best_static} {knob_means[best_static]:.3f}")
print(f"oracle knob-router (rep-mean basis): {oracle_knob:.3f}  (+{oracle_knob-knob_means[best_static]:.3f})")
print(f"best-of-reps single knob ({rbk}, luck): {rb:.3f}  (+{rb-knob_means[best_static]:.3f})")
print(f"oracle all runs (union w/ lottery)  : {oracle_all:.3f}  (+{oracle_all-knob_means[best_static]:.3f})")

# best-of-K curves for the 5-rep knobs
print("\nbest-of-K rep curve (expected max of K random reps):")
for k in ['anchor_D1k','C1_D5k','C2_DS1k','C3_L1kCode']:
    reps_n = len(next(iter([matrix[k]])) and matrix[k])
    row = []
    for K in range(1, reps_n+1):
        vals = []
        for combo in itertools.combinations(range(reps_n), K):
            vals.append(mean([max(M[t][k][i] for i in combo) for t in tasks]))
        row.append(f"K={K}:{mean(vals):.3f}")
    print(f"  {k:12s} " + "  ".join(row))

# ============ 2. VARIANCE DECOMPOSITION ============
print("\n===== variance decomposition (score cells) =====")
cells = [(t, k, s) for t in tasks for k in matrix for s in M[t][k]]
grand = mean([c[2] for c in cells])
task_mean = {t: mean([s for k in matrix for s in M[t][k]]) for t in tasks}
tk_mean = {(t,k): mean(M[t][k]) for t in tasks for k in matrix}
ss_total = sum((s-grand)**2 for _,_,s in cells)
ss_task  = sum((task_mean[t]-grand)**2 for t,_,_ in cells)
ss_knob_within = sum((tk_mean[(t,k)]-task_mean[t])**2 for t,k,_ in cells)
ss_rep   = sum((s-tk_mean[(t,k)])**2 for t,k,s in cells)
print(f"SS task {ss_task/ss_total:.1%} | SS knob(within task) {ss_knob_within/ss_total:.1%} | SS rep(residual) {ss_rep/ss_total:.1%}")

# ============ 3. TASK OUTCOME CLASSES ============
allscores = {t: [s for k in matrix for s in M[t][k]] for t in tasks}
classes = {}
for t in tasks:
    v = allscores[t]
    if all(x >= PASS for x in v): c = 'stable_pass'
    elif all(x < PASS for x in v):
        c = 'never_pass' if all(x == 0 for x in v) else 'stable_low'
    else:
        pr = mean([1 if x >= PASS else 0 for x in v])
        c = 'mostly_pass' if pr >= .8 else ('mostly_fail' if pr <= .2 else 'coin')
    classes[t] = c
from collections import Counter
print("\n===== outcome classes (30 runs/task, pass=score>=0.5) =====")
print(Counter(classes.values()).most_common())
json.dump(classes, open('/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad/classes.json','w'))

# ============ 4. KNOB AFFINITY: non-overlapping rep separation ============
print("\n===== knob affinity: tasks where knob A's WORST rep > knob B's BEST rep =====")
aff = []
for t in tasks:
    for a in matrix:
        for b in matrix:
            if a >= b: continue
            if min(M[t][a]) > max(M[t][b]) and min(M[t][a]) - max(M[t][b]) >= 0.5:
                aff.append((t, 'A>' , a, b))
            elif min(M[t][b]) > max(M[t][a]) and min(M[t][b]) - max(M[t][a]) >= 0.5:
                aff.append((t, 'B>', b, a))
win_count = Counter()
task_aff = {}
for t, d, w, l in aff:
    win_count[w] += 1
    task_aff.setdefault(t, []).append((w, l))
print("clean separations (winner-knob counts):", win_count.most_common())
print(f"tasks with >=1 clean knob separation: {len(task_aff)}")
for t, pairs in sorted(task_aff.items()):
    ws = Counter(w for w,_ in pairs)
    print(f"  {t}: {dict(ws)}  (vs {sorted(set(l for _,l in pairs))})")
json.dump({t: pairs for t, pairs in task_aff.items()},
          open('/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad/knob_affinity.json','w'))

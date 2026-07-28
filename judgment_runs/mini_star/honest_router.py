"""HONEST routable headroom: cross-validated oracle router.
Pick the knob per task using rep subset A; score it on held-out rep subset B.
Anything above static-best on HELD-OUT reps is real routable signal, not noise-chasing."""
import json, itertools, random, statistics, sys
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')

SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
M = json.load(open(f'{SC}/score_matrix.json'))
tasks = sorted(M)
KNOBS = list(next(iter(M.values())).keys())
def mean(xs): return sum(xs)/len(xs)
random.seed(11)

# only knobs with >=4 reps can be split 2/2+
usable = [k for k in KNOBS if len(M[tasks[0]][k]) >= 4]
print("knobs usable for CV split (>=4 reps):", usable)

static_best_k = max(KNOBS, key=lambda k: mean([mean(M[t][k]) for t in tasks]))
print("static best knob:", static_best_k)

TRIALS = 300
gains_cv, gains_naive, held_static = [], [], []
for _ in range(TRIALS):
    sel_idx, ev_idx = {}, {}
    for k in usable:
        n = len(M[tasks[0]][k])
        idx = list(range(n)); random.shuffle(idx)
        h = n // 2
        sel_idx[k], ev_idx[k] = idx[:h], idx[h:]
    # router picks on SELECT half, scored on EVAL half
    r_cv, r_naive, s_static = [], [], []
    for t in tasks:
        pick = max(usable, key=lambda k: mean([M[t][k][i] for i in sel_idx[k]]))
        r_cv.append(mean([M[t][pick][i] for i in ev_idx[pick]]))
        # naive oracle: pick and score on the SAME (select) half
        r_naive.append(mean([M[t][pick][i] for i in sel_idx[pick]]))
        s_static.append(mean([M[t][static_best_k][i] for i in ev_idx[static_best_k]])
                        if static_best_k in ev_idx else mean(M[t][static_best_k]))
    gains_cv.append(mean(r_cv)); gains_naive.append(mean(r_naive)); held_static.append(mean(s_static))

print(f"\nheld-out static best      : {mean(held_static):.3f}")
print(f"CV router (honest)        : {mean(gains_cv):.3f}   gain {mean(gains_cv)-mean(held_static):+.3f}")
print(f"same-half router (naive)  : {mean(gains_naive):.3f}   gain {mean(gains_naive)-mean(held_static):+.3f}  <-- inflated by noise-chasing")
print(f"CV router std over trials : {statistics.pstdev(gains_cv):.3f}")

# How much of the naive gain is pure noise? bootstrap a NULL: all knobs identical
# (permute knob labels within task) -> router gain that a signal-free world produces
null = []
for _ in range(TRIALS):
    sel_idx, ev_idx = {}, {}
    for k in usable:
        n = len(M[tasks[0]][k]); idx = list(range(n)); random.shuffle(idx); h = n//2
        sel_idx[k], ev_idx[k] = idx[:h], idx[h:]
    r = []
    for t in tasks:
        pool = [x for k in usable for x in M[t][k]]
        random.shuffle(pool)
        # rebuild fake knobs from the same pooled scores (destroys knob identity)
        fake = {}
        c = 0
        for k in usable:
            n = len(M[t][k]); fake[k] = pool[c:c+n]; c += n
        pick = max(usable, key=lambda k: mean([fake[k][i] for i in sel_idx[k]]))
        r.append(mean([fake[pick][i] for i in ev_idx[pick]]))
    null.append(mean(r))
print(f"NULL (knob labels destroyed): {mean(null):.3f}  -> signal-free CV router gain {mean(null)-mean(held_static):+.3f}")
print(f"REAL routable signal        : {mean(gains_cv)-mean(null):+.3f}")

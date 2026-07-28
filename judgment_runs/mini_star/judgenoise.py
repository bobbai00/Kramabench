"""(a) string_approx coins: same answer, different score? (b) answer-cluster self-consistency."""
import json, os, sys, re, itertools
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
from collections import Counter

SCRATCH = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
M = json.load(open(f'{SCRATCH}/score_matrix.json'))
classes = json.load(open(f'{SCRATCH}/classes.json'))
tasks = sorted(M)
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'

# answer_type per task
feat = {}
for wl in ['archeology','astronomy','biomedical','environment','legal','wildfire']:
    for t in json.load(open(f'/home/bob/Desktop/bobflow/Kramabench/workload/{wl}.json')):
        if t.get('id'): feat[t['id']] = t.get('answer_type')

arms = []
for k, b in BASES.items():
    for i in range(6):
        a = f'{b}Replicate{i}'
        if os.path.isdir(f'{ROOT}/{a}'):
            s = kb.answer_scores(a)
            if len(s) >= 90:
                arms.append((k, a, s))

def get_answer(arm, t):
    p = f'{ROOT}/{arm}/{t}/answer.json'
    try:
        return json.load(open(p)).get('answer')
    except Exception:
        return None

def norm(ans):
    if ans is None: return None
    s = str(ans).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    try:
        f = float(s.replace(',', ''))
        return round(f, 6)
    except ValueError:
        return s

# ---- (a) same-answer-different-score on string_approx / list_approx coins
print("===== judge-noise check: identical normalized answer scored BOTH pass and fail =====")
noisy = []
for t in tasks:
    if classes[t] != 'coin': continue
    by_ans = {}
    for k, a, s in arms:
        if t not in s: continue
        na = norm(get_answer(a, t))
        by_ans.setdefault(na, []).append(s[t])
    for na, scores in by_ans.items():
        if na is None: continue
        if any(x >= 0.5 for x in scores) and any(x < 0.5 for x in scores):
            noisy.append((t, feat.get(t), str(na)[:60], [round(x,2) for x in scores]))
for row in noisy:
    print(f"  {row[0]:24s} {row[1]:20s} ans={row[2]!r} scores={row[3]}")
print(f"tasks with same-answer flip: {len(set(r[0] for r in noisy))} / 31 coins")

# ---- (b) answer-cluster self-consistency of 3 reps within each knob
print("\n===== answer-cluster vote-of-3 (pick modal normalized answer, take its score) =====")
def sc_vote(vals):  # vals: list of (norm_answer, score)
    cnt = Counter(v[0] for v in vals if v[0] is not None)
    if not cnt: return 0.0
    modal, mc = cnt.most_common(1)[0]
    if mc == 1:   # all distinct -> random pick
        return sum(v[1] for v in vals)/len(vals)
    return max(s for a, s in vals if a == modal)

for k in BASES:
    karms = [(a, s) for kk, a, s in arms if kk == k]
    if len(karms) < 3: continue
    singles, votes = [], []
    for t in tasks:
        pool = [(norm(get_answer(a, t)), s[t]) for a, s in karms if t in s]
        if len(pool) < 3: continue
        singles.append(sum(1 if s>=0.5 else 0 for _, s in pool)/len(pool))
        combos = list(itertools.combinations(pool, 3))
        votes.append(sum(1 if sc_vote(c)>=0.5 else 0 for c in combos)/len(combos))
    print(f"  {k:12s} single={sum(singles)/len(singles):.3f}  vote3={sum(votes)/len(votes):.3f}  (+{sum(votes)/len(votes)-sum(singles)/len(singles):.3f})  n={len(singles)}")

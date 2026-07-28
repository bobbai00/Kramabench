"""Realizable (non-oracle) strategies: knob-diverse agreement cascade vs static best.
Simulated by resampling actual runs from the matrix. No ground truth used for routing."""
import json, os, sys, re, itertools, random, statistics
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb

SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
classes = json.load(open(f'{SC}/classes.json'))
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'

# runs[knob] = list of dicts task -> (score, cost, norm_answer)
def norm(a):
    if a is None: return None
    s = re.sub(r'\s+', ' ', str(a).strip().lower())
    try: return round(float(s.replace(',', '')), 6)
    except ValueError: return s

runs = {}
for k, b in BASES.items():
    lst = []
    for i in range(6):
        arm = f'{b}Replicate{i}'
        if not os.path.isdir(f'{ROOT}/{arm}'): continue
        sc = kb.answer_scores(arm)
        if len(sc) < 90: continue
        cost = {r['task_id']: r['cost'] for r in kb.load_cost_stats(arm)}
        d = {}
        for t, s in sc.items():
            try: ans = json.load(open(f'{ROOT}/{arm}/{t}/answer.json')).get('answer')
            except Exception: ans = None
            d[t] = (s, cost.get(t, 0.0), norm(ans))
        lst.append(d)
    runs[k] = lst

tasks = sorted(set.intersection(*[set(d) for k in runs for d in runs[k]]))
print(f"tasks={len(tasks)}  knobs={len(runs)}  runs={sum(len(v) for v in runs.values())}")

def mean(xs): return sum(xs)/len(xs)
random.seed(7)
TRIALS = 400

def sample(k, t):
    return random.choice(runs[k])[t]

def strat_static(k):
    a, c = [], []
    for t in tasks:
        s, co, _ = sample(k, t); a.append(s); c.append(co)
    return mean(a), mean(c)

def strat_vote3(k):
    """3 reps of same knob, modal answer wins."""
    a, c = [], []
    for t in tasks:
        picks = [sample(k, t) for _ in range(3)]
        c.append(sum(p[1] for p in picks))
        from collections import Counter
        cnt = Counter(p[2] for p in picks if p[2] is not None)
        if cnt and cnt.most_common(1)[0][1] > 1:
            modal = cnt.most_common(1)[0][0]
            a.append(max(p[0] for p in picks if p[2] == modal))
        else:
            a.append(random.choice(picks)[0])
    return mean(a), mean(c)

def strat_cascade(k1, k2, k3):
    """Run k1; run k2; if normalized answers agree -> stop. Else run k3 and take
    majority (or k3's answer if all differ). Knob-diverse, no ground truth used."""
    a, c = [], []
    for t in tasks:
        p1, p2 = sample(k1, t), sample(k2, t)
        cost = p1[1] + p2[1]
        if p1[2] is not None and p1[2] == p2[2]:
            a.append(max(p1[0], p2[0])); c.append(cost); continue
        p3 = sample(k3, t); cost += p3[1]
        from collections import Counter
        cnt = Counter(p[2] for p in (p1, p2, p3) if p[2] is not None)
        if cnt and cnt.most_common(1)[0][1] > 1:
            modal = cnt.most_common(1)[0][0]
            a.append(max(p[0] for p in (p1,p2,p3) if p[2] == modal))
        else:
            a.append(p3[0])
        c.append(cost)
    return mean(a), mean(c)

def repeat(fn, *args):
    res = [fn(*args) for _ in range(TRIALS)]
    return mean([r[0] for r in res]), mean([r[1] for r in res])

print("\n=== realizable strategies (no ground truth in the loop) ===")
print(f"{'strategy':46s} {'acc':>6s} {'$/task':>8s} {'$/pass':>8s}")
out = []
for k in BASES:
    a, c = repeat(strat_static, k)
    out.append((f'static {k}', a, c))
for k in ['anchor_D1k', 'C5_DS2k', 'C3_L1kCode']:
    a, c = repeat(strat_vote3, k)
    out.append((f'vote3 same-knob {k}', a, c))
for combo in [('anchor_D1k','C1_D5k','C2_DS1k'), ('anchor_D1k','C5_DS2k','C3_L1kCode'),
              ('C5_DS2k','C3_L1kCode','C1_D5k'), ('anchor_D1k','C3_L1kCode','C5_DS2k')]:
    a, c = repeat(strat_cascade, *combo)
    out.append((f'cascade {"+".join(combo)}', a, c))
for name, a, c in sorted(out, key=lambda x: -x[1]):
    print(f"{name:46s} {a:6.3f} {c:8.4f} {c/max(a,1e-9):8.4f}")

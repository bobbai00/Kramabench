"""Fair CV router: same knob pool for router and static baseline. Plus
'confident-only' routing (route only where select-half separation is large)."""
import json, random, statistics, sys
SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
M = json.load(open(f'{SC}/score_matrix.json'))
tasks = sorted(M)
ALL = list(next(iter(M.values())).keys())
USABLE = [k for k in ALL if len(M[tasks[0]][k]) >= 4]   # anchor,C1,C2,C4,C3
def mean(xs): return sum(xs)/len(xs)
random.seed(11)
TRIALS = 400

def run(margin=None, pool=USABLE):
    """margin=None -> always route. margin=x -> route only if select-half best
    exceeds select-half of the default knob by >= x, else use default."""
    default = max(pool, key=lambda k: mean([mean(M[t][k]) for t in tasks]))
    router, static, nullr = [], [], []
    for _ in range(TRIALS):
        sel, ev = {}, {}
        for k in pool:
            n = len(M[tasks[0]][k]); idx = list(range(n)); random.shuffle(idx); h = n//2
            sel[k], ev[k] = idx[:h], idx[h:]
        r, s, nl = [], [], []
        for t in tasks:
            selmean = {k: mean([M[t][k][i] for i in sel[k]]) for k in pool}
            best = max(pool, key=lambda k: selmean[k])
            pick = best if (margin is None or selmean[best] - selmean[default] >= margin) else default
            r.append(mean([M[t][pick][i] for i in ev[pick]]))
            s.append(mean([M[t][default][i] for i in ev[default]]))
            # null: destroy knob identity by pooling & reshuffling this task's scores
            poolsc = [x for k in pool for x in M[t][k]]; random.shuffle(poolsc)
            fake, c = {}, 0
            for k in pool:
                n = len(M[t][k]); fake[k] = poolsc[c:c+n]; c += n
            fsel = {k: mean([fake[k][i] for i in sel[k]]) for k in pool}
            fbest = max(pool, key=lambda k: fsel[k])
            fpick = fbest if (margin is None or fsel[fbest] - fsel[default] >= margin) else default
            nl.append(mean([fake[fpick][i] for i in ev[fpick]]))
        router.append(mean(r)); static.append(mean(s)); nullr.append(mean(nl))
    return default, mean(static), mean(router), mean(nullr), statistics.pstdev(router)

print(f"{'margin':>8s} {'default':12s} {'static':>7s} {'router':>7s} {'null':>7s} {'router-null':>12s}")
for m in [None, 0.2, 0.34, 0.5, 0.7, 1.0]:
    d, s, r, n, sd = run(m)
    lab = 'always' if m is None else f'>= {m}'
    print(f"{lab:>8s} {d:12s} {s:7.3f} {r:7.3f} {n:7.3f} {r-n:+12.3f}   (router sd {sd:.3f})")

# per-task: which tasks would a large-margin router move, and does it help held-out?
print("\n--- tasks where select-half margin >= 0.7 fires most often (400 trials) ---")
from collections import Counter
fire = Counter(); helped = Counter(); hurt = Counter()
default = max(USABLE, key=lambda k: mean([mean(M[t][k]) for t in tasks]))
for _ in range(TRIALS):
    sel, ev = {}, {}
    for k in USABLE:
        n = len(M[tasks[0]][k]); idx = list(range(n)); random.shuffle(idx); h = n//2
        sel[k], ev[k] = idx[:h], idx[h:]
    for t in tasks:
        selmean = {k: mean([M[t][k][i] for i in sel[k]]) for k in USABLE}
        best = max(USABLE, key=lambda k: selmean[k])
        if selmean[best] - selmean[default] >= 0.7:
            fire[t] += 1
            gain = mean([M[t][best][i] for i in ev[best]]) - mean([M[t][default][i] for i in ev[default]])
            if gain > 0.1: helped[t] += 1
            elif gain < -0.1: hurt[t] += 1
for t, f in fire.most_common(15):
    print(f"  {t:24s} fires {f/TRIALS:5.0%}  helped {helped[t]/max(f,1):4.0%}  hurt {hurt[t]/max(f,1):4.0%}")

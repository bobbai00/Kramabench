#!/usr/bin/env python
"""
Pilot contrastive learning + eval (end-to-end).

Learns a selector from the latest-vs-delta CONTRAST: given a task's trajectory
features, predict whether the full-delta trajectory helps over latest-core
(label = ΔQ > eps). Evaluated with leave-one-out CV against three baselines:
always-latest, always-delta, and the per-task oracle. Reports the accuracy /
token trade-off — the metric the learned context selector ultimately targets.

N is small (pilot), so this is a PLUMBING + directional-signal demo, not a
significant result. No sklearn in the venv → tiny numpy L2-logistic-regression.

Usage:
    python bobflow_context_learning/train/train_eval.py [--eps 0.02]
"""
import argparse
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "bobflow_context_learning", "data")

# Structural features hypothesized to predict "trajectory helps":
FEATURES = ["n_bundles", "n_failed", "frac_failed", "n_superseded", "n_deleted", "max_churn", "mean_churn"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X, y, l2=1.0, lr=0.1, iters=2000):
    """Standardized L2-regularized logistic regression via gradient descent."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((len(Xs), 1)), Xs])  # bias
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = sigmoid(Xs @ w)
        grad = Xs.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:] / len(y)
        w -= lr * grad
    return w, mu, sd


def predict(w, mu, sd, x):
    xs = np.hstack([[1.0], (x - mu) / sd])
    return float(sigmoid(xs @ w))


def auc(y, scores):
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02, help="ΔQ threshold for 'delta helped'")
    args = ap.parse_args()

    contrast = {r["task_id"]: r for r in json.load(open(os.path.join(DATA, "contrast.json")))["rows"]}
    feats = {t["task_id"]: t for t in json.load(open(os.path.join(DATA, "features.json")))["tasks"]}

    rows = []
    for tid, c in contrast.items():
        if c.get("delta_q") is None or tid not in feats:
            continue
        tf = feats[tid]["task_features"]
        rows.append({
            "task_id": tid,
            "x": np.array([float(tf.get(k, 0)) for k in FEATURES]),
            "y": int(c["delta_q"] > args.eps),
            "dq": c["delta_q"], "ql": c["q_latest"], "qd": c["q_delta"],
            "tl": c["tok_latest"] or 0, "td": c["tok_delta"] or 0,
        })
    n = len(rows)
    if n < 4:
        print(f"[train] only {n} complete tasks — need both arms done. Re-run after data-gen finishes.")
        return

    y = np.array([r["y"] for r in rows])
    print(f"[train] {n} tasks complete | label balance: delta-helped={int(y.sum())}, not={int(n - y.sum())} (eps={args.eps})")

    # ---- baselines (accuracy / token trade-off) ----
    acc_latest = np.mean([r["ql"] for r in rows]); tok_latest = np.mean([r["tl"] for r in rows])
    acc_delta = np.mean([r["qd"] for r in rows]); tok_delta = np.mean([r["td"] for r in rows])
    # oracle: per task take the better arm (tie -> latest, cheaper)
    acc_oracle = np.mean([max(r["ql"], r["qd"]) for r in rows])
    tok_oracle = np.mean([r["td"] if r["qd"] > r["ql"] else r["tl"] for r in rows])

    # ---- learned selector, leave-one-out CV ----
    loo_scores, loo_choice_acc, sel_acc, sel_tok = [], [], [], []
    for i in range(n):
        tr = [r for j, r in enumerate(rows) if j != i]
        Xtr = np.array([r["x"] for r in tr]); ytr = np.array([r["y"] for r in tr])
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            # degenerate fold (one class) — fall back to majority
            p = float(ytr.mean())
        else:
            w, mu, sd = fit_logreg(Xtr, ytr)
            p = predict(w, mu, sd, rows[i]["x"])
        loo_scores.append(p)
        use_delta = p >= 0.5
        loo_choice_acc.append(int(use_delta) == rows[i]["y"])
        sel_acc.append(rows[i]["qd"] if use_delta else rows[i]["ql"])
        sel_tok.append(rows[i]["td"] if use_delta else rows[i]["tl"])

    print("\n===== EVAL: accuracy / token trade-off (mean over tasks) =====")
    print(f"  {'policy':16s} {'accuracy':>9s} {'tokens':>10s} {'acc/1k-tok':>11s}")
    for name, a, t in [("always-latest", acc_latest, tok_latest), ("always-delta", acc_delta, tok_delta),
                       ("learned (LOO)", np.mean(sel_acc), np.mean(sel_tok)), ("oracle", acc_oracle, tok_oracle)]:
        print(f"  {name:16s} {a:9.3f} {t:10,.0f} {a/(t/1000):11.4f}")

    print("\n===== SELECTOR quality (leave-one-out) =====")
    print(f"  arm-choice accuracy: {np.mean(loo_choice_acc):.3f}  (did it pick the better arm?)")
    print(f"  AUC (ΔQ>eps):        {auc(y, np.array(loo_scores)):.3f}")

    # ---- feature signal (full-data fit, for interpretation) ----
    if 0 < y.sum() < n:
        X = np.array([r["x"] for r in rows])
        w, mu, sd = fit_logreg(X, y)
        print("\n===== feature coefficients (standardized; + ⇒ predicts 'trajectory helps') =====")
        for k, c in sorted(zip(FEATURES, w[1:]), key=lambda kv: -abs(kv[1])):
            print(f"  {k:16s} {c:+.3f}")
    print("\n  NOTE: pilot N — directional/plumbing demo, not a significant result.")


if __name__ == "__main__":
    main()

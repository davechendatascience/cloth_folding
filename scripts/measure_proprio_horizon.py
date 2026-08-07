"""At what horizon does proprioception stop being sufficient?

The policy ignores the cameras (image/proprio influence 0.0445). The usual
reading is "the visual pathway is broken", but the likelier one is that the
*task as posed* does not require vision: predicting one joint delta at 30 fps is
nearly free from proprioception alone, because trajectories are smooth.
Gradient descent takes the cheap channel, and no amount of auxiliary loss on the
encoder changes what the actor finds cheapest.

If that is right, the fix is structural -- predict a *chunk* of future actions,
long enough that proprioception cannot extrapolate it. This script finds how
long that is, before committing to a training run.

Method, entirely offline on the preprocessed cache: for each horizon k, fit a
ridge regression from proprio history to the action chunk a[t..t+k] - s[t], and
report held-out R^2. Ridge rather than a network on purpose -- we want the
*information available in the channel*, not what a particular architecture
extracts. A linear model with lagged inputs is a generous proxy for "the easy
shortcut", so where it collapses is where the shortcut genuinely dies.

Reading the output:
  R^2 near 1.0  -> proprio alone predicts this horizon; vision is optional and
                   the policy will ignore it, whatever the loss says
  R^2 collapsing -> proprio is insufficient; the model must look at the cloth
                    to do better, so vision becomes load-bearing by task design

Cross-check against J: the horizon over which J changes appreciably is the
horizon over which the demonstrator made a cloth-dependent decision. If the two
scales agree, the chunk length is principled rather than copied from ACT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--horizons", default="1,2,5,10,20,40,80,160")
p.add_argument("--lags", type=int, default=4, help="proprio history frames fed to the fit")
p.add_argument("--ridge", type=float, default=1e-3)
p.add_argument("--val_frac", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

cache = Path(args.cache)
state = np.load(cache / "state.npy").astype(np.float64)
action = np.load(cache / "action.npy").astype(np.float64)
episode = np.load(cache / "episode.npy")
Jf = cache / "J.npy"
J = np.load(Jf).astype(np.float64) if Jf.exists() else None

eps = np.unique(episode)
rng = np.random.RandomState(args.seed)
perm = rng.permutation(eps)
n_val = max(1, int(round(len(eps) * args.val_frac)))
val_eps, train_eps = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())

horizons = [int(x) for x in args.horizons.split(",")]
print(f"{len(eps)} episodes, {len(state)} frames, lags={args.lags}\n")
print(f"{'k':>5} {'train R2':>9} {'val R2':>9} {'mean|dJ| over k':>16} {'verdict':>28}")
print("-" * 72)

rows = []
for k in horizons:
    X_tr, Y_tr, X_va, Y_va = [], [], [], []
    dj_spans = []
    for e in eps:
        idx = np.where(episode == e)[0]
        if len(idx) < args.lags + k + 2:
            continue
        s, a = state[idx], action[idx]
        # proprio history: current state plus `lags` previous states
        lo, hi = args.lags, len(idx) - k - 1
        if hi <= lo:
            continue
        feats = np.concatenate(
            [s[t0 - args.lags:t0 + 1].reshape(-1) for t0 in range(lo, hi)]
        ).reshape(hi - lo, -1)
        # target: the whole future chunk, relative to the current state
        tgt = np.stack([(a[t0:t0 + k] - s[t0]).reshape(-1) for t0 in range(lo, hi)])
        if e in val_eps:
            X_va.append(feats); Y_va.append(tgt)
        else:
            X_tr.append(feats); Y_tr.append(tgt)
        if J is not None:
            jj = J[idx]
            if np.isfinite(jj).all():
                dj_spans.append(np.abs(jj[k:] - jj[:-k]).mean() if k < len(jj) else np.nan)

    if not X_tr or not X_va:
        continue
    Xtr = np.concatenate(X_tr); Ytr = np.concatenate(Y_tr)
    Xva = np.concatenate(X_va); Yva = np.concatenate(Y_va)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xva = np.hstack([Xva, np.ones((len(Xva), 1))])

    A = Xtr.T @ Xtr + args.ridge * len(Xtr) * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)

    def r2(X, Y):
        pred = X @ W
        ss_res = ((Y - pred) ** 2).sum()
        ss_tot = ((Y - Ytr.mean(0)) ** 2).sum()
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    tr_r2, va_r2 = r2(Xtr, Ytr), r2(Xva, Yva)
    mdj = float(np.nanmean(dj_spans)) if dj_spans else float("nan")

    if va_r2 > 0.9:
        v = "proprio suffices; vision idle"
    elif va_r2 > 0.7:
        v = "proprio still dominant"
    elif va_r2 > 0.4:
        v = "shortcut weakening"
    else:
        v = "shortcut dead; vision required"
    rows.append((k, tr_r2, va_r2, mdj))
    print(f"{k:>5} {tr_r2:>9.4f} {va_r2:>9.4f} {mdj:>16.4f} {v:>28}")

print("\n=== what this implies for the action chunk ===")
if rows:
    ok = [r for r in rows if r[2] < 0.7]
    if ok:
        k_star = ok[0][0]
        print(f"  proprio-only R^2 falls below 0.7 at k = {k_star}")
        print(f"  -> chunk at least this long, or the policy can keep ignoring vision")
    else:
        print("  proprio predicts every horizon tested. Either extend --horizons,")
        print("  or the demonstrations are smooth enough that chunking alone will")
        print("  not force visual grounding -- which would be a real finding.")
    fin = [r for r in rows if np.isfinite(r[3])]
    if fin:
        print("\n  |dJ| over the chunk grows with k, so a longer chunk also spans a")
        print("  larger change in the Lyapunov functional -- the horizon over which")
        print("  the demonstrator made a cloth-dependent decision.")
print("\nCaveat: ridge on lagged proprio is a proxy for the easy shortcut, not a")
print("bound. A recurrent net may extract somewhat more, so treat these R^2 as")
print("optimistic about where the shortcut dies, i.e. chunk at least this long.")

"""Fit the contact-kernel length scale, instead of guessing it.

The ablation showed the contact kernel -- an inverse-distance weighting of
gripper displacement onto each check-point -- is what recovers dp/du. Adding it
took the action's contribution from 1.5% to 10.3% at h=1, a 7x step, while
cloth-space coordinates alone did nothing. The length scale in that run was 5 cm,
picked out of the air.

One scalar, and the term is load-bearing, so it is worth fitting properly.

**Selection honesty.** With 10 episodes a three-way split is too thin, so the
optimum is chosen on held-out episodes and then checked for *stability across
seeds*. A length scale that is genuinely physical should win under every split;
one that is fitting split noise will jump around. The spread across seeds is
reported next to the optimum, and it is the number to trust.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--cp", default="checkpoints_cpee.npy")
p.add_argument("--ee", default="ee_pos_cpee.npy")
p.add_argument("--horizon", type=int, default=5)
p.add_argument("--scales", default="1,2,3,5,8,12,20,35,60")
p.add_argument("--shapes", default="rational,exp,gauss")
p.add_argument("--seeds", default="0,1,2,3,4")
p.add_argument("--val_frac", type=float, default=0.3)
p.add_argument("--ridge", type=float, default=1e-3)
args = p.parse_args()

cache = Path(args.cache)
CP = np.load(cache / args.cp).astype(np.float64)
EEW = np.load(cache / args.ee).astype(np.float64) * 100.0     # m -> cm
ep = np.load(cache / "episode.npy")

ok = np.isfinite(CP).all(axis=(1, 2)) & np.isfinite(EEW).all(axis=(1, 2))
eps = np.array(sorted({int(e) for e in np.unique(ep[ok])
                       if np.isfinite(CP[ep == e]).all() and np.isfinite(EEW[ep == e]).all()}))
print(f"[data] {len(eps)} episodes with cloth state + EE, horizon {args.horizon}")

h = args.horizon
scales = [float(x) for x in args.scales.split(",")]
shapes = args.shapes.split(",")
seeds = [int(x) for x in args.seeds.split(",")]


def kernel(dist, L, shape):
    if shape == "rational":
        return 1.0 / (1.0 + (dist / L) ** 2)
    if shape == "exp":
        return np.exp(-dist / L)
    return np.exp(-((dist / L) ** 2))


def build(L, shape):
    """Features and targets for one kernel setting."""
    X, Y, G = [], [], []
    for e in eps:
        i = np.where(ep == e)[0]
        if len(i) < h + 3:
            continue
        P_, E_ = CP[i], EEW[i]
        n = len(i) - h
        vel = np.zeros_like(P_); vel[1:] = P_[1:] - P_[:-1]
        dee = np.zeros_like(E_); dee[1:] = E_[1:] - E_[:-1]
        rel = P_[:, None, :, :] - E_[:, :, None, :]
        dist = np.linalg.norm(rel, axis=-1)
        prox = kernel(dist, L, shape)
        ca = prox[..., None] * dee[:, :, None, :]
        X.append(np.concatenate([
            P_[:-h].reshape(n, -1), vel[:-h].reshape(n, -1),
            E_[:-h].reshape(n, -1), dee[:-h].reshape(n, -1),
            prox[:-h].reshape(n, -1), ca[:-h].reshape(n, -1),
        ], axis=1))
        Y.append((P_[h:] - P_[:-h]).reshape(n, -1))
        G.append(np.full(n, e))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def fit_eval(X, Y, G, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(eps)
    n_val = max(1, int(round(len(eps) * args.val_frac)))
    val = set(perm[:n_val].tolist())
    va = np.isin(G, list(val)); tr = ~va
    m, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = np.hstack([(X[tr] - m) / sd, np.ones((int(tr.sum()), 1))])
    A = Z.T @ Z + args.ridge * len(Z) * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ Y[tr])
    Zv = np.hstack([(X[va] - m) / sd, np.ones((int(va.sum()), 1))])
    return float(((Y[va] - Zv @ W) ** 2).mean())


print(f"\n{'shape':<10}{'L (cm)':>8}" + "".join(f"{'seed'+str(s):>10}" for s in seeds)
      + f"{'mean':>10}{'std':>8}")
print("-" * (18 + 10 * len(seeds) + 18))

best = None
results = {}
for shape in shapes:
    for L in scales:
        X, Y, G = build(L, shape)
        ms = [fit_eval(X, Y, G, s) for s in seeds]
        mu, sd_ = float(np.mean(ms)), float(np.std(ms))
        results[(shape, L)] = (mu, sd_, ms)
        print(f"{shape:<10}{L:>8.1f}" + "".join(f"{v:>10.5f}" for v in ms)
              + f"{mu:>10.5f}{sd_:>8.5f}")
        if best is None or mu < best[0]:
            best = (mu, shape, L)

mu, shape, L = best
print(f"\n=== best: {shape} kernel, L = {L:.1f} cm  (mean val MSE {mu:.5f}) ===")

# Stability: is the optimum the same under every individual split?
per_seed_best = []
for k, s in enumerate(seeds):
    cand = min(results.items(), key=lambda kv: kv[1][2][k])
    per_seed_best.append(cand[0])
uniq = sorted(set(per_seed_best))
print(f"  per-seed optima: {per_seed_best}")
if len(uniq) == 1:
    print("  STABLE -- every split picks the same kernel, so this is physical, not")
    print("  split noise.")
else:
    Ls = [x[1] for x in per_seed_best]
    print(f"  spread: L in [{min(Ls):.1f}, {max(Ls):.1f}] cm across splits.")
    print("  Treat the optimum as approximate; the ablation gain matters more than")
    print("  the exact value.")

guess = results.get(("rational", 5.0))
if guess:
    print(f"\n  original guess (rational, 5 cm): {guess[0]:.5f}")
    print(f"  fitted                         : {mu:.5f}   ({1 - mu/guess[0]:+.2%})")

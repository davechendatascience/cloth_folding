"""Can cloth state be predicted as a dynamical system?

    p_{t+1} = f(p_t, u_t)

This is the first question, before perception and before control, because it is
answerable from *state alone* -- no vision, no grasping, no policy. If the cloth
does not admit a learnable forward model at this horizon, nothing downstream
works: the controller needs dp/du, the observer needs a process model to
propagate belief through occlusion, and pick-and-place needs the settle to be
predictable.

It is also where damping matters most. An under-damped plant answers one action
with ringing, so `u` maps to a distribution of outcomes; critically damped, it
settles smoothly and the map is nearly static. That claim is testable by fitting
the same model to data from both plants.

**Baselines are the whole game here, and they are unusually strong.** At 30 fps
the cloth barely moves between frames, so:

  * `dp = 0`      (nothing happens) is already excellent by MSE
  * `dp = dp_prev` (persistence) exploits smoothness
  * `phase -> dp`  is the confound that invalidated three earlier results in
                   this project -- stereotyped demonstrations make the cloth's
                   motion a function of how far through the episode you are

A model is only interesting if it beats all three. Reporting R^2 against the
mean would look impressive and mean nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--cp", default="checkpoints_cp.npy")
p.add_argument("--horizon", type=int, default=1, help="predict p_{t+h} - p_t")
p.add_argument("--val_frac", type=float, default=0.3)
p.add_argument("--ridge", type=float, default=1e-3)
p.add_argument("--ablate", action="store_true",
               help="Fit nested feature sets to isolate what the ACTION contributes. "
                    "If commanding the arm does not improve prediction, there is no "
                    "dp/du to invert and model-based control is impossible -- the "
                    "cloth would be predictable but not controllable.")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

cache = Path(args.cache)
CP = np.load(cache / args.cp).astype(np.float64)      # (N, 6, 3) cm
act = np.load(cache / "action.npy").astype(np.float64)   # (N, 12) joint targets
st = np.load(cache / "state.npy").astype(np.float64)     # (N, 12) joint positions
ep = np.load(cache / "episode.npy")

ok = np.isfinite(CP).all(axis=(1, 2))
eps = np.array(sorted({int(e) for e in np.unique(ep[ok])
                       if np.isfinite(CP[ep == e]).all()}))
if len(eps) < 3:
    raise SystemExit(f"only {len(eps)} episodes with recorded cloth state")

rng = np.random.RandomState(args.seed)
perm = rng.permutation(eps)
n_val = max(1, int(round(len(eps) * args.val_frac)))
val_eps, train_eps = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())
print(f"[data] {len(eps)} episodes with cloth state -> train {len(train_eps)} / val {len(val_eps)}")

h = args.horizon
X, Y, PH, PREV, G, BLOCKS = [], [], [], [], [], []
for e in eps:
    i = np.where(ep == e)[0]
    if len(i) < h + 3:
        continue
    P_, A_, S_ = CP[i], act[i], st[i]
    n = len(i) - h
    dp = (P_[h:] - P_[:-h]).reshape(n, -1)                    # target
    # features: current cloth config, joint state, commanded action, and the
    # action *delta* (what the arm is about to do, which is what moves cloth)
    # Cloth has momentum: the system is second-order, so the state is
    # (position, velocity). Omitting velocity handed the model a first-order
    # state and let plain persistence -- which is nothing but a velocity
    # extrapolation -- beat it by 2x at h=1 and 23x at h=20.
    vel = np.zeros_like(P_)
    vel[1:] = P_[1:] - P_[:-1]
    blocks = {
        "p": P_[:-h].reshape(n, -1),
        "vel": vel[:-h].reshape(n, -1),
        "s": S_[:-h],
        "a": A_[:-h],
        "a-s": A_[:-h] - S_[:-h],
    }
    feats = np.concatenate([blocks[k] for k in ("p", "vel", "s", "a", "a-s")], axis=1)
    BLOCKS.append(blocks)
    # CAUSAL velocity extrapolation. The obvious "prev = dp[t-1]" leaks the
    # future for h > 1: dp[t-1] = p[t+h-1] - p[t-1] overlaps h-1 of the h frames
    # being predicted, so at h=20 the "baseline" sees 19 of the 20 future frames
    # and looks unbeatable. This uses only p[t] and p[t-1].
    prev = (h * vel[:-h]).reshape(n, -1)
    X.append(feats); Y.append(dp); PREV.append(prev)
    PH.append(np.linspace(0.0, 1.0, n))
    G.append(np.full(n, e))

X = np.concatenate(X); Y = np.concatenate(Y)
PREV = np.concatenate(PREV); PH = np.concatenate(PH); G = np.concatenate(G)
tr = np.isin(G, list(train_eps)); va = ~tr
print(f"[data] transitions: train {int(tr.sum())} / val {int(va.sum())}, horizon {h} step(s)")


def mse(pred, true):
    return float(((true - pred) ** 2).mean())


def r2_vs(pred, true, base):
    """R^2 measured against a *baseline*, not against the mean."""
    return 1.0 - mse(pred, true) / max(mse(base, true), 1e-12)


zero = np.zeros_like(Y)
mu_tr = Y[tr].mean(0)
mean_pred = np.repeat(mu_tr[None], len(Y), 0)

# phase-only: mean dp as a function of normalised episode phase (train only)
Gk = 50
grid = np.linspace(0, 1, Gk)
curve = np.stack([
    np.interp(grid, np.sort(PH[tr]), Y[tr][np.argsort(PH[tr]), c]) for c in range(Y.shape[1])
], axis=-1)
phase_pred = np.stack([np.interp(PH, grid, curve[:, c]) for c in range(Y.shape[1])], axis=-1)


def fit_ridge(Xtr, Ytr, lam):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = np.hstack([(Xtr - m) / s, np.ones((len(Xtr), 1))])
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ Ytr)
    return lambda Xq: np.hstack([(Xq - m) / s, np.ones((len(Xq), 1))]) @ W


f = fit_ridge(X[tr], Y[tr], args.ridge)
pred = f(X)

print(f"\n=== predicting dp over {h} step(s), held-out episodes ===")
print(f"{'predictor':<34}{'MSE (cm^2)':>13}{'vs zero':>10}")
print("-" * 57)
rows = [
    ("zero (nothing happens)", zero),
    ("train mean dp", mean_pred),
    ("velocity extrapolation (causal)", PREV),
    ("phase only (the clock)", phase_pred),
    ("ridge on (p, dp/dt, s, a, a-s)", pred),
]
for name, q in rows:
    print(f"{name:<34}{mse(q[va], Y[va]):>13.5f}{r2_vs(q[va], Y[va], zero[va]):>10.4f}")

best_base = min(mse(zero[va], Y[va]), mse(PREV[va], Y[va]), mse(phase_pred[va], Y[va]))
gain = 1.0 - mse(pred[va], Y[va]) / max(best_base, 1e-12)

print(f"\n  strongest trivial baseline MSE : {best_base:.5f}")
print(f"  model MSE                      : {mse(pred[va], Y[va]):.5f}")
print(f"  improvement over it            : {gain:+.4f}")

if args.ablate:
    cat = {k: np.concatenate([b[k] for b in BLOCKS]) for k in BLOCKS[0]}
    sets = [
        ("cloth only        (p, vel)", ("p", "vel")),
        ("+ arm pose        (p, vel, s)", ("p", "vel", "s")),
        ("+ action          (p, vel, s, a)", ("p", "vel", "s", "a")),
        ("+ action delta    (full)", ("p", "vel", "s", "a", "a-s")),
        ("action only       (s, a, a-s)", ("s", "a", "a-s")),
    ]
    print("\n=== ablation: what does the ACTION contribute? ===")
    print(f"{'feature set':<36}{'MSE':>11}{'vs cloth-only':>15}")
    print("-" * 62)
    base_mse = None
    for name, keys in sets:
        Xs = np.concatenate([cat[k] for k in keys], axis=1)
        fk = fit_ridge(Xs[tr], Y[tr], args.ridge)
        m = mse(fk(Xs)[va], Y[va])
        if base_mse is None:
            base_mse = m
        print(f"{name:<36}{m:>11.5f}{1 - m/max(base_mse,1e-12):>15.4f}")
    print("\n  A large gain from adding the action means dp/du is real and the")
    print("  cloth is CONTROLLABLE, not merely predictable.")

print("\n=== verdict ===")
if gain >= 0.30:
    print("  Cloth state IS predictable as a dynamical system beyond the trivial")
    print("  baselines. A forward model, and therefore dp/du, is learnable.")
elif gain >= 0.10:
    print("  Weakly predictable. Real signal, but a linear model on 6 check-points")
    print("  captures little -- try more data, a nonlinear model, or a longer")
    print("  horizon where the motion is larger relative to the noise.")
else:
    print("  NOT predictable beyond doing nothing. Either the cloth barely moves at")
    print("  this horizon (raise --horizon), or 6 check-points are too coarse a")
    print("  state to be Markovian -- the cloth's configuration is not captured by")
    print("  6 points, so p_{t+1} genuinely does not depend only on p_t.")

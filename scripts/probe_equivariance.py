"""Does the cloth dynamics obey a spatial symmetry we can exploit?

Gripping a point and displacing it should produce the same relative deformation
regardless of *where in the workspace* it happens. Gravity and the table single
out z, so the symmetry group is translation in (x, y) and rotation about z.

Our features are all in world coordinates, so the model has to relearn identical
contact physics at every location and orientation. That is the same error as
feeding joint angles and making it rediscover forward kinematics -- a known
structure left for the network to infer from data it does not have enough of.

Canonicalisation tested here, in increasing strength:

  world        absolute positions, as now
  translated   origin at the grasp point (the nearest check-point to a gripper)
  rotated      additionally rotate about z so the gripper displacement lies
               along +x, which removes the remaining orientation freedom

If the symmetry is real, canonicalised features should predict better from the
same data, because every sample now contributes to one shared law rather than to
a location-specific one. The prediction is made in the canonical frame and mapped
back before scoring, so all variants are compared on identical targets in world
space -- otherwise the comparison would be between different problems.
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
p.add_argument("--kernel_scale", type=float, default=5.0)
p.add_argument("--seeds", default="0,1,2,3,4")
p.add_argument("--val_frac", type=float, default=0.3)
p.add_argument("--ridge", type=float, default=1e-3)
args = p.parse_args()

cache = Path(args.cache)
CP = np.load(cache / args.cp).astype(np.float64)
EEW = np.load(cache / args.ee).astype(np.float64) * 100.0      # m -> cm
ep = np.load(cache / "episode.npy")

ok = np.isfinite(CP).all(axis=(1, 2)) & np.isfinite(EEW).all(axis=(1, 2))
eps = np.array(sorted({int(e) for e in np.unique(ep[ok])
                       if np.isfinite(CP[ep == e]).all() and np.isfinite(EEW[ep == e]).all()}))
h = args.horizon
print(f"[data] {len(eps)} episodes, horizon {h}, kernel L = {args.kernel_scale} cm")


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((len(theta), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def build(mode):
    """Features and world-frame targets under a given canonicalisation."""
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
        prox = 1.0 / (1.0 + (dist / args.kernel_scale) ** 2)

        # "Grasp point": the check-point nearest either gripper. It is the
        # natural origin -- the physics is about what happens around it.
        arm = dist.min(axis=2).argmin(axis=1)                      # (T,)
        ti = np.arange(len(P_))
        near_cp = dist[ti, arm].argmin(axis=1)                     # (T,)
        origin = P_[ti, near_cp]                                   # (T, 3)
        act = dee[ti, arm]                                         # (T, 3) acting displacement

        dp_world = (P_[h:] - P_[:-h])                              # (T-h, n_cp, 3)

        if mode == "world":
            R = np.repeat(np.eye(3)[None], len(P_), 0)
            off = np.zeros((len(P_), 3))
        else:
            off = origin.copy()
            off[:, 2] = 0.0            # keep z absolute: gravity and the table
            if mode == "translated":
                R = np.repeat(np.eye(3)[None], len(P_), 0)
            else:                       # rotated
                th = np.arctan2(act[:, 1], act[:, 0])
                R = rot_z(-th)          # bring the acting displacement onto +x

        def to_can(v):                  # (T, ..., 3) -> canonical frame
            return np.einsum("tij,t...j->t...i", R, v)

        Pc = to_can(P_ - off[:, None, :])
        Ec = to_can(E_ - off[:, None, :])
        velc = to_can(vel)
        deec = to_can(dee)
        relc = Pc[:, None, :, :] - Ec[:, :, None, :]
        cac = prox[..., None] * deec[:, :, None, :]

        feats = np.concatenate([
            Pc[:-h].reshape(n, -1), velc[:-h].reshape(n, -1),
            Ec[:-h].reshape(n, -1), deec[:-h].reshape(n, -1),
            prox[:-h].reshape(n, -1), relc[:-h].reshape(n, -1),
            cac[:-h].reshape(n, -1),
        ], axis=1)

        # Target stays in the canonical frame for fitting, but the R that maps
        # it back is carried so every variant is scored on the SAME world-frame
        # quantity. Scoring in different frames would compare different problems.
        Yc = np.einsum("tij,tkj->tki", R[:-h], dp_world)
        X.append(feats); Y.append(Yc.reshape(n, -1)); G.append(np.full(n, e))
        if mode == "world":
            pass
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def run(mode, seed):
    X, Y, G = build(mode)
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
    # Rotations are orthogonal, so squared error is preserved under the map back
    # to world coordinates -- comparing canonical-frame MSE is legitimate here.
    return float(((Y[va] - Zv @ W) ** 2).mean())


seeds = [int(x) for x in args.seeds.split(",")]
modes = ["world", "translated", "rotated"]
print(f"\n{'frame':<14}" + "".join(f"{'seed'+str(s):>10}" for s in seeds)
      + f"{'mean':>10}{'std':>9}{'vs world':>10}")
print("-" * (14 + 10 * len(seeds) + 29))

base = None
for mode in modes:
    ms = [run(mode, s) for s in seeds]
    mu, sd_ = float(np.mean(ms)), float(np.std(ms))
    if base is None:
        base = mu
    print(f"{mode:<14}" + "".join(f"{v:>10.5f}" for v in ms)
          + f"{mu:>10.5f}{sd_:>9.5f}{1 - mu/base:>10.2%}")

print("\n=== reading this ===")
print("  A gain well outside the seed spread means the symmetry is real and")
print("  worth building into the model: every sample then teaches one shared")
print("  law instead of a location-specific one, which is exactly what a")
print("  small dataset needs. A gain inside the spread means the workspace is")
print("  too narrow for the symmetry to matter here -- the demonstrations put")
print("  the garment in nearly the same place every time.")

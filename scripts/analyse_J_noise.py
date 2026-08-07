"""Is the J signal that BC consumes bigger than the noise in producing it?

Two replays of the same episode, with the same recorded actions, do not produce
the same J. Measured on episode 0: final J 1.770 vs 1.062 across two runs of
*identical* configuration. PhysX GPU particle simulation is not bit-reproducible
(atomic ordering), so this is expected in kind -- the question is magnitude.

The endpoint spread is the wrong statistic to panic about. BC does not consume
J; it consumes:

  * ``dJ[t] = J[t+1] - J[t]``  -> the AWR weight ``w = exp(-dJ/beta)``
  * ``J[t]``                   -> the auxiliary head target ``||J_hat - J||``

A trajectory-level divergence that grows smoothly leaves per-step dJ largely
intact, while white per-step noise destroys it. Those two cases have very
different consequences and identical endpoint spread, so they must be
distinguished directly.

Reports, for each episode present in both runs:
  signal  -- std of dJ within a run (what the weighting is trying to see)
  noise   -- std of the paired difference dJ_A - dJ_B (what corrupts it)
  SNR     -- signal / noise; below ~1 the AWR weights are mostly noise
  corr    -- correlation of dJ_A with dJ_B; how much of the per-step
             structure actually replicates
  J corr  -- correlation of J_A with J_B; the auxiliary head only needs the
             level to be predictable, which is a much weaker requirement

Usage:
  analyse_J_noise.py --cache DIR --a J.npy --b J_camB.npy [--episodes 0,1,2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--a", default="J.npy")
p.add_argument("--b", default="J_camB.npy")
p.add_argument("--episodes", default="", help="comma list; default = all shared")
args = p.parse_args()

cache = Path(args.cache)
A = np.load(cache / args.a)
B = np.load(cache / args.b)
ep = np.load(cache / "episode.npy")

if args.episodes:
    eps = [int(x) for x in args.episodes.split(",")]
else:
    shared = np.isfinite(A) & np.isfinite(B)
    eps = sorted(set(ep[shared].tolist()))

if not eps:
    raise SystemExit("no episodes labelled in both files -- nothing to compare")

print(f"comparing {args.a} vs {args.b} over {len(eps)} episode(s)\n")
hdr = (f"{'ep':>4} {'n':>5} {'J_A end':>8} {'J_B end':>8} {'sig(dJ)':>9} "
       f"{'noise':>9} {'SNR':>6} {'corr dJ':>8} {'corr J':>7}")
print(hdr)
print("-" * len(hdr))

rows = []
for e in eps:
    r = np.where(ep == e)[0]
    a, b = A[r], B[r]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        continue
    a, b = a[m], b[m]
    da, db = np.diff(a), np.diff(b)

    signal = float(da.std())
    noise = float((da - db).std())
    snr = signal / noise if noise > 0 else float("inf")
    cdj = float(np.corrcoef(da, db)[0, 1]) if da.std() > 0 and db.std() > 0 else np.nan
    cj = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    rows.append((e, len(a), a[-1], b[-1], signal, noise, snr, cdj, cj))
    print(f"{e:>4} {len(a):>5} {a[-1]:>8.3f} {b[-1]:>8.3f} {signal:>9.4f} "
          f"{noise:>9.4f} {snr:>6.2f} {cdj:>8.3f} {cj:>7.3f}")

if not rows:
    raise SystemExit("no episode had enough overlapping finite frames")

arr = np.array([(r[4], r[5], r[6], r[7], r[8]) for r in rows], dtype=float)
sig, noi, snr, cdj, cj = arr.mean(axis=0)
print("\n=== summary ===")
print(f"  mean dJ signal std   : {sig:.4f}")
print(f"  mean dJ noise std    : {noi:.4f}")
print(f"  mean SNR             : {snr:.2f}")
print(f"  mean corr(dJ_A,dJ_B) : {cdj:.3f}")
print(f"  mean corr(J_A, J_B)  : {cj:.3f}")

print("\n=== what this means for the two consumers ===")
# AWR weighting needs per-step dJ to replicate. Thresholds are judgement calls,
# stated explicitly so the verdict can be argued with rather than trusted.
if snr >= 2 and cdj >= 0.5:
    print("  AWR dJ weighting : USABLE -- per-step descent replicates.")
elif snr >= 1:
    print("  AWR dJ weighting : MARGINAL -- weights carry real signal but are")
    print("                     noticeably corrupted. Consider smoothing dJ over")
    print("                     a window, or averaging J over repeated replays.")
else:
    print("  AWR dJ weighting : NOT USABLE as-is -- the per-step weights would be")
    print("                     mostly simulator noise. Smooth dJ, or relabel on")
    print("                     CPU if that proves deterministic.")

if cj >= 0.9:
    print("  J_hat aux head   : USABLE -- the J *level* is highly reproducible,")
    print("                     which is all this target needs.")
elif cj >= 0.7:
    print("  J_hat aux head   : MARGINAL -- level partly reproducible.")
else:
    print("  J_hat aux head   : NOT USABLE -- even the level does not replicate.")

print("\nNote: correlated trajectory-level drift inflates endpoint spread while")
print("leaving per-step dJ intact. Judge on SNR and corr(dJ), not on J_end.")

"""Log the six check-point 3D positions on every success check.

Needed because the graded J metric and the camera disagree. On checkpoint 12500,
`dist(p[2], p[3])` fell 42.64 -> 20.95 in a single check interval and
`dist(p[0], p[4])` fell 26.71 -> 17.72, while the top camera showed **zero**
pixels changing by more than 25 grey levels across those same intervals -- and a
max-over-40x40-block analysis, which would catch a localised sleeve moving,
stayed at 0.4-0.9 for the whole episode. A do-nothing control on the identical
garment and seed held J at 7.36 -> 7.43 and never jumped, so the policy causes
these transitions; what is unknown is whether they correspond to any real change
in garment configuration.

Three hypotheses, which the raw positions separate immediately:

* the moving points lie outside the top camera's crop (the garment extends past
  the frame edges), so a real fold is happening off-screen
* the motion is mostly vertical, which a top-down projection barely renders
* the simulation particles diverge from the rendered mesh under gripper contact,
  in which case J is responding to contact noise and is not a progress signal

`get_object_particle_position` is looked up in module globals at call time, so
replacing the module attribute intercepts every check without touching LeHome's
code. Activated by LEHOME_LOG_CP=1.
"""

from __future__ import annotations

import os

import numpy as np


def install() -> bool:
    if not os.environ.get("LEHOME_LOG_CP"):
        return False

    import lehome.utils.success_checker_chanllege as sc

    orig = sc.get_object_particle_position
    n = {"i": 0}

    def wrapped(particle_object, index_list):
        p = orig(particle_object, index_list)
        try:
            a = np.asarray(p, dtype=float).reshape(len(index_list), 3)
            n["i"] += 1
            pos = " ".join(f"p{j}=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"
                           for j, v in enumerate(a))
            # Spread tells us the working scale without assuming cm vs m.
            print(f"[CP {n['i']:04d}] idx={list(index_list)} {pos} "
                  f"bbox={np.ptp(a, axis=0).round(3).tolist()}", flush=True)
        except Exception as e:  # never let logging break an eval
            print(f"[CP] logging failed: {e}", flush=True)
        return p

    sc.get_object_particle_position = wrapped
    print("[log_checkpoints] ACTIVE - logging check-point positions", flush=True)
    return True

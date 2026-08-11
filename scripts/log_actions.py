"""Log joint state and commanded action every step. Activated by LEHOME_LOG_ACT=1.

The only question this answers: does the arm move, and does the policy ask it to?

`observation.state` is the 12 measured joint positions and the action is 12 joint
position targets, so the two together separate three cases that look identical
from a camera:

* action ~= state, state static      -> policy commands "stay put"; nothing moves
* action far from state, state static -> policy commands motion, the arm cannot
                                         follow (saturation, limits, bad targets)
* action far from state, state follows -> the arm really is moving

No new metric, no aggregation -- raw joint travel in radians.
"""

from __future__ import annotations

import os

import numpy as np

from scripts.eval_policy.registry import PolicyRegistry


def install() -> bool:
    if not os.environ.get("LEHOME_LOG_ACT"):
        return False

    cls = PolicyRegistry._registry["lerobot"]
    orig = cls.select_action
    n = {"i": 0}

    def wrapped(self, observation):
        a = orig(self, observation)
        try:
            s = np.asarray(observation["observation.state"], dtype=float).reshape(-1)
            act = np.asarray(a, dtype=float).reshape(-1)
            n["i"] += 1
            if n["i"] % 10 == 1:  # every 10th step is plenty at 600 steps
                print(f"[ACT {n['i']:04d}] "
                      f"state={np.array2string(s, precision=3, separator=',')} "
                      f"action={np.array2string(act, precision=3, separator=',')} "
                      f"|a-s|={np.abs(act - s).sum():.4f}", flush=True)
        except Exception as e:
            print(f"[ACT] logging failed: {e}", flush=True)
        return a

    cls.select_action = wrapped
    print("[log_actions] ACTIVE - logging joint state and commanded action", flush=True)
    return True

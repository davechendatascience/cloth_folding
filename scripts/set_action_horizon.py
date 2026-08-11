"""Override the policy's open-loop action horizon. LEHOME_N_ACTION_STEPS=10.

SmolVLA defaults to `n_action_steps=50`: it predicts a 50-step chunk and executes
all of it before looking at the world again. Over a 600-step episode that is only
**12 observations**, and 12 opportunities to correct.

That matters here because the failure mode is specific. The checkpoint predicts
demonstration actions to 0.0051 MSE, beating a persistence baseline 8.7x, and
replaying recorded actions through this same environment folds the garment
(J 7.27 -> 0.00, success). So both the policy and the environment work; what
fails is the two together. That is compounding error -- accurate one step at a
time on states from the demonstration distribution, with no recovery behaviour
once its own actions carry it off that distribution, and 50-step open-loop
segments give drift maximum room to accumulate.

Shortening the horizon is the direct intervention: it costs nothing at training
time (this is purely an inference-time config), and it trades compute for
closed-loop correction. The chunk is still predicted at `chunk_size=50`; only
the number of steps consumed before re-planning changes.

`n_action_steps` sets the action queue's `maxlen` at `reset()` and the slice
taken from each chunk, so it must be set before the first reset -- done here at
policy construction.
"""

from __future__ import annotations

import os

from scripts.eval_policy.registry import PolicyRegistry


def install() -> int | None:
    raw = os.environ.get("LEHOME_N_ACTION_STEPS", "").strip()
    if not raw:
        return None
    n = int(raw)

    cls = PolicyRegistry._registry["lerobot"]
    orig_init = cls.__init__

    def wrapped(self, *a, **kw):
        orig_init(self, *a, **kw)
        inner = getattr(self, "policy", None)
        cfg = getattr(inner, "config", None)
        if cfg is None or not hasattr(cfg, "n_action_steps"):
            print("[set_action_horizon] could not find policy.config.n_action_steps",
                  flush=True)
            return
        was = cfg.n_action_steps
        cfg.n_action_steps = n
        print(f"[set_action_horizon] n_action_steps {was} -> {n} "
              f"(chunk_size {getattr(cfg, 'chunk_size', '?')} unchanged); "
              f"re-plans every {n} steps, {600 // n} observations per 600-step episode",
              flush=True)

    cls.__init__ = wrapped
    return n

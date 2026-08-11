"""Vision ablation for LeHome's official evaluation.

Success above zero does not by itself show a policy uses its cameras. The
garment's reset **position is fixed** (`soft_reset_pos_range` has min == max on
x, y and z); only orientation varies (roll +-20 deg, pitch +-36 deg, yaw 0), and
texture/light randomisation are both disabled. Ten of the twelve eval garments
are in the training set and the seed is fixed. A policy that replays a phase
template therefore gets the garment's location for free -- and the demos here
are ~69% predictable from phase alone (phase -> action R^2 0.688; adding
proprioception reaches only 0.725).

So the ablation is the control: run the identical protocol with the visual
input destroyed. If the success rate does not move, the policy is not using
vision, whatever the loss says. This is the test that exposed the behaviour
cloned policies, whose closed-loop J moved 0.009 under randomised texture and
lighting while the frozen baseline's own run-to-run noise was 0.086.

Two modes:

``frozen``  hold the first frame of the episode for its whole duration. The
            imagery stays in-distribution -- correct exposure, real garment,
            plausible scene -- but carries no information about the current
            state. This is the honest test, and the default.
``blank``   zero the images. Decisive but out-of-distribution, so a drop in
            success can mean "relies on vision" *or* "was handed garbage it
            never saw in training". Read it only alongside ``frozen``.

Activated by the ``LEHOME_ABLATE`` environment variable so that the official
argument parser, its `policy_type == "lerobot"` kwargs branch, and `main()` are
all untouched -- the registered `lerobot` class is swapped for a subclass that
corrupts the observation and then defers to the real implementation.
"""

from __future__ import annotations

import os

import numpy as np

from scripts.eval_policy.lerobot_policy import LeRobotPolicy
from scripts.eval_policy.registry import PolicyRegistry


class _AblatedLeRobotPolicy(LeRobotPolicy):
    """LeRobotPolicy with the camera channels destroyed before inference."""

    MODE = "frozen"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._held: dict[str, np.ndarray] = {}

    def reset(self):
        # Per-episode: the held frame must come from *this* episode, or the
        # ablation would leak a frame across garments.
        self._held = {}
        return super().reset()

    def select_action(self, observation):
        obs = dict(observation)
        for k, v in observation.items():
            if "images" not in k:
                continue
            if self.MODE == "blank":
                obs[k] = np.zeros_like(v)
            else:
                if k not in self._held:
                    self._held[k] = np.array(v, copy=True)
                obs[k] = self._held[k]
        return super().select_action(obs)


class _BlankPolicy(_AblatedLeRobotPolicy):
    MODE = "blank"


class _NoopPolicy(LeRobotPolicy):
    """Hold the current joint configuration: the do-nothing control.

    Without this, a graded J means nothing. Cloth dropped on a table settles on
    its own, and an arm thrashing near it will move check-points around, so
    "J fell from 7.4 to 4.9" is only evidence of manipulation if doing nothing
    scores worse. This is the official-harness equivalent of the frozen-arm
    baseline (J = 7.118) measured on the old custom harness -- that number came
    from a different protocol and different garments, so it cannot be reused
    here directly.

    Commanding the current state as the position target is exactly "stay put",
    since the action space is 12 joint position targets.
    """

    def select_action(self, observation):
        return np.asarray(observation["observation.state"], dtype=np.float32).reshape(-1)


def install() -> str | None:
    """Swap the registered `lerobot` policy for an ablated one, if requested.

    Returns the active mode, or None when ablation is off.
    """
    mode = os.environ.get("LEHOME_ABLATE", "").strip().lower()
    if not mode:
        return None
    if mode not in ("frozen", "blank", "noop"):
        raise SystemExit(
            f"LEHOME_ABLATE must be 'frozen', 'blank' or 'noop', got {mode!r}")

    cls = {"blank": _BlankPolicy, "noop": _NoopPolicy}.get(
        mode, _AblatedLeRobotPolicy)
    # register_policy() refuses to overwrite, so replace the entry directly.
    PolicyRegistry._registry["lerobot"] = cls
    print(f"[ablate_vision] ACTIVE mode={mode} -- 'lerobot' now {cls.__name__}. "
          f"Camera input is destroyed; any success is NOT vision-driven.")
    return mode

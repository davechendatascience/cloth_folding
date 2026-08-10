"""LeHome cloth folding: evaluation objective and simulator access.

The project now finetunes a pretrained VLA (pi0) via LeRobot rather than
training a custom policy. What lives here is what survives that change and is
still needed to *judge* a policy:

* ``math.garment_functional`` -- J, whose zero set is LeHome's own success
  predicate (verified over 300 configurations).
* ``tasks.isaac_garment_backend`` -- the live garment environment.
* ``tasks.isaac_app`` -- a guarded Isaac launcher (Kit ignores SIGTERM and
  leaves a 300%-CPU orphan on any unguarded failure).

The retired stack -- behaviour cloning, PPO with damped updates, the reward
shaping, the mock backend, grasp retrieval -- is in git history. Its findings,
which are the durable part, are in README.md, LEVERS.md and docs/.
"""

from __future__ import annotations

__all__ = ["GarmentFoldFunctional", "GarmentFunctionalCfg", "GARMENT_CONDITIONS"]
__version__ = "0.2.0"


def __getattr__(name: str):
    if name in __all__:
        from . import math as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

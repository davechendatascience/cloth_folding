"""The Lyapunov functional J.

``GarmentFoldFunctional`` is a continuous relaxation of LeHome's own
``success_checker_garment_fold``: its zero set is identical to that boolean,
verified over 300 random configurations. That equivalence is why it can score a
policy regardless of how the policy was produced.

Caveat recorded in LEVERS.md: away from zero the relaxation is exploitable --
crumpling the garment zeroes the three "must be close" terms and reaches J~1.19
without folding anything. J == 0 is sound; J as a dense reward is not.
"""
from .garment_functional import (
    GARMENT_CONDITIONS,
    GarmentFoldFunctional,
    GarmentFunctionalCfg,
)

__all__ = ["GARMENT_CONDITIONS", "GarmentFoldFunctional", "GarmentFunctionalCfg"]

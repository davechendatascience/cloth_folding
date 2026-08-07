"""Continuous J for the *real* LeHome garment-fold success metric.

LeHome scores folding with a boolean: ``success_checker_garment_fold()`` checks
a handful of pairwise distances between garment check-points against
per-garment thresholds. A boolean is useless as a Lyapunov functional -- it is
neither continuous nor informative about *how far* from folded the garment is.

This module turns that predicate into a functional ``J >= 0`` with

    J(x) = 0   <=>   every LeHome condition is satisfied   <=>   x in G

by summing the *margin violations* of the individual conditions. Each condition
``d <= t`` contributes ``relu(d - t)``; each ``d >= t`` contributes
``relu(t - d)``. So J measures, in centimetres, how much total distance the
garment still has to move before LeHome would call it folded.

This is a strictly better fit for the spec's Sec. 2.1 requirements than a
mask-IoU proxy, because the goal set is *identical* to the official one rather
than merely correlated with it -- there is no way to reduce J to zero without
actually passing the challenge's own check.

Conditions transcribed from
``lehome/utils/success_checker_chanllege.py`` (check_top_sleeve,
check_pant_long, check_pant_short). Units are centimetres: LeHome multiplies
mesh points by 100 in ``get_object_particle_position``, and the thresholds are
scaled by the garment's ``init_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

# (index_a, index_b, comparison, threshold_slot)
#   "le" -> condition is dist <= success_distance[slot]
#   "ge" -> condition is dist >= success_distance[slot]
Condition = Tuple[int, int, str, int]

GARMENT_CONDITIONS: Dict[str, List[Condition]] = {
    # check_top_sleeve: fold the sleeves in, keep the body spread.
    "top-long-sleeve": [
        (0, 4, "le", 0),
        (2, 3, "le", 1),
        (1, 5, "le", 2),
        (0, 1, "ge", 3),
        (4, 5, "ge", 4),
    ],
    "top-short-sleeve": [
        (0, 4, "le", 0),
        (2, 3, "le", 1),
        (1, 5, "le", 2),
        (0, 1, "ge", 3),
        (4, 5, "ge", 4),
    ],
    # check_pant_long: bring the cuffs together, keep the legs extended.
    "long-pant": [
        (0, 4, "le", 0),
        (0, 2, "ge", 1),
        (1, 3, "ge", 2),
        (1, 5, "le", 3),
    ],
    # check_pant_short: bring the waist/hem together, keep the legs apart.
    "short-pant": [
        (0, 1, "le", 0),
        (4, 5, "le", 1),
        (0, 4, "ge", 2),
        (1, 5, "ge", 3),
    ],
}


@dataclass
class GarmentFunctionalCfg:
    """Weights for the continuous garment functional."""

    scale: float = 10.0
    """Normaliser in centimetres. J is reported in units of this length, so
    ``scale=10`` means J=1 is "10 cm of total margin violation remaining"."""

    weights: Sequence[float] | None = None
    """Optional per-condition weights. ``None`` weights all conditions equally.

    Worth setting: the "must stay apart" (``ge``) conditions are satisfied by
    the *initial* flat garment and only get violated by crumpling, whereas the
    "must come together" (``le``) conditions are what folding actually
    achieves. Equal weighting therefore spends much of J's dynamic range on
    conditions that start satisfied."""


class GarmentFoldFunctional:
    """J for a LeHome garment, from its check-point positions.

    Args:
        garment_type: one of :data:`GARMENT_CONDITIONS`.
        success_distance: LeHome's per-condition thresholds, already scaled by
            the garment's ``init_scale`` (as ``success_checker_garment_fold``
            does).
        cfg: normalisation/weights.
    """

    def __init__(
        self,
        garment_type: str,
        success_distance: Sequence[float],
        cfg: GarmentFunctionalCfg | None = None,
    ) -> None:
        if garment_type not in GARMENT_CONDITIONS:
            raise ValueError(
                f"unknown garment_type {garment_type!r}; "
                f"expected one of {sorted(GARMENT_CONDITIONS)}"
            )
        self.garment_type = garment_type
        self.conditions = GARMENT_CONDITIONS[garment_type]
        self.cfg = cfg or GarmentFunctionalCfg()

        needed = max(slot for *_, slot in self.conditions) + 1
        if len(success_distance) < needed:
            raise ValueError(
                f"{garment_type} needs {needed} thresholds, got {len(success_distance)}"
            )
        self.success_distance = list(success_distance)

        if self.cfg.weights is not None and len(self.cfg.weights) != len(self.conditions):
            raise ValueError(
                f"weights has {len(self.cfg.weights)} entries, "
                f"{garment_type} has {len(self.conditions)} conditions"
            )

    # ------------------------------------------------------------------ compute

    def __call__(
        self, points: torch.Tensor, return_components: bool = False
    ):
        """Evaluate J from check-point positions.

        Args:
            points: ``(P, 3)`` or ``(B, P, 3)`` check-point positions **in
                centimetres**, ordered as LeHome's ``check_points``.
            return_components: also return per-condition margins.
        Returns:
            ``J`` of shape ``(B,)``, optionally with a component dict.
        """
        p = torch.as_tensor(points)
        if p.dim() == 2:
            p = p.unsqueeze(0)
        if p.dim() != 3 or p.shape[-1] != 3:
            raise ValueError(f"points must be (P,3) or (B,P,3), got {tuple(p.shape)}")

        total = p.new_zeros(p.shape[0])
        comps: Dict[str, torch.Tensor] = {}
        for k, (i, j, cmp_, slot) in enumerate(self.conditions):
            d = torch.linalg.vector_norm(p[:, i] - p[:, j], dim=-1)
            t = self.success_distance[slot]
            # Margin violation: how far this condition is from being satisfied.
            margin = (d - t) if cmp_ == "le" else (t - d)
            violation = margin.clamp_min(0.0)
            w = 1.0 if self.cfg.weights is None else float(self.cfg.weights[k])
            total = total + w * violation
            if return_components:
                comps[f"c{k+1}_{cmp_}_{i}{j}"] = violation
                comps[f"c{k+1}_dist"] = d

        j = total / self.cfg.scale
        if return_components:
            comps["J"] = j
            return j, comps
        return j

    # ------------------------------------------------------------------ helpers

    def is_success(self, points: torch.Tensor) -> torch.Tensor:
        """Boolean success, matching LeHome's checker exactly (J == 0)."""
        return self(points) <= 0.0

    @property
    def num_conditions(self) -> int:
        return len(self.conditions)

    @property
    def required_points(self) -> int:
        """Highest check-point index referenced, plus one."""
        return max(max(i, j) for i, j, _, _ in self.conditions) + 1

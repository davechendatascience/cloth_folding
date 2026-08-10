"""Retrieve joint configurations that grasp, instead of solving IK for them.

Our DLS grasp holds cloth in 0 of 60 randomised states. The demonstrations hold
constantly. The measured difference is posture, not reach: our wrist arrives
about 30 degrees from where the demos hold it (x-axis 97.8 vs 129.4 degrees to
world -z, z-axis 169.2 vs 140.6, y identical -- a rotation about y), with roughly
double the variance, and 3.5 cm further from the cloth.

The arm has 5 controllable joints, so orientation is not freely assignable and
position-only IK is a kinematic necessity rather than a mistake. We cannot
*command* the demonstrators' wrist pose. So we retrieve it.

Every demo grasp frame carries a joint configuration that is simultaneously
reachable, correctly oriented, and proven to hold cloth. Indexing those by
gripper position gives an inverse kinematics restricted to the manifold the
demonstrations actually use -- with no null-space lottery, because the solution
is a recorded one rather than a solver's choice.

Locally-weighted linear regression rather than plain nearest neighbour: the
library is a sparse sample of a smooth map, and snapping to the nearest recorded
pose would quantise targets to wherever a demonstrator happened to be. A local
linear fit interpolates between neighbours while staying on their manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class GraspRetrievalCfg:
    k: int = 24
    """Neighbours in the local fit. Too few overfits noisy demo frames; too many
    averages across distinct wrist postures and reintroduces the very posture
    variance this exists to remove."""
    ridge: float = 1e-2
    """Regularisation on the local fit, in normalised units."""
    max_dist_cm: float = 12.0
    """Refuse to extrapolate beyond this from the nearest library sample. Outside
    the demonstrated manifold there is no evidence the pose grasps, and a
    confident wrong answer is worse than an abstention."""
    bandwidth_cm: float = 8.0
    """Gaussian weighting scale on neighbour distance."""


class GraspRetriever:
    """Maps a desired gripper position to a demonstrated joint configuration.

    Args:
        library: path to ``grasp_library.npz`` from ``extract_grasp_library.py``.
        cfg: retrieval hyperparameters.
    """

    def __init__(self, library, cfg: Optional[GraspRetrievalCfg] = None) -> None:
        self.cfg = cfg or GraspRetrievalCfg()
        d = np.load(Path(library), allow_pickle=False)
        self.q = np.asarray(d["q"], dtype=np.float64)              # (N, 12)
        self.ee_pos = np.asarray(d["ee_pos"], dtype=np.float64)    # (N, 3) cm
        self.arm = np.asarray(d["arm"], dtype=np.int64)            # (N,)
        self.grip_val = np.asarray(d["grip_val"], dtype=np.float64)
        self.cp_pos = np.asarray(d["cp_pos"], dtype=np.float64)
        if len(self.q) == 0:
            raise ValueError(f"empty grasp library: {library}")
        # Per-arm index: the two arms occupy disjoint workspaces (arm0 spans
        # x -48..-10 cm, arm1 spans 7..44), so mixing them would interpolate
        # across a gap that no configuration occupies.
        self._by_arm = {a: np.where(self.arm == a)[0] for a in (0, 1)}

    # ------------------------------------------------------------------ query
    def arm_for(self, target_cm: np.ndarray) -> Optional[int]:
        """Which arm can reach this point, by nearest library sample."""
        best, best_d = None, np.inf
        for a, idx in self._by_arm.items():
            if not len(idx):
                continue
            d = float(np.linalg.norm(self.ee_pos[idx] - target_cm, axis=-1).min())
            if d < best_d:
                best, best_d = a, d
        return best

    def nearest_distance(self, target_cm: np.ndarray, arm: int) -> float:
        idx = self._by_arm.get(arm, np.array([], dtype=int))
        if not len(idx):
            return float("inf")
        return float(np.linalg.norm(self.ee_pos[idx] - target_cm, axis=-1).min())

    def query(self, target_cm: np.ndarray, arm: int):
        """Joint configuration placing the gripper body near ``target_cm``.

        Returns ``(q, grip_closed, distance)`` or ``None`` when the target lies
        outside the demonstrated manifold -- abstaining is deliberate, since a
        pose with no supporting evidence is not known to grasp.
        """
        target_cm = np.asarray(target_cm, dtype=np.float64).reshape(3)
        idx = self._by_arm.get(arm, np.array([], dtype=int))
        if not len(idx):
            return None
        d = np.linalg.norm(self.ee_pos[idx] - target_cm, axis=-1)
        if d.min() > self.cfg.max_dist_cm:
            return None
        k = min(self.cfg.k, len(idx))
        sel = idx[np.argsort(d)[:k]]
        dk = np.linalg.norm(self.ee_pos[sel] - target_cm, axis=-1)

        # Locally-weighted linear fit q ~ A @ (x - target) + b, evaluated at the
        # target so b is the answer. Weighting keeps nearer demonstrations
        # dominant without discarding the rest.
        w = np.exp(-(dk / self.cfg.bandwidth_cm) ** 2)
        X = self.ee_pos[sel] - target_cm
        A = np.hstack([X, np.ones((len(sel), 1))])
        W = np.diag(w)
        lhs = A.T @ W @ A + self.cfg.ridge * np.eye(A.shape[1]) * max(w.sum(), 1e-9)
        coef = np.linalg.solve(lhs, A.T @ W @ self.q[sel])
        q = coef[-1]                       # the intercept == prediction at X=0

        gi = 5 if arm == 0 else 11
        grip = float(np.average(self.grip_val[sel], weights=w))
        q[gi] = grip
        return q, grip, float(dk.min())

    # ------------------------------------------------------------- diagnostics
    def coverage(self, points_cm: np.ndarray, arm: int) -> float:
        """Fraction of ``points_cm`` inside the demonstrated manifold."""
        return float(np.mean([
            self.nearest_distance(p, arm) <= self.cfg.max_dist_cm for p in points_cm
        ]))

    def loo_error(self, arm: int, n: int = 200, seed: int = 0) -> dict:
        """Leave-one-out joint error, in radians.

        The honest check on the local fit: hold out a library sample, predict its
        joint configuration from the rest, compare. A large error means the map
        from gripper position to joint configuration is not single-valued at this
        neighbourhood size -- distinct wrist postures reaching the same point --
        which would defeat the entire point of retrieving posture.
        """
        idx = self._by_arm.get(arm, np.array([], dtype=int))
        if len(idx) < self.cfg.k + 2:
            return {}
        rng = np.random.RandomState(seed)
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
        errs = []
        for i in pick:
            keep = self._by_arm[arm]
            keep = keep[keep != i]
            saved = self._by_arm[arm]
            self._by_arm[arm] = keep
            try:
                r = self.query(self.ee_pos[i], arm)
            finally:
                self._by_arm[arm] = saved
            if r is None:
                continue
            q, _, _ = r
            arm_ids = [0, 1, 2, 3, 4] if arm == 0 else [6, 7, 8, 9, 10]
            errs.append(np.abs(q[arm_ids] - self.q[i][arm_ids]))
        if not errs:
            return {}
        E = np.stack(errs)
        return {"n": len(E), "mean_rad": float(E.mean()),
                "p90_rad": float(np.percentile(E, 90)),
                "mean_deg": float(np.degrees(E.mean()))}

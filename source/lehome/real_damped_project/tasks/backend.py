"""The LeHome coupling seam.

The spec's task class talks to LeHome through a small, fixed set of calls
(Sec. 4.2 of the newer spec, Sec. 4.2.2-4.2.4 of the earlier one):

    reset_env_ids, get_end_effector_positions, get_end_effector_velocities,
    set_end_effector_targets, render_cameras, get_proprioception,
    compute_cloth_error, check_done

:class:`LeHomeBackend` states that contract explicitly. Two implementations:

* :class:`IsaacLeHomeBackend` -- thin adapter over LeHome's ``FoldGarmentEnv``.
  Requires ``lehome`` + Isaac Lab to be installed; imported lazily.
* :class:`MockFoldGarmentBackend` -- a self-contained, GPU-capable damped
  mass-spring cloth with a critically-damped EE impedance model. It exists so
  the reward, controller, policy and PPO loop can be exercised end-to-end
  *before* the simulator stack is available, and so regressions in the RL
  machinery can be caught in seconds rather than in a multi-hour Isaac run.

The mock is a genuine dissipative system -- Rayleigh-damped springs, an
over-dampable second-order EE -- so it is a fair testbed for the damping and
monotonicity claims. It is emphatically *not* a substitute for Newton cloth:
no self-collision, no friction, no contact, no bending resistance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import torch

from ..math.cloth_functional import (
    ClothErrorFunctional,
    ClothFunctionalCfg,
    make_flat_cloth,
    make_folded_target,
)


@runtime_checkable
class LeHomeBackend(Protocol):
    """Everything the damped task needs from the underlying cloth environment."""

    num_envs: int
    num_arms: int
    dt: float

    def reset_env_ids(self, env_ids: torch.Tensor) -> None: ...
    def get_end_effector_positions(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    def get_end_effector_velocities(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    def set_end_effector_targets(self, x_cmd: torch.Tensor) -> None: ...
    def simulate(self) -> None: ...
    def render_cameras(self) -> torch.Tensor: ...
    def get_proprioception(self) -> torch.Tensor: ...
    def compute_cloth_error(self) -> torch.Tensor: ...
    def check_done(self) -> Tuple[torch.Tensor, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


@dataclass
class MockClothCfg:
    """Configuration for the damped mass-spring stand-in cloth."""

    num_envs: int = 8
    rows: int = 9
    cols: int = 9
    size: Tuple[float, float] = (0.30, 0.30)
    dt: float = 1.0 / 60.0
    substeps: int = 4

    # --- cloth material ---
    vertex_mass: float = 0.01
    stiffness: float = 60.0
    """Structural spring constant k (N/m)."""
    rayleigh_alpha: float = 4.0
    """Mass-proportional damping (1/s). D = alpha*M + beta*K (Sec. 2.2)."""
    rayleigh_beta: float = 0.005
    """Stiffness-proportional damping (s).

    Kept small on purpose. Explicit integration of the beta*K*v term is stable
    only while ``h * beta * k * lambda_max(L) / m < 2``; with the defaults here
    that caps beta near 0.01, and the more intuitive value 0.08 blows the sheet
    up within a few frames. :meth:`MockFoldGarmentBackend._required_substeps`
    enforces the bound at construction rather than leaving it to be discovered
    as a NaN downstream.
    """
    gravity: float = -0.5
    """Deliberately weak: the mock has no ground contact model, so full g would
    just drop the sheet. Enough to break symmetry, not enough to dominate."""

    # --- end effector impedance (mirrors DampedImpedanceController gains) ---
    ee_mass: float = 1.0
    ee_stiffness: float = 200.0
    ee_damping_ratio: float = 1.0

    # --- episode ---
    max_episode_steps: int = 200
    success_threshold: float = 0.02
    """Episode succeeds when J drops below this."""
    reset_noise: float = 0.02
    """Uniform xy jitter (m) applied to the initial sheet, so the task is not
    a single fixed trajectory."""

    # --- cameras ---
    image_res: int = 64
    seed: int = 0

    functional: ClothFunctionalCfg = field(default_factory=ClothFunctionalCfg)

    @property
    def num_verts(self) -> int:
        return self.rows * self.cols


class MockFoldGarmentBackend:
    """Damped mass-spring cloth with two impedance-controlled grippers.

    State per environment:
        ``verts`` (N,3), ``vel`` (N,3), ``ee_pos`` (2,3), ``ee_vel`` (2,3),
        ``ee_cmd`` (2,3).

    The two arms grip the two left-edge corners; the goal is to fold the left
    half onto the right half, matching :func:`make_folded_target`.
    """

    def __init__(self, cfg: MockClothCfg, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = cfg.num_envs
        self.num_arms = 2
        self.dt = cfg.dt
        self._gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

        fcfg = cfg.functional
        if fcfg.grid_shape is None:
            fcfg.grid_shape = (cfg.rows, cfg.cols)

        self._rest = make_flat_cloth(cfg.rows, cfg.cols, cfg.size, device=self.device)
        self._target = make_folded_target(cfg.rows, cfg.cols, cfg.size, device=self.device)
        self.functional = ClothErrorFunctional(fcfg, self._target).to(self.device)

        # Gripped vertices: the two corners of the left edge (row 0 / row-1, col 0).
        self._grip_idx = torch.tensor(
            [0, (cfg.rows - 1) * cfg.cols], dtype=torch.long, device=self.device
        )

        # Rest lengths for structural springs.
        g = self._rest.view(cfg.rows, cfg.cols, 3)
        self._rest_len_x = torch.linalg.vector_norm(g[:, 1:] - g[:, :-1], dim=-1)  # (R, C-1)
        self._rest_len_y = torch.linalg.vector_norm(g[1:, :] - g[:-1, :], dim=-1)  # (R-1, C)

        self.substeps = self._required_substeps()

        b, n = cfg.num_envs, cfg.num_verts
        self.verts = self._rest.unsqueeze(0).repeat(b, 1, 1)
        self.vel = torch.zeros(b, n, 3, device=self.device)
        self.ee_pos = torch.zeros(b, 2, 3, device=self.device)
        self.ee_vel = torch.zeros(b, 2, 3, device=self.device)
        self.ee_cmd = torch.zeros(b, 2, 3, device=self.device)
        self.episode_step = torch.zeros(b, dtype=torch.long, device=self.device)

        self.reset_env_ids(torch.arange(b, device=self.device))

    # --------------------------------------------------------------- properties

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (6, self.cfg.image_res, self.cfg.image_res)

    @property
    def proprio_dim(self) -> int:
        return 6 + 6 + 6  # ee_pos, ee_vel, ee_cmd

    @property
    def target_verts(self) -> torch.Tensor:
        return self._target

    # ------------------------------------------------------------------- stability

    def _required_substeps(self, max_substeps: int = 256) -> int:
        """Smallest sub-step count that keeps explicit integration stable.

        Three explicit-Euler conditions must hold for the sub-step ``h``:

        * springs      ``h * sqrt(k * L_max / m) < 2``
        * beta damping ``h * beta * k * L_max / m  < 2``
        * alpha damping``h * alpha                 < 2``
        * EE impedance ``h * omega_n_ee            < 2``

        ``L_max ~= 8`` is the largest eigenvalue of the 5-point grid Laplacian.
        A safety factor of 4 keeps us away from the marginal case, where the
        scheme is technically stable but visibly ringing -- which would
        contaminate the very damping behaviour this backend exists to model.
        """
        import math
        import warnings

        c = self.cfg
        safety, l_max = 4.0, 8.0
        m = c.vertex_mass

        limits = [
            2.0 / max(math.sqrt(c.stiffness * l_max / m), 1e-12),
            2.0 / max(c.rayleigh_beta * c.stiffness * l_max / m, 1e-12),
            2.0 / max(c.rayleigh_alpha, 1e-12),
            2.0 / max(math.sqrt(c.ee_stiffness / c.ee_mass), 1e-12),
        ]
        h_max = min(limits) / safety
        needed = max(1, math.ceil(c.dt / h_max))

        if needed > max_substeps:
            warnings.warn(
                f"mock cloth needs {needed} sub-steps for stability but is capped at "
                f"{max_substeps}; reduce stiffness ({c.stiffness}), rayleigh_beta "
                f"({c.rayleigh_beta}) or dt ({c.dt}). The sheet may diverge.",
                stacklevel=3,
            )
            return max_substeps
        return max(needed, c.substeps)

    # ------------------------------------------------------------------- lifecycle

    def reset_env_ids(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        k = env_ids.numel()

        noise = (
            torch.rand(k, 1, 3, generator=self._gen).to(self.device) * 2.0 - 1.0
        ) * self.cfg.reset_noise
        noise[..., 2] = 0.0  # keep the sheet on its plane at reset

        self.verts[env_ids] = self._rest.unsqueeze(0) + noise
        self.vel[env_ids] = 0.0
        self.episode_step[env_ids] = 0

        grip = self.verts[env_ids][:, self._grip_idx, :]  # (k, 2, 3)
        self.ee_pos[env_ids] = grip
        self.ee_cmd[env_ids] = grip
        self.ee_vel[env_ids] = 0.0

    # ------------------------------------------------------------------- getters

    def get_end_effector_positions(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.ee_pos if env_ids is None else self.ee_pos[env_ids]

    def get_end_effector_velocities(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.ee_vel if env_ids is None else self.ee_vel[env_ids]

    def set_end_effector_targets(self, x_cmd: torch.Tensor) -> None:
        if x_cmd.shape != self.ee_cmd.shape:
            raise ValueError(f"x_cmd shape {tuple(x_cmd.shape)} != {tuple(self.ee_cmd.shape)}")
        self.ee_cmd = x_cmd.to(self.device, self.ee_cmd.dtype)

    # -------------------------------------------------------------------- physics

    def _spring_forces(self, p: torch.Tensor) -> torch.Tensor:
        """Structural spring force on each vertex, ``(B, R, C, 3)`` in, same out."""
        k = self.cfg.stiffness
        f = torch.zeros_like(p)

        # Along columns (x direction).
        d = p[:, :, 1:] - p[:, :, :-1]
        length = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
        force = k * (length - self._rest_len_x.unsqueeze(0).unsqueeze(-1)) * (d / length)
        f[:, :, :-1] += force
        f[:, :, 1:] -= force

        # Along rows (y direction).
        d = p[:, 1:, :] - p[:, :-1, :]
        length = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
        force = k * (length - self._rest_len_y.unsqueeze(0).unsqueeze(-1)) * (d / length)
        f[:, :-1, :] += force
        f[:, 1:, :] -= force
        return f

    def _graph_laplacian(self, v: torch.Tensor) -> torch.Tensor:
        """Graph Laplacian ``(L v)_i = sum_j (v_i - v_j)`` over grid neighbours.

        This is the linearised stiffness operator applied to velocities, used
        for the stiffness-proportional (beta) half of Rayleigh damping, whose
        force is ``-beta * K * v``.

        The sign convention matters and is easy to get backwards: the operator
        must return ``+L v``, not ``-L v``. With the opposite sign the "damping"
        term becomes an energy *source* and the sheet diverges within a few
        frames -- mildly enough (~1.5x per sub-step) to look like an ordinary
        CFL problem rather than a sign error.
        """
        out = torch.zeros_like(v)
        d = v[:, :, 1:] - v[:, :, :-1]
        out[:, :, :-1] -= d
        out[:, :, 1:] += d
        d = v[:, 1:, :] - v[:, :-1, :]
        out[:, :-1, :] -= d
        out[:, 1:, :] += d
        return out

    def simulate(self) -> None:
        """Advance one control step (``substeps`` physics sub-steps)."""
        c = self.cfg
        h = c.dt / self.substeps
        m = c.vertex_mass
        b, rows, cols = self.num_envs, c.rows, c.cols

        # --- EE impedance: M x_ddot + D x_dot + K (x - x_cmd) = 0 -----------
        kd = 2.0 * c.ee_damping_ratio * (c.ee_stiffness * c.ee_mass) ** 0.5
        for _ in range(self.substeps):
            acc = (-c.ee_stiffness * (self.ee_pos - self.ee_cmd) - kd * self.ee_vel) / c.ee_mass
            self.ee_vel = self.ee_vel + h * acc
            self.ee_pos = self.ee_pos + h * self.ee_vel

        # --- cloth: Rayleigh-damped mass-spring, semi-implicit Euler --------
        p = self.verts.view(b, rows, cols, 3)
        v = self.vel.view(b, rows, cols, 3)
        for _ in range(self.substeps):
            f = self._spring_forces(p)
            f = f - c.rayleigh_alpha * m * v                       # alpha * M * v
            f = f - c.rayleigh_beta * c.stiffness * self._graph_laplacian(v)  # beta * K * v
            f[..., 2] += m * c.gravity
            v = v + h * (f / m)
            p = p + h * v

        self.verts = p.reshape(b, -1, 3)
        self.vel = v.reshape(b, -1, 3)

        # --- gripped vertices are kinematically driven by the EEs -----------
        self.verts[:, self._grip_idx, :] = self.ee_pos
        self.vel[:, self._grip_idx, :] = self.ee_vel

        self.episode_step += 1

    # -------------------------------------------------------------- observations

    def _splat(self, pts: torch.Tensor, ax0: int, ax1: int, bounds, sigma: float) -> torch.Tensor:
        """Soft top-down/side occupancy image, ``(B, R, R)``."""
        r = self.cfg.image_res
        a0, a1, b0, b1 = bounds
        xs = torch.linspace(a0, a1, r, device=pts.device, dtype=pts.dtype)
        ys = torch.linspace(b0, b1, r, device=pts.device, dtype=pts.dtype)
        two_s2 = 2.0 * sigma * sigma
        wx = torch.exp(-((pts[..., ax0:ax0 + 1] - xs.view(1, 1, -1)) ** 2) / two_s2)
        wy = torch.exp(-((pts[..., ax1:ax1 + 1] - ys.view(1, 1, -1)) ** 2) / two_s2)
        return -torch.expm1(-torch.einsum("bnx,bny->bxy", wx, wy))

    def render_cameras(self) -> torch.Tensor:
        """``(B, 6, H, W)``: two virtual cameras x (cloth, grippers, target)."""
        c = self.cfg
        sx, sy = c.size
        top_bounds = (-sx, sx, -sy, sy)
        side_bounds = (-sx, sx, -0.1, 0.3)
        sigma = 1.5 * (2.0 * sx) / c.image_res

        tgt = self._target.unsqueeze(0).expand(self.num_envs, -1, -1)
        top = torch.stack(
            [
                self._splat(self.verts, 0, 1, top_bounds, sigma),
                self._splat(self.ee_pos, 0, 1, top_bounds, sigma * 2),
                self._splat(tgt, 0, 1, top_bounds, sigma),
            ],
            dim=1,
        )
        side = torch.stack(
            [
                self._splat(self.verts, 0, 2, side_bounds, sigma),
                self._splat(self.ee_pos, 0, 2, side_bounds, sigma * 2),
                self._splat(tgt, 0, 2, side_bounds, sigma),
            ],
            dim=1,
        )
        return torch.cat([top, side], dim=1)

    def get_proprioception(self) -> torch.Tensor:
        """``(B, 18)``. Contains no cloth state -- see Sec. 3.1."""
        return torch.cat(
            [self.ee_pos.flatten(1), self.ee_vel.flatten(1), self.ee_cmd.flatten(1)], dim=-1
        )

    def compute_cloth_error(self) -> torch.Tensor:
        return self.functional(self.verts)

    def check_done(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(terminated, truncated)``, each ``(B,)`` bool."""
        j = self.compute_cloth_error()
        terminated = j < self.cfg.success_threshold
        truncated = self.episode_step >= self.cfg.max_episode_steps
        return terminated, truncated


# ---------------------------------------------------------------------------
# Isaac / LeHome backend
# ---------------------------------------------------------------------------


class IsaacLeHomeBackend:
    """Adapter over LeHome's ``FoldGarmentEnv``.

    Constructed only when ``lehome`` and Isaac Lab are importable. The method
    bodies are one-line delegations to the LeHome API named in the spec; the
    exact call signatures must be reconciled against the installed
    ``lehome-challenge`` revision before first use -- see ``README.md``.
    """

    def __init__(self, cfg) -> None:  # pragma: no cover - requires Isaac Lab
        try:
            from lehome.tasks.fold_garment import FoldGarmentEnv  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "LeHome is not installed. Install lehome-challenge + Isaac Lab (see README), "
                "or use MockFoldGarmentBackend for offline testing."
            ) from exc

        self.env = FoldGarmentEnv(cfg)
        self.num_envs = cfg.scene.num_envs
        self.num_arms = 2
        self.dt = self.env.dt

    def reset_env_ids(self, env_ids):  # pragma: no cover
        return self.env.reset_env_ids(env_ids)

    def get_end_effector_positions(self, env_ids=None):  # pragma: no cover
        return self.env.get_end_effector_positions(env_ids)

    def get_end_effector_velocities(self, env_ids=None):  # pragma: no cover
        return self.env.get_end_effector_velocities(env_ids)

    def set_end_effector_targets(self, x_cmd):  # pragma: no cover
        return self.env.set_end_effector_targets(x_cmd)

    def simulate(self):  # pragma: no cover
        return self.env.step_physics()

    def render_cameras(self):  # pragma: no cover
        return self.env.render_cameras()

    def get_proprioception(self):  # pragma: no cover
        return self.env.get_proprioception()

    def compute_cloth_error(self):  # pragma: no cover
        return self.env.compute_cloth_error()

    def check_done(self):  # pragma: no cover
        return self.env.check_done()

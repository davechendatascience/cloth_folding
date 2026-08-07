"""Damped impedance controller for the bimanual SO-ARM101 (spec Sec. 2.2, 3.2).

Two jobs, matching the two roles the spec gives this class:

1. **Command shaping (this file).** Turn raw policy deltas into commanded EE
   positions through a clipped, low-pass-filtered integrator. The resulting map
   ``u -> x_cmd`` is bounded and Lipschitz,

       || x_cmd(u) - x_cmd(v) || <= L || u - v ||,     L = beta_smooth

   which is what Sec. 3.2 asks for, and is what supports the contraction
   property of the closed-loop map F.

2. **Gain synthesis (this file, consumed by the task cfg).** Report the
   physical impedance gains ``(K, D)`` for the EE dynamics

       M x_ddot + D x_dot + K (x - x_cmd) = 0,     D = 2 * zeta * sqrt(K * M)

   with ``zeta >= 1`` (critically or over-damped). The *dynamics* themselves are
   integrated by Isaac Lab's OSC / impedance actuator; this class only supplies
   the gains and the reference ``x_cmd``.

Why each shaping step is non-expansive (so the composition is Lipschitz):

* elementwise clip to ``[-max_delta, max_delta]`` -- projection onto a box,
  1-Lipschitz;
* first-order low pass ``d <- (1-b) d_prev + b d`` -- ``b``-Lipschitz in the new
  input, and it removes the high-frequency content that causes EE ringing;
* integration ``x_cmd <- x_cmd + d`` -- 1-Lipschitz in ``d``;
* optional projection onto the workspace box -- projection onto a convex set,
  1-Lipschitz.

Pure torch: no Isaac Lab dependency, so it is directly unit-testable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class DampedImpedanceController:
    """Clipped, low-pass integrator producing commanded EE positions.

    Args:
        damping_ratio: zeta for the EE impedance. Must be >= 1 for the
            critically/over-damped behaviour the spec requires.
        stiffness: nominal Cartesian stiffness K (N/m).
        max_delta: per-step position change bound (m). Sets the Lipschitz
            bound on one integrator step and caps EE speed at
            ``max_delta / dt``.
        dt: control period (s).
        ee_mass: apparent Cartesian mass M (kg), used for gain synthesis.
        smoothing: low-pass coefficient ``beta`` in (0, 1]. 1.0 disables
            filtering; smaller values damp the command path harder.
        workspace_bounds: optional ``(lo, hi)``, each broadcastable to
            ``(..., 3)``, clamping commanded positions.
    """

    def __init__(
        self,
        damping_ratio: float = 1.0,
        stiffness: float = 200.0,
        max_delta: float = 0.02,
        dt: float = 1.0 / 60.0,
        ee_mass: float = 1.0,
        smoothing: float = 0.5,
        max_lead: Optional[float] = 0.05,
        workspace_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        if damping_ratio < 1.0:
            raise ValueError(
                f"damping_ratio must be >= 1 (critical/over-damped); got {damping_ratio}. "
                "An under-damped EE contradicts the non-ringing requirement (Sec. 2.2)."
            )
        if stiffness <= 0.0:
            raise ValueError("stiffness must be > 0")
        if max_delta <= 0.0:
            raise ValueError("max_delta must be > 0")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if ee_mass <= 0.0:
            raise ValueError("ee_mass must be > 0")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError(f"smoothing must be in (0, 1]; got {smoothing}")

        self.damping_ratio = float(damping_ratio)
        self.stiffness = float(stiffness)
        self.max_delta = float(max_delta)
        self.dt = float(dt)
        self.ee_mass = float(ee_mass)
        self.smoothing = float(smoothing)
        self.max_lead = None if max_lead is None else float(max_lead)
        self.workspace_bounds = workspace_bounds

        if self.max_lead is not None and self.max_lead <= 0.0:
            raise ValueError("max_lead must be > 0 or None")

        self._x_cmd: Optional[torch.Tensor] = None
        self._delta_prev: Optional[torch.Tensor] = None

    # ------------------------------------------------------------ gain synthesis

    @property
    def damping(self) -> float:
        """D = 2 * zeta * sqrt(K * M) -- the Rayleigh-consistent EE damping."""
        return 2.0 * self.damping_ratio * math.sqrt(self.stiffness * self.ee_mass)

    @property
    def natural_frequency(self) -> float:
        """omega_n = sqrt(K / M), rad/s."""
        return math.sqrt(self.stiffness / self.ee_mass)

    @property
    def lipschitz_constant(self) -> float:
        """L in ``||x_cmd(u) - x_cmd(v)|| <= L ||u - v||`` for one step."""
        return self.smoothing

    def is_sampling_stable(self) -> bool:
        """Whether ``dt`` resolves the impedance dynamics.

        A discrete integrator cannot represent a mode faster than ~1/(2 dt).
        We require at least ~10 samples per natural period, which is the
        practical condition for the discretised closed loop to stay dissipative
        rather than pick up numerical ringing.
        """
        return self.natural_frequency * self.dt < 2.0 * math.pi / 10.0

    # ----------------------------------------------------------------- lifecycle

    def reset(self, x_init: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> None:
        """Initialise (or re-initialise) the commanded EE positions.

        Args:
            x_init: ``(num_envs, num_arms, 3)`` initial EE positions. When
                ``env_ids`` is given, this may instead be ``(len(env_ids),
                num_arms, 3)`` -- only those rows are written.
            env_ids: optional 1-D indices of environments to reset. ``None``
                resets every environment.
        """
        x_init = torch.as_tensor(x_init)
        if x_init.dim() != 3 or x_init.shape[-1] != 3:
            raise ValueError(f"x_init must be (num_envs, num_arms, 3), got {tuple(x_init.shape)}")

        if env_ids is None or self._x_cmd is None:
            self._x_cmd = x_init.clone()
            self._delta_prev = torch.zeros_like(self._x_cmd)
            return

        rows = x_init if x_init.shape[0] == env_ids.numel() else x_init[env_ids]
        self._x_cmd[env_ids] = rows.to(self._x_cmd.dtype).to(self._x_cmd.device)
        self._delta_prev[env_ids] = 0.0

    @property
    def x_cmd(self) -> Optional[torch.Tensor]:
        """Current commanded EE positions, or ``None`` before the first reset."""
        return self._x_cmd

    # ---------------------------------------------------------------------- step

    def step(self, x_current: torch.Tensor, delta_cmd: torch.Tensor) -> torch.Tensor:
        """Push policy deltas through the damped, clipped integrator.

        Args:
            x_current: ``(num_envs, num_arms, 3)`` measured EE positions. Used
                only to lazily initialise the command if :meth:`reset` was never
                called.
            delta_cmd: ``(num_envs, num_arms, 3)`` desired Cartesian deltas.
        Returns:
            ``(num_envs, num_arms, 3)`` new commanded EE positions.
        """
        x_current = torch.as_tensor(x_current)
        delta_cmd = torch.as_tensor(delta_cmd)
        if delta_cmd.shape != x_current.shape:
            raise ValueError(
                f"delta_cmd shape {tuple(delta_cmd.shape)} != x_current shape {tuple(x_current.shape)}"
            )

        if self._x_cmd is None:
            self.reset(x_current)
        elif self._x_cmd.shape != x_current.shape:
            raise ValueError(
                f"controller was reset with shape {tuple(self._x_cmd.shape)} but got "
                f"{tuple(x_current.shape)}; call reset() after changing num_envs"
            )

        delta = delta_cmd.to(self._x_cmd.dtype).to(self._x_cmd.device)
        delta = delta.clamp(-self.max_delta, self.max_delta)

        # First-order low pass: the command-path damping.
        b = self.smoothing
        delta = (1.0 - b) * self._delta_prev + b * delta
        self._delta_prev = delta

        x_cmd = self._x_cmd + delta

        # --- anti-windup -----------------------------------------------------
        # The command is an integrator, which adds 90 degrees of phase lag. Fed
        # from any proportional feedback (which is what a competent policy is),
        # that lag closes an oscillatory loop around the second-order EE plant:
        # x_cmd keeps accumulating while the EE lags, overshoots, and the error
        # flips sign -- a sustained limit cycle, even though the impedance
        # itself is critically damped for a fixed setpoint.
        #
        # Measured before this leash: commanding a 0.30 m move drove x_cmd to
        # 0.31 past a 0.15 m goal and left the EE ringing with ~0.09 m amplitude
        # indefinitely.
        #
        # Projecting the command into a ball of radius `max_lead` around the
        # measured pose bounds the accumulated lead, which is exactly the
        # classic integrator anti-windup. Projection onto a convex set is
        # non-expansive, so the Lipschitz guarantee of Sec. 3.2 is preserved.
        if self.max_lead is not None:
            lead = x_cmd - x_current.to(x_cmd)
            dist = torch.linalg.vector_norm(lead, dim=-1, keepdim=True)
            scale = (self.max_lead / dist.clamp_min(1e-9)).clamp(max=1.0)
            x_cmd = x_current.to(x_cmd) + lead * scale

            # The projection alone can yank the command further than one step's
            # travel, breaking the `max_delta / dt` speed bound that the clip
            # exists to provide. Rate-limiting the correction keeps both
            # guarantees: the lead stays bounded (no windup) *and* the command
            # never moves faster than max_delta per step.
            x_cmd = self._x_cmd + (x_cmd - self._x_cmd).clamp(-self.max_delta, self.max_delta)

        if self.workspace_bounds is not None:
            lo, hi = self.workspace_bounds
            x_cmd = torch.maximum(torch.minimum(x_cmd, hi.to(x_cmd)), lo.to(x_cmd))

        self._x_cmd = x_cmd
        return x_cmd

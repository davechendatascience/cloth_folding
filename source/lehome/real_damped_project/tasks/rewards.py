"""Reward as discrete Lyapunov descent (spec Sec. 2.3).

    r_t = r_task + r_mono + r_vel + r_act

with

    r_task = -J(x_t)
    r_mono = -lambda_up * dJ           if J(x_t) < J_near and dJ >  epsilon
             +lambda_down             if J(x_t) < J_near and dJ < -epsilon
             0                        otherwise
    r_vel  = -lambda_v      * ||x_dot_ee||
    r_act  = -lambda_da     * ||a_t - a_{t-1}||

where ``dJ = J(x_{t+1}) - J(x_t)``.

Two fidelity notes, both places where the two spec revisions disagree and this
implementation follows the *newer* one (`real_analysis_..._spec.md`):

* **Near-goal gate.** Sec. 2.3 states the condition as "If J(x_t) < J_near",
  i.e. the gate is on the *previous* J -- we ask for monotone descent once the
  trajectory has entered the near-goal region. The earlier spec's snippet gated
  on the new J instead. ``near_gate`` exposes both; default is ``"prev"``,
  per Sec. 2.3. This matters: gating on the new J cannot penalise the step that
  *leaves* the near-goal region, which is exactly the ringing we want to punish.

* **Action term.** Sec. 2.3 defines ``r_act`` on the action *change*
  ``||a_t - a_{t-1}||``. The earlier spec's snippet used ``||a_{t-1}||``, which
  penalises motion rather than jerk. The change norm is used here.

The point of ``r_mono`` is the real-analysis argument: if a policy makes
``{J(x_t)}`` eventually monotone non-increasing, then -- being bounded below by
0 -- it converges by the monotone convergence theorem. ``r_task`` is what pushes
that limit toward 0; ``r_mono`` is what forbids the oscillation.

Pure torch, batched over envs, no simulator dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class LyapunovRewardCfg:
    """Reward hyperparameters (spec Sec. 5.1)."""

    j_near: float = 0.02
    """Threshold below which monotone descent is enforced."""
    epsilon: float = 1.0e-3
    """Dead-band on dJ; changes smaller than this are treated as noise."""
    lambda_up: float = 10.0
    """Penalty scale on error *increase* near the goal. Should dominate
    lambda_down so that ringing is never net-profitable."""
    lambda_down: float = 1.0
    """Bonus for a significant error decrease near the goal."""
    lambda_v: float = 0.1
    """EE velocity penalty (physical damping)."""
    lambda_delta_a: float = 0.01
    """Action-change penalty (action-space damping)."""
    near_gate: str = "prev"
    """Which J gates the near-goal region: ``"prev"`` (Sec. 2.3), ``"next"``
    (earlier spec), or ``"either"`` (gate if either side is near)."""
    mono_mode: str = "proportional"
    """How the near-goal *descent* is rewarded.

    ``"constant"`` is Sec. 2.3 exactly as written: ``r_mono = +lambda_down``, a
    flat bonus, while the ascent penalty ``-lambda_up * dJ`` is proportional.
    That asymmetry is exploitable. Oscillating with amplitude ``a`` earns
    ``lambda_down - lambda_up * a`` per cycle, which is *positive* whenever

        a < lambda_down / lambda_up

    -- 0.1 with the spec's defaults (lambda_down=1, lambda_up=10), i.e. for any
    oscillation smaller than 5x the whole near-goal band. A policy maximising
    return would therefore learn to ring around the goal forever, which is the
    exact behaviour the design exists to prevent.

    ``"proportional"`` (default) makes the bonus proportional too,
    ``r_mono = -lambda_down * dJ``, so a full cycle nets ``(lambda_down -
    lambda_up) * a < 0`` whenever ``lambda_up > lambda_down``. The spec's
    intent -- descent rewarded, ascent punished harder -- is preserved, and the
    exploit closes.
    """

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        if self.j_near <= 0.0:
            raise ValueError("j_near must be > 0")
        for name in ("lambda_up", "lambda_down", "lambda_v", "lambda_delta_a"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.near_gate not in ("prev", "next", "either"):
            raise ValueError(f"near_gate must be prev|next|either, got {self.near_gate!r}")
        if self.mono_mode not in ("proportional", "constant"):
            raise ValueError(f"mono_mode must be proportional|constant, got {self.mono_mode!r}")

        import warnings

        if self.lambda_up < self.lambda_down:
            # Not fatal, but it inverts the intended incentive.
            warnings.warn(
                f"lambda_up ({self.lambda_up}) < lambda_down ({self.lambda_down}): a policy can "
                "profit from oscillating J up and down near the goal, defeating r_mono.",
                stacklevel=2,
            )
        if self.mono_mode == "constant":
            breakeven = self.lambda_down / max(self.lambda_up, 1e-12)
            warnings.warn(
                "mono_mode='constant' reproduces Sec. 2.3 literally, but the flat down-bonus "
                f"makes oscillation profitable for any amplitude below {breakeven:.4g}. "
                "Use mono_mode='proportional' unless you are deliberately reproducing the spec.",
                stacklevel=2,
            )


class LyapunovDescentReward:
    """Stateful reward term holding ``prev_J`` and ``prev_action`` per env.

    Args:
        cfg: reward hyperparameters.
        num_envs: number of parallel environments.
        action_dim: policy action dimension.
        device / dtype: storage for the internal buffers.
    """

    def __init__(
        self,
        cfg: LyapunovRewardCfg,
        num_envs: int,
        action_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self.prev_J = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device, dtype=dtype)
        # False until the env has produced its first J, so the first step of an
        # episode contributes no r_mono (dJ is undefined across a reset).
        self.has_prev_J = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # ---------------------------------------------------------------- lifecycle

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Clear ``prev_J`` / ``prev_action`` for the given envs (all if None).

        Crossing an episode boundary must not generate a spurious dJ: the cloth
        teleports back to its initial configuration, which is not a dynamical
        transition and carries no Lyapunov meaning.
        """
        if env_ids is None:
            self.prev_J.zero_()
            self.prev_action.zero_()
            self.has_prev_J.zero_()
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self.prev_J[env_ids] = 0.0
            self.prev_action[env_ids] = 0.0
            self.has_prev_J[env_ids] = False

    # ------------------------------------------------------------------ compute

    def compute(
        self,
        j: torch.Tensor,
        ee_vel: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate the total reward for one environment step.

        Args:
            j: ``(num_envs,)`` cloth error J(x_{t+1}) at the post-step state.
            ee_vel: ``(num_envs, num_arms, 3)`` EE velocities.
            action: ``(num_envs, action_dim)`` action a_t just applied.
        Returns:
            ``(reward, components)``, reward is ``(num_envs,)``.
        """
        c = self.cfg
        j = j.to(self.device, self.dtype)
        action = action.to(self.device, self.dtype)
        ee_vel = ee_vel.to(self.device, self.dtype)

        # --- task term: Lyapunov descent -----------------------------------
        r_task = -j

        # --- monotone convergence term -------------------------------------
        delta_j = j - self.prev_J
        if c.near_gate == "prev":
            near = self.prev_J < c.j_near
        elif c.near_gate == "next":
            near = j < c.j_near
        else:
            near = (self.prev_J < c.j_near) | (j < c.j_near)
        near = near & self.has_prev_J

        r_mono = torch.zeros_like(j)
        inc = near & (delta_j > c.epsilon)
        dec = near & (delta_j < -c.epsilon)
        r_mono = torch.where(inc, -c.lambda_up * delta_j, r_mono)
        if c.mono_mode == "proportional":
            # -lambda_down * dJ, and dJ < 0 here, so this is a positive bonus
            # proportional to the progress made.
            down = -c.lambda_down * delta_j
        else:
            down = torch.full_like(j, c.lambda_down)
        r_mono = torch.where(dec, down, r_mono)

        # --- damping terms --------------------------------------------------
        vel_norm = torch.linalg.vector_norm(ee_vel.flatten(1), dim=-1)
        r_vel = -c.lambda_v * vel_norm

        delta_a = action - self.prev_action
        r_act = -c.lambda_delta_a * torch.linalg.vector_norm(delta_a, dim=-1)

        reward = r_task + r_mono + r_vel + r_act

        # --- roll state forward ---------------------------------------------
        self.prev_J = j.clone()
        self.prev_action = action.clone()
        self.has_prev_J = torch.ones_like(self.has_prev_J)

        components = {
            "r_task": r_task,
            "r_mono": r_mono,
            "r_vel": r_vel,
            "r_act": r_act,
            "J": j,
            "delta_J": delta_j,
            # Emitted so downstream metrics use the same masks the reward used,
            # rather than re-deriving them and mis-handling the first step of an
            # episode (where dJ is undefined).
            "near_mask": near,
            "mono_violation": inc,
            "ee_speed": vel_norm,
        }
        return reward, components

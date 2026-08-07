"""``LeHomeFoldGarmentRealDampedEnv`` -- the damped cloth-folding RL task (Sec. 4).

Wraps a LeHome fold-garment backend with:

* Newton cloth damping (configured through the cfg, applied by the backend),
* the damped impedance controller for SO-ARM101,
* the real-analysis-guided reward built on J, its monotonicity, and smoothness.

The spec's method names (``_reset_idx``, ``_pre_step``, ``_post_step``) are kept
verbatim, and a standard ``reset``/``step`` pair is layered on top so the env is
usable by any RL runner without Isaac Lab's ``RLTaskEnv`` being importable.

Observation policy (Sec. 3.1): the agent sees images and proprioception only.
Cloth vertices and keypoints are read *only* to evaluate J for the reward. This
is enforced structurally -- :meth:`_get_observations` never touches the cloth
state -- and asserted in the test suite.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..control.impedance_controller import DampedImpedanceController
from .backend import LeHomeBackend, MockFoldGarmentBackend
from .rewards import LyapunovDescentReward, LyapunovRewardCfg


class LeHomeFoldGarmentRealDampedEnv:
    """Vectorised cloth-folding environment with Lyapunov-descent reward.

    Args:
        cfg: a config built by
            :func:`..tasks.cfg.build_lehome_real_damped_cfg`.
        backend: the cloth simulator. If ``None``, one is constructed from
            ``cfg.backend`` (mock or Isaac, per ``cfg.use_mock_backend``).
        device: torch device for the RL-side tensors.
    """

    def __init__(
        self,
        cfg: Any,
        backend: Optional[LeHomeBackend] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        if backend is None:
            backend = self._build_backend(cfg, self.device)
        self.backend = backend

        self.num_envs = backend.num_envs
        self.num_arms = backend.num_arms
        self.action_dim = 3 * self.num_arms  # Sec. 3.1: 3D delta per arm

        self.controller = DampedImpedanceController(
            damping_ratio=cfg.damping_ratio,
            stiffness=cfg.stiffness,
            max_delta=cfg.max_delta,
            dt=backend.dt,
            ee_mass=cfg.ee_mass,
            smoothing=cfg.action_smoothing,
        )
        if not self.controller.is_sampling_stable():
            import warnings

            warnings.warn(
                f"dt={backend.dt:.4g}s is coarse for omega_n="
                f"{self.controller.natural_frequency:.1f} rad/s; the discretised impedance "
                "may ring even though zeta >= 1. Reduce stiffness or dt.",
                stacklevel=2,
            )

        self.reward_fn = LyapunovDescentReward(
            cfg.reward,
            num_envs=self.num_envs,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._last_extras: Dict[str, torch.Tensor] = {}

        self.reset()

    # ------------------------------------------------------------------ construction

    @staticmethod
    def _build_backend(cfg: Any, device: torch.device) -> LeHomeBackend:
        if getattr(cfg, "use_mock_backend", True):
            return MockFoldGarmentBackend(cfg.backend, device=device)
        from .backend import IsaacLeHomeBackend

        return IsaacLeHomeBackend(cfg.backend)

    # ------------------------------------------------------------------ spaces

    @property
    def observation_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """``{"images": (C,H,W), "proprio": (P,)}`` -- what the policy consumes."""
        return {
            "images": tuple(self.backend.image_shape),
            "proprio": (self.backend.proprio_dim,),
        }

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.action_dim,)

    # ------------------------------------------------------------------ spec API

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset the selected environments (Sec. 4.2, ``_reset_idx``)."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.backend.reset_env_ids(env_ids)

        x_init = self.backend.get_end_effector_positions()
        self.controller.reset(x_init, env_ids=env_ids)

        # prev_J / prev_action must not carry across an episode boundary.
        self.reward_fn.reset(env_ids)
        self.prev_action[env_ids] = 0.0
        self.episode_reward[env_ids] = 0.0

    def _pre_step(self, actions: torch.Tensor) -> None:
        """Route actions through the damped controller onto the robot (Sec. 4.2)."""
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"actions must be ({self.num_envs}, {self.action_dim}), got {tuple(actions.shape)}"
            )
        actions = actions.clamp(-1.0, 1.0)  # policy output is normalised

        # Normalised action -> metres, then through the clipped/filtered integrator.
        delta = actions.view(self.num_envs, self.num_arms, 3) * self.cfg.max_delta
        x_current = self.backend.get_end_effector_positions()
        x_cmd = self.controller.step(x_current, delta)
        self.backend.set_end_effector_targets(x_cmd)

        self._pending_action = actions

    def _post_step(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Observations, reward, and termination after physics (Sec. 4.2).

        Returns ``(obs, rewards, terminated, truncated, extras)``.
        """
        obs = self._get_observations()

        j = self.backend.compute_cloth_error().to(self.device)
        ee_vel = self.backend.get_end_effector_velocities().to(self.device)

        rewards, components = self.reward_fn.compute(j, ee_vel, self._pending_action)
        self.prev_action = self._pending_action.clone()
        self.episode_reward += rewards

        terminated, truncated = self.backend.check_done()
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)

        extras = {"log": {k: v.detach() for k, v in components.items()}}
        extras["success"] = terminated.clone()
        self._last_extras = extras
        return obs, rewards, terminated, truncated, extras

    # ------------------------------------------------------------------ observations

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Images + proprioception. Deliberately excludes all cloth state."""
        return {
            "images": self.backend.render_cameras().to(self.device),
            "proprio": self.backend.get_proprioception().to(self.device),
        }

    # ------------------------------------------------------------------ gym-ish API

    def reset(self) -> Dict[str, torch.Tensor]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._pending_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        return self._get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """One environment step: pre-step, physics, post-step, auto-reset."""
        self._pre_step(actions)
        self.backend.simulate()
        obs, rewards, terminated, truncated, extras = self._post_step()

        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            extras["episode_reward"] = self.episode_reward[done_ids].clone()
            # The pre-reset observation is the true final state of the old
            # episode. A time-limit truncation is not a real terminal state, so
            # the value bootstrap must be taken here -- *before* the reset --
            # not from the fresh observation that replaces it below.
            extras["final_observation"] = {k: v.clone() for k, v in obs.items()}

            self._reset_idx(done_ids)
            # Observations for reset envs must reflect the new episode.
            fresh = self._get_observations()
            for key in obs:
                obs[key][done_ids] = fresh[key][done_ids]

        return obs, rewards, terminated, truncated, extras

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def cloth_error(self) -> torch.Tensor:
        """J(x_t) -- for logging and analysis only, never an observation."""
        return self.backend.compute_cloth_error().to(self.device)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:  # pragma: no cover
            close()

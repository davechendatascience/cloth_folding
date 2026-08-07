"""Adapter: LeHome's ``GarmentEnv`` -> the :class:`LeHomeBackend` protocol.

Bridges the spec's assumed interface (Cartesian EE deltas, continuous cloth
error, stacked images) onto what LeHome actually provides (12-D joint position
targets, a boolean success checker, a LeRobot observation dict).

Design decisions, each forced by a measurement rather than a preference:

* **Differential IK on the simulator's Jacobian, not LeHome's URDF solver.**
  ``lehome.utils.bimanual_ik_solver`` disagrees with the simulator: at q≈0 its
  FK places the gripper 0.452 m from the base while the sim's ``gripper`` body
  is at 0.386 m, and a commanded 5 cm move drove the arm 0.394 m to the wrong
  place. Distances-from-base are rotation-invariant, so that gap is a genuine
  kinematic mismatch, not a frame convention error.
  ``DifferentialIKController`` derives its Jacobian from the articulation
  itself, so it is consistent by construction.

* **``ik_method="dls"`` -- damped least squares.** Levenberg-Marquardt damping
  on the IK solve. This is a fourth damping layer the spec never names, and it
  is what keeps the solution bounded near kinematic singularities -- precisely
  where the URDF solver was flinging the arm through 90-degree joint jumps for
  5 cm commands.

* **``command_type="position"`` with ``use_relative_mode=True``.** The arm has
  5 controllable joints, so orientation is not freely assignable; position-only
  IK is the honest choice. Relative mode makes the controller consume exactly
  the 3D-delta-per-arm that Sec. 3.1 specifies, giving ``action_dim = 6``.

* **Continuous J from :mod:`..math.garment_functional`.** LeHome's checker
  returns a boolean, which cannot serve as a Lyapunov functional. The
  functional's zero set is identical to that boolean.

Must be constructed inside an :func:`..tasks.isaac_app.isaac_app` block: the
Isaac Lab and LeHome imports below require the Kit runtime to be live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch

from ..math.garment_functional import GarmentFoldFunctional, GarmentFunctionalCfg

# Measured on the live simulator (scripts/measure_joint_damping.py) at the zero
# pose, K=17.8, D=0.60. D_crit = 2K/omega_n, i.e. the damping that puts each
# joint at zeta=1.
#
#   joint           zeta    omega_n    J        D_crit   settle(steps)
#   shoulder_pan    0.567   33.23      0.01612  1.071    15
#   shoulder_lift   0.529   27.76      0.02310  1.283    27
#   elbow_flex      0.649   32.80      0.01654  1.085    17
#   wrist_flex      0.933   56.24      0.00563  0.633    11
#   wrist_roll      >=1     -          -        -        12
#
# CAVEAT: J is configuration-dependent and these were measured at q=0 with the
# other joints held. The single-DOF second-order fit also ignores inter-joint
# coupling. Treat these as a good starting point, not exact values -- re-measure
# at a representative folding pose before relying on them quantitatively.
MEASURED_CRITICAL_DAMPING: dict = {
    "shoulder_pan": 1.071,
    "shoulder_lift": 1.283,
    "elbow_flex": 1.085,
    "wrist_flex": 0.633,
    # wrist_roll showed no overshoot at D=0.60 (already >= critical); left alone.
}


def actuator_overrides(
    joint_names_expr: Sequence[str],
    stiffness,
    damping,
    damping_overrides: dict,
    stiffness_scale: float = 1.0,
):
    """Compute ``(stiffness, damping)`` for one actuator group.

    Isaac Lab resolves a dict-valued gain with ``strict=True``: **every key must
    match at least one joint in that actuator group**, or it raises
    ``"Not all regular expressions are matched"``. The SO101 splits its joints
    across two groups (``sts3215-arm`` and ``sts3215-gripper``), so handing the
    full arm-joint dict to the gripper group is an error, not a no-op. Keys are
    therefore filtered per group.

    Partial coverage *within* a group is fine -- unmatched joints keep their
    existing value, which is what we want for ``wrist_roll`` (no measured
    overshoot, so already >= critical, so left alone).

    ``stiffness_scale`` scales K; damping scales as sqrt(K) to hold
    ``zeta = D / (2 sqrt(K J))`` fixed.

    Args:
        joint_names_expr: the group's joint name patterns.
        stiffness / damping: current values (float or dict).
        damping_overrides: ``{joint_name: D}`` desired damping.
        stiffness_scale: multiplier on K.
    Returns:
        ``(new_stiffness, new_damping)``.
    """
    import re

    def scale(v, f):
        if f == 1.0:
            return v
        return {k: x * f for k, x in v.items()} if isinstance(v, dict) else v * f

    mine = {
        j: d
        for j, d in (damping_overrides or {}).items()
        if any(re.fullmatch(e, j) for e in joint_names_expr)
    }

    new_stiffness = scale(stiffness, stiffness_scale)
    root = stiffness_scale**0.5
    if mine:
        new_damping = {j: d * root for j, d in mine.items()}
    else:
        new_damping = scale(damping, root)
    return new_stiffness, new_damping


@dataclass
class IsaacGarmentCfg:
    """Configuration for the real LeHome garment backend."""

    garment_name: str = "Top_Long_Seen_0"
    garment_version: str = "Release"
    device: str = "cpu"

    # --- control ---
    decimation: int = 20
    """Physics steps per policy action.

    LeHome ships ``decimation=1`` (90 Hz), but measured 2% settling is 11-27
    control steps, so the policy issues up to 27 commands before the plant has
    responded to the first. The action's effect is then unobservable within its
    own step, which breaks credit assignment regardless of reward design.

    20 gives ~4.5 Hz, roughly one action per settling time once
    :attr:`joint_damping` brings the slow joints to critical (worst settling
    drops from 27 to ~19 steps). Raise ``stiffness_scale`` if that is too
    sluggish -- settling scales as 1/sqrt(K).
    """
    joint_damping: Optional[dict] = None
    """Per-joint actuator damping, ``{joint_name: D}``. ``None`` uses
    :data:`MEASURED_CRITICAL_DAMPING`; pass ``{}`` to keep LeHome's values.

    Must be per-joint, not a scalar. Gains are uniform (K=17.8, D=0.60) but
    inertia varies 4x across the arm (0.0056-0.0231 kg m^2), and zeta ~
    1/sqrt(J), so one global value cannot suit all joints: D=1.283 would put
    shoulder_lift at exactly critical but drive wrist_flex to zeta=2.03,
    over-damping the one joint that was already fine.

    Why this matters beyond comfort: if the *plant* overshoots, J(x_t) is
    non-monotone no matter what the policy does, so the spec's monotone
    convergence argument is undermined below the level the reward can reach.
    """
    stiffness_scale: float = 1.0
    """Multiply K (and D by sqrt of it, preserving zeta).

    Settling time scales as 1/omega_n = sqrt(J/K), so 4x stiffness halves it
    and allows roughly half the decimation. Costs contact realism: a stiffer
    arm pushes harder against the cloth."""
    ik_lambda: float = 0.05
    """DLS damping. Larger = smaller, safer steps near singularities."""
    max_delta: float = 0.02
    """Cartesian delta bound per policy step, metres."""

    # --- observations ---
    image_size: int = 84
    """Cameras are 480x640; downsampled to this square for the policy."""
    use_depth: bool = False

    functional: GarmentFunctionalCfg = field(default_factory=GarmentFunctionalCfg)


class IsaacGarmentBackend:
    """LeHomeBackend implementation over LeHome's ``GarmentEnv``."""

    def __init__(self, cfg: IsaacGarmentCfg) -> None:
        import gymnasium as gym
        import lehome.tasks  # noqa: F401  (registers the gym ids)
        from isaaclab.controllers import (
            DifferentialIKController,
            DifferentialIKControllerCfg,
        )
        from lehome.tasks.bedroom.challenge_garment_loader import ChallengeGarmentLoader
        from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        env_cfg = GarmentEnvCfg()
        env_cfg.garment_name = cfg.garment_name
        env_cfg.garment_version = cfg.garment_version
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = cfg.device
        env_cfg.decimation = cfg.decimation
        env_cfg.sim.render_interval = cfg.decimation

        damping = MEASURED_CRITICAL_DAMPING if cfg.joint_damping is None else cfg.joint_damping
        for robot in (env_cfg.left_robot, env_cfg.right_robot):
            for act in robot.actuators.values():
                new_k, new_d = actuator_overrides(
                    list(act.joint_names_expr), act.stiffness, act.damping,
                    damping, cfg.stiffness_scale,
                )
                act.stiffness, act.damping = new_k, new_d

        # --- garment metadata, for J -----------------------------------------
        loader = ChallengeGarmentLoader(env_cfg.garment_cfg_base_path)
        gconf = loader.load_garment_config(cfg.garment_name, cfg.garment_version)
        self.garment_type = loader.get_garment_type(cfg.garment_name)
        self.check_points: Sequence[int] = list(gconf["check_point"])
        scale = float(gconf["scale"][0])
        thresholds = [float(d) * scale for d in gconf["success_distance"]]
        self.functional = GarmentFoldFunctional(self.garment_type, thresholds, cfg.functional)

        # --- env -------------------------------------------------------------
        self.env = gym.make("LeHome-BiSO101-Direct-Garment-v2", cfg=env_cfg).unwrapped
        # Required before stepping; DirectRLEnv never calls it, and
        # GarmentObject.reset() raises AttributeError without it.
        self.env.initialize_obs()

        self.num_envs = 1
        self.num_arms = 2
        self.dt = float(env_cfg.sim.dt) * cfg.decimation

        self.arms = (self.env.left_arm, self.env.right_arm)
        body_names = list(self.env.left_arm.data.body_names)
        self.ee_body_idx = body_names.index("gripper")
        # Fixed-base articulation: the Jacobian omits the root body.
        self.jacobi_body_idx = self.ee_body_idx - 1
        joint_names = list(self.env.left_arm.data.joint_names)
        self.arm_joint_ids = [joint_names.index(n) for n in joint_names if n != "gripper"]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": cfg.ik_lambda},
        )
        self.ik = [
            DifferentialIKController(ik_cfg, num_envs=1, device=cfg.device) for _ in range(2)
        ]

        self._joint_targets = torch.zeros(1, 12, device=self.device)
        self._last_obs = None
        self.episode_step = torch.zeros(1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ helpers

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        c = 3 * 3 + (1 if self.cfg.use_depth else 0)
        return (c, self.cfg.image_size, self.cfg.image_size)

    @property
    def proprio_dim(self) -> int:
        return 12 + 12 + 6  # joint pos, joint vel, both EE positions

    def _ee_pos_w(self, arm_i: int) -> torch.Tensor:
        return self.arms[arm_i].data.body_pos_w[:, self.ee_body_idx, :]

    def _ee_quat_w(self, arm_i: int) -> torch.Tensor:
        return self.arms[arm_i].data.body_quat_w[:, self.ee_body_idx, :]

    def _ee_pose_b(self, arm_i: int):
        """EE pose in the arm's root frame -- what DifferentialIKController wants."""
        from isaaclab.utils.math import subtract_frame_transforms

        art = self.arms[arm_i]
        root_pos, root_quat = art.data.root_pos_w, art.data.root_quat_w
        return subtract_frame_transforms(
            root_pos, root_quat, self._ee_pos_w(arm_i), self._ee_quat_w(arm_i)
        )

    # ------------------------------------------------------------------ protocol

    def reset_env_ids(self, env_ids: torch.Tensor) -> None:
        self.env.reset()
        for ik in self.ik:
            ik.reset()
        self._joint_targets = torch.cat(
            [self.arms[0].data.joint_pos, self.arms[1].data.joint_pos], dim=-1
        ).clone()
        self.episode_step.zero_()

    def get_end_effector_positions(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.stack([self._ee_pos_w(0)[0], self._ee_pos_w(1)[0]], dim=0).unsqueeze(0)

    def get_end_effector_velocities(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = [a.data.body_lin_vel_w[:, self.ee_body_idx, :][0] for a in self.arms]
        return torch.stack(v, dim=0).unsqueeze(0)

    def set_end_effector_targets(self, x_cmd: torch.Tensor) -> None:
        """Convert commanded EE positions to joint targets via damped-LS IK.

        ``x_cmd`` is absolute (the controller integrates deltas), so the delta
        handed to the relative-mode IK is ``x_cmd - x_current``.
        """
        x_cmd = x_cmd.to(self.device)
        for i, art in enumerate(self.arms):
            delta = (x_cmd[:, i, :] - self._ee_pos_w(i)).clamp(
                -self.cfg.max_delta, self.cfg.max_delta
            )
            ee_pos_b, ee_quat_b = self._ee_pose_b(i)
            self.ik[i].set_command(delta, ee_pos_b, ee_quat_b)

            jac = art.root_physx_view.get_jacobians()[
                :, self.jacobi_body_idx, :3, self.arm_joint_ids
            ]
            q = art.data.joint_pos[:, self.arm_joint_ids]
            q_des = self.ik[i].compute(ee_pos_b, ee_quat_b, jac, q)

            base = 0 if i == 0 else 6
            for k, jid in enumerate(self.arm_joint_ids):
                self._joint_targets[:, base + jid] = q_des[:, k]

    def simulate(self) -> None:
        obs, _, _, _, _ = self.env.step(self._joint_targets)
        self._last_obs = obs
        self.episode_step += 1

    # --------------------------------------------------------------- observations

    def render_cameras(self) -> torch.Tensor:
        """``(1, 9, S, S)`` -- three RGB cameras, downsampled and normalised."""
        import torch.nn.functional as F

        mats = []
        for name in ("top_camera", "left_camera", "right_camera"):
            rgb = getattr(self.env, name).data.output["rgb"]  # (1, H, W, 3), uint8
            x = rgb.permute(0, 3, 1, 2).float()
            if x.max() > 1.5:
                x = x / 255.0
            mats.append(
                F.interpolate(
                    x, size=(self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear", align_corners=False,
                )
            )
        return torch.cat(mats, dim=1).to(self.device)

    def get_proprioception(self) -> torch.Tensor:
        """Robot state only -- no cloth information (Sec. 3.1)."""
        q = torch.cat([self.arms[0].data.joint_pos, self.arms[1].data.joint_pos], dim=-1)
        dq = torch.cat([self.arms[0].data.joint_vel, self.arms[1].data.joint_vel], dim=-1)
        ee = self.get_end_effector_positions().flatten(1)
        return torch.cat([q, dq, ee], dim=-1).to(self.device)

    # --------------------------------------------------------------------- reward

    def check_point_positions_cm(self) -> torch.Tensor:
        """Check-point particle positions in centimetres, as LeHome measures them.

        Deliberately does **not** call ``GarmentObject.get_current_mesh_points()``.
        That method unconditionally does::

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed_mesh_points)

        building a 14746-point Open3D cloud that is then discarded whenever
        ``visualize=False, save=False`` -- and it rigid-transforms all 14746
        points when we need 6. Measured at **154 ms per call, 21.6% of every
        environment step**.

        Here we read the raw points once, index the 6 check-points, and
        transform only those. Same numbers, none of the waste.
        """
        import numpy as np

        obj = self.env.object
        idx = self.check_points

        try:
            if getattr(obj, "_device", "cpu") == "cpu":
                pts = obj._get_points_pose().detach().cpu().numpy()[idx]
                pos, ori = obj.get_world_pose()
                cp = obj.transform_points(
                    pts,
                    pos.detach().cpu().numpy(),
                    ori.detach().cpu().numpy(),
                    obj.get_world_scale().detach().cpu().numpy(),
                )
            else:
                cp = (
                    obj._cloth_prim_view.get_world_positions()
                    .squeeze(0)[idx]
                    .detach()
                    .cpu()
                    .numpy()
                )
        except Exception:
            # Fall back to LeHome's own accessor if the internals shift.
            pts, *_ = obj.get_current_mesh_points()
            cp = np.asarray(pts)[idx]

        cp = np.asarray(cp) * 100.0
        return torch.as_tensor(cp, dtype=torch.float32, device=self.device).unsqueeze(0)

    def compute_cloth_error(self) -> torch.Tensor:
        return self.functional(self.check_point_positions_cm())

    def check_done(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(terminated, truncated)``.

        LeHome's ``_get_dones`` returns ``(time_out, time_out)`` -- success does
        not terminate. Terminating on success is what the RL formulation needs,
        so it is derived from J instead, and the two are kept distinct so the
        GAE bootstrap is handled correctly.
        """
        terminated = self.compute_cloth_error() <= 0.0
        max_steps = int(self.env.max_episode_length)
        truncated = self.episode_step >= max_steps
        return terminated.to(self.device), truncated.to(self.device)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # pragma: no cover
            pass

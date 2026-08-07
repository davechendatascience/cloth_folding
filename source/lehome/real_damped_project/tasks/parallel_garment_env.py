"""A LeHome garment env that actually supports ``num_envs > 1``.

LeHome's ``GarmentEnv`` is single-environment for three separate reasons, all
fixed here:

1. **It never clones.** ``_setup_scene()`` never calls
   ``self.scene.clone_environments()``. Isaac Lab's ``DirectRLEnv`` calls
   ``_setup_scene()`` but leaves replication to the env, so ``num_envs=8``
   silently produces 8 empty env origins. This is the actual root cause -- not
   a PhysX limitation, as the prim paths made it look.

2. **Absolute prim paths.** Robots and cameras live at ``/World/Robot/...``,
   outside the ``/World/envs/env_.*/`` namespace the cloner replicates. Cloning
   them requires only changing the configured paths.

3. **Single-prim particle wrappers.** ``SingleClothPrim`` wraps exactly one
   cloth and ``get_object_particle_position()`` reads one. But
   ``isaacsim.core.prims.ClothPrim`` is a *view* class taking a
   ``prim_paths_expr`` regex and returning ``(num_envs, P, 3)`` from
   ``get_world_positions()`` -- so batched reads need no new code, just the
   right class.

The lifecycle constraint discovered the hard way (see LEVERS.md): cloths must
exist **before** ``sim.reset()``, because the physics tensor view is built once
there and a later-created cloth is never registered with it. Hence the garment
is created inside ``_setup_scene`` and cloned there, never afterwards.

Why this matters: ``num_envs`` is the only lever that fixes GPU utilisation at
its root (one env of 14.7k particles leaves the GB10 at 62% compute / 0% memory
bandwidth), and it is simultaneously what turns an 83-day on-policy RL budget
into roughly a day.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


DEFAULT_NUM_ENVS = 4
"""Chosen from the measured scaling curve, not from a guess.

Step time is affine in N and does not parallelise -- cloth simulation is serial
across envs, so throughput asymptotes rather than scaling. Measured at
decimation 3 on GPU:

    with cameras:     t(N) = 223 ms + 119.0 ms x N   -> ceiling  8.4 env-steps/s
    without cameras:  t(N) =  47 ms +  54.3 ms x N   -> ceiling 18.4 env-steps/s

| N  | cameras: step / env-steps-s | no cameras: step / env-steps-s |
|----|-----------------------------|--------------------------------|
| 1  |   348.7 ms  /  2.87         |    101.4 ms  /   9.86          |
| 4  |   700.2 ms  /  5.71         |    264.4 ms  /  15.13          |
| 16 |  2132.3 ms  /  7.50         |    (~916 ms  /  ~17.5, pred)   |

Startup is 17.4s at N<=4 and 44.8s at N=16; peak RSS 7.8 / 8.3 / 10.7 GB.

N=4 captures 82% of the no-camera ceiling. N=16 would add only ~1.16x for 4x
the envs, 2.6x the startup, and 4x the replicated geometry once the static
scene has to be per-env. Bad trade at every margin.

Note the per-env term more than halves when rendering is off (119 -> 54.3 ms):
``TiledCamera`` renders per env, so camera cost scales with N as well. Rendering
was both the fixed *and* a large part of the marginal cost.

The earlier "10-60x" estimate for this lever was wrong by an order of
magnitude. It assumed per-env work would overlap on the GPU; it does not. The
largest single win here was not parallelism at all -- it was not rendering
images that nothing reads (3.44x at N=1).
"""


def build_parallel_cfg(base_cfg, num_envs: int = DEFAULT_NUM_ENVS, env_spacing: float = 3.0):
    """Rewrite a ``GarmentEnvCfg`` so its assets live under the env namespace.

    Isaac Lab's cloner replicates whatever sits under ``/World/envs/env_.*/``.
    LeHome places everything at absolute paths, so nothing is replicated; this
    moves the robots (and, with them, the wrist/top cameras that are their
    children) into the namespace.

    The bedroom scene and the dome light stay global on purpose: they are
    static, shared, and replicating them would multiply geometry for nothing.
    """
    base_cfg.scene.num_envs = num_envs
    base_cfg.scene.env_spacing = env_spacing
    # replicate_physics=False is required for particle cloth, and this is the
    # single least obvious line in the file.
    #
    # With True, PhysX replicates env_0's physics structure across envs instead
    # of parsing each env's prims. That path covers articulations and rigid
    # bodies -- not particle cloths. The failure is silent and cost four builds
    # to localise: cloning genuinely succeeds (all N Garment/mesh prims exist on
    # the stage, none instanceable), but the particle-cloth physics view reports
    # count=1, because PhysX only ever parsed one cloth. USD prims present,
    # physics objects absent.
    #
    # False makes PhysX parse every env separately: slower to start up and more
    # memory, which is the price of N real cloths.
    base_cfg.scene.replicate_physics = False

    def reroot(path: str) -> str:
        """``/World/Robot/Left_Robot`` -> ``/World/envs/env_.*/Left_Robot``.

        The intermediate ``Robot/`` level is dropped deliberately. Isaac Lab
        spawns an asset by creating its leaf under each cloned env origin, so a
        path like ``/World/envs/env_.*/Robot/Left_Robot`` makes it look for a
        parent ``Robot`` prim that cloning never creates:

            RuntimeError: Unable to find source prim path:
            '/World/envs/env_.*/Robot'. Please create the prim before spawning.
        """
        return path.replace("/World/Robot/", "/World/envs/env_.*/", 1)

    base_cfg.left_robot.prim_path = reroot(base_cfg.left_robot.prim_path)
    base_cfg.right_robot.prim_path = reroot(base_cfg.right_robot.prim_path)
    for cam in ("top_camera", "left_wrist", "right_wrist"):
        c = getattr(base_cfg, cam, None)
        if c is not None and hasattr(c, "prim_path"):
            c.prim_path = reroot(c.prim_path)
    return base_cfg


GARMENT_SUBPATH = "Garment"
"""Per-env prim name passed to ``GarmentObject``."""

GARMENT_MESH_SUBPATH = "Garment/mesh"
"""What the batched view must match.

``GarmentObject`` builds a hierarchy: given ``prim_path=".../Garment"`` the
actual cloth prim lands at ``.../Garment/mesh`` (visible in LeHome's own
single-env case as ``/World/Object/Top_Long_Seen_0/mesh``). Pointing
``ClothPrim`` at the parent silently finds no cloths and
``create_particle_cloth_view`` returns None, surfacing only as
``'NoneType' object has no attribute 'count'`` inside ``initialize()``."""


def make_parallel_env_class():
    """Build the ParallelGarmentEnv subclass.

    Deferred behind a function because the LeHome and Isaac Lab imports require
    a live Kit runtime, so this module must stay importable without one (the
    test suite imports it to check the config rewriting).
    """
    from isaacsim.core.prims import ClothPrim
    from lehome.tasks.bedroom.garment_bi_v2 import GarmentEnv

    class ParallelGarmentEnv(GarmentEnv):
        """``GarmentEnv`` with per-env garments and a batched cloth view."""

        def _create_garment_object(self):
            """Create the garment inside env_0 so cloning replicates it.

            LeHome hardcodes ``/World/Object/{garment_name}``, which sits
            outside the namespace Isaac Lab's cloner touches. The result is not
            an error -- it is N environments quietly sharing one cloth, which
            trains happily on identical data. Measured before this override:
            ``particle_positions()`` returned ``(1, 14746, 3)`` for
            ``num_envs=4``.

            This is a narrow reimplementation of the base method: the original
            spends most of its length deleting a pre-existing prim, which
            cannot exist in a freshly built scene. The parts that matter --
            validation and the texture/light randomisation configs -- are kept.
            """
            from lehome.assets.object.Garment import GarmentObject

            prim_path = f"/World/envs/env_0/{GARMENT_SUBPATH}"
            try:
                self.object = GarmentObject(
                    prim_path=prim_path,
                    particle_config=self.particle_config,
                    garment_config=self.garment_config,
                    rng=self.garment_rng,
                )
            except Exception as e:  # pragma: no cover - needs Isaac
                raise RuntimeError(
                    f"Failed to create GarmentObject at {prim_path}: {e}"
                ) from e

            self._validate_created_object()
            self.texture_cfg = self.particle_config.objects.get("texture_randomization", {})
            self.light_cfg = self.particle_config.objects.get("light_randomization", {})

        def _setup_scene(self):
            # Robots, cameras, static scene, and the env_0 garment -- the base
            # implementation, which now writes into /World/envs/env_.*/ because
            # build_parallel_cfg rewrote the paths.
            super()._setup_scene()

            n = self.scene.num_envs
            if n > 1:
                # Replicate env_0 (robots + garment) into env_1..N-1. Must
                # happen here: after sim.reset() the physics view is fixed and
                # new cloths can never register with it.
                # copy_from_source stays at the default. It was briefly set to
                # True on the theory that inherited clones were the reason only
                # one cloth simulated; a stage dump disproved that (all N
                # Garment/mesh prims exist, IsInstanceable()=False, and the
                # count was still 1). The real cause was replicate_physics --
                # see build_parallel_cfg.
                self.scene.clone_environments()
                try:
                    self.scene.filter_collisions(global_prim_paths=["/World/Scene"])
                except Exception:
                    # Older Isaac Lab signatures differ; collision filtering is
                    # an optimisation, not a correctness requirement.
                    pass

        def initialize_obs(self):
            """LeHome's hook, extended to build the batched cloth view.

            Called after ``gym.make(...).unwrapped`` and before stepping. The
            single-cloth ``self.object`` stays valid for env_0 so the base
            class's own logic keeps working; the view is what we read from.
            """
            super().initialize_obs()

            expr = f"/World/envs/env_.*/{GARMENT_MESH_SUBPATH}"
            try:
                self.cloth_view = ClothPrim(prim_paths_expr=expr, name="garment_view")
                self.cloth_view.initialize(self._physics_sim_view_or_none())
            except Exception as exc:  # pragma: no cover - needs Isaac
                self.cloth_view = None
                print(f"[ParallelGarmentEnv] batched cloth view unavailable: {exc}")

            self._capture_initial_particles()

        skip_images: bool = False
        """Return proprio-only observations and never touch the cameras.

        Set this for J labelling and any demo replay: those read particle
        positions and replay recorded actions, so every image the env produces
        is discarded. LeHome's ``_get_observations`` nonetheless renders three
        640x480 cameras, copies four buffers to host, and converts a depth map
        to uint16 on every single step.

        This must be set *together* with ``AppLauncher(enable_cameras=False)``.
        Disabling the renderer alone crashes, because ``_get_observations``
        reads ``self.top_camera.data.output["rgb"]`` unconditionally and the
        sensor buffer is invalid without a render product:

            sensor_base.py: self._is_outdated[outdated_env_ids] = False
            RuntimeError: CUDA error: an illegal memory access was encountered

        which then poisons the CUDA context and takes GpuParticleClothView down
        with it -- so the symptom points at the cloth, not the cameras.
        """

        def _get_observations(self):
            if not self.skip_images:
                return super()._get_observations()

            # Same proprio the full path builds, without the camera reads.
            # Kept batched (no squeeze(0)): the base implementation assumes a
            # single env, which is exactly what this class exists to lift.
            left = torch.cat(
                [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
            )
            right = torch.cat(
                [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
            )
            joint_pos = torch.cat([left, right], dim=1)
            return {
                "action": self.actions.detach().cpu().numpy(),
                "observation.state": joint_pos.detach().cpu().numpy(),
            }

        def _capture_initial_particles(self):
            """Store the as-built particle configuration for every env.

            Captured once, after the view exists and before any stepping. Each
            env's clone already sits at its own origin, so the stored array is
            directly re-writable at reset with no origin arithmetic -- which is
            the whole reason for snapshotting all N rather than broadcasting
            env_0's.
            """
            if getattr(self, "cloth_view", None) is None:
                self._initial_particles = None
                return
            pts = self.cloth_view.get_world_positions()
            self._initial_particles = torch.as_tensor(pts).clone()

        def _garment_xform(self, i: int):
            """Cached SingleXFormPrim for env i's garment."""
            from isaacsim.core.prims import SingleXFormPrim

            if not hasattr(self, "_garment_xforms"):
                self._garment_xforms = {}
            if i not in self._garment_xforms:
                self._garment_xforms[i] = SingleXFormPrim(
                    f"/World/envs/env_{i}/{GARMENT_SUBPATH}"
                )
            return self._garment_xforms[i]

        def set_garment_poses(self, poses, env_ids=None):
            """Pose each env's garment independently.

            ``poses``: ``(len(env_ids), 6)`` as ``[x, y, z, roll, pitch, yaw]``
            with angles in degrees -- the same convention as LeHome's
            ``set_all_pose``, and the same layout as ``garment_info.json``.
            Positions are env-local; the env origin is added here.

            This exists because LeHome's reset poses ``self.object``, which
            wraps env_0 alone, so the clones keep their as-built configuration.
            Measured at N=4 before this method: env_0 centroid z = 0.5292 while
            envs 1-3 sat at ~0.20. Silent, and fatal -- demo replay from an
            unmatched garment pose reduces J by 0.7% instead of 90%.

            The three-step sequence (identity -> initial points -> target pose)
            mirrors ``GarmentObject.set_all_pose`` deliberately: that ordering
            is what makes the Xform transform carry the particles with it.
            """
            from isaacsim.core.utils.rotations import euler_angles_to_quat

            if getattr(self, "cloth_view", None) is None:
                raise RuntimeError("no batched cloth view; cannot pose per-env")
            if self._initial_particles is None:
                raise RuntimeError("initial particles were never captured")

            poses = np.asarray(poses, dtype=np.float32).reshape(-1, 6)
            ids = list(range(self.scene.num_envs)) if env_ids is None else list(env_ids)
            if len(poses) != len(ids):
                raise ValueError(f"{len(poses)} poses for {len(ids)} envs")

            origins = self.scene.env_origins.detach().cpu().numpy()

            # 1. zero the Xforms, so writing particle positions lands in a known
            #    frame rather than composing with whatever pose is current.
            for i in ids:
                self._garment_xform(i).set_world_pose(
                    np.zeros(3, dtype=np.float32),
                    euler_angles_to_quat(np.zeros(3), degrees=True),
                )

            # 2. restore the as-built particle configuration for those envs.
            idx = torch.as_tensor(ids, dtype=torch.long)
            self.cloth_view.set_world_positions(
                self._initial_particles[idx].to(self.device), indices=idx
            )

            # 3. move each Xform to its target; the particles follow.
            for k, i in enumerate(ids):
                pos = poses[k, :3] + origins[i]
                self._garment_xform(i).set_world_pose(
                    pos.astype(np.float32),
                    euler_angles_to_quat(poses[k, 3:], degrees=True),
                )

        def _physics_sim_view_or_none(self):
            try:
                from isaacsim.core.simulation_manager import SimulationManager

                return SimulationManager.get_physics_sim_view()
            except Exception:  # pragma: no cover
                return None

        def particle_positions(self) -> Optional[torch.Tensor]:
            """``(num_envs, P, 3)`` particle positions, or ``None``.

            Falls back to the single-cloth path when the view is unavailable,
            so a one-env run behaves exactly as before.
            """
            if getattr(self, "cloth_view", None) is not None:
                return self.cloth_view.get_world_positions()
            if self.object is None:
                return None
            pts = self.object._get_points_pose().detach()
            return pts.unsqueeze(0)

    return ParallelGarmentEnv

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


def build_parallel_cfg(base_cfg, num_envs: int, env_spacing: float = 3.0):
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
    base_cfg.scene.replicate_physics = True

    def reroot(path: str) -> str:
        return path.replace("/World/Robot/", "/World/envs/env_.*/Robot/", 1)

    base_cfg.left_robot.prim_path = reroot(base_cfg.left_robot.prim_path)
    base_cfg.right_robot.prim_path = reroot(base_cfg.right_robot.prim_path)
    for cam in ("top_camera", "left_wrist", "right_wrist"):
        c = getattr(base_cfg, cam, None)
        if c is not None and hasattr(c, "prim_path"):
            c.prim_path = reroot(c.prim_path)
    return base_cfg


GARMENT_SUBPATH = "Garment"
"""Per-env garment prim name; the batched view matches
``/World/envs/env_.*/<GARMENT_SUBPATH>``."""


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
                self.scene.clone_environments(copy_from_source=False)
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

            expr = f"/World/envs/env_.*/{GARMENT_SUBPATH}"
            try:
                self.cloth_view = ClothPrim(prim_paths_expr=expr, name="garment_view")
                self.cloth_view.initialize(self._physics_sim_view_or_none())
            except Exception as exc:  # pragma: no cover - needs Isaac
                self.cloth_view = None
                print(f"[ParallelGarmentEnv] batched cloth view unavailable: {exc}")

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

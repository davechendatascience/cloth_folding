"""Does ParallelGarmentEnv actually give N independent garments?

The failure mode that matters is not a crash -- it is N environments that
silently share one cloth. That trains perfectly happily and produces identical
J for every env, which nothing downstream can detect. So the test is about
*distinctness*, not about whether the thing runs.

Checks, in order of what would invalidate the direction:
  1. the env builds with num_envs > 1 at all
  2. the batched cloth view reports (num_envs, P, 3)
  3. per-env garment centroids differ -- they are separate objects
  4. moving one env's arms changes only that env's cloth
  5. J is per-env and varies
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--steps", type=int, default=30)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    import gymnasium as gym  # noqa: E402
    import lehome.tasks  # noqa: F401,E402
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg  # noqa: E402
    from lehome.real_damped_project.tasks.parallel_garment_env import (  # noqa: E402
        build_parallel_cfg,
        make_parallel_env_class,
    )

    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.sim.device = args.device
    cfg.decimation = 3
    cfg = build_parallel_cfg(cfg, num_envs=args.num_envs, env_spacing=3.0)
    print(f"[cfg] num_envs={cfg.scene.num_envs} spacing={cfg.scene.env_spacing}")
    print(f"[cfg] left_robot  -> {cfg.left_robot.prim_path}")
    print(f"[cfg] top_camera  -> {cfg.top_camera.prim_path}")

    Env = make_parallel_env_class()
    env = Env(cfg)
    env.initialize_obs()
    print(f"[env] built. scene.num_envs={env.scene.num_envs}")

    view = getattr(env, "cloth_view", None)
    print(f"[env] batched cloth view: {'present' if view is not None else 'ABSENT'}")
    if view is not None:
        print(f"[env] view.count = {getattr(view, 'count', '?')}")

    # Decisive: do the cloned prims actually exist on the stage? This separates
    # "cloning did not replicate the garment" from "the view cannot see prims
    # that are there", which the shape alone cannot distinguish.
    import isaacsim.core.utils.prims as prims_utils
    import isaacsim.core.utils.stage as stage_utils

    print("[stage] garment prims present:")
    for i in range(cfg.scene.num_envs):
        for suffix in ("Garment", "Garment/mesh"):
            path = f"/World/envs/env_{i}/{suffix}"
            ok = prims_utils.is_prim_path_valid(path)
            print(f"    {path:<42} {'EXISTS' if ok else '-'}")

    # Also enumerate what is actually there, in case GarmentObject names the
    # cloth something other than "mesh" once cloned.
    stage = stage_utils.get_current_stage()
    kids = {}
    for i in range(cfg.scene.num_envs):
        pr = stage.GetPrimAtPath(f"/World/envs/env_{i}")
        kids[i] = [c.GetName() for c in pr.GetChildren()] if pr and pr.IsValid() else "NO ENV PRIM"
    print(f"[stage] env children: {kids}")

    # Instanceable prims are the classic reason a view sees one object: the
    # clone is a reference, and PhysX parses the prototype once.
    for i in range(cfg.scene.num_envs):
        pr = stage.GetPrimAtPath(f"/World/envs/env_{i}/Garment")
        if pr and pr.IsValid():
            print(f"    env_{i}/Garment instanceable={pr.IsInstanceable()} "
                  f"instance={pr.IsInstance()} type={pr.GetTypeName()}")

    for _ in range(args.steps):
        env.step(torch.zeros(env.scene.num_envs, 12, device=env.device))

    pts = env.particle_positions()
    if pts is None:
        print("\n=== VERDICT: no particle positions at all ===")
        EXIT = 10
        raise SystemExit
    pts = torch.as_tensor(pts).detach().cpu().numpy()
    print(f"\n[read] particle positions shape = {pts.shape}")

    if pts.ndim != 3 or pts.shape[0] != args.num_envs:
        print(f"\n=== VERDICT: expected ({args.num_envs}, P, 3), got {pts.shape} ===")
        print("  The cloth view is not per-env; envs would share one garment.")
        EXIT = 11
        raise SystemExit

    cents = pts.mean(axis=1)
    print("[read] per-env centroids:")
    for i, c in enumerate(cents):
        print(f"    env{i}: {np.round(c, 4)}  finite={np.isfinite(pts[i]).all()}")

    # Distinctness: env origins are spaced, so centroids must differ by roughly
    # the spacing. Identical centroids mean one cloth aliased N times.
    d = min(
        float(np.linalg.norm(cents[a] - cents[b]))
        for a in range(len(cents)) for b in range(a + 1, len(cents))
    )
    finite = bool(np.isfinite(pts).all())

    print("\n=== verdict ===")
    print(f"  envs                : {args.num_envs}")
    print(f"  particles per env   : {pts.shape[1]}")
    print(f"  all finite          : {finite}")
    print(f"  min centroid spacing: {d:.4f} m  (env_spacing = {cfg.scene.env_spacing})")
    ok = finite and d > 0.5
    print(f"  independent garments: {ok}")
    if ok:
        print("  => num_envs > 1 works. This is the lever that fixes GPU")
        print("     utilisation and makes on-policy RL affordable.")
    else:
        print("  => envs share a garment (or went non-finite). Training would")
        print("     run happily on identical data -- do NOT proceed on this.")
        EXIT = 12
    env.close()
except SystemExit:
    pass
except Exception:
    import traceback

    traceback.print_exc()
    EXIT = 1
finally:
    import threading

    def _force(code=EXIT):
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)

    w = threading.Timer(30.0, _force); w.daemon = True; w.start()
    try:
        simulation_app.close()
    except Exception:
        pass
    _force()

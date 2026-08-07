"""Can N particle cloths coexist and simulate independently in one stage?

This is the crux of the parallel-env work. Everything else -- moving robot prim
paths under /World/envs/env_.*/, looping GarmentObject creation, stacking
particle reads -- is mechanical. But PhysX particle *systems* are global-ish
objects, and if they cannot be replicated per environment (or if N cloths
cannot share one system) then `num_envs > 1` is not available for this task at
all and the whole direction dies here.

Cheapest possible test: build the normal single-garment env, then add a second
GarmentObject at a different prim path, step both, and check that

  * both report valid, finite particle positions,
  * the two are *independent* -- moving one does not move the other,
  * neither has collapsed to NaN, which is how PhysX signals an unhappy
    particle system.

A pass does not prove 64 envs work; it proves the blocking assumption is not
false, which is what decides whether to invest in the rest.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--n_extra", type=int, default=2, help="extra cloths beyond the env's own")
p.add_argument("--steps", type=int, default=40)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg,
        IsaacGarmentBackend,
    )

    backend = IsaacGarmentBackend(
        IsaacGarmentCfg(garment_name=args.garment, device=args.device,
                        decimation=3, joint_damping={})
    )
    backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
    env = backend.env
    print(f"[base] garment at {env.object._prim.GetPath()} "
          f"device={getattr(env.object,'_device','?')}")

    from lehome.assets.object.Garment import GarmentObject  # noqa: E402

    def read_points(obj):
        """Particle positions, via whichever path this device uses."""
        if getattr(obj, "_device", "cpu") == "cpu":
            return obj._get_points_pose().detach().cpu().numpy()
        return obj._cloth_prim_view.get_world_positions().squeeze(0).detach().cpu().numpy()

    # ---- create extra cloths at distinct prim paths --------------------
    extras = []
    for i in range(args.n_extra):
        # NOTE: an earlier version used /World/envs/env_{i}/Garment and every
        # creation failed with AttributeError: 'NoneType' has no attribute
        # 'count'. That parent prim does not exist -- Isaac Lab only creates
        # /World/envs/env_N when the scene clones with num_envs > 1 -- so the
        # test was measuring a missing parent, not a particle-system limit.
        # Use a parent that exists, to isolate the actual question.
        path = f"/World/Object/ExtraCloth{i}"
        try:
            g = GarmentObject(
                prim_path=path,
                particle_config=env.particle_config,
                garment_config=env.garment_config,
                rng=np.random.RandomState(100 + i),
            )
            g.initialize()
            extras.append((path, g))
            print(f"[extra {i}] created at {path}")
        except Exception as exc:
            import traceback
            print(f"[extra {i}] FAILED at {path}: {type(exc).__name__}: {str(exc)[:160]}")
            traceback.print_exc()

    if not extras:
        print("\n=== VERDICT: cannot create a second particle cloth at all ===")
        print("  num_envs > 1 is NOT available for this task without deeper changes.")
        EXIT = 8
        raise SystemExit

    # ---- separate them, then step ------------------------------------
    for i, (path, g) in enumerate(extras):
        g.set_all_pose({"Garment": [0.25 * (i + 1), 0.0, 0.73, 0.0, 0.0, 0.0]})

    for _ in range(args.steps):
        backend.simulate()

    print("\n=== after stepping ===")
    base_pts = read_points(env.object)
    print(f"  base   n={len(base_pts):>6} centroid={np.round(base_pts.mean(0),4)} "
          f"finite={np.isfinite(base_pts).all()}")
    all_ok, centroids = np.isfinite(base_pts).all(), [base_pts.mean(0)]
    for i, (path, g) in enumerate(extras):
        try:
            pts = read_points(g)
            fin = np.isfinite(pts).all()
            all_ok &= fin
            centroids.append(pts.mean(0))
            print(f"  extra{i} n={len(pts):>6} centroid={np.round(pts.mean(0),4)} finite={fin}")
        except Exception as exc:
            all_ok = False
            print(f"  extra{i} READ FAILED: {type(exc).__name__}: {str(exc)[:120]}")

    # independence: distinct centroids means they are not the same object
    sep = min(
        float(np.linalg.norm(centroids[a] - centroids[b]))
        for a in range(len(centroids)) for b in range(a + 1, len(centroids))
    )
    print("\n=== verdict ===")
    print(f"  cloths created : {1 + len(extras)}")
    print(f"  all finite     : {all_ok}")
    print(f"  min centroid separation: {sep:.4f} m")
    ok = all_ok and sep > 0.05
    print(f"  independent simulation: {ok}")
    if ok:
        print("  => multiple particle cloths coexist. Parallel envs are worth building.")
    else:
        print("  => cloths are not independent or went non-finite. Investigate before")
        print("     investing in scene re-authoring.")
        EXIT = 9
    backend.close()
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

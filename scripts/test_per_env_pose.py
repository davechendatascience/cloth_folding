"""Do per-env garment poses actually take effect, independently?

Parallel envs are worthless for this task without this. LeHome's reset poses
``self.object`` -- the single wrapper around env_0 -- so clones keep their
as-built configuration. Measured at N=4: env_0 centroid z = 0.5292, envs 1-3 at
~0.20. And an unmatched garment pose is precisely the bug that dropped demo
replay from 90% J reduction to 0.7%, so N envs starting from the wrong
configuration is worse than one env starting from the right one.

The check is that each env's cloth lands where *that env* was told to go, and
nowhere near where the others were told to go:

  1. request N distinct poses
  2. each env's centroid tracks its own requested offset
  3. changing one env's pose does not move the others
  4. nothing goes non-finite (PhysX's way of reporting an unhappy cloth)

Note (3) is the real content. A pose call that moved every env identically
would still pass (2) if the requested poses happened to share an offset.
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
p.add_argument("--settle", type=int, default=15, help="steps after each pose call")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    import lehome.tasks  # noqa: F401,E402
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg  # noqa: E402
    from lehome.real_damped_project.tasks.parallel_garment_env import (  # noqa: E402
        build_parallel_cfg,
        make_parallel_env_class,
    )

    N = args.num_envs
    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.sim.device = args.device
    cfg.decimation = 3
    cfg = build_parallel_cfg(cfg, num_envs=N, env_spacing=3.0)

    env = make_parallel_env_class()(cfg)
    env.initialize_obs()
    print(f"[env] num_envs={env.scene.num_envs} view.count="
          f"{getattr(getattr(env, 'cloth_view', None), 'count', '?')}")

    origins = env.scene.env_origins.detach().cpu().numpy()

    def centroids():
        pts = torch.as_tensor(env.particle_positions()).detach().cpu().numpy()
        return pts.mean(axis=1), np.isfinite(pts).all()

    def step(n):
        for _ in range(n):
            env.step(torch.zeros(N, 12, device=env.device))

    # ---- 1/2: distinct poses, each env tracks its own -------------------
    # Deliberately different in x, y and z so a single shared offset cannot
    # masquerade as per-env control.
    poses = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        poses[i, :3] = [0.10 * i, -0.05 * i, 0.75 + 0.05 * i]
    env.set_garment_poses(poses)
    step(args.settle)

    c, finite = centroids()
    local = c - origins  # strip the env origin: what remains is the pose we asked for
    print("\n[per-env] centroid minus env origin (requested xyz in brackets):")
    for i in range(N):
        print(f"    env{i}: {np.round(local[i], 4)}   [{np.round(poses[i, :3], 3)}]")

    # The cloth settles under gravity, so z will not match the request and the
    # absolute xy carries the garment's own centroid offset. What must hold is
    # that the *differences between envs* track the differences we requested.
    d_req = poses[:, :2] - poses[0, :2]
    d_got = local[:, :2] - local[0, :2]
    xy_err = float(np.abs(d_req - d_got).max())
    print(f"\n[per-env] max |requested - actual| inter-env xy offset: {xy_err:.4f} m")

    # ---- 3: move ONE env, confirm the others hold still ------------------
    before, _ = centroids()
    moved = np.array(poses)
    moved[1, :3] = [0.60, 0.40, 0.85]          # only env 1
    env.set_garment_poses(moved[1:2], env_ids=[1])
    step(args.settle)
    after, finite2 = centroids()

    delta = np.linalg.norm(after - before, axis=1)
    print("\n[isolation] centroid movement after re-posing env 1 only:")
    for i in range(N):
        tag = "<- moved" if i == 1 else ""
        print(f"    env{i}: {delta[i]:.4f} m {tag}")

    others = np.delete(delta, 1)
    moved_ok = delta[1] > 0.10
    others_ok = others.max() < 0.05

    # ---- 4: are the envs physically EQUIVALENT, not merely distinct? -----
    # Independence is not enough. If the static bedroom is spawned once at
    # /World/Scene -- outside the env namespace the cloner replicates -- then
    # env_0's garment rests on the furniture and every other env's falls to the
    # floor. The envs would be independent, correctly posed, and still not
    # interchangeable, which silently corrupts anything trained across them.
    #
    # Identical initial conditions must settle to the same height. This check
    # was added after an earlier version of this script reported "usable" while
    # env_0 sat 0.33 m above envs 1-3.
    rest_z = local[:, 2]
    z_spread = float(rest_z.max() - rest_z.min())
    print(f"\n[equivalence] per-env settled z: {np.round(rest_z, 4)}")
    print(f"[equivalence] spread: {z_spread:.4f} m   (want < 0.05)")
    equivalent = z_spread < 0.05

    print("\n=== verdict ===")
    print(f"  all finite                 : {bool(finite and finite2)}")
    print(f"  inter-env xy offset error  : {xy_err:.4f} m   (want < 0.02)")
    print(f"  targeted env moved         : {moved_ok}  ({delta[1]:.4f} m)")
    print(f"  untargeted envs held still : {others_ok}  (max {others.max():.4f} m)")
    print(f"  envs settle alike          : {equivalent}  (z spread {z_spread:.4f} m)")
    posing_ok = bool(finite and finite2) and xy_err < 0.02 and moved_ok and others_ok
    print(f"  per-env posing works       : {posing_ok}")
    ok = posing_ok and equivalent
    print(f"  envs usable for training   : {ok}")
    if ok:
        print("  => parallel envs are usable for labelling and RL resets.")
    elif posing_ok:
        print("  => posing works but envs are NOT physically equivalent. The")
        print("     static scene is global (/World/Scene), so only env_0 has the")
        print("     furniture the garment rests on. Independent-but-unequal envs")
        print("     corrupt training silently -- do NOT proceed on this.")
        EXIT = 14
    else:
        print("  => poses are shared or ineffective. Envs would train from the")
        print("     wrong garment configuration -- do NOT proceed on this.")
        EXIT = 13
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

"""Closed-loop verification of IsaacGarmentBackend's Cartesian control.

The same test that exposed LeHome's URDF solver (commanded 5 cm, moved 39 cm in
the wrong direction). Differential IK on the simulator's own Jacobian should be
consistent by construction -- but "should be" is exactly the assumption that
failed last time, and the Jacobian indexing (`ee_body_idx - 1` for a fixed-base
articulation) and the root-frame transform are both easy to get subtly wrong in
a way that still produces smooth, plausible motion.

Also smoke-tests the rest of the backend protocol: J, cameras, proprioception,
termination.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--settle", type=int, default=40, help="policy steps per commanded move")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
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

    cfg = IsaacGarmentCfg(garment_name=args.garment, device=args.device)
    print(f"[cfg] decimation={cfg.decimation} ik_lambda={cfg.ik_lambda} "
          f"max_delta={cfg.max_delta} damping=per-joint")
    backend = IsaacGarmentBackend(cfg)
    backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
    print(f"[ok] backend built. garment_type={backend.garment_type} dt={backend.dt:.4f}s")

    # ---------------- protocol smoke ----------------
    print("\n=== protocol shapes ===")
    imgs = backend.render_cameras()
    prop = backend.get_proprioception()
    ee = backend.get_end_effector_positions()
    vel = backend.get_end_effector_velocities()
    J0 = backend.compute_cloth_error()
    print(f"  images  {tuple(imgs.shape)}  range=[{imgs.min():.3f},{imgs.max():.3f}]"
          f"  expected {backend.image_shape}")
    print(f"  proprio {tuple(prop.shape)}  expected (1,{backend.proprio_dim})")
    print(f"  ee_pos  {tuple(ee.shape)}  {np.round(ee[0].cpu().numpy(),4).tolist()}")
    print(f"  ee_vel  {tuple(vel.shape)}")
    print(f"  J       {float(J0):.4f}")
    term, trunc = backend.check_done()
    print(f"  done    terminated={bool(term.any())} truncated={bool(trunc.any())}")

    # ---------------- closed-loop Cartesian ----------------
    print("\n=== closed-loop Cartesian accuracy ===")
    moves = [
        ("+z 5cm", np.array([0.0, 0.0, 0.05])),
        ("-z 5cm", np.array([0.0, 0.0, -0.05])),
        ("+y 5cm", np.array([0.0, 0.05, 0.0])),
        ("+x 3cm", np.array([0.03, 0.0, 0.0])),
    ]
    results = []
    for label, d in moves:
        start = backend.get_end_effector_positions().clone()
        target = start.clone()
        target[0, 0] += torch.tensor(d, dtype=torch.float32, device=start.device)
        target[0, 1] += torch.tensor(d, dtype=torch.float32, device=start.device)

        for _ in range(args.settle):
            backend.set_end_effector_targets(target)
            backend.simulate()

        end = backend.get_end_effector_positions()
        for ai, aname in enumerate(("left", "right")):
            moved = float(torch.linalg.vector_norm(end[0, ai] - start[0, ai]))
            err = float(torch.linalg.vector_norm(end[0, ai] - target[0, ai]))
            cmd = float(np.linalg.norm(d))
            results.append((label, aname, cmd, moved, err))
            status = "OK " if err < 0.01 else ("WARN" if err < 0.03 else "BAD ")
            print(f"  [{status}] {label:8s} {aname:5s}  commanded={cmd:.3f}  "
                  f"moved={moved:.4f}  residual_err={err:.4f} m")

    print("\n=== verdict ===")
    bad = [r for r in results if r[4] >= 0.03]
    warn = [r for r in results if 0.01 <= r[4] < 0.03]
    max_err = max(r[4] for r in results)
    # Overshoot check: did any move travel far beyond what was commanded?
    runaway = [r for r in results if r[3] > 3 * r[2]]
    print(f"  max residual error : {max_err:.4f} m")
    print(f"  ok/warn/bad        : {len(results)-len(warn)-len(bad)}/{len(warn)}/{len(bad)}")
    print(f"  runaway moves      : {len(runaway)}  (LeHome's solver had 8x overshoot)")
    if max_err < 0.01 and not runaway:
        print("  => Cartesian control is TRUSTWORTHY")
    elif max_err < 0.03 and not runaway:
        print("  => usable but imprecise; check ik_lambda / settle steps")
    else:
        print("  => NOT trustworthy; do not build on this")
        EXIT = 2

    # ---------------- does J respond to motion? ----------------
    J1 = backend.compute_cloth_error()
    print(f"\n  J before moves = {float(J0):.4f}   after = {float(J1):.4f}   "
          f"delta = {float(J1-J0):+.4f}")

    backend.close()
except Exception:
    import traceback

    traceback.print_exc()
    EXIT = 1
finally:
    try:
        simulation_app.close()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT)

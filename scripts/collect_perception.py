"""Collect (image, cloth state) pairs with no episode phase to exploit.

The one link never honestly established in this project. The `Ĵ` head scored
R^2 0.87 and it was reading the clock: a phase-only predictor scored 0.810 on
the same target, and within-band R^2 was NEGATIVE, i.e. classification rather
than regression. Demonstration frames all carry a well-defined episode phase, so
that confound is unavoidable in demo-derived data.

Randomised poses have no episode, so there is nothing to read but the cloth.

Grasping is *not* required here -- that was the dependency error in the design
doc. Configuration diversity comes from randomised placement plus gentle arm
sweeps, both of which already work; only action-conditioned *dynamics* needs a
working grasp.

Records per frame:
  image        (9, S, S) uint8 -- three cameras, same layout the policy sees
  checkpoints  (n_cp, 3) cm    -- ground truth, INCLUDING occluded points
  ee           (2, 3) m        -- gripper positions, so occlusion can be derived
  pose_id                      -- which randomised placement this came from

The occluded labels are the point of doing this in simulation: perception can be
supervised to infer configuration it cannot see, which is the observer problem
and is exactly what real-world datasets cannot provide.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--poses", type=int, default=300, help="randomised placements")
p.add_argument("--settle", type=int, default=18, help="steps to settle after placing")
p.add_argument("--sweep", type=int, default=26, help="gentle arm sweep steps (no grasp)")
p.add_argument("--post", type=int, default=8, help="steps to re-settle after the sweep")
p.add_argument("--record_from", type=int, default=8,
               help="start recording this many settle steps in, once the cloth has "
                    "stopped free-falling and the frames are informative")
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--image_size", type=int, default=84)
p.add_argument("--seed", type=int, default=0)
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
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    rng = np.random.RandomState(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Critical damping: this is generated data, so we are free to choose the
    # cleaner plant. Only demo-imitation was forced onto the under-damped one.
    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device,
        decimation=args.decimation, image_size=args.image_size))
    print(f"[env] dt={backend.dt:.4f}s garment={backend.garment_type}", flush=True)

    n_cp = len(backend.check_points)
    C, H, W = backend.image_shape
    per_pose = (args.settle - args.record_from) + args.sweep + args.post
    total = args.poses * per_pose
    print(f"[out] {args.poses} poses x {per_pose} recorded = {total} frames, "
          f"images {(C, H, W)}", flush=True)

    images = np.memmap(out / "images.u8", dtype=np.uint8, mode="w+", shape=(total, C, H, W))
    CP = np.full((total, n_cp, 3), np.nan, dtype=np.float32)
    EE = np.full((total, 2, 3), np.nan, dtype=np.float32)
    POSE = np.full(total, -1, dtype=np.int64)
    POSE6 = np.full((args.poses, 6), np.nan, dtype=np.float32)

    def snap(i, pose_i):
        images[i] = (backend.render_cameras()[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        CP[i] = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
        EE[i] = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        POSE[i] = pose_i

    write = 0
    t0 = time.time()
    for s in range(args.poses):
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        # Deliberately wider than the demonstrations (x in [-0.079, 0.039],
        # y in [-0.019, 0.040], rx/ry within +-20 deg, rz always 0) so the model
        # sees configurations the demos never contain.
        pose = [float(rng.uniform(-0.13, 0.09)), float(rng.uniform(-0.09, 0.09)), 0.73,
                float(rng.uniform(-28, 28)), float(rng.uniform(-28, 28)),
                float(rng.uniform(-35, 35))]
        POSE6[s] = pose
        backend.set_garment_pose(pose)

        for k in range(args.settle):
            backend.simulate()
            if k >= args.record_from:
                snap(write, s); write += 1

        # Gentle sweep: push the cloth around to crumple it. No grasp needed --
        # the aim is configuration diversity, not a controlled displacement.
        ee_now = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        arm = int(rng.randint(2))
        cp = int(rng.randint(n_cp))
        cur = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
        goal = cur[cp] / 100.0 + np.array(
            [rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(-0.01, 0.02)],
            dtype=np.float32)
        x_cmd = torch.tensor(ee_now, dtype=torch.float32).unsqueeze(0)
        for k in range(args.sweep):
            a = (k + 1) / args.sweep
            x = x_cmd.clone()
            x[0, arm] = torch.tensor((1 - a) * ee_now[arm] + a * goal, dtype=torch.float32)
            backend.set_end_effector_targets(x)
            backend.simulate()
            snap(write, s); write += 1

        for _ in range(args.post):
            backend.simulate()
            snap(write, s); write += 1

        if s % 10 == 0 or s == args.poses - 1:
            el = time.time() - t0
            rate = (s + 1) / max(el, 1e-9)
            eta = (args.poses - s - 1) / max(rate, 1e-9) / 60
            print(f"  pose {s:>4}/{args.poses}  frames {write:>6}  "
                  f"{rate*60:.1f} poses/min  ETA {eta:.0f} min", flush=True)

    images.flush()
    np.save(out / "checkpoints_cm.npy", CP)
    np.save(out / "ee_pos_m.npy", EE)
    np.save(out / "pose_id.npy", POSE)
    np.save(out / "pose6.npy", POSE6)
    (out / "meta.json").write_text(json.dumps({
        "n_frames": int(write), "image_shape": [C, H, W], "n_check_points": n_cp,
        "poses": args.poses, "garment": args.garment, "seed": args.seed,
        "damping": "critical", "note": "randomised placements; no episode phase exists",
    }, indent=2))
    fin = np.isfinite(CP[:write]).all(axis=(1, 2))
    print(f"\n[done] {write} frames from {args.poses} poses, "
          f"{int(fin.sum())} with finite cloth state -> {out}", flush=True)
    backend.close()
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

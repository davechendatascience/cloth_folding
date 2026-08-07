"""Collect cloth perception + dynamics data that is free of the phase confound.

Every previous dataset here came from demonstrations, and every frame in a
demonstration has a well-defined episode phase. That single fact invalidated
three results in a row:

    phase -> action   R^2 0.688   (vs 0.725 for a proprio ridge)
    phase -> J        R^2 0.810   (vs 0.863 for the visual encoder)
    BC closed loop    indistinguishable from a frozen arm

A model can satisfy any of those objectives by reading the clock. Randomised
configurations remove the confound by construction: there is no episode, so
there is no phase to exploit.

One procedure yields everything the factored design needs:

    (image, p)          perception -- image to cloth configuration
    (p, u, dp)          dynamics   -- the cloth Jacobian / world model
    visibility          occlusion  -- which check-points the arm covers
    both damping modes  the A/B on whether critical damping makes the
                        action -> outcome map materially more predictable

Why damping belongs in the data-generation step, not just the controller: an
under-damped plant answers the same action with ringing and overshoot, so one
`u` maps to a distribution of outcomes and the forward model must learn the
transient. Critically damped, the same action settles smoothly to a predictable
configuration and the map is nearly static. `--damping` runs both so the claim
is measured rather than assumed.

Each sample is a *targeted* perturbation, not random flailing: the arm is sent
to a randomly chosen check-point, grasps, and displaces it. Random joint motion
would mostly miss the cloth and record dp ~ 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", required=True, help="output directory")
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--samples", type=int, default=200, help="randomised configurations")
p.add_argument("--settle", type=int, default=25, help="steps to settle after placing")
p.add_argument("--approach", type=int, default=30, help="steps to reach the check-point")
p.add_argument("--move", type=int, default=40, help="steps of displacement while grasping")
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--damping", default="critical", choices=["critical", "demo"],
               help="'demo' = LeHome's original under-damped joints (joint_damping={}); "
                    "'critical' = our measured per-joint critical damping. Run both to "
                    "test whether damping makes the forward model more learnable.")
p.add_argument("--grip_closed", type=float, default=-0.17,
               help="gripper joint value for a closed grasp. Demonstrations are "
                    "bimodal at ~-0.145 and ~+0.5; which pole is 'closed' is not "
                    "documented, so it is a parameter and the collector reports "
                    "|dp| so the right value is chosen by measurement.")
p.add_argument("--grip_open", type=float, default=0.6)
p.add_argument("--lift", type=float, default=0.05, help="approach height above the target, m")
p.add_argument("--image_size", type=int, default=84)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

# Images are the point of this dataset, so rendering stays on.
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

    cfg = IsaacGarmentCfg(
        garment_name=args.garment, device=args.device, decimation=args.decimation,
        image_size=args.image_size,
    )
    if args.damping == "demo":
        cfg.joint_damping = {}
    backend = IsaacGarmentBackend(cfg)
    print(f"[env] damping={args.damping}  dt={backend.dt:.4f}s  garment={backend.garment_type}")

    GRIP_IDX = (5, 11)   # joint order ends with "gripper"; two arms of 6

    def set_grip(arm_i, value):
        backend._joint_targets[:, GRIP_IDX[arm_i]] = float(value)

    n_cp = len(backend.check_points)
    per_sample = args.approach + args.move
    total = args.samples * per_sample
    C, H, W = backend.image_shape
    print(f"[out] {args.samples} samples x {per_sample} steps = {total} frames, images {(C,H,W)}")

    images = np.memmap(out / "images.u8", dtype=np.uint8, mode="w+", shape=(total, C, H, W))
    P = np.full((total, n_cp, 3), np.nan, dtype=np.float32)   # check-points, cm
    EE = np.full((total, 2, 3), np.nan, dtype=np.float32)     # both EE positions, m
    Q = np.full((total, 12), np.nan, dtype=np.float32)        # joint targets
    SAMPLE = np.full(total, -1, dtype=np.int64)
    GRASPED = np.zeros(total, dtype=np.int64)                 # which check-point is held
    PHASE_FREE = np.ones(total, dtype=np.uint8)               # marker: no episode phase

    def snap(i, sample_i, grasped_cp):
        images[i] = (backend.render_cameras()[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        P[i] = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
        EE[i] = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        Q[i] = backend._joint_targets[0].detach().cpu().numpy()
        SAMPLE[i] = sample_i
        GRASPED[i] = grasped_cp

    write = 0
    kept = 0
    for s in range(args.samples):
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))

        # Randomised placement, deliberately wider than the demonstrations
        # (x in [-0.079,0.039], y in [-0.019,0.040], rx/ry within +-20 deg) so the
        # perception model sees configurations the demos never contain. rz is
        # always 0 in the dataset; randomising it adds orientation diversity.
        pose = [float(rng.uniform(-0.12, 0.08)), float(rng.uniform(-0.08, 0.08)), 0.73,
                float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25)),
                float(rng.uniform(-30, 30))]
        backend.set_garment_pose(pose)
        for _ in range(args.settle):
            backend.simulate()

        p0 = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
        if not np.isfinite(p0).all():
            print(f"  sample {s}: non-finite cloth after settle, skipped")
            continue

        # Target a random check-point with a random arm, then displace it.
        cp = int(rng.randint(n_cp))
        arm = int(rng.randint(2))
        target = p0[cp] / 100.0                       # cm -> m
        disp = np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10),
                         rng.uniform(0.0, 0.08)], dtype=np.float32)

        ee_now = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        x_cmd = torch.tensor(ee_now, dtype=torch.float32).unsqueeze(0)
        above = target + np.array([0.0, 0.0, args.lift], dtype=np.float32)

        # Approach from above with the gripper open, then descend onto the
        # check-point. Driving straight at it from the side sweeps the cloth
        # away before the grasp -- measured |dp| of 0.15 cm, i.e. no useful
        # interaction at all.
        set_grip(arm, args.grip_open)
        half = max(args.approach // 2, 1)
        for k in range(args.approach):
            if k < half:
                a = (k + 1) / half
                goal = (1 - a) * ee_now[arm] + a * above
            else:
                a = (k - half + 1) / max(args.approach - half, 1)
                goal = (1 - a) * above + a * target
            x = x_cmd.clone()
            x[0, arm] = torch.tensor(goal, dtype=torch.float32)
            backend.set_end_effector_targets(x)
            set_grip(arm, args.grip_open)
            backend.simulate()
            snap(write, s, -1); write += 1

        # Close, let the grasp settle, then displace while holding.
        for _ in range(5):
            set_grip(arm, args.grip_closed)
            backend.simulate()

        for k in range(args.move):
            a = (k + 1) / args.move
            x = x_cmd.clone()
            x[0, arm] = torch.tensor(target + a * disp, dtype=torch.float32)
            backend.set_end_effector_targets(x)
            set_grip(arm, args.grip_closed)
            backend.simulate()
            snap(write, s, cp); write += 1

        p1 = P[write - 1]
        moved = float(np.linalg.norm(p1[cp] - p0[cp]))
        kept += 1
        if s % 10 == 0 or s < 3:
            print(f"  sample {s:>4}: cp{cp} arm{arm}  |dp| = {moved:6.2f} cm  "
                  f"frames {write}/{total}", flush=True)

    images.flush()
    np.save(out / "checkpoints_cm.npy", P)
    np.save(out / "ee_pos_m.npy", EE)
    np.save(out / "joint_targets.npy", Q)
    np.save(out / "sample_index.npy", SAMPLE)
    np.save(out / "grasped_cp.npy", GRASPED)
    np.save(out / "phase_free.npy", PHASE_FREE)
    (out / "meta.json").write_text(json.dumps({
        "n_frames": int(write), "image_shape": [C, H, W], "n_check_points": n_cp,
        "damping": args.damping, "garment": args.garment, "seed": args.seed,
        "samples_requested": args.samples, "samples_kept": kept,
        "approach_steps": args.approach, "move_steps": args.move,
        "note": "randomised configurations; no episode phase exists in this data",
    }, indent=2))

    fin = np.isfinite(P[:write]).all(axis=(1, 2))
    print(f"\n[done] {write} frames, {kept}/{args.samples} samples, "
          f"{int(fin.sum())} with finite cloth state -> {out}")
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

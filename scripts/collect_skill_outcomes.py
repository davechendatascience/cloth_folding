"""Collect probe skills labelled with CONTACT MODES and OUTCOMES.

The shift: do not try to recover cloth physics or reconstruct configuration.
Identify the parts of state that change which next actions are reliable. Two
cloth states are equivalent if the same skill has the same outcome distribution
from both -- an action-conditioned state abstraction, not a visual one.

So the labels are not "where are the check-points". They are:

  mode m_t     per frame, from privileged state:
                 grasped  -- the nearest check-point tracks the gripper
                 slipping -- in contact, but not tracking
                 free     -- no contact
  outcome      per skill, the thing a planner needs:
                 did the grasp hold, for how long, and what displacement resulted

Both are computed exactly from simulator state and are cheap. Neither requires
knowing Young's modulus, bending stiffness, or the hidden layer topology -- the
ill-posed inverse problem this deliberately avoids.

**Failures are the point.** A hand-written grasp succeeds rarely here (measured
|dp| of 0.16-1.27 cm), and LeHome uses no particle attachment -- holding relies
entirely on adhesion 0.1 / friction 0.5. That gives a naturally balanced
success/failure set, which is exactly what an outcome predictor needs and what a
reconstruction dataset cannot provide.

Probe parameters are randomised so outcomes vary: approach direction, descent
depth, gripper closure, lateral offset from the target point, and lift height.
The learnable question becomes P(grasp holds | image, probe parameters), which is
judged by the criterion that matters rather than by reconstruction error.

Saves incrementally. An earlier collector wrote its arrays only at the end and
270 poses were lost when it was stopped.
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
p.add_argument("--skills", type=int, default=400)
p.add_argument("--settle", type=int, default=16)
p.add_argument("--approach", type=int, default=22)
p.add_argument("--close", type=int, default=6)
p.add_argument("--move", type=int, default=28)
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--image_size", type=int, default=84)
p.add_argument("--track_tol", type=float, default=0.4,
               help="cm/step tolerance for 'the check-point is tracking the gripper'")
p.add_argument("--contact_r", type=float, default=13.0,
               help="contact radius in cm, measured BODY-to-check-point. The gripper "
                    "body origin sits ~9 cm from the fingertips, so 6 cm was below "
                    "anything physically achievable -- the demos never get the body "
                    "within 7.3 cm while manipulating the cloth perfectly well.")
p.add_argument("--gripper_offset", type=float, default=0.09,
               help="metres from the gripper BODY origin to the contact point. "
                    "Commanding the body to the check-point drives the fingertips ~9 cm "
                    "through the cloth into the table; the IK then saturates against "
                    "collision, which produced a hard 56.7 cm floor and 8% grasp rate. "
                    "Calibrated from demo frames where cloth tracks a gripper: the "
                    "grasped point sits 7.3-9.8 cm from the body origin.")
p.add_argument("--save_every", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

MODE_FREE, MODE_CONTACT, MODE_SLIP, MODE_GRASP = 0, 1, 2, 3
EXIT = 0
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    rng = np.random.RandomState(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device,
        decimation=args.decimation, image_size=args.image_size))
    n_cp = len(backend.check_points)
    C, H, W = backend.image_shape
    GRIP = (5, 11)
    per_skill = args.approach + args.close + args.move
    total = args.skills * per_skill
    print(f"[env] garment={backend.garment_type} n_cp={n_cp}", flush=True)
    print(f"[out] {args.skills} skills x {per_skill} steps = {total} frames", flush=True)

    images = np.memmap(out / "images.u8", dtype=np.uint8, mode="w+", shape=(total, C, H, W))
    CP = np.full((total, n_cp, 3), np.nan, dtype=np.float32)
    EE = np.full((total, 2, 3), np.nan, dtype=np.float32)
    MODE = np.full(total, -1, dtype=np.int8)          # per-frame contact mode
    SKILL = np.full(total, -1, dtype=np.int64)
    # Probe parameters: what the planner would choose. These are the "u".
    PARAMS = np.full((args.skills, 9), np.nan, dtype=np.float32)
    # Outcomes: what a skill-transition model predicts.
    OUT = np.full((args.skills, 6), np.nan, dtype=np.float32)
    OUT_NAMES = ["grasp_held_frac", "held_at_end", "cp_displacement_cm",
                 "ee_cp_final_gap_cm", "max_track_run", "target_cp"]

    def snap(i, skill_i, mode):
        images[i] = (backend.render_cameras()[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        CP[i] = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
        EE[i] = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        MODE[i] = mode
        SKILL[i] = skill_i

    def save():
        images.flush()
        np.save(out / "checkpoints_cm.npy", CP); np.save(out / "ee_pos_m.npy", EE)
        np.save(out / "mode.npy", MODE); np.save(out / "skill_id.npy", SKILL)
        np.save(out / "params.npy", PARAMS); np.save(out / "outcomes.npy", OUT)
        (out / "meta.json").write_text(json.dumps({
            "image_shape": [C, H, W], "n_check_points": n_cp,
            "outcome_names": OUT_NAMES,
            "param_names": ["target_cp", "arm", "off_x", "off_y", "depth",
                            "grip_closed", "disp_x", "disp_y", "disp_z"],
            "mode_names": {"0": "free", "1": "contact", "2": "slipping", "3": "grasped"},
            "track_tol_cm": args.track_tol, "contact_r_cm": args.contact_r,
            "frames_written": int(write), "skills_done": int(done),
        }, indent=2))

    write = 0
    done = 0
    t0 = time.time()
    for s in range(args.skills):
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        pose = [float(rng.uniform(-0.13, 0.09)), float(rng.uniform(-0.09, 0.09)), 0.73,
                float(rng.uniform(-28, 28)), float(rng.uniform(-28, 28)),
                float(rng.uniform(-35, 35))]
        backend.set_garment_pose(pose)
        for _ in range(args.settle):
            backend.simulate()

        p0 = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
        if not np.isfinite(p0).all():
            continue

        # --- randomised probe parameters -- these make outcomes VARY ---------
        cp = int(rng.randint(n_cp))
        arm = int(rng.randint(2))
        off = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)], dtype=np.float32)
        depth = float(rng.uniform(-0.015, 0.010))     # below/above the cloth surface
        grip_c = float(rng.choice([-0.17, -0.05, 0.2, 0.6]))
        disp = np.array([rng.uniform(-0.09, 0.09), rng.uniform(-0.09, 0.09),
                         rng.uniform(0.0, 0.07)], dtype=np.float32)
        PARAMS[s] = [cp, arm, off[0], off[1], depth, grip_c, disp[0], disp[1], disp[2]]

        target = p0[cp] / 100.0 + np.array(
            [off[0], off[1], args.gripper_offset + depth], dtype=np.float32)
        above = target + np.array([0.0, 0.0, 0.07], dtype=np.float32)
        ee0 = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        x_cmd = torch.tensor(ee0, dtype=torch.float32).unsqueeze(0)

        def step_to(goal, grip, skill_i):
            global write
            x = x_cmd.clone()
            x[0, arm] = torch.tensor(goal, dtype=torch.float32)
            backend.set_end_effector_targets(x)
            backend._joint_targets[:, GRIP[arm]] = grip
            backend.simulate()
            # Mode from privileged state: does the target check-point move with
            # the gripper? This is the whole labelling scheme, and it needs no
            # material parameters.
            cur = backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)
            eec = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)]) * 100.0
            gap = float(np.linalg.norm(cur[cp] - eec[arm]))
            if write > 0 and SKILL[write - 1] == skill_i:
                dcp = cur[cp] - CP[write - 1][cp]
                dee = eec[arm] - EE[write - 1][arm] * 100.0
                tracking = float(np.linalg.norm(dcp - dee)) < args.track_tol
            else:
                tracking = False
            if gap > args.contact_r:
                mode = MODE_FREE
            elif tracking:
                mode = MODE_GRASP
            elif np.linalg.norm(dcp if write > 0 else 0) > 1e-3:
                mode = MODE_SLIP
            else:
                mode = MODE_CONTACT
            snap(write, skill_i, mode); write += 1
            return gap

        for k in range(args.approach):
            a = (k + 1) / args.approach
            half = args.approach // 2
            goal = ((1 - (k + 1) / half) * ee0[arm] + ((k + 1) / half) * above
                    if k < half else
                    (1 - (k - half + 1) / max(args.approach - half, 1)) * above
                    + ((k - half + 1) / max(args.approach - half, 1)) * target)
            step_to(goal, 0.6, s)
        for _ in range(args.close):
            step_to(target, grip_c, s)
        start = write
        for k in range(args.move):
            a = (k + 1) / args.move
            gap = step_to(target + a * disp, grip_c, s)

        # --- outcome labels --------------------------------------------------
        seg = MODE[start:write]
        held = (seg == MODE_GRASP)
        runs, cur_run = 0, 0
        for v in held:
            cur_run = cur_run + 1 if v else 0
            runs = max(runs, cur_run)
        p1 = CP[write - 1]
        OUT[s] = [float(held.mean()), float(held[-3:].mean() > 0.5),
                  float(np.linalg.norm(p1[cp] - p0[cp])), gap, float(runs), float(cp)]
        done += 1

        if s % args.save_every == 0 or s == args.skills - 1:
            save()
            el = time.time() - t0
            rate = (s + 1) / max(el, 1e-9)
            ok_frac = np.nanmean(OUT[:s + 1, 1])
            print(f"  skill {s:>4}/{args.skills}  frames {write:>6}  "
                  f"held_at_end={ok_frac:.2f}  {rate*60:.1f}/min  "
                  f"ETA {(args.skills-s-1)/max(rate,1e-9)/60:.0f} min", flush=True)

    save()
    print(f"\n[done] {write} frames, {done} skills -> {out}", flush=True)
    print(f"  grasp held at end: {np.nanmean(OUT[:done,1]):.2%}  "
          f"mean displacement {np.nanmean(OUT[:done,2]):.2f} cm", flush=True)
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

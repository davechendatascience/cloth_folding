"""Phase A: can friction-only grasping be made reliable in LeHome?

Everything in the cloth-folding literature we surveyed assumes grasping is a
solved primitive -- SpeedFolding on a real YuMi, BiFold via SoftGym's picker,
which attaches to the nearest particle by construction. LeHome has **no particle
attachment**: holding depends entirely on `adhesion 0.1` / `friction 0.5` /
`particle_friction_scale 0.6`. An earlier sweep here measured a best hold rate of
**20%**, and the primitive that displaced the most cloth held **0%** -- it was
shoving, not carrying. That is why the earlier skill thread was abandoned, and it
is the prerequisite for porting a pick-and-place method.

But the demonstrations *do* grasp, so the physics permits it. The question is
what conditions they hit that our synthesised primitives missed. Two independent
measurements point at approach depth:

    gripper body -> cloth distance
      replay (works)             0.065 m
      demo grasps (work)         0.100 m     [runs/grasp_orientation.json]
      old synthesised primitives 0.135 m     <- failed
      trained policies (fail)    0.095-0.126 m

The gripper *body* origin sits ~9 cm from the actual contact point (calibrated on
4,926 demo frames), so a body at 0.135 m leaves the fingers ~4 cm above the
fabric. Everything that works gets the body inside ~0.10 m.

This sweeps descend depth and dwell, and measures hold rate directly: grasp a
check-point, lift, and ask whether the cloth came with the gripper. Success
criterion is a hold rate clearly above the 20% ceiling; if nothing clears it, the
port is not viable without changing the cloth parameters, and that is worth
knowing before building anything on top.

Reported per condition: contact fraction, held-at-end fraction, and how far the
targeted check-point actually travelled -- because a primitive can look invariant
by doing nothing, which is how the earlier profile was misread.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--trials", type=int, default=8, help="randomised cloth states per condition")
p.add_argument("--depths", default="0.06,0.08,0.10,0.13",
               help="gripper-body target distance above the check-point, metres")
p.add_argument("--dwell", default="10,30", help="sim steps to dwell before lifting")
p.add_argument("--lift", type=float, default=0.12, help="lift height, metres")
p.add_argument("--out", default=None)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=False, device="cuda").app

EXIT = 0
try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name="Top_Long_Seen_0", device="cuda", decimation=3,
        joint_damping={}, skip_images=True))
    print("[grasp] backend up", flush=True)

    depths = [float(x) for x in args.depths.split(",")]
    dwells = [int(x) for x in args.dwell.split(",")]

    def cloth_pts():
        return np.asarray(backend.env.object.get_current_mesh_points()[0],
                          dtype=float).reshape(-1, 3)

    def settle(n=40):
        for _ in range(n):
            backend.simulate()

    rows = []
    for depth in depths:
        for dwell in dwells:
            held = contact = 0
            travels = []
            for t in range(args.trials):
                backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
                settle(40)
                pts = cloth_pts()
                # target the check-point that is most exposed (highest z), the
                # one a human would pick
                cps = backend.check_point_positions_cm()[0].detach().cpu().numpy() / 100.0
                tgt = cps[int(np.argmax(cps[:, 2]))]
                p_before = tgt.copy()

                # approach from directly above, descend to `depth`, dwell, lift
                for phase, z_off, steps in (("above", 0.22, 25),
                                            ("descend", depth, 30),
                                            ("dwell", depth, dwell),
                                            ("lift", depth + args.lift, 35)):
                    # set_end_effector_targets wants (num_envs, 2, 3): a
                    # position per arm. Arm 0 goes to the target; arm 1 is
                    # commanded to hold its current position so only one hand
                    # is under test.
                    for _ in range(steps):
                        right_now = backend._ee_pos_w(1)[0].detach()
                        cmd = torch.stack([
                            torch.tensor([tgt[0], tgt[1], tgt[2] + z_off],
                                         dtype=torch.float32, device=backend.device),
                            right_now.to(torch.float32),
                        ]).unsqueeze(0)
                        backend.set_end_effector_targets(cmd)
                        backend.simulate()
                    if phase == "dwell":
                        ee = backend._ee_pos_w(0)[0].detach().cpu().numpy()
                        d = float(np.min(np.linalg.norm(cloth_pts() - ee, axis=1)))
                        if d <= 0.09:
                            contact += 1

                pts_after = cloth_pts()
                # did the targeted point come up with the gripper?
                near = pts_after[np.argmin(np.linalg.norm(pts_after - p_before, axis=1))]
                travel = float(np.linalg.norm(near - p_before))
                travels.append(travel)
                if near[2] - p_before[2] > 0.04:      # rose >4 cm = carried
                    held += 1

            rows.append(dict(depth=depth, dwell=dwell, trials=args.trials,
                             contact=contact / args.trials, held=held / args.trials,
                             travel_cm=float(np.mean(travels) * 100)))
            print(f"[grasp] depth {depth:.2f} m dwell {dwell:>3}  "
                  f"contact {contact}/{args.trials}  held {held}/{args.trials}  "
                  f"travel {np.mean(travels)*100:5.2f} cm", flush=True)

    print("\n[grasp] ==== summary (prior best hold rate was 20%) ====", flush=True)
    for r in sorted(rows, key=lambda r: -r["held"]):
        print(f"  depth {r['depth']:.2f} dwell {r['dwell']:>3}  held {100*r['held']:5.1f}%  "
              f"contact {100*r['contact']:5.1f}%  travel {r['travel_cm']:5.2f} cm", flush=True)
    if args.out:
        open(args.out, "w").write(json.dumps(rows, indent=2))
    backend.close()
except Exception:
    import traceback
    traceback.print_exc(); sys.stderr.flush(); EXIT = 1
finally:
    import threading

    def _force(code=EXIT):
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)

    t = threading.Timer(30.0, _force); t.daemon = True; t.start()
    try:
        app.close()
    except Exception:
        pass
    _force()

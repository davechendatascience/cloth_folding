"""Do our IK-driven grasps hold the wrist the way the demonstrations do?

`lift` holds the cloth in 0 of 60 states while the demonstrations produce 4,926
frames where cloth tracks a gripper. Reach is no longer the problem (the 9 cm
body-to-contact offset fixed that; there is now transient contact in 33% of
states) so the difference is in *how* the hand arrives.

The arm has 5 controllable joints, so orientation is not freely assignable and
position-only IK is a kinematic necessity rather than an oversight. But
orientation is still *determined* by which solution the DLS solver picks, and the
demonstrations -- recorded joint trajectories -- may sit in a completely
different part of that solution space.

This runs both in one session and compares the gripper's axes at the moment that
matters:

  demos: frames where a check-point provably tracks a gripper (real grasps)
  ours:  the moment our lift primitive closes its gripper

Reported as the angle between each gripper body axis and world -z, which is
convention-free: what matters is whether the two distributions differ, not which
axis is nominally "the approach axis".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--demo_episodes", type=int, default=4)
p.add_argument("--our_trials", type=int, default=12)
p.add_argument("--gripper_offset", type=float, default=0.09)
p.add_argument("--track_tol", type=float, default=0.4)
p.add_argument("--contact_r", type=float, default=13.0)
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--json_out", default="")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    def quat_to_axes(q):
        """wxyz quaternion -> the three body axes as world-frame column vectors."""
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    DOWN = np.array([0.0, 0.0, -1.0])

    def axis_angles(q):
        """Angle (deg) between each body axis and world -z."""
        A = quat_to_axes(q)
        return [float(np.degrees(np.arccos(np.clip(A[:, k] @ DOWN, -1, 1)))) for k in range(3)]

    cache = Path(args.cache)
    action = np.load(cache / "action.npy")
    episode = np.load(cache / "episode.npy")
    ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
    gnames = list(ginfo.keys())

    # Demo replay uses the ORIGINAL under-damped plant: that is the plant the
    # trajectories were recorded on, and replaying elsewhere degrades the fold.
    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device, decimation=3,
        joint_damping={}, skip_images=True))
    n_cp = len(backend.check_points)
    GRIP = (5, 11)

    def cps():
        return backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)

    def ee(a):
        return backend._ee_pos_w(a)[0].detach().cpu().numpy() * 100.0

    def quat(a):
        return backend._ee_quat_w(a)[0].detach().cpu().numpy()

    # ---------------- demos ----------------
    demo_ang, demo_gap = [], []
    for e in range(args.demo_episodes):
        rows = np.where(episode == e)[0]
        if not len(rows):
            continue
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        gi, li = e // args.eps_per_garment, e % args.eps_per_garment
        if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
            backend.set_garment_pose(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
            for _ in range(5):
                backend.simulate()
        prev_cp = cps(); prev_ee = [ee(0), ee(1)]
        for a_i in action[rows]:
            backend.set_joint_targets(torch.as_tensor(a_i, dtype=torch.float32))
            backend.simulate()
            cur_cp = cps(); cur_ee = [ee(0), ee(1)]
            for arm in (0, 1):
                d = np.linalg.norm(cur_cp - cur_ee[arm], axis=-1)
                dcp = cur_cp - prev_cp
                dee = cur_ee[arm] - prev_ee[arm]
                held = (np.linalg.norm(dcp - dee[None], axis=-1) < args.track_tol) \
                    & (d < args.contact_r) & (np.linalg.norm(dee) > 0.15)
                if held.any():
                    demo_ang.append(axis_angles(quat(arm)))
                    demo_gap.append(float(d[held].min()))
            prev_cp, prev_ee = cur_cp, cur_ee
        print(f"  demo ep{e}: {len(demo_ang)} grasp frames so far", flush=True)

    # ---------------- ours ----------------
    our_ang, our_gap = [], []
    rng = np.random.RandomState(0)
    for t in range(args.our_trials):
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        pose = [float(rng.uniform(-0.13, 0.09)), float(rng.uniform(-0.09, 0.09)), 0.73,
                float(rng.uniform(-28, 28)), float(rng.uniform(-28, 28)),
                float(rng.uniform(-35, 35))]
        backend.set_garment_pose(pose)
        for _ in range(16):
            backend.simulate()
        p0 = cps()
        if not np.isfinite(p0).all():
            continue
        order = rng.permutation(n_cp)
        cp, arm = int(order[0]), 0
        for c in order:
            if p0[c, 0] < -8:
                cp, arm = int(c), 0; break
            if p0[c, 0] > 5:
                cp, arm = int(c), 1; break
        tgt = p0[cp] / 100.0 + np.array([0, 0, args.gripper_offset], dtype=np.float32)

        def go(goal, grip, steps):
            e0 = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
            x0 = torch.tensor(e0, dtype=torch.float32).unsqueeze(0)
            for k in range(steps):
                a = (k + 1) / steps
                x = x0.clone()
                x[0, arm] = torch.tensor((1 - a) * e0[arm] + a * goal, dtype=torch.float32)
                backend.set_end_effector_targets(x)
                backend._joint_targets[:, GRIP[arm]] = grip
                backend.simulate()

        go(tgt + np.array([0, 0, 0.07], np.float32), 0.6, 14)
        go(tgt, 0.6, 10)
        our_ang.append(axis_angles(quat(arm)))          # at the moment of closure
        our_gap.append(float(np.linalg.norm(cps()[cp] - ee(arm))))
        print(f"  our trial {t}: gap {our_gap[-1]:.1f} cm", flush=True)

    D = np.array(demo_ang); O = np.array(our_ang)
    print(f"\n=== gripper axis vs world -z, at the grasp moment ===")
    print(f"{'axis':<8}{'demos (n=%d)' % len(D):>22}{'ours (n=%d)' % len(O):>22}{'diff':>9}")
    print("-" * 61)
    res = {}
    for k, nm in enumerate("xyz"):
        if len(D) and len(O):
            dm, ds = D[:, k].mean(), D[:, k].std()
            om, os_ = O[:, k].mean(), O[:, k].std()
            res[nm] = dict(demo_mean=float(dm), demo_sd=float(ds),
                           our_mean=float(om), our_sd=float(os_))
            print(f"{nm:<8}{dm:>14.1f} +-{ds:>5.1f}{om:>14.1f} +-{os_:>5.1f}{om-dm:>9.1f}")
    if len(D) and len(O):
        print(f"\n  gap at grasp: demos {np.mean(demo_gap):.1f} cm   ours {np.mean(our_gap):.1f} cm")
        worst = max(res, key=lambda k: abs(res[k]["our_mean"] - res[k]["demo_mean"]))
        delta = abs(res[worst]["our_mean"] - res[worst]["demo_mean"])
        print(f"\n  largest axis discrepancy: {worst}-axis, {delta:.0f} degrees")
        if delta > 30:
            print("  => our wrist arrives in a materially different orientation. The DLS")
            print("     solver is picking a solution the demonstrator never uses, which")
            print("     is a plausible reason closure never pinches the cloth.")
        else:
            print("  => orientation is NOT the difference. Look elsewhere: descent depth,")
            print("     closure force, or approach speed.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"axes": res, "demo_gap_cm": float(np.mean(demo_gap)) if demo_gap else None,
             "our_gap_cm": float(np.mean(our_gap)) if our_gap else None}, indent=2))
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

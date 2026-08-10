"""Extract joint configurations that provably grasp cloth, from demo replay.

Our scripted grasp holds in 0 of 60 states. The demonstrations hold constantly --
367 verified grasp frames in 4 episodes. The measured difference is not reach but
*posture*: our wrist arrives ~30 degrees from where the demos hold it (x-axis
97.8 vs 129.4 deg to world -z, z-axis 169.2 vs 140.6, y identical -- a rotation
about y), with double the variance, and 3.5 cm further from the cloth.

The arm has 5 controllable joints, so orientation is not freely assignable and
position-only IK is a kinematic necessity. We therefore cannot *command* the
demonstrators' wrist pose. But we can *retrieve* it: every grasp frame carries a
joint configuration that is reachable, correctly oriented, and known to hold
cloth. Indexing those by gripper position gives an IK restricted to the manifold
the demonstrations actually use.

A grasp frame is one where, for some arm:
  * a check-point moves with that gripper (|dp - dee| < track_tol),
  * the gripper is genuinely moving (so co-incidental agreement is excluded),
  * and the check-point is within contact range of the gripper BODY -- which
    must exceed ~9 cm, since the body origin sits that far from the fingertips.

Also records the gripper joint value at each grasp. We have been guessing which
pole of the bimodal demo distribution means "closed"; the grasp frames answer it.
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
p.add_argument("--out", required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--track_tol", type=float, default=0.4, help="cm/step")
p.add_argument("--contact_r", type=float, default=13.0, help="cm, body-to-check-point")
p.add_argument("--min_move", type=float, default=0.15, help="cm/step gripper motion")
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

    cache = Path(args.cache)
    action = np.load(cache / "action.npy")
    state = np.load(cache / "state.npy")
    episode = np.load(cache / "episode.npy")
    ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
    gnames = list(ginfo.keys())

    # ORIGINAL under-damped plant: these are the trajectories' own plant, and
    # replaying elsewhere degrades the fold.
    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device, decimation=3,
        joint_damping={}, skip_images=True))
    n_cp = len(backend.check_points)

    def cps():
        return backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)

    def eep(a):
        return backend._ee_pos_w(a)[0].detach().cpu().numpy() * 100.0

    def eeq(a):
        return backend._ee_quat_w(a)[0].detach().cpu().numpy()

    lib = {k: [] for k in ("q", "ee_pos", "ee_quat", "cp_pos", "arm", "grip_val",
                           "cp_idx", "episode", "frame")}
    t_frames = 0

    for e in range(args.episodes):
        rows = np.where(episode == e)[0]
        if not len(rows):
            continue
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        gi, li = e // args.eps_per_garment, e % args.eps_per_garment
        if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
            backend.set_garment_pose(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
            for _ in range(5):
                backend.simulate()

        prev_cp = cps()
        prev_ee = [eep(0), eep(1)]
        for t, a_i in enumerate(action[rows]):
            backend.set_joint_targets(torch.as_tensor(a_i, dtype=torch.float32))
            backend.simulate()
            cur_cp = cps()
            cur_ee = [eep(0), eep(1)]
            if not np.isfinite(cur_cp).all():
                prev_cp, prev_ee = cur_cp, cur_ee
                continue
            for arm in (0, 1):
                dee = cur_ee[arm] - prev_ee[arm]
                if np.linalg.norm(dee) < args.min_move:
                    continue
                dcp = cur_cp - prev_cp
                d = np.linalg.norm(cur_cp - cur_ee[arm], axis=-1)
                held = (np.linalg.norm(dcp - dee[None], axis=-1) < args.track_tol) \
                    & (d < args.contact_r)
                if not held.any():
                    continue
                cp_i = int(np.where(held, d, np.inf).argmin())   # the closest held point
                lib["q"].append(state[rows[t]].copy())
                lib["ee_pos"].append(cur_ee[arm].copy())
                lib["ee_quat"].append(eeq(arm).copy())
                lib["cp_pos"].append(cur_cp[cp_i].copy())
                lib["arm"].append(arm)
                lib["grip_val"].append(float(state[rows[t]][5 if arm == 0 else 11]))
                lib["cp_idx"].append(cp_i)
                lib["episode"].append(e)
                lib["frame"].append(int(t))
                t_frames += 1
            prev_cp, prev_ee = cur_cp, cur_ee
        print(f"  ep{e:>3}: {t_frames} grasp frames total", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.asarray(v) for k, v in lib.items()}
    np.savez(out / "grasp_library.npz", **arrs)

    n = len(arrs["q"])
    print(f"\n[done] {n} verified grasp configurations -> {out/'grasp_library.npz'}")
    if n:
        gv = arrs["grip_val"]
        arm = arrs["arm"]
        gap = np.linalg.norm(arrs["cp_pos"] - arrs["ee_pos"], axis=-1)
        print(f"\n  gripper joint value while grasping:")
        print(f"    mean {gv.mean():+.4f}  median {np.median(gv):+.4f}  "
              f"p10 {np.percentile(gv,10):+.4f}  p90 {np.percentile(gv,90):+.4f}")
        print(f"    -> this is what 'closed' means; we had been guessing between "
              f"the demo distribution's two poles")
        print(f"\n  body-to-grasped-point gap: mean {gap.mean():.1f}  "
              f"p10 {np.percentile(gap,10):.1f}  p90 {np.percentile(gap,90):.1f} cm")
        for a in (0, 1):
            m = arm == a
            if m.sum():
                P_ = arrs["ee_pos"][m]
                print(f"  arm{a}: n={int(m.sum()):>5}  gripper x {P_[:,0].min():.0f}..{P_[:,0].max():.0f}"
                      f"  y {P_[:,1].min():.0f}..{P_[:,1].max():.0f}"
                      f"  z {P_[:,2].min():.0f}..{P_[:,2].max():.0f} cm")
        print("\n  This is the reachable-and-holding manifold. Retrieval against it")
        print("  gives position AND orientation together, with no IK null-space lottery.")
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

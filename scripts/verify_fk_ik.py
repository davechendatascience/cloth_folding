"""Verify FK/IK against the live simulator before trusting Cartesian control.

Three failure modes this is designed to catch, all of which produce motion that
looks plausible while being wrong:

1. **Frame mismatch.** The IK solver targets the URDF frame
   ``gripper_frame_link``; the simulator reports ``body_pos_w`` for bodies named
   ``gripper`` / ``jaw``. If these are offset, every Cartesian command is
   biased by a constant vector.
2. **Wrong base pose.** ``solve_bimanual_ik_simple`` defaults to base poses
   ``[1.15,-2.3,0.5]`` / ``[1.65,-2.3,0.5]``, but ``garment_bi_cfg_v2`` places
   the arms at ``(-0.23,-0.25,0.5)`` / ``(0.23,-0.25,0.5)`` with a 180-degree
   z-rotation. Using the defaults silently solves in the wrong frame.
3. **5-DOF reachability.** The arm has 5 controllable joints, so orientation is
   not free. Position-only IK can still fail or return large joint jumps.

Run from the lehome-challenge directory with its venv.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--samples", type=int, default=8)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    import gymnasium as gym  # noqa: E402
    import lehome.tasks  # noqa: F401,E402
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg  # noqa: E402
    from lehome.utils.bimanual_ik_solver import BimanualIKSolver  # noqa: E402

    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device

    env = gym.make("LeHome-BiSO101-Direct-Garment-v2", cfg=cfg).unwrapped
    env.initialize_obs()
    for _ in range(3):
        env.step(torch.zeros(1, 12, device=env.device))

    # ---- base poses straight from the cfg, not the solver's defaults -------
    lp = cfg.left_robot.init_state.pos
    lq = cfg.left_robot.init_state.rot
    rp = cfg.right_robot.init_state.pos
    rq = cfg.right_robot.init_state.rot
    print(f"\n[cfg] left  base pos={lp} quat(wxyz)={lq}")
    print(f"[cfg] right base pos={rp} quat(wxyz)={rq}")

    solver = BimanualIKSolver(
        urdf_path="Assets/robots/so101_new_calib.urdf",
        left_base_pose=(list(lp), list(lq)),
        right_base_pose=(list(rp), list(rq)),
    )

    body_names = env.left_arm.data.body_names
    print(f"\n[sim] bodies: {body_names}")

    def quat_to_R(q_wxyz):
        w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def fk_world(q_rad, arm):
        """FK in world frame.

        RobotKinematics.forward_kinematics takes **degrees** (it applies
        np.deg2rad internally) while Isaac Lab reports joint_pos in radians.
        Passing radians straight through is wrong by a factor of 57.3 and still
        returns a plausible-looking pose -- convert explicitly.
        """
        q_deg = np.degrees(np.asarray(q_rad, dtype=np.float64)[:5])
        T = solver.solver.forward_kinematics(q_deg)
        local = np.asarray(T)[:3, 3]
        base_pos = np.array(lp if arm == "left" else rp, dtype=np.float64)
        base_quat = np.array(lq if arm == "left" else rq, dtype=np.float64)
        return base_pos + quat_to_R(base_quat) @ local, local

    # ---- 1. which sim body does the URDF target frame correspond to? ------
    print("\n=== 1. FK frame identification ===")
    for arm, art in (("left", env.left_arm), ("right", env.right_arm)):
        q = art.data.joint_pos[0].detach().cpu().numpy()
        fkw, local = fk_world(q, arm)

        print(f"\n  [{arm}] joints(rad)={np.round(q, 3)}")
        print(f"  [{arm}] FK(base frame)={np.round(local, 4)}  ->  world={np.round(fkw, 4)}")
        best = (None, 1e9)
        for bi, bn in enumerate(body_names):
            bp = art.data.body_pos_w[0, bi].detach().cpu().numpy()
            err = float(np.linalg.norm(bp - fkw))
            if err < best[1]:
                best = (bn, err)
            print(f"      body {bn:<10s} pos={np.round(bp, 4)}  |FK-body|={err:.4f} m")
        print(f"  [{arm}] closest sim body to URDF gripper_frame_link: "
              f"{best[0]} ({best[1]:.4f} m)")

    # ---- 2. IK round-trip: pos -> joints -> FK -> pos ----------------------
    print("\n=== 2. IK round-trip (position only) ===")
    rng = np.random.RandomState(0)
    for arm, art in (("left", env.left_arm), ("right", env.right_arm)):
        gi = body_names.index("gripper")
        ee0 = art.data.body_pos_w[0, gi].detach().cpu().numpy().astype(np.float64)
        q0 = art.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
        print(f"\n  [{arm}] current gripper world pos = {np.round(ee0, 4)}")

        ok, fails, errs = 0, 0, []
        for k in range(args.samples):
            delta = rng.uniform(-0.05, 0.05, size=3)
            target = ee0 + delta
            sol = solver.solve_ik(target, arm=arm, initial_joints=q0, state_unit="rad")
            if sol is None:
                fails += 1
                print(f"      sample {k}: target={np.round(target,4)} -> IK FAILED")
                continue
            achieved, _ = fk_world(np.asarray(sol, dtype=np.float64), arm)
            err = float(np.linalg.norm(achieved - target))
            errs.append(err)
            ok += 1
            print(f"      sample {k}: |target-FK(IK(target))| = {err:.5f} m  "
                  f"dq_max={np.abs(np.asarray(sol)[:5]-q0[:5]).max():.3f} rad")
        if errs:
            print(f"  [{arm}] round-trip error: mean={np.mean(errs):.5f}  max={np.max(errs):.5f}  "
                  f"ok={ok}/{args.samples} failed={fails}")

    # ---- 3. closed loop: command IK joints, step sim, measure -------------
    print("\n=== 3. Closed-loop: apply IK joints and step the simulator ===")
    gi = body_names.index("gripper")
    ee0 = env.left_arm.data.body_pos_w[0, gi].detach().cpu().numpy().astype(np.float64)
    target = ee0 + np.array([0.0, 0.0, 0.05])
    q0 = env.left_arm.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    sol = solver.solve_ik(target, arm="left", initial_joints=q0, state_unit="rad")
    if sol is None:
        print("  IK failed for a 5 cm +z move -- cannot close the loop.")
    else:
        act = torch.zeros(1, 12, device=env.device)
        act[0, :6] = torch.tensor(np.asarray(sol)[:6], dtype=torch.float32, device=env.device)
        act[0, 6:] = env.right_arm.data.joint_pos[0]
        for i in range(60):
            env.step(act)
        ee1 = env.left_arm.data.body_pos_w[0, gi].detach().cpu().numpy().astype(np.float64)
        print(f"  start ={np.round(ee0,4)}")
        print(f"  target={np.round(target,4)}")
        print(f"  actual={np.round(ee1,4)}   |target-actual|={np.linalg.norm(ee1-target):.5f} m")
        print(f"  moved ={np.linalg.norm(ee1-ee0):.5f} m (commanded 0.05)")

    env.close()
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
    os._exit(EXIT)  # Kit will not unwind on its own

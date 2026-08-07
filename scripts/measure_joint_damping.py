"""Measure the SO101 joints' actual damping ratio from a step response.

Everything in the damping hierarchy hangs off one unknown: the effective joint
inertia J. The configured gains are known (K=17.8, D=0.60), so given the
measured natural frequency,

    J     = K / omega_n^2
    zeta  = D / (2 sqrt(K J)) = D * omega_n / (2 K)
    D_crit= 2 sqrt(K J)       = 2 K / omega_n

i.e. the whole picture follows from omega_n, which the damped oscillation
period gives directly. If the response shows no overshoot the joint is already
>= critically damped and we report a lower bound instead.

Method: hard-reset the arm to zero, command a constant joint-position step,
log joint_pos every control step, then estimate:

  * overshoot ratio  -> zeta      (zeta = -ln(OS) / sqrt(pi^2 + ln^2(OS)))
  * peak spacing     -> omega_d   -> omega_n = omega_d / sqrt(1 - zeta^2)
  * 2% settling time -> steps the policy issues before the plant settles

Run from lehome-challenge with its venv.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--amp", type=float, default=0.20, help="step amplitude, rad")
p.add_argument("--steps", type=int, default=300)
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

    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = gym.make("LeHome-BiSO101-Direct-Garment-v2", cfg=cfg).unwrapped
    env.initialize_obs()

    arm = env.left_arm
    joint_names = arm.data.joint_names
    K = 17.8
    D = 0.60
    dt = cfg.sim.dt * cfg.decimation
    print(f"\n[cfg] K={K}  D={D}  sim.dt={cfg.sim.dt:.5f}  decimation={cfg.decimation}  "
          f"control dt={dt:.5f}s ({1/dt:.1f} Hz)")
    print(f"[cfg] episode_length_s={cfg.episode_length_s} -> "
          f"{int(cfg.episode_length_s/dt)} control steps/episode")
    print(f"[cfg] J for zeta=1: D^2/(4K) = {D*D/(4*K):.6f} kg m^2\n")

    def analyse(y: np.ndarray, target: float, name: str):
        """Estimate zeta / omega_n from a unit-ish step response."""
        y = np.asarray(y, dtype=np.float64)
        y0 = y[0]
        span = target - y0
        if abs(span) < 1e-9:
            return None
        yn = (y - y0) / span                      # normalised: 0 -> 1
        y_ss = float(np.mean(yn[-30:]))

        # peaks / troughs of the error signal
        e = yn - y_ss
        peaks = [i for i in range(1, len(e) - 1)
                 if abs(e[i]) > abs(e[i - 1]) and abs(e[i]) > abs(e[i + 1]) and abs(e[i]) > 1e-4]

        overshoot = float(np.max(yn) - y_ss) if np.max(yn) > y_ss + 1e-4 else 0.0
        result = {"name": name, "y_ss": y_ss, "overshoot": overshoot}

        if overshoot > 1e-4 and len(peaks) >= 2:
            OS = overshoot / max(abs(y_ss), 1e-9)
            OS = min(max(OS, 1e-6), 0.999)
            lnOS = math.log(OS)
            zeta = -lnOS / math.sqrt(math.pi**2 + lnOS**2)
            # damped period from successive same-sign peak spacing
            period_steps = 2.0 * float(np.mean(np.diff(peaks[:4]))) if len(peaks) >= 2 else None
            if period_steps and period_steps > 0:
                omega_d = 2 * math.pi / (period_steps * dt)
                omega_n = omega_d / math.sqrt(max(1 - zeta**2, 1e-9))
            else:
                omega_n = None
            result.update(zeta=zeta, omega_n=omega_n, damped=False)
        else:
            result.update(zeta=None, omega_n=None, damped=True)

        # 2% settling
        band = 0.02
        settle = len(yn)
        for i in range(len(yn) - 1, -1, -1):
            if abs(yn[i] - y_ss) > band:
                settle = i + 1
                break
        result["settle_steps"] = settle
        result["settle_s"] = settle * dt
        return result

    print("=" * 78)
    print(f"{'joint':<14}{'overshoot':>10}{'zeta':>8}{'omega_n':>10}{'J':>11}"
          f"{'D_crit':>9}{'settle':>9}")
    print("=" * 78)

    rows = []
    for ji, jname in enumerate(joint_names):
        if jname == "gripper":
            continue
        # hard reset both arms to zero
        z = torch.zeros_like(arm.data.joint_pos)
        arm.write_joint_state_to_sim(z, torch.zeros_like(z))
        env.right_arm.write_joint_state_to_sim(
            torch.zeros_like(env.right_arm.data.joint_pos),
            torch.zeros_like(env.right_arm.data.joint_pos),
        )
        for _ in range(40):
            env.step(torch.zeros(1, 12, device=env.device))

        act = torch.zeros(1, 12, device=env.device)
        act[0, ji] = args.amp
        traj = []
        for _ in range(args.steps):
            env.step(act)
            traj.append(float(arm.data.joint_pos[0, ji]))

        r = analyse(np.array(traj), args.amp, jname)
        if r is None:
            continue
        if r["omega_n"]:
            J = K / r["omega_n"] ** 2
            Dc = 2 * K / r["omega_n"]
            rows.append((jname, r["zeta"], r["omega_n"], J, Dc, r["settle_steps"]))
            print(f"{jname:<14}{r['overshoot']:>10.4f}{r['zeta']:>8.3f}"
                  f"{r['omega_n']:>10.2f}{J:>11.5f}{Dc:>9.3f}{r['settle_steps']:>9d}")
        else:
            rows.append((jname, None, None, None, None, r["settle_steps"]))
            print(f"{jname:<14}{r['overshoot']:>10.4f}{'>=1':>8}{'n/a':>10}"
                  f"{'n/a':>11}{'n/a':>9}{r['settle_steps']:>9d}")

    print("=" * 78)
    meas = [r for r in rows if r[1] is not None]
    if meas:
        zmean = float(np.mean([r[1] for r in meas]))
        dmax = float(np.max([r[4] for r in meas]))
        smax = int(np.max([r[5] for r in rows]))
        print(f"\nmean measured zeta      : {zmean:.3f}   "
              f"({'UNDER-damped' if zmean < 0.95 else 'critical or over'})")
        print(f"D for zeta=1 (worst jnt): {dmax:.3f}   (configured D = {D})")
        print(f"worst 2% settling       : {smax} control steps = {smax*dt:.3f} s")
        print(f"\n=> with decimation={cfg.decimation}, the policy issues ~{smax} "
              f"commands before the plant settles.")
        target_dec = max(1, int(round(smax)))
        print(f"=> one action per settling time needs decimation ~{target_dec} "
              f"({1/(dt*target_dec):.1f} Hz policy rate)")
    else:
        print("\nAll joints non-oscillatory: already >= critically damped.")

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
    os._exit(EXIT)

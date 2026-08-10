"""Which candidate actions have outcomes invariant to cloth state?

A skill is an action whose result does not depend on wrinkle-level detail. That
is what makes quasi-static primitives work where continuous control does not, and
it is measurable directly: run each primitive from many randomised cloth states
and look at the spread of its outcome.

  low variance   -> a genuine skill; usable open-loop, needs little perception
  high variance  -> cloth state matters here; this is where perception earns its
                    cost and a closed-loop controller is justified

The output decides the skill library empirically, and it maps where perception is
actually needed -- rather than assuming it is needed everywhere, which is the
assumption that cost the behaviour-cloning thread.

**The obvious way to fool this experiment** is a primitive that does nothing: it
will have beautifully low outcome variance. So displacement and delta-J are
reported alongside, and a no-op primitive is included as the floor. Low variance
with near-zero displacement is inactivity, not skill.

Critical damping throughout: an under-damped plant rings, so the same primitive
from the same state lands differently and outcome variance would measure the
controller rather than the cloth. We are free to choose the clean plant because
this is generated data.

Two constants that are not free parameters:
  GRIPPER_OFFSET  the gripper BODY origin sits ~9 cm from the contact point.
                  Commanding the body to a check-point drives the fingertips
                  through the cloth into the table; the IK then saturates against
                  collision. Calibrated from 4,926 demo frames where cloth tracks
                  a gripper (body-to-grasped-point 7.3-9.8 cm).
  CONTACT_R       contact radius measured body-to-check-point, so it must exceed
                  that offset. The demos never bring the body within 7.3 cm while
                  manipulating the cloth perfectly well.
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
p.add_argument("--states", type=int, default=40, help="randomised cloth states per primitive")
p.add_argument("--settle", type=int, default=16)
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--gripper_offset", type=float, default=0.09)
p.add_argument("--contact_r", type=float, default=13.0)
p.add_argument("--track_tol", type=float, default=0.4)
p.add_argument("--save_every", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

# No images needed: this measures outcomes from privileged state only, and
# rendering is ~3x the per-step cost.
launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
        _install_proprio_only_observations,
    )

    rng = np.random.RandomState(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = IsaacGarmentCfg(garment_name=args.garment, device=args.device,
                          decimation=args.decimation, skip_images=True)
    backend = IsaacGarmentBackend(cfg)
    n_cp = len(backend.check_points)
    GRIP = (5, 11)
    print(f"[env] garment={backend.garment_type} n_cp={n_cp} (critical damping, no render)",
          flush=True)

    def cps():
        return backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)

    def ees():
        return np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)]) * 100.0

    def move_to(goal_m, arm, grip, steps):
        """Drive one gripper toward a world target, returning per-step tracking."""
        ee0 = np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)])
        x_cmd = torch.tensor(ee0, dtype=torch.float32).unsqueeze(0)
        track = []
        prev_cp, prev_ee = cps(), ees()
        for k in range(steps):
            a = (k + 1) / steps
            x = x_cmd.clone()
            x[0, arm] = torch.tensor((1 - a) * ee0[arm] + a * goal_m, dtype=torch.float32)
            backend.set_end_effector_targets(x)
            backend._joint_targets[:, GRIP[arm]] = grip
            backend.simulate()
            cur_cp, cur_ee = cps(), ees()
            d = np.linalg.norm(cur_cp - cur_ee[arm], axis=-1)
            dcp = cur_cp - prev_cp
            dee = cur_ee[arm] - prev_ee[arm]
            held = (np.linalg.norm(dcp - dee[None], axis=-1) < args.track_tol) & (d < args.contact_r)
            track.append(held)
            prev_cp, prev_ee = cur_cp, cur_ee
        return np.array(track)          # (steps, n_cp)

    # ---- the candidate primitives -------------------------------------------
    # Each takes the settled cloth state and returns (name, executor). Executors
    # return the per-step tracking array over the *outcome* phase.
    def prim_noop(p0, rng_):
        """Floor. Low outcome variance here means inactivity, not skill."""
        def run():
            for _ in range(40):
                backend.simulate()
            return np.zeros((1, n_cp), dtype=bool), -1
        return run

    def _reachable_cp(p0):
        """Pick a check-point and the arm whose side it is on.

        Measured reach: arm0 spans x -48..-10 cm, arm1 spans x 7..44 cm, so the
        band between them is unreachable and the choice of arm is not free.
        """
        order = rng.permutation(n_cp)
        for c in order:
            x = p0[c, 0]
            if x < -8:
                return int(c), 0
            if x > 5:
                return int(c), 1
        return int(order[0]), (0 if p0[order[0], 0] < 0 else 1)

    def prim_lift(p0, rng_):
        """Grasp a check-point and lift straight up."""
        cp, arm = _reachable_cp(p0)
        tgt = p0[cp] / 100.0 + np.array([0, 0, args.gripper_offset], dtype=np.float32)
        def run():
            move_to(tgt + np.array([0, 0, 0.07], np.float32), arm, 0.6, 14)
            move_to(tgt, arm, 0.6, 10)
            for _ in range(5):
                backend._joint_targets[:, GRIP[arm]] = -0.17
                backend.simulate()
            return move_to(tgt + np.array([0, 0, 0.08], np.float32), arm, -0.17, 24), cp
        return run

    def prim_drag(p0, rng_):
        """Grasp and drag laterally, keeping close to the table."""
        cp, arm = _reachable_cp(p0)
        tgt = p0[cp] / 100.0 + np.array([0, 0, args.gripper_offset], dtype=np.float32)
        dx = float(rng_.uniform(-0.08, 0.08)); dy = float(rng_.uniform(-0.08, 0.08))
        def run():
            move_to(tgt + np.array([0, 0, 0.07], np.float32), arm, 0.6, 14)
            move_to(tgt, arm, 0.6, 10)
            for _ in range(5):
                backend._joint_targets[:, GRIP[arm]] = -0.17
                backend.simulate()
            return move_to(tgt + np.array([dx, dy, 0.01], np.float32), arm, -0.17, 24), cp
        return run

    def prim_fold(p0, rng_):
        """Grasp a check-point and carry it toward the cloth centroid: the
        canonical fold motion, and the one J actually rewards."""
        cp, arm = _reachable_cp(p0)
        centre = p0.mean(axis=0) / 100.0
        tgt = p0[cp] / 100.0 + np.array([0, 0, args.gripper_offset], dtype=np.float32)
        goal = np.array([centre[0], centre[1], tgt[2] + 0.04], dtype=np.float32)
        def run():
            move_to(tgt + np.array([0, 0, 0.07], np.float32), arm, 0.6, 14)
            move_to(tgt, arm, 0.6, 10)
            for _ in range(5):
                backend._joint_targets[:, GRIP[arm]] = -0.17
                backend.simulate()
            return move_to(goal, arm, -0.17, 28), cp
        return run

    def prim_push(p0, rng_):
        """No grasp: sweep the open gripper across the cloth. A useful contrast --
        if pushing is as invariant as grasping, the grasp is not buying anything."""
        cp, arm = _reachable_cp(p0)
        tgt = p0[cp] / 100.0 + np.array([0, 0, args.gripper_offset - 0.005], dtype=np.float32)
        dx = float(rng_.uniform(-0.08, 0.08)); dy = float(rng_.uniform(-0.08, 0.08))
        def run():
            move_to(tgt + np.array([0, 0, 0.07], np.float32), arm, 0.6, 14)
            return move_to(tgt + np.array([dx, dy, 0.0], np.float32), arm, 0.6, 28), cp
        return run

    PRIMS = [("noop", prim_noop), ("lift", prim_lift), ("drag", prim_drag),
             ("fold", prim_fold), ("push", prim_push)]

    rows = []
    t0 = time.time()
    for pi, (name, factory) in enumerate(PRIMS):
        for s in range(args.states):
            # Same seed sequence per primitive => the SAME cloth states are used
            # for every primitive, so differences are the primitive, not the draw.
            r2 = np.random.RandomState(args.seed * 1000 + s)
            backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
            pose = [float(r2.uniform(-0.13, 0.09)), float(r2.uniform(-0.09, 0.09)), 0.73,
                    float(r2.uniform(-28, 28)), float(r2.uniform(-28, 28)),
                    float(r2.uniform(-35, 35))]
            backend.set_garment_pose(pose)
            for _ in range(args.settle):
                backend.simulate()
            p0 = cps()
            if not np.isfinite(p0).all():
                continue
            J0 = float(backend.compute_cloth_error())

            track, cp = factory(p0, r2)()
            p1 = cps()
            J1 = float(backend.compute_cloth_error())
            if not np.isfinite(p1).all():
                continue

            held = track[:, cp] if cp >= 0 and track.shape[1] > cp else np.zeros(1, bool)
            rows.append(dict(
                prim=name, state=s, target_cp=int(cp),
                held_frac=float(held.mean()), held_at_end=float(held[-3:].mean() > 0.5),
                disp_cm=float(np.linalg.norm(p1[cp] - p0[cp])) if cp >= 0 else 0.0,
                cloth_move_cm=float(np.linalg.norm(p1 - p0, axis=-1).mean()),
                J0=J0, J1=J1, dJ=J1 - J0,
            ))
        el = time.time() - t0
        n_done = (pi + 1) * args.states
        print(f"  [{name}] {args.states} states done  "
              f"({el/60:.1f} min elapsed, ~{el/max(n_done,1)*(len(PRIMS)*args.states-n_done)/60:.0f} min left)",
              flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))

    (out / "rows.json").write_text(json.dumps(rows, indent=1))

    # ---- the profile ---------------------------------------------------------
    print(f"\n{'primitive':<8}{'n':>4}{'held%':>8}{'disp cm':>10}{'sd':>8}"
          f"{'cloth cm':>10}{'dJ':>9}{'sd(dJ)':>9}{'CV':>7}")
    print("-" * 73)
    summary = {}
    for name, _ in PRIMS:
        r = [x for x in rows if x["prim"] == name]
        if not r:
            continue
        disp = np.array([x["disp_cm"] for x in r])
        dJ = np.array([x["dJ"] for x in r])
        cm = np.array([x["cloth_move_cm"] for x in r])
        held = np.array([x["held_at_end"] for x in r])
        # Coefficient of variation on dJ: spread relative to effect. Low CV with
        # a real effect is the definition of an invariant skill.
        cv = float(dJ.std() / max(abs(dJ.mean()), 1e-6))
        summary[name] = dict(n=len(r), held=float(held.mean()), disp=float(disp.mean()),
                             disp_sd=float(disp.std()), cloth=float(cm.mean()),
                             dJ=float(dJ.mean()), dJ_sd=float(dJ.std()), cv=cv)
        print(f"{name:<8}{len(r):>4}{held.mean()*100:>7.0f}%{disp.mean():>10.2f}"
              f"{disp.std():>8.2f}{cm.mean():>10.2f}{dJ.mean():>9.3f}{dJ.std():>9.3f}{cv:>7.2f}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== reading this ===")
    noop = summary.get("noop", {})
    print(f"  noop moves the cloth {noop.get('cloth', float('nan')):.2f} cm and shifts J by "
          f"{noop.get('dJ', float('nan')):+.3f} -- that is the floor. Any primitive near it")
    print("  is inactive, however low its variance.")
    print("  A skill = large |dJ| or displacement AND low CV. High CV means cloth")
    print("  state decides the outcome, so perception is required for that action.")
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

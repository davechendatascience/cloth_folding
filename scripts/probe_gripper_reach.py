"""Do the grippers ever reach the cloth? LEHOME_PROBE_REACH=1.

Replaying the exact recorded joint trajectory does not fold the garment here:
only 2.23% of pixels change against the recording's 53.85%, and the garment's
footprint holds flat for all 315 steps. The task tolerates 7.2-9.9 cm on
check-point distances, so this is not a precision failure -- the cloth is barely
displaced at all. The obvious question is whether the hands ever arrive.

Logs, per step, the distance from each gripper body to the **nearest cloth
particle** (all 14,544, not just the six check-points -- a gripper can be on the
fabric while far from every check-point).

Two calibrations matter when reading the output:

* the gripper *body* origin sits about **9 cm** from the actual contact point,
  measured over 4,926 demo frames where cloth tracked a gripper. So a body-to-
  cloth distance near 0.09 m means touching; near zero means the fingers are
  through the fabric.
* LeHome uses no particle attachment. Holding depends entirely on
  `adhesion 0.1` / `friction 0.5`, so proximity is necessary but not sufficient.

Hooks `stabilize_garment_after_reset` to capture the live env once per episode,
then `get_object_particle_position` for a per-step tick -- that is called every
step now the success-checker throttle is repaired.
"""

from __future__ import annotations

import os

import numpy as np

_ENV = {"env": None}


def install() -> bool:
    if not os.environ.get("LEHOME_PROBE_REACH"):
        return False

    import scripts.utils.common as common
    import lehome.utils.success_checker_chanllege as sc

    orig_stab = common.stabilize_garment_after_reset

    def stab(env, args, num_steps: int = 20):
        _ENV["env"] = env
        return orig_stab(env, args, num_steps)

    common.stabilize_garment_after_reset = stab

    orig_pos = sc.get_object_particle_position
    st = {"n": 0, "best": [9e9, 9e9], "hist": []}

    def wrapped(particle_object, index_list):
        out = orig_pos(particle_object, index_list)
        st["n"] += 1
        env = _ENV["env"]
        if env is None:
            return out
        try:
            pts = np.asarray(particle_object.get_current_mesh_points()[0],
                             dtype=float).reshape(-1, 3)
            ds = []
            for arm in (env.left_arm, env.right_arm):
                names = arm.data.body_names if hasattr(arm.data, "body_names") else arm.body_names
                gi = list(names).index("gripper")
                ee = arm.data.body_pos_w[0, gi, :].detach().cpu().numpy().astype(float)
                d = float(np.min(np.linalg.norm(pts - ee, axis=1)))
                ds.append(d)
            st["best"] = [min(st["best"][i], ds[i]) for i in (0, 1)]
            st["hist"].append(ds)
            with open(os.environ.get("LEHOME_REACH_OUT", "/tmp/reach.tsv"), "a") as fh:
                fh.write(f"{st['n']}\t{ds[0]:.5f}\t{ds[1]:.5f}\n")
            if st["n"] % 30 == 1:
                print(f"[reach {st['n']:04d}] L {ds[0]:.3f} m   R {ds[1]:.3f} m   "
                      f"(min so far L {st['best'][0]:.3f} R {st['best'][1]:.3f})",
                      flush=True)
        except Exception as e:
            if st["n"] % 200 == 1:
                print(f"[reach] failed: {type(e).__name__}: {e}", flush=True)
        return out

    sc.get_object_particle_position = wrapped

    import atexit

    def report():
        h = np.array(st["hist"]) if st["hist"] else None
        if h is None or not len(h):
            return
        print("\n[reach] ==== summary ====", flush=True)
        for i, side in enumerate(("left", "right")):
            col = h[:, i]
            print(f"[reach] {side:5}  min {col.min():.3f} m   median {np.median(col):.3f} m   "
                  f"steps within 9cm(contact): {(col <= 0.09).sum()}/{len(col)}   "
                  f"within 15cm: {(col <= 0.15).sum()}/{len(col)}", flush=True)
        print("[reach] gripper body origin sits ~9 cm from the contact point "
              "(calibrated on 4,926 demo frames)", flush=True)

    atexit.register(report)
    print("[probe_gripper_reach] ACTIVE", flush=True)
    return True

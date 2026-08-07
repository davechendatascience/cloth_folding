"""Measure real simulator throughput, to size any on-policy RL plan.

At num_envs=1 the only question that matters is: how many policy steps per
second, and therefore how many wall-clock days for a realistic sample budget?

Breaks the cost down by component, because the remedy differs:
  * physics-dominated  -> lower decimation, or accept fewer samples
  * render-dominated   -> fewer/smaller cameras (LeHome ships 3x 480x640)
  * J-dominated        -> cache particle reads

Measured at two decimations so the physics and per-step fixed costs separate:
  t(dec) = t_fixed + dec * t_physics_substep
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--steps", type=int, default=60)
p.add_argument("--warmup", type=int, default=8)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
results = {}
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg,
        IsaacGarmentBackend,
    )

    def bench(decimation: int):
        t_build0 = time.time()
        cfg = IsaacGarmentCfg(
            garment_name=args.garment, device=args.device, decimation=decimation
        )
        backend = IsaacGarmentBackend(cfg)
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        build_s = time.time() - t_build0

        target = backend.get_end_effector_positions().clone()
        for _ in range(args.warmup):
            backend.set_end_effector_targets(target)
            backend.simulate()

        acc = {"ik": 0.0, "sim": 0.0, "img": 0.0, "prop": 0.0, "J": 0.0}
        t0 = time.time()
        for i in range(args.steps):
            # tiny wander so IK/physics do real work
            target[0, :, 2] += 0.002 * (1 if (i // 10) % 2 == 0 else -1)

            t = time.time(); backend.set_end_effector_targets(target); acc["ik"] += time.time() - t
            t = time.time(); backend.simulate();                      acc["sim"] += time.time() - t
            t = time.time(); backend.render_cameras();                acc["img"] += time.time() - t
            t = time.time(); backend.get_proprioception();            acc["prop"] += time.time() - t
            t = time.time(); backend.compute_cloth_error();           acc["J"] += time.time() - t
        total = time.time() - t0

        rate = args.steps / total
        print(f"\n--- decimation={decimation} (control dt={backend.dt:.4f}s) ---")
        print(f"  build                : {build_s:6.1f} s")
        print(f"  policy steps/sec     : {rate:6.2f}")
        print(f"  physics steps/sec    : {rate*decimation:7.1f}")
        print(f"  ms per policy step   : {1000*total/args.steps:7.1f}")
        for k in ("ik", "sim", "img", "prop", "J"):
            ms = 1000 * acc[k] / args.steps
            print(f"    {k:<5s}: {ms:7.2f} ms  ({100*acc[k]/total:5.1f}%)")
        # realtime factor: sim-seconds simulated per wall-second
        print(f"  realtime factor      : {rate*backend.dt:6.3f}x "
              f"({'slower' if rate*backend.dt < 1 else 'faster'} than realtime)")
        backend.close()
        return rate, build_s, acc, total

    r20, *_ = bench(20)
    results[20] = r20
    try:
        r1, *_ = bench(1)
        results[1] = r1
    except Exception as exc:  # second build can be flaky in Kit
        print(f"\n[warn] decimation=1 bench failed: {exc!r}")

    # ---------------- projection ----------------
    print("\n" + "=" * 66)
    print("PROJECTED WALL-CLOCK AT num_envs=1")
    print("=" * 66)
    rate = results[20]
    print(f"  measured: {rate:.2f} policy steps/sec\n")
    print(f"  {'budget (policy steps)':<26}{'hours':>10}{'days':>10}")
    for budget in (1e5, 1e6, 1e7, 1e8):
        h = budget / rate / 3600
        print(f"  {budget:<26.0e}{h:>10.1f}{h/24:>10.1f}")
    print("\n  Typical visual-RL-from-scratch budgets for contact-rich")
    print("  deformable manipulation are 1e7-1e8 steps.")
    ep = 60.0 / (20 / 90)
    print(f"\n  episode = {ep:.0f} policy steps -> "
          f"{rate/ep*3600:.0f} episodes/hour")

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

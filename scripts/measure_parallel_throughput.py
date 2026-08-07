"""How does simulator throughput actually scale with num_envs?

LEVERS.md has carried a "potentially 10-60x" estimate for parallel envs since
before they worked. That number has never been measured, and it is load-bearing:
it is the difference between an 83-day on-policy RL budget and a one-day one,
and it is the justification for paying the cost of replicating the bedroom
geometry per env.

Reports, per num_envs:
  * startup    -- scene build time. replicate_physics=False parses every env
                  separately, so this is expected to grow with N and may be
                  what caps useful N, not the step rate.
  * step time  -- wall seconds per policy step (all envs advanced together)
  * throughput -- env-steps/s = num_envs / step_time. This is the number that
                  matters for sample budgets.
  * efficiency -- throughput / (N x single-env throughput). 1.0 is linear
                  scaling; well under 1.0 means the per-step Python/marshalling
                  overhead or rendering dominates.
  * peak RSS   -- N copies of cloth + robots + 3 TiledCameras per env is not
                  obviously affordable at N=16.

Run one process per N: num_envs is fixed at scene build and Kit cannot rebuild
it in place.

Caveat this script cannot remove: if anything else is using the GPU, absolute
numbers are depressed. Run the same N twice at the ends of a sweep to bound the
drift -- ratios between N measured under equal contention remain meaningful even
when absolutes are not.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--json_out", default="")
p.add_argument("--no_cameras", action="store_true",
               help="launch without camera rendering. Labelling and J-only\n                    replay never read images, so if the fixed per-step cost\n                    is rendering this removes it outright.")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=not args.no_cameras,
                       device=args.device)
simulation_app = launcher.app

EXIT = 0
out = {"num_envs": args.num_envs, "decimation": args.decimation,
       "cameras": not args.no_cameras, "ok": False}
try:
    import torch  # noqa: E402
    import lehome.tasks  # noqa: F401,E402
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg  # noqa: E402
    from lehome.real_damped_project.tasks.parallel_garment_env import (  # noqa: E402
        build_parallel_cfg,
        make_parallel_env_class,
    )

    N = args.num_envs
    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.sim.device = args.device
    cfg.decimation = args.decimation
    cfg = build_parallel_cfg(cfg, num_envs=N, env_spacing=3.0)

    t0 = time.time()
    env = make_parallel_env_class()(cfg)
    # Must accompany enable_cameras=False. LeHome's _get_observations reads
    # top_camera.data.output["rgb"] every step; without a render product that
    # read is an illegal GPU access which poisons the CUDA context and then
    # surfaces as a GpuParticleClothView error, pointing at the wrong thing.
    env.skip_images = args.no_cameras
    env.initialize_obs()
    startup = time.time() - t0
    out["startup_s"] = round(startup, 2)
    view = getattr(env, "cloth_view", None)
    out["view_count"] = int(getattr(view, "count", 0)) if view is not None else 0
    print(f"[N={N}] startup {startup:.1f}s  view.count={out['view_count']}")

    # A view.count below N means the envs are not all simulating, and any
    # throughput measured here would be for fewer cloths than claimed.
    if out["view_count"] != N:
        print(f"[N={N}] ABORT: view.count={out['view_count']} != num_envs={N}; "
              "throughput would be measured on the wrong number of cloths.")
        EXIT = 20
        raise SystemExit

    act = torch.zeros(N, 12, device=env.device)
    for _ in range(args.warmup):
        env.step(act)

    # Per-step times, not just a total: the spread tells us whether a slow mean
    # is a steady cost or an occasional stall (e.g. a render or a sync).
    times = []
    for _ in range(args.steps):
        t = time.time()
        env.step(act)
        times.append(time.time() - t)
    times = np.array(times)

    # Particle-read cost, measured separately -- it is O(N) host-side work per
    # step and a candidate next bottleneck once N is large.
    t = time.time()
    for _ in range(10):
        _ = env.particle_positions()
    read_s = (time.time() - t) / 10

    step_s = float(np.median(times))
    out.update(
        step_s_median=round(step_s, 4),
        step_s_mean=round(float(times.mean()), 4),
        step_s_p90=round(float(np.percentile(times, 90)), 4),
        throughput_env_steps_s=round(N / step_s, 2),
        particle_read_s=round(read_s, 4),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        ok=True,
    )

    print(f"[N={N}] step median {step_s*1000:.1f} ms  (mean {times.mean()*1000:.1f}, "
          f"p90 {np.percentile(times,90)*1000:.1f})")
    print(f"[N={N}] throughput {out['throughput_env_steps_s']:.2f} env-steps/s")
    print(f"[N={N}] particle read {read_s*1000:.1f} ms/call")
    print(f"[N={N}] peak RSS {out['peak_rss_gb']:.1f} GB")
    print("RESULT " + json.dumps(out))
    env.close()
except SystemExit:
    pass
except Exception:
    import traceback

    traceback.print_exc()
    out["error"] = "exception"
    print("RESULT " + json.dumps(out))
    EXIT = 1
finally:
    if args.json_out:
        try:
            with open(args.json_out, "w") as f:
                json.dump(out, f)
        except Exception:
            pass
    import threading

    def _force(code=EXIT):
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)

    w = threading.Timer(30.0, _force); w.daemon = True; w.start()
    try:
        simulation_app.close()
    except Exception:
        pass
    _force()

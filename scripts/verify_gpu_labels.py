"""Check that GPU-labelled J agrees with CPU-labelled J on the same episode.

GPU and CPU PhysX are different solvers, not merely different hardware, and the
particle-position read takes a different code path on each
(``_cloth_prim_view.get_world_positions()`` vs the CPU transform). If they
disagree, the dataset carries labels from two different physics regimes --
inconsistent in a way that nothing downstream can detect, because every
consumer just reads J.npy.

That would be worse than being slow, so it is worth one episode of compute to
rule out before accepting a 2x speedup on 240 of them.

Replays a specific already-CPU-labelled episode on the GPU and reports the
per-frame difference against the stored reference.
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
p.add_argument("--reference", required=True, help="snapshot of the CPU J labels")
p.add_argument("--episode", type=int, required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--tol", type=float, default=0.10, help="acceptable mean |dJ|")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")  # required on GB10; see memory notes
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg,
        IsaacGarmentBackend,
    )

    cache = Path(args.cache)
    action = np.load(cache / "action.npy")
    episode = np.load(cache / "episode.npy")
    ref = np.load(args.reference)

    rows = np.where(episode == args.episode)[0]
    ref_J = ref[rows]
    if not np.isfinite(ref_J).all():
        raise SystemExit(f"episode {args.episode} is not fully labelled in the reference")
    print(f"[ref] episode {args.episode}: {len(rows)} frames, "
          f"J {ref_J[0]:.3f} -> {ref_J[-1]:.3f} (min {ref_J.min():.3f})")

    ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
    gnames = list(ginfo.keys())

    backend = IsaacGarmentBackend(
        IsaacGarmentCfg(garment_name=args.garment, device=args.device,
                        decimation=args.decimation, joint_damping={})
    )
    backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
    gi, li = args.episode // args.eps_per_garment, args.episode % args.eps_per_garment
    backend.set_garment_pose(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
    for _ in range(5):
        backend.simulate()

    got = np.empty(len(rows), dtype=np.float32)
    for i, a in enumerate(action[rows]):
        backend.set_joint_targets(torch.as_tensor(a, dtype=torch.float32))
        backend.simulate()
        got[i] = float(backend.compute_cloth_error())

    d = np.abs(got - ref_J)
    print(f"\n[gpu] J {got[0]:.3f} -> {got[-1]:.3f} (min {got.min():.3f})")
    print("\n=== agreement ===")
    print(f"  mean |dJ| = {d.mean():.4f}   max |dJ| = {d.max():.4f}")
    print(f"  |dJ| at final frame = {d[-1]:.4f}")
    print(f"  correlation = {np.corrcoef(got, ref_J)[0,1]:.4f}")
    ok = d.mean() < args.tol
    print(f"\n  agree within tol={args.tol}: {ok}")
    if not ok:
        print("  => GPU and CPU physics diverge. Labels from the two paths are NOT\n"
              "     interchangeable; relabel everything with one solver.")
        EXIT = 7
    else:
        print("  => GPU labels are consistent with CPU labels; the speedup is safe to take.")
    backend.close()
except SystemExit:
    raise
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

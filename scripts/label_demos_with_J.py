"""Label every demonstration frame with the Lyapunov functional J.

The demonstrations carry images, joint states and actions -- but no cloth
state, so no J. Without J the imitation loss cannot reference the quantity the
whole design is about, which is exactly why the first BC attempt optimised
action similarity and learned to ignore the cameras (image influence /
proprio influence = 0.11).

This replays each episode's recorded actions through the simulator, with the
garment placed at that episode's recorded pose, and logs J at every step. It
works because replay is faithful: verified 2/3 episodes reach J = 0.000
(runs/reachability.json).

Writes ``J.npy`` alongside the preprocessed cache, aligned frame-for-frame with
``state.npy`` / ``action.npy``, which unlocks:

  * ``w(dJ) = exp(-dJ/beta)``   -- imitate the descending transitions (AWR)
  * ``lambda_j * ||J_hat - J||`` -- force camera -> deformable topology grounding
  * any later offline-RL stage, which needs per-transition reward

Resumable: an existing partial ``J.npy`` plus ``J_done.npy`` is continued
rather than restarted, since a full pass is several hours.
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
p.add_argument("--cache", required=True, help="preprocess.py output dir")
p.add_argument("--dataset", required=True, help="source LeRobot dir (garment_info.json)")
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--with_images", action="store_true",
               help="render cameras (default off: labelling never reads them)")
p.add_argument("--decimation", type=int, default=3)
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--render_interval", type=int, default=0,
               help="0 = render once per policy step (the working default). "
                    "Do NOT set this very high: the TiledCameras still exist "
                    "and Isaac Lab's step path waits on renders that never "
                    "arrive. Measured at 100000: 3 minutes, 0 episodes, GPU at "
                    "0%%, plus a stray zenity dialog. Moderate values untested.")
p.add_argument("--max_episodes", type=int, default=0, help="0 = all")
p.add_argument("--episode_min", type=int, default=0,
               help="Restrict to episodes >= this. Lets a second labeller work "
                    "the far end of the range while a first works up from the "
                    "start, without redoing each other's episodes.")
p.add_argument("--episode_max", type=int, default=10**9, help="Restrict to episodes < this.")
p.add_argument("--out_suffix", default=None,
               help="override the J/J_done filename suffix. Used to re-label\n                    already-done episodes into a separate file for A/B checks\n                    without touching the real labels.")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num_shards", type=int, default=1,
               help="Split episodes across independent processes. One process "
                    "uses ~1.3 cores and 7GB, so a 20-core box runs 6 shards "
                    "comfortably and turns 5.5h into ~1h. Each shard writes its "
                    "own files; merge_J_shards.py combines them.")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher  # noqa: E402

# Labelling replays recorded actions and reads particle positions -- it never
# reads an image. Rendering three 640x480 cameras per step was 3.44x of the
# per-step cost (348.7 -> 101.4 ms at N=1). --with_images restores the old
# behaviour if a future variant of this script needs frames.
launcher = AppLauncher(headless=True, enable_cameras=args.with_images,
                       device=args.device)
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
    n = len(action)

    # resume support -- a full pass is hours
    sfx = args.out_suffix if args.out_suffix is not None else (
        "" if args.num_shards == 1 else f"_shard{args.shard}")
    jf_path, df_path = cache / f"J{sfx}.npy", cache / f"J_done{sfx}.npy"
    J = np.full(n, np.nan, dtype=np.float32)
    done = np.zeros(0, dtype=np.int64)
    if jf_path.exists() and df_path.exists():
        J = np.load(jf_path)
        done = np.load(df_path)
        print(f"[resume] {len(done)} episodes already labelled in this shard")

    ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
    gnames = list(ginfo.keys())

    eps = np.unique(episode)
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    # Interleave rather than block-split, so every shard sees a mix of garments
    # and a crash loses coverage evenly instead of an entire garment variant.
    eps = eps[(eps >= args.episode_min) & (eps < args.episode_max)]
    eps = eps[args.shard :: args.num_shards]
    todo = [e for e in eps if e not in done]
    print(f"[data] {len(eps)} episodes, {len(todo)} to label, {n} frames total")

    # LeHome's original gains: the demonstrations were recorded on that plant,
    # and replaying on our critically-damped one measurably degrades the fold
    # (J_end 2.200 vs 0.211 on the one episode that failed). J labels must
    # describe what the demonstrator actually achieved.
    backend = IsaacGarmentBackend(
        IsaacGarmentCfg(garment_name=args.garment, device=args.device,
                        decimation=args.decimation, joint_damping={},
                        skip_images=not args.with_images,
                        render_interval=args.render_interval or None)
    )
    print(f"[env] dt={backend.dt:.4f}s ({1/backend.dt:.1f} Hz), original damping")

    t0 = time.time()
    for k, e in enumerate(todo):
        rows = np.where(episode == e)[0]
        acts = action[rows]

        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        gi, li = int(e) // args.eps_per_garment, int(e) % args.eps_per_garment
        if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
            backend.set_garment_pose(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
            for _ in range(5):
                backend.simulate()

        for i, a in enumerate(acts):
            backend.set_joint_targets(torch.as_tensor(a, dtype=torch.float32))
            backend.simulate()
            J[rows[i]] = float(backend.compute_cloth_error())

        done = np.append(done, e)
        np.save(jf_path, J)
        np.save(df_path, done)

        span = J[rows]
        rate = (k + 1) / max(time.time() - t0, 1e-9)
        eta = (len(todo) - k - 1) / max(rate, 1e-9) / 60
        print(f"  ep{int(e):>4} n={len(rows):>4}  J {span[0]:6.3f} -> {span[-1]:6.3f} "
              f"min={span.min():6.3f}  | {k+1}/{len(todo)}  ETA {eta:.0f} min", flush=True)

    labelled = int(np.isfinite(J).sum())
    print(f"\n[done] shard {args.shard}: {labelled}/{n} frames -> {jf_path}")
    if labelled:
        fin = J[np.isfinite(J)]
        print(f"  J range [{fin.min():.3f}, {fin.max():.3f}]  mean {fin.mean():.3f}")
        print(f"  frames reaching J=0: {(fin <= 0).sum()} ({(fin<=0).mean()*100:.1f}%)")
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

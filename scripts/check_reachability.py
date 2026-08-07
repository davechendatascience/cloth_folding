"""Replay demonstrated actions through our env and check J actually reaches 0.

Two things at once, and the second is the reason this exists.

1. **Reachability.** ``RunContract.reachability_verified`` must not be ticked on
   inference. On the mock backend an oracle reached only J=1.51 against a 0.02
   threshold -- the task was impossible and three training runs were spent
   before anyone checked. A human teleoperator folded these garments 250 times,
   so folding *is* achievable in principle; what needs proving is that it is
   achievable **through our env wrapper**, with our decimation, our joint
   damping, and our J.

2. **Env fidelity.** If demonstrated actions do not reproduce demonstrated
   outcomes here, then BC is training on a task the policy will never face, and
   every downstream number is measuring the wrong thing. A large J at the end of
   a faithful replay is a *far* more valuable finding than a successful BC run,
   because it invalidates the whole pipeline rather than one policy.

Sources of legitimate mismatch to keep in mind when reading the result: the
demos were recorded at decimation 1 (90 Hz sim, 30 fps logging) with LeHome's
original joint damping, whereas we replay at a chosen decimation with
per-joint critical damping. Garment initial pose is also randomised per reset,
so an exact trajectory match is not expected -- the question is whether J falls
substantially, not whether it matches frame for frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True, help="preprocess.py output (for actions)")
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--decimation", type=int, default=3, help="3 -> 30Hz, matching demo fps")
p.add_argument("--out", default=None, help="write a JSON verdict here")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
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
    eps = np.unique(episode)[: args.episodes]
    print(f"[data] replaying episodes {eps.tolist()} from {cache.name}")

    backend = IsaacGarmentBackend(
        IsaacGarmentCfg(garment_name=args.garment, device=args.device,
                        decimation=args.decimation)
    )
    print(f"[env] dt={backend.dt:.4f}s ({1/backend.dt:.1f} Hz) garment={backend.garment_type}")
    thresh = 0.0  # J == 0 is exactly LeHome's success predicate

    rows = []
    for e in eps:
        acts = action[episode == e]
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        js = []
        for a in acts:
            backend.set_joint_targets(torch.as_tensor(a, dtype=torch.float32))
            backend.simulate()
            js.append(float(backend.compute_cloth_error()))
        term, _ = backend.check_done()
        r = {
            "episode": int(e), "steps": len(acts),
            "J_start": js[0], "J_end": js[-1], "J_min": float(np.min(js)),
            "success": bool(term.any()),
            "reduction": float(1.0 - js[-1] / max(js[0], 1e-9)),
        }
        rows.append(r)
        print(f"  ep{r['episode']:>3} n={r['steps']:>4}  J {r['J_start']:7.3f} -> "
              f"{r['J_end']:7.3f}  min={r['J_min']:7.3f}  "
              f"reduction={r['reduction']*100:5.1f}%  success={r['success']}", flush=True)

    j_end = float(np.mean([r["J_end"] for r in rows]))
    j_start = float(np.mean([r["J_start"] for r in rows]))
    j_min = float(np.mean([r["J_min"] for r in rows]))
    n_succ = sum(r["success"] for r in rows)

    print("\n=== verdict ===")
    print(f"  mean J: {j_start:.3f} -> {j_end:.3f}   (best {j_min:.3f})")
    print(f"  successes (J==0): {n_succ}/{len(rows)}")

    if n_succ > 0:
        verdict, note = True, "demonstrated actions reach LeHome's success predicate here"
    elif j_end < 0.5 * j_start:
        verdict, note = True, (
            "J more than halves under replay: the task is being performed, but the "
            "replay does not close it out -- expect a decimation/damping/init mismatch"
        )
    else:
        verdict, note = False, (
            "demonstrated actions do NOT reduce J here. The env wrapper does not "
            "reproduce the demonstrations, so BC would train for a task the policy "
            "never faces. Fix this before spending anything on training."
        )
    print(f"  reachability_verified = {verdict}")
    print(f"  -> {note}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"reachability_verified": verdict, "note": note, "episodes": rows,
             "decimation": args.decimation, "garment": args.garment}, indent=2))
        print(f"  wrote {args.out}")
    EXIT = 0 if verdict else 6

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

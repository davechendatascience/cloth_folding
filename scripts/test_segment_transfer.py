"""Do demo trajectory segments transfer, or do they only replay?

Progress in the demonstrations is concentrated: 11.9% of frames carry 80% of the
J descent, arriving as ~3 bursts of ~15 frames at consistent phases (~20%, ~45%,
~75% through the episode). Those bursts are the natural skill unit -- much larger
than the 24-step primitives that failed the invariance profile, and matching the
scale at which the demonstrations demonstrably work.

But a segment that only works on the cloth state it was recorded from is not a
skill, it is playback. The distinction is exactly the action-conditioned
abstraction criterion: a skill has a similar outcome from *equivalent* states.

So the test is a cross-transfer:

  SELF   drive episode A to burst k, execute A's own burst k       (control)
  CROSS  drive episode A to burst k, execute EPISODE B's burst k   (the test)

If CROSS reduces J comparably to SELF, the segment is a transferable skill and a
planner can compose these. If only SELF works, we have a playback system and the
"skill" framing does not apply.

FROZEN is included as the floor: hold position for the same number of frames.
Without it, any J change during the burst window could be settling rather than
the action -- which is how the earlier primitive profile nearly read as success.
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
p.add_argument("--pairs", type=int, default=12, help="(A,B) episode pairs to test")
p.add_argument("--burst", type=int, default=0, help="which burst index to test (0-based)")
p.add_argument("--lead", type=int, default=6, help="frames of lead-in before the burst")
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
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
    J = np.load(cache / "J.npy")
    action = np.load(cache / "action.npy")
    episode = np.load(cache / "episode.npy")
    ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
    gnames = list(ginfo.keys())

    def bursts_of(e):
        """Contiguous windows carrying the steepest J descent."""
        rows = np.where(episode == e)[0]
        j = J[rows]
        if not np.isfinite(j).all() or len(j) < 30:
            return None, None
        dj = np.diff(j)
        thr = min(np.percentile(dj, 12), -1e-4)
        idx = np.where(dj <= thr)[0]
        if not len(idx):
            return None, None
        runs, s, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > 5:
                runs.append((s, prev)); s = i
            prev = i
        runs.append((s, prev))
        runs = [(a, b) for a, b in runs if b - a >= 4]
        return rows, runs

    eps = [e for e in np.unique(episode) if np.isfinite(J[episode == e]).all()]
    cand = []
    for e in eps:
        rows, runs = bursts_of(e)
        if runs and len(runs) > args.burst:
            cand.append((int(e), rows, runs[args.burst]))
    print(f"[data] {len(cand)} episodes have a burst #{args.burst}", flush=True)

    # ORIGINAL under-damped plant: these are recorded trajectories.
    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device, decimation=3,
        joint_damping={}, skip_images=True))

    def drive_to(e, rows, upto):
        """Replay episode e from reset up to frame `upto`, return J there."""
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        gi, li = e // args.eps_per_garment, e % args.eps_per_garment
        if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
            backend.set_garment_pose(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
            for _ in range(5):
                backend.simulate()
        for a_i in action[rows[:upto]]:
            backend.set_joint_targets(torch.as_tensor(a_i, dtype=torch.float32))
            backend.simulate()
        return float(backend.compute_cloth_error())

    def run_segment(acts):
        for a_i in acts:
            backend.set_joint_targets(torch.as_tensor(a_i, dtype=torch.float32))
            backend.simulate()
        return float(backend.compute_cloth_error())

    rng = np.random.RandomState(args.seed)
    results = []
    for t in range(args.pairs):
        ia, ib = rng.choice(len(cand), size=2, replace=False)
        eA, rowsA, (sA, gA) = cand[ia]
        eB, rowsB, (sB, gB) = cand[ib]
        startA = max(sA - args.lead, 0)
        segA = action[rowsA[startA:gA + 1]]
        segB = action[rowsB[max(sB - args.lead, 0):gB + 1]]
        n = min(len(segA), len(segB))
        if n < 6:
            continue

        row = {"A": eA, "B": eB, "n": int(n)}
        # SELF: A's own burst
        j0 = drive_to(eA, rowsA, startA)
        row["J0"] = j0
        row["self"] = run_segment(segA[:n]) - j0
        # CROSS: B's burst on A's state
        j0c = drive_to(eA, rowsA, startA)
        row["cross"] = run_segment(segB[:n]) - j0c
        # FROZEN floor: hold A's pose for the same duration
        j0f = drive_to(eA, rowsA, startA)
        hold = np.repeat(action[rowsA[startA]][None], n, axis=0)
        row["frozen"] = run_segment(hold) - j0f
        results.append(row)
        print(f"  pair {t:>3}: A={eA} B={eB} n={n}  "
              f"self {row['self']:+.3f}  cross {row['cross']:+.3f}  frozen {row['frozen']:+.3f}",
              flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))

    if results:
        S = np.array([r["self"] for r in results])
        C = np.array([r["cross"] for r in results])
        F = np.array([r["frozen"] for r in results])
        print(f"\n=== burst #{args.burst}, {len(results)} pairs ===")
        print(f"{'condition':<10}{'mean dJ':>10}{'sd':>9}{'better than frozen':>21}")
        print("-" * 50)
        for nm, v in (("frozen", F), ("self", S), ("cross", C)):
            print(f"{nm:<10}{v.mean():>+10.3f}{v.std():>9.3f}"
                  f"{(v < F).mean()*100 if nm != 'frozen' else 0:>20.0f}%")
        print(f"\n  self vs frozen : {S.mean()-F.mean():+.3f}")
        print(f"  cross vs frozen: {C.mean()-F.mean():+.3f}")
        print(f"  cross retains   {100*(C.mean()-F.mean())/min(S.mean()-F.mean(),-1e-9):.0f}% "
              f"of self's effect" if S.mean() < F.mean() else "")
        print("\n=== verdict ===")
        if S.mean() >= F.mean() - 0.05:
            print("  Even SELF barely beats frozen: the burst windows are not where the")
            print("  action does the work, or replay is too chaotic to reproduce them.")
        elif C.mean() <= F.mean() - 0.5 * (F.mean() - S.mean()):
            print("  CROSS transfers. These segments are skills: another episode's burst")
            print("  reduces J on this cloth state, so a planner can compose them.")
        else:
            print("  CROSS does not transfer. The segments are playback, not skills --")
            print("  they work only on the state they were recorded from.")
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

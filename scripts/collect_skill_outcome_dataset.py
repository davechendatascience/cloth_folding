"""Build the (state, skill, outcome) dataset that a skill-selection policy needs.

The policy this serves does not model cloth physics:

    skill library  = demo progress bursts (~3 per episode, ~15 frames each,
                     carrying 80% of the J descent)
    outcome model  = P(dJ | state, skill)
    policy         = execute the skill predicted to reduce J most

Nothing here requires identifying material parameters, reconstructing the mesh,
or solving grasping from scratch. The bursts already grasp -- they are recorded
demonstration behaviour, the one thing in this project that has never failed a
measurement. What is unknown is *which* burst works from *which* state, and that
is learnable from outcomes alone.

Pilot evidence, 14 cross-transfer pairs: a burst on its own state gives
dJ = -1.68; a foreign burst gives -0.617 and beats frozen in 64% of cases. So
outcomes vary by (state, skill) pair -- which is precisely what makes the
selection problem non-trivial and worth learning.

Each trial records:
  state    check-point positions before the skill (privileged; the teacher
           signal) plus the image, so a vision model can be trained later to
           replace it
  skill    which library segment was executed, and its identifying features
  outcome  dJ, plus whether the cloth was actually held during it

The state is deliberately recorded in BOTH forms. Privileged state makes the
outcome model trainable immediately; the image makes it deployable later. That
is the teacher/student split -- privileged information helps discover the latent
factors, and is not needed at deployment.
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
p.add_argument("--cache", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--out", required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda")
p.add_argument("--trials", type=int, default=400)
p.add_argument("--lead", type=int, default=6)
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--image_size", type=int, default=84)
p.add_argument("--save_every", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
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

    # ---- skill library: the progress bursts ---------------------------------
    def bursts_of(e):
        rows = np.where(episode == e)[0]
        j = J[rows]
        if not np.isfinite(j).all() or len(j) < 30:
            return None, []
        dj = np.diff(j)
        thr = min(np.percentile(dj, 12), -1e-4)
        idx = np.where(dj <= thr)[0]
        if not len(idx):
            return rows, []
        runs, s, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > 5:
                runs.append((s, prev)); s = i
            prev = i
        runs.append((s, prev))
        return rows, [(a, b) for a, b in runs if b - a >= 4]

    LIB = []
    for e in np.unique(episode):
        rows, runs = bursts_of(e)
        if rows is None:
            continue
        for k, (a, b) in enumerate(runs):
            LIB.append(dict(ep=int(e), k=k, start=int(a), end=int(b),
                            rows=rows, dJ_demo=float(J[rows[b]] - J[rows[max(a - args.lead, 0)]])))
    print(f"[lib] {len(LIB)} skill segments from {len(np.unique(episode))} episodes", flush=True)

    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device, decimation=3,
        joint_damping={}, image_size=args.image_size))
    n_cp = len(backend.check_points)
    C, H, W = backend.image_shape

    def cps():
        return backend.check_point_positions_cm().detach().cpu().numpy().reshape(n_cp, 3)

    def ees():
        return np.stack([backend._ee_pos_w(a)[0].detach().cpu().numpy() for a in (0, 1)]) * 100.0

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    IMG = np.memmap(out / "state_images.u8", dtype=np.uint8, mode="w+",
                    shape=(args.trials, C, H, W))
    ST = np.full((args.trials, n_cp, 3), np.nan, dtype=np.float32)
    EE = np.full((args.trials, 2, 3), np.nan, dtype=np.float32)
    SKILL = np.full((args.trials, 4), np.nan, dtype=np.float32)   # ep, k, len, dJ_demo
    OUT = np.full((args.trials, 4), np.nan, dtype=np.float32)     # dJ, J0, J1, held_frac
    rng = np.random.RandomState(args.seed)

    def save(n):
        IMG.flush()
        np.save(out / "state_cp.npy", ST); np.save(out / "state_ee.npy", EE)
        np.save(out / "skill.npy", SKILL); np.save(out / "outcome.npy", OUT)
        (out / "meta.json").write_text(json.dumps({
            "trials_done": int(n), "image_shape": [C, H, W], "n_check_points": n_cp,
            "skill_fields": ["episode", "burst_index", "n_frames", "dJ_in_demo"],
            "outcome_fields": ["dJ", "J_before", "J_after", "held_frac"],
            "library_size": len(LIB),
            "note": "state recorded as BOTH privileged check-points and image: "
                    "privileged trains the outcome model now, image deploys it later",
        }, indent=2))

    t0 = time.time()
    done = 0
    for t in range(args.trials):
        # A state is produced by driving some episode partway -- this samples
        # the distribution of *mid-fold* configurations a policy would actually
        # face, which randomised placement alone does not reach.
        drv = LIB[rng.randint(len(LIB))]
        rows = drv["rows"]
        upto = max(drv["start"] - args.lead, 0)
        e = drv["ep"]
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        gi, li = e // args.eps_per_garment, e % args.eps_per_garment
        if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
            backend.set_garment_pose(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
            for _ in range(5):
                backend.simulate()
        for a_i in action[rows[:upto]]:
            backend.set_joint_targets(torch.as_tensor(a_i, dtype=torch.float32))
            backend.simulate()

        s_cp = cps()
        if not np.isfinite(s_cp).all():
            continue
        J0 = float(backend.compute_cloth_error())
        ST[t] = s_cp
        EE[t] = ees() / 100.0
        IMG[t] = (backend.render_cameras()[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # Execute a skill: 50% its own (the demo continuation), 50% a random
        # other. Mixing both is what makes the outcome model learn selection
        # rather than memorise "the matching one always wins".
        use = drv if rng.rand() < 0.5 else LIB[rng.randint(len(LIB))]
        urows = use["rows"]
        seg = action[urows[max(use["start"] - args.lead, 0): use["end"] + 1]]
        SKILL[t] = [use["ep"], use["k"], len(seg), use["dJ_demo"]]

        prev_cp, prev_ee = cps(), ees()
        held = []
        for a_i in seg:
            backend.set_joint_targets(torch.as_tensor(a_i, dtype=torch.float32))
            backend.simulate()
            cur_cp, cur_ee = cps(), ees()
            hit = False
            for arm in (0, 1):
                dee = cur_ee[arm] - prev_ee[arm]
                if np.linalg.norm(dee) < 0.15:
                    continue
                d = np.linalg.norm(cur_cp - cur_ee[arm], axis=-1)
                if ((np.linalg.norm(cur_cp - prev_cp - dee[None], axis=-1) < 0.4)
                        & (d < 13.0)).any():
                    hit = True
            held.append(hit)
            prev_cp, prev_ee = cur_cp, cur_ee
        J1 = float(backend.compute_cloth_error())
        OUT[t] = [J1 - J0, J0, J1, float(np.mean(held)) if held else 0.0]
        done += 1

        if t % args.save_every == 0 or t == args.trials - 1:
            save(done)
            el = time.time() - t0
            v = OUT[:t + 1, 0]; v = v[np.isfinite(v)]
            print(f"  trial {t:>4}/{args.trials}  dJ mean {v.mean():+.3f}  "
                  f"improving {np.mean(v < -0.3):.0%}  "
                  f"{(t+1)/max(el,1e-9)*60:.1f}/min  ETA {(args.trials-t-1)/max((t+1)/max(el,1e-9),1e-9)/60:.0f} min",
                  flush=True)

    save(done)
    v = OUT[:, 0]; v = v[np.isfinite(v)]
    print(f"\n[done] {done} trials -> {out}")
    print(f"  dJ: mean {v.mean():+.3f}  sd {v.std():.3f}  improving (dJ<-0.3) {np.mean(v<-0.3):.0%}")
    print("  This is the dataset a skill-selection policy learns from: predict dJ")
    print("  from (state, skill), then choose the skill with the best predicted dJ.")
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

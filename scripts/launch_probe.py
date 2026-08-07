"""Does Isaac Sim launch headless on this box, and does the LeHome garment env build?

Run with lehome-challenge's venv, from the lehome-challenge directory (the
garment/particle config paths in GarmentEnvCfg are relative to it).
"""

from __future__ import annotations

import argparse
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--garment", default=None, help="e.g. Top_Long_Unseen_0")
p.add_argument("--device", default="cpu")
p.add_argument("--steps", type=int, default=5)
args = p.parse_args()

t0 = time.time()
print("[1] launching Isaac Sim (headless)...", flush=True)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = app_launcher.app
print(f"[1] app up in {time.time()-t0:.1f}s", flush=True)

# Everything below needs the kit runtime alive (pxr, omni, lehome).
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

print("[2] importing lehome tasks...", flush=True)
import lehome.tasks  # noqa: F401,E402

ids = [k for k in gym.registry if "LeHome" in k]
print(f"[2] registered LeHome envs: {ids}", flush=True)

from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg  # noqa: E402
from lehome.tasks.bedroom.challenge_garment_loader import ChallengeGarmentLoader  # noqa: E402

cfg = GarmentEnvCfg()
loader = ChallengeGarmentLoader(cfg.garment_cfg_base_path)
if args.garment is None:
    print("[3] no --garment given; stopping before env construction.", flush=True)
    simulation_app.close()
    sys.exit(0)

gcfg = loader.load_garment_config(args.garment, cfg.garment_version)
gtype = loader.get_garment_type(args.garment)
print(f"[3] garment={args.garment} type={gtype}", flush=True)
print(f"    check_point={gcfg.get('check_point')}", flush=True)
print(f"    success_distance={gcfg.get('success_distance')}  scale={gcfg.get('scale')}", flush=True)

print(f"[4] building env for garment={args.garment} device={args.device}...", flush=True)
cfg.garment_name = args.garment
cfg.scene.num_envs = 1
cfg.sim.device = args.device

env = gym.make("LeHome-BiSO101-Direct-Garment-v2", cfg=cfg).unwrapped
print(f"[4] env built in {time.time()-t0:.1f}s", flush=True)
print(f"    action_space={env.action_space}  obs_space={env.observation_space}", flush=True)

# LeHome-specific: sets the garment's initial particle positions. DirectRLEnv
# does not call it, and GarmentObject.reset() raises AttributeError without it.
env.initialize_obs()
print("[5] initialize_obs() OK", flush=True)

inner = env
for i in range(args.steps):
    a = torch.zeros(1, 12, device=inner.device)
    obs, rew, term, trunc, info = env.step(a)
    keys = list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
    print(f"    step {i}: rew={float(torch.as_tensor(rew).float().mean()):.4f} "
          f"term={bool(torch.as_tensor(term).any())} trunc={bool(torch.as_tensor(trunc).any())} "
          f"obs={keys}", flush=True)

# Camera shapes, needed to size the policy's image encoder.
for name in ("top_camera", "left_camera", "right_camera"):
    cam = getattr(inner, name, None)
    if cam is not None:
        out = cam.data.output
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        print(f"    {name}: {shapes}", flush=True)

# EE pose source for the Cartesian controller.
print(f"    left_arm bodies : {inner.left_arm.data.body_names}", flush=True)
print(f"    body_pos_w shape: {tuple(inner.left_arm.data.body_pos_w.shape)}", flush=True)
print(f"    joint names     : {inner.left_arm.data.joint_names}", flush=True)

# The thing J needs: particle positions.
obj = inner.object
print(f"[6] garment object: {type(obj).__name__}", flush=True)
print(f"    check_points={getattr(obj,'check_points',None)}", flush=True)
print(f"    success_distance={getattr(obj,'success_distance',None)}", flush=True)
print(f"    init_scale={getattr(obj,'init_scale',None)}", flush=True)
try:
    pts, *rest = obj.get_current_mesh_points()
    print(f"    mesh points: shape={getattr(pts,'shape',None)}  (extra returns: {len(rest)})", flush=True)

    # The exact quantity J consumes: check-points in centimetres.
    import numpy as np
    idx = gcfg["check_point"]
    cp = np.asarray(pts)[idx] * 100.0
    print(f"    check-point positions (cm):\n{cp}", flush=True)

    from lehome_real_damped_shim import evaluate_J  # noqa: F401
except ImportError:
    # Evaluate J inline rather than depending on the project package being
    # importable inside lehome-challenge's venv.
    import itertools
    scaled = [d * float(gcfg["scale"][0]) for d in gcfg["success_distance"]]
    conds = [(0, 4, "le", 0), (2, 3, "le", 1), (1, 5, "le", 2),
             (0, 1, "ge", 3), (4, 5, "ge", 4)]
    total = 0.0
    print(f"    thresholds (scaled, cm): {[round(s,2) for s in scaled]}", flush=True)
    for k, (i, j_, cmp_, slot) in enumerate(conds):
        d = float(np.linalg.norm(cp[i] - cp[j_]))
        t = scaled[slot]
        v = max(0.0, (d - t) if cmp_ == "le" else (t - d))
        total += v
        print(f"      cond{k+1}: d({i},{j_})={d:7.2f} {cmp_} {t:6.2f} -> violation {v:7.2f}", flush=True)
    print(f"    J (cm of violation / 10) = {total/10.0:.4f}   success={total<=0}", flush=True)
except Exception as exc:
    print(f"    get_current_mesh_points FAILED: {exc!r}", flush=True)

env.close()
simulation_app.close()
print(f"[done] total {time.time()-t0:.1f}s", flush=True)

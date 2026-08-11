"""Does the checkpoint reproduce demonstration actions on demonstration frames?

No simulator. This separates the two hypotheses that closed-loop evaluation
cannot tell apart:

* the policy never learned the mapping  -> error here is large
* the policy learned it and the *eval* is still wrong -> error here is small

Reported against baselines that must be beaten for the number to mean anything,
because this project has repeatedly produced impressive-looking regression
metrics from predictors that ignore the observation:

    predict the mean action      what a model that ignores everything achieves
    predict the current state    "don't move" -- at 30 fps this is a strong
                                 baseline and beating it is the real bar

Both are computed on the same frames, so the comparison is exact.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--root", default="/home/edge-host/Documents/GitHub/lehome-challenge/"
                                 "Datasets/example/top_long_merged")
p.add_argument("--repo_id", default="lehome/dataset_challenge_merged")
p.add_argument("--n", type=int, default=120)
p.add_argument("--device", default="cuda")
p.add_argument("--task", default="fold the garment on the table")
args = p.parse_args()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

ds = LeRobotDataset(args.repo_id, root=args.root, revision="main", video_backend="pyav")
policy = SmolVLAPolicy.from_pretrained(args.ckpt).to(args.device).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
print(f"[offline] {args.ckpt}\n[offline] dataset {len(ds)} frames", flush=True)

rng = np.random.RandomState(0)
idx = rng.choice(len(ds), args.n, replace=False)

pred, true, state = [], [], []
for k, i in enumerate(idx):
    s = ds[int(i)]
    obs = {"observation.state": s["observation.state"], "task": args.task}
    for cam in ("top_rgb", "left_rgb", "right_rgb"):
        obs[f"observation.images.{cam}"] = s[f"observation.images.{cam}"]
    if hasattr(policy, "reset"):
        policy.reset()          # chunk queue must not leak across sampled frames
    with torch.inference_mode():
        a = post(policy.select_action(pre(obs)))
    pred.append(a.reshape(-1).float().cpu().numpy())
    true.append(s["action"].numpy().reshape(-1))
    state.append(s["observation.state"].numpy().reshape(-1))
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{len(idx)}", flush=True)

pred, true, state = np.array(pred), np.array(true), np.array(state)
mse = lambda x: float(((true - x) ** 2).mean())  # noqa: E731

print("\n=== action prediction, MSE in rad^2 ===")
print(f"  policy                {mse(pred):.6f}")
print(f"  predict current state {mse(state):.6f}   <- 'do not move'")
print(f"  predict mean action   {mse(np.repeat(true.mean(0, keepdims=True), len(true), 0)):.6f}")
print(f"\n  policy vs state baseline : "
      f"{'BEATS' if mse(pred) < mse(state) else 'LOSES TO'} it "
      f"({mse(state) / max(mse(pred), 1e-12):.2f}x)")
per = ((true - pred) ** 2).mean(axis=0)
names = ["L_pan", "L_lift", "L_elbow", "L_wflex", "L_wroll", "L_grip",
         "R_pan", "R_lift", "R_elbow", "R_wflex", "R_wroll", "R_grip"]
print("\n  per-joint MSE:", "  ".join(f"{n}={v:.4f}" for n, v in zip(names, per)))
print(f"\n  mean |pred - true| = {np.abs(true - pred).mean():.4f} rad")
print(f"  action std in data = {true.std():.4f} rad")

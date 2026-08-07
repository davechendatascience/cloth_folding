"""Does the policy actually use the cameras, or only proprioception?

This is the measurement that diagnosed the first BC run. Camera, spatial
attention, joint control and physics all checked out individually; the policy
simply did not use what the attention computed, so the camera -> deformable
topology -> manipulation chain was severed at one link.

That original diagnosis reported 0.11 from an ad-hoc check whose perturbation
scheme was never recorded. This script measures the *same checkpoint*
(runs/bc_top_long/best.pt) at **0.0445**. The two numbers are not comparable and
0.11 should not be used as a threshold. The baseline that counts is 0.0445,
because it is produced by this code -- compare new checkpoints against that, via
--baseline.

Method: perturb one modality at a time and measure how much the predicted action
moves.

    influence(m) = E || pi(o with m perturbed) - pi(o) ||

Perturbation is scaled per modality by that modality's own standard deviation in
the data, so the two numbers are comparable. Perturbing images by an absolute
epsilon and proprio by the same absolute epsilon would compare nothing -- the
modalities live on different scales, and the ratio would just report that.

Reported:
  image influence   -- action movement under image perturbation
  proprio influence -- action movement under proprio perturbation
  ratio             -- image/proprio. Higher is better; compare to --baseline,
                       not to the historical 0.11.
  j_head R^2        -- if the checkpoint has the auxiliary head, how well J is
                       predicted from vision alone. This is the direct check on
                       whether the aux loss did its job: predicting J *requires*
                       representing check-point geometry, so a high R^2 with a
                       low influence ratio would mean the encoder learned the
                       geometry and the actor still ignored it.

A caveat this cannot escape: influence measures sensitivity, not correctness. A
policy could be highly sensitive to images and still act wrongly. It is a
necessary condition for visual control, not a sufficient one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lehome.real_damped_project.data.dataset import LeHomeDemoDataset, split_episodes
from lehome.real_damped_project.policy.vision_attention_policy import VisionAttentionPolicy

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--device", default="cuda")
p.add_argument("--batches", type=int, default=40)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--seq_len", type=int, default=16)
p.add_argument("--sigma", type=float, default=0.5,
               help="perturbation size in units of each modality's own std")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--baseline", type=float, default=0.0445,
               help="ratio for the pre-fix checkpoint measured with THIS script. "
                    "The historical 0.11 came from a different method.")
p.add_argument("--json_out", default="")
args = p.parse_args()

device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
torch.manual_seed(args.seed)

ck = torch.load(args.ckpt, map_location=device, weights_only=False)
# Checkpoints store hyperparameters under "args" and weights under "policy".
cfg = ck.get("args", ck.get("cfg", {}))

_, val_eps = split_episodes(args.cache, cfg.get("val_frac", 0.1), cfg.get("seed", 0))
# The checkpoint may have been trained on a re-split labelled pool; prefer the
# episode list it recorded, so we evaluate on genuinely held-out episodes.
if "val_eps" in ck:
    val_eps = ck["val_eps"]

ds = LeHomeDemoDataset(args.cache, args.seq_len, episodes=val_eps,
                       delta_target=not cfg.get("absolute_target", False))
if ck.get("state_mean") is not None:
    ds.state_mean = np.asarray(ck["state_mean"], dtype=np.float32)
    ds.state_std = np.asarray(ck["state_std"], dtype=np.float32)

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=2, drop_last=True)

# Must mirror train_bc's construction exactly, including squash=False --
# the targets are unbounded deltas, and a mismatched squash would change the
# action scale and therefore both influence numbers.
policy = VisionAttentionPolicy(
    image_channels=ds.image_shape[0], proprio_dim=ds.proprio_dim,
    action_dim=ds.action_dim,
    feature_dim=cfg.get("feature_dim", 256), hidden_dim=cfg.get("hidden_dim", 256),
    squash=False,
    predict_j=cfg.get("lambda_j", 0.0) > 0.0,
).to(device)
policy.load_state_dict(ck.get("policy", ck.get("model", ck.get("state_dict"))))
policy.eval()

img_d, prop_d, j_true, j_pred_all = [], [], [], []

with torch.no_grad():
    for bi, batch in enumerate(dl):
        if bi >= args.batches:
            break
        img, prop = batch[0].to(device), batch[1].to(device)
        J = batch[-1].to(device) if ds.dJ is not None and len(batch) > 3 else None

        want_j = policy.j_head is not None
        if want_j:
            base, _, _, jp = policy.forward_sequence(img, prop, return_j=True)
            j_pred_all.append(jp.flatten().cpu())
            if J is not None:
                j_true.append(J.flatten().cpu())
        else:
            base = policy.forward_sequence(img, prop)[0]

        # Scale each perturbation by that modality's own spread in this batch,
        # so the two influences are on comparable footing.
        i_std = img.std().clamp_min(1e-6)
        p_std = prop.std().clamp_min(1e-6)

        pert_img = policy.forward_sequence(
            img + args.sigma * i_std * torch.randn_like(img), prop)[0]
        pert_prop = policy.forward_sequence(
            img, prop + args.sigma * p_std * torch.randn_like(prop))[0]

        img_d.append((pert_img - base).norm(dim=-1).mean().cpu())
        prop_d.append((pert_prop - base).norm(dim=-1).mean().cpu())

img_inf = float(torch.stack(img_d).mean())
prop_inf = float(torch.stack(prop_d).mean())
ratio = img_inf / prop_inf if prop_inf > 0 else float("inf")

out = {"ckpt": args.ckpt, "image_influence": round(img_inf, 6),
       "proprio_influence": round(prop_inf, 6), "ratio": round(ratio, 4),
       "sigma": args.sigma, "val_episodes": len(val_eps)}

print(f"  image influence   : {img_inf:.6f}")
print(f"  proprio influence : {prop_inf:.6f}")
print(f"  ratio (image/prop): {ratio:.4f}   baseline {args.baseline:.4f} "
      f"({ratio/max(args.baseline,1e-9):.2f}x)")

if j_pred_all and j_true:
    jp = torch.cat(j_pred_all).numpy()
    jt = torch.cat(j_true).numpy()
    m = np.isfinite(jp) & np.isfinite(jt)
    if m.sum() > 10 and jt[m].std() > 0:
        ss_res = float(((jt[m] - jp[m]) ** 2).sum())
        ss_tot = float(((jt[m] - jt[m].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        corr = float(np.corrcoef(jt[m], jp[m])[0, 1])
        out.update(j_r2=round(r2, 4), j_corr=round(corr, 4))
        print(f"  J_hat R^2 (vision only): {r2:.4f}   corr {corr:.4f}")

print()
rel = ratio / max(args.baseline, 1e-9)
out["baseline"] = args.baseline
out["vs_baseline"] = round(rel, 3)
if ratio >= 0.5:
    print("VERDICT: the policy uses the cameras. The severed link is repaired.")
elif rel >= 3.0:
    print(f"VERDICT: {rel:.1f}x the pre-fix baseline -- the visual pathway is")
    print("         materially stronger, but still well below parity with proprio.")
elif rel >= 1.5:
    print(f"VERDICT: {rel:.1f}x baseline. Real but modest; weigh against the")
    print("         J_hat R^2 before trusting it.")
else:
    print("VERDICT: still proprio-dominated. The fix did not take -- do not")
    print("         proceed to RL finetuning on this checkpoint.")

if args.json_out:
    Path(args.json_out).write_text(json.dumps(out, indent=2))

"""Can vision recover the cloth configuration, not just the scalar J?

This decides whether the factored design is buildable:

    vision -> check-point positions   <- THIS SCRIPT
    dJ/dp  -> desired dp              <- analytic, verified differentiable
    dp/du  -> required EE motion      <- the cloth Jacobian, the missing link
    IK     -> joint commands          <- verified 8/8 to 0.0000 m

We already know vision predicts the *scalar* J at R^2 0.87 from the
attention-pooled context alone. Positions are the full-rank version of that
signal and a strictly harder target, so it has to be measured rather than
assumed.

**The output is deliberately not fragmented.** A head emitting 18 independent
coordinates lets each error propagate separately into the J gradient and the
Jacobian, and they compound. It is also the wrong space: J consumes only five
*pairwise distances* between specific check-points, so a common-mode error --
the whole garment estimated 2 cm off -- is invisible to J while dominating a
position-wise loss. One shared trunk emits the whole configuration at once, and
the metrics are reported worst-fragmentation-last:

  1. J(p_hat) vs J(p)   -- end to end, the quantity that actually matters
  2. pairwise distances -- exactly what J consumes
  3. per-point position -- most fragmented, reported but not optimised for

The loss combines position error with the J-relevant distances, so the outputs
are coupled during training rather than only at evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--cp", default="checkpoints_cp.npy", help="recorded positions")
p.add_argument("--garment_type", default="top-long-sleeve")
p.add_argument("--device", default="cuda")
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--feature_dim", type=int, default=256)
p.add_argument("--lambda_dist", type=float, default=1.0,
               help="weight on the pairwise-distance term that couples the outputs")
p.add_argument("--val_frac", type=float, default=0.25)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--json_out", default="")
args = p.parse_args()

torch.manual_seed(args.seed)
dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

cache = Path(args.cache)
meta = json.loads((cache / "meta.json").read_text())
c, h, w = meta["image_shape"]
n = meta["n_frames"]
images = np.memmap(cache / "images.u8", dtype=np.uint8, mode="r", shape=(n, c, h, w))
episode = np.load(cache / "episode.npy")
CP = np.load(cache / args.cp)

ok = np.isfinite(CP).all(axis=(1, 2))
eps = np.array(sorted({int(e) for e in np.unique(episode[ok])
                       if np.isfinite(CP[episode == e]).all()}))
if len(eps) < 4:
    raise SystemExit(f"only {len(eps)} fully-recorded episodes; record more first")

rng = np.random.RandomState(args.seed)
perm = rng.permutation(eps)
n_val = max(1, int(round(len(eps) * args.val_frac)))
val_eps, train_eps = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())
tr_idx = np.where(np.isin(episode, list(train_eps)) & ok)[0]
va_idx = np.where(np.isin(episode, list(val_eps)) & ok)[0]
print(f"[data] {len(eps)} recorded episodes -> train {len(train_eps)} / val {len(val_eps)}")
print(f"[data] frames: train {len(tr_idx)} / val {len(va_idx)}")

P = CP.shape[1]
mu = CP[tr_idx].reshape(-1, 3).mean(0)
sd = CP[tr_idx].reshape(-1, 3).std(0) + 1e-6
print(f"[data] {P} check-points; position mean {mu.round(2)} std {sd.round(2)} (cm)")

from lehome.real_damped_project.math.garment_functional import GARMENT_CONDITIONS  # noqa: E402

pairs = [(i, j) for (i, j, _, _) in GARMENT_CONDITIONS[args.garment_type]]
print(f"[data] J consumes {len(pairs)} pairwise distances: {pairs}")


def pdists(x):  # (B,P,3) -> (B,len(pairs))
    return torch.stack([torch.linalg.vector_norm(x[:, i] - x[:, j], dim=-1)
                        for i, j in pairs], dim=-1)


class ConfigNet(nn.Module):
    """Shared trunk -> whole configuration at once (not 18 separate heads)."""

    def __init__(self, in_ch, feat, n_pts):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, feat, 1),
        )
        self.score = nn.Conv2d(feat, 1, 1)
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(True),
                                  nn.Linear(256, n_pts * 3))
        self.n_pts = n_pts

    def forward(self, img):
        f = self.enc(img)
        b, d, hh, ww = f.shape
        att = torch.softmax(self.score(f).view(b, hh * ww), -1)
        z = torch.bmm(f.view(b, d, hh * ww), att.unsqueeze(-1)).squeeze(-1)
        return self.head(z).view(b, self.n_pts, 3)


net = ConfigNet(c, args.feature_dim, P).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
print(f"[net] {sum(q.numel() for q in net.parameters())/1e6:.2f}M params")

mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)


def batches(idx, bs, shuffle=True):
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for s in range(0, len(order) - bs + 1, bs):
        sel = np.sort(idx[order[s:s + bs]])
        img = torch.from_numpy(np.asarray(images[sel], dtype=np.float32) / 255.0).to(dev)
        tgt = torch.from_numpy(CP[sel].astype(np.float32)).to(dev)
        yield img, tgt


for ep_i in range(args.epochs):
    net.train()
    tot = k = 0.0
    for img, tgt in batches(tr_idx, args.batch_size):
        pred = net(img) * sd_t + mu_t
        loss = nn.functional.mse_loss(pred, tgt)
        # Couple the outputs: the distances J consumes must also be right, so
        # the head cannot trade a good average against a wrong relative geometry.
        loss = loss + args.lambda_dist * nn.functional.mse_loss(pdists(pred), pdists(tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += float(loss.detach()); k += 1
    print(f"[{ep_i+1:>3}/{args.epochs}] train loss {tot/max(k,1):.4f}", flush=True)

# ---- evaluation, least-fragmented metric first ---------------------------
net.eval()
PR, TG = [], []
with torch.no_grad():
    for img, tgt in batches(va_idx, args.batch_size, shuffle=False):
        PR.append((net(img) * sd_t + mu_t).cpu())
        TG.append(tgt.cpu())
pred = torch.cat(PR); true = torch.cat(TG)


def r2(a, b):
    ss_res = float(((a - b) ** 2).sum())
    ss_tot = float(((b - b.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


from lehome.real_damped_project.math.garment_functional import (  # noqa: E402
    GarmentFoldFunctional, GarmentFunctionalCfg)

gconf_th = [10.0] * len(pairs)
fn = GarmentFoldFunctional(args.garment_type, gconf_th, GarmentFunctionalCfg())
J_pred = fn(pred).numpy()
J_true = fn(true).numpy()

d_pred, d_true = pdists(pred), pdists(true)

print("\n=== vision -> cloth configuration (held-out episodes) ===")
print(f"  1. J(p_hat) vs J(p)      R2 = {r2(torch.tensor(J_pred), torch.tensor(J_true)):>7.4f}"
      f"   corr = {np.corrcoef(J_pred, J_true)[0,1]:.4f}")
print(f"  2. pairwise distances    R2 = {r2(d_pred, d_true):>7.4f}"
      f"   mean abs err = {float((d_pred-d_true).abs().mean()):.2f} cm")
print(f"  3. per-point positions   R2 = {r2(pred, true):>7.4f}"
      f"   mean abs err = {float((pred-true).abs().mean()):.2f} cm")

j_r2 = r2(torch.tensor(J_pred), torch.tensor(J_true))
d_r2 = r2(d_pred, d_true)
print("\n=== verdict ===")
if j_r2 >= 0.7 and d_r2 >= 0.6:
    print("  Vision recovers the configuration well enough to drive the controller.")
    print("  The factored design is buildable; next unknown is the cloth Jacobian.")
elif j_r2 >= 0.5:
    print("  Partial. J is recoverable through positions but the relative geometry")
    print("  is loose -- the descent direction dJ/dp would be noisy. Consider more")
    print("  recorded episodes before trusting a controller built on this.")
else:
    print("  Vision does NOT recover the configuration. Predicting the scalar J")
    print("  (R2 0.87) does not extend to the geometry the controller needs, so the")
    print("  factored design needs a different perception target.")

if args.json_out:
    Path(args.json_out).write_text(json.dumps(
        {"j_r2": j_r2, "dist_r2": d_r2, "pos_r2": r2(pred, true),
         "n_train_eps": len(train_eps), "n_val_eps": len(val_eps)}, indent=2))

"""Can vision recover cloth configuration? Measured against real baselines.

The link never honestly established here. The `Ĵ` head scored R^2 0.87 and was
reading the clock -- a phase-only predictor scored 0.810 on the same target, and
within-band R^2 was NEGATIVE, meaning it classified "early/unfolded" vs
"late/folded" with no resolving power inside a band. That is useless for control,
which needs dJ/dp to point somewhere near the goal.

This runs on randomised-pose data, so there is no episode phase. But two weaker
confounds remain and each gets a control:

  constant        predict the training mean -- the floor
  frame index     each pose still has an internal settle -> sweep -> settle
                  trajectory, so a within-pose clock exists
  gripper only    during the sweep the cloth follows the arm, so EE position
                  partially predicts cloth state with no vision at all

Vision has to beat all three. Reported in the order that matters, least
fragmented first, because J consumes only five pairwise distances and a
common-mode error (whole garment off by 2 cm) is invisible to J while dominating
a position-wise loss:

  1. J(p_hat) vs J(p)     end to end, the quantity that decides success
  2. pairwise distances   exactly what J consumes
  3. per-point positions  most fragmented, reported last

Plus granularity -- R^2 within narrow J bands. Negative there means
classification, not regression, and that is the failure mode that made the 0.87
look real.

Occlusion is the other axis: error is stratified by gripper-to-check-point
distance, separating points that are visible, covered by a passing arm, or held.
Ground truth exists for occluded points, which is the whole reason to do this in
simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="collect_perception.py output dir")
p.add_argument("--garment_type", default="top-long-sleeve")
p.add_argument("--device", default="cuda")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--feature_dim", type=int, default=256)
p.add_argument("--lambda_dist", type=float, default=1.0)
p.add_argument("--val_frac", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--json_out", default="")
args = p.parse_args()

torch.manual_seed(args.seed)
dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

d = Path(args.data)
meta = json.loads((d / "meta.json").read_text())
C, H, W = meta["image_shape"]
n = meta["n_frames"]
images = np.memmap(d / "images.u8", dtype=np.uint8, mode="r", shape=(n, C, H, W))
CP = np.load(d / "checkpoints_cm.npy")[:n]
EE = np.load(d / "ee_pos_m.npy")[:n] * 100.0          # m -> cm
POSE = np.load(d / "pose_id.npy")[:n]

ok = np.isfinite(CP).all(axis=(1, 2)) & np.isfinite(EE).all(axis=(1, 2))
poses = np.array(sorted(set(POSE[ok].tolist())))
rng = np.random.RandomState(args.seed)
perm = rng.permutation(poses)
n_val = max(1, int(round(len(poses) * args.val_frac)))
# Split by POSE, never by frame: frames within a pose are near-duplicates and a
# frame-level split would leak the answer.
val_p = set(perm[:n_val].tolist())
va = np.where(np.isin(POSE, list(val_p)) & ok)[0]
tr = np.where(~np.isin(POSE, list(val_p)) & ok)[0]
print(f"[data] {len(poses)} poses -> train {len(poses)-n_val} / val {n_val}")
print(f"[data] frames: train {len(tr)} / val {len(va)}")

P = CP.shape[1]
mu = CP[tr].reshape(-1, 3).mean(0)
sd = CP[tr].reshape(-1, 3).std(0) + 1e-6

from lehome.real_damped_project.math.garment_functional import (  # noqa: E402
    GARMENT_CONDITIONS, GarmentFoldFunctional, GarmentFunctionalCfg)

pairs = [(i, j) for (i, j, _, _) in GARMENT_CONDITIONS[args.garment_type]]
fn = GarmentFoldFunctional(args.garment_type, [10.0] * len(pairs), GarmentFunctionalCfg())


def pdists(x):
    return torch.stack([torch.linalg.vector_norm(x[:, i] - x[:, j], dim=-1)
                        for i, j in pairs], dim=-1)


class ConfigNet(nn.Module):
    """Shared trunk -> whole configuration at once, not 18 independent heads."""

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
        b, dd, hh, ww = f.shape
        att = torch.softmax(self.score(f).view(b, hh * ww), -1)
        z = torch.bmm(f.view(b, dd, hh * ww), att.unsqueeze(-1)).squeeze(-1)
        return self.head(z).view(b, self.n_pts, 3)


net = ConfigNet(C, args.feature_dim, P).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)
print(f"[net] {sum(q.numel() for q in net.parameters())/1e6:.2f}M params")


def batches(idx, bs, shuffle=True):
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for s in range(0, len(order) - bs + 1, bs):
        sel = np.sort(idx[order[s:s + bs]])
        img = torch.from_numpy(np.asarray(images[sel], dtype=np.float32) / 255.0).to(dev)
        tgt = torch.from_numpy(CP[sel].astype(np.float32)).to(dev)
        yield img, tgt


for e in range(args.epochs):
    net.train()
    tot = k = 0.0
    for img, tgt in batches(tr, args.batch_size):
        pred = net(img) * sd_t + mu_t
        loss = nn.functional.mse_loss(pred, tgt) \
            + args.lambda_dist * nn.functional.mse_loss(pdists(pred), pdists(tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += float(loss.detach()); k += 1
    print(f"[{e+1:>3}/{args.epochs}] train {tot/max(k,1):.4f}", flush=True)

net.eval()
PR = []
with torch.no_grad():
    for img, _ in batches(va, args.batch_size, shuffle=False):
        PR.append((net(img) * sd_t + mu_t).cpu())
pred = torch.cat(PR)
nv = len(pred)
sel = np.sort(va)[:nv]
true = torch.tensor(CP[sel], dtype=torch.float32)


def r2(a, b):
    return 1.0 - float(((a - b) ** 2).sum()) / max(float(((b - b.mean(0)) ** 2).sum()), 1e-9)


# ---- baselines ----------------------------------------------------------
const = torch.tensor(np.repeat(CP[tr].mean(0)[None], nv, 0), dtype=torch.float32)

# within-pose frame index: mean configuration as a function of position in the
# pose's own settle -> sweep -> settle trajectory
fidx = np.zeros(len(CP), dtype=np.float64)
for q in poses:
    i = np.where(POSE == q)[0]
    fidx[i] = np.linspace(0, 1, len(i))
G = 40
grid = np.linspace(0, 1, G)
curve = np.stack([np.interp(grid, np.sort(fidx[tr]),
                            CP[tr].reshape(len(tr), -1)[np.argsort(fidx[tr]), c])
                  for c in range(P * 3)], axis=-1)
fpred = torch.tensor(np.stack([np.interp(fidx[sel], grid, curve[:, c])
                               for c in range(P * 3)], axis=-1).reshape(nv, P, 3),
                     dtype=torch.float32)

# gripper-only: ridge from both EE positions to the configuration
Xtr = np.hstack([EE[tr].reshape(len(tr), -1), (EE[tr].reshape(len(tr), -1)) ** 2])
Xva = np.hstack([EE[sel].reshape(nv, -1), (EE[sel].reshape(nv, -1)) ** 2])
m_, s_ = Xtr.mean(0), Xtr.std(0) + 1e-8
Ztr = np.hstack([(Xtr - m_) / s_, np.ones((len(Xtr), 1))])
Zva = np.hstack([(Xva - m_) / s_, np.ones((nv, 1))])
Wg = np.linalg.solve(Ztr.T @ Ztr + 1e-2 * len(Ztr) * np.eye(Ztr.shape[1]),
                     Ztr.T @ CP[tr].reshape(len(tr), -1))
gpred = torch.tensor((Zva @ Wg).reshape(nv, P, 3), dtype=torch.float32)

rows = [("constant (train mean)", const), ("within-pose frame index", fpred),
        ("gripper position only", gpred), ("VISION", pred)]

print(f"\n{'predictor':<28}{'J R2':>9}{'dist R2':>10}{'pos R2':>9}{'dist MAE':>11}")
print("-" * 67)
res = {}
for name, q in rows:
    jr = r2(torch.tensor(fn(q).numpy()), torch.tensor(fn(true).numpy()))
    dr = r2(pdists(q), pdists(true))
    pr = r2(q, true)
    mae = float((pdists(q) - pdists(true)).abs().mean())
    res[name] = (jr, dr, pr)
    print(f"{name:<28}{jr:>9.4f}{dr:>10.4f}{pr:>9.4f}{mae:>11.2f}")

best_base = max(res[k][1] for k in list(res)[:-1])
gain = res["VISION"][1] - best_base
print(f"\n  best non-vision baseline (dist R2): {best_base:.4f}")
print(f"  vision                            : {res['VISION'][1]:.4f}")
print(f"  gain from seeing the cloth        : {gain:+.4f}")

# ---- granularity --------------------------------------------------------
Jt = fn(true).numpy()
print("\n=== granularity: dist R2 within narrow J bands ===")
print("  (negative => classification, not regression -- the 0.87 failure mode)")
for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 8)]:
    m = (Jt >= lo) & (Jt < hi)
    if m.sum() > 50:
        print(f"  J in [{lo},{hi}): n={int(m.sum()):>5}  R2={r2(pdists(pred)[m], pdists(true)[m]):>8.4f}")

# ---- occlusion ----------------------------------------------------------
rel = true[:, None, :, :] - torch.tensor(EE[sel], dtype=torch.float32)[:, :, None, :]
gdist = torch.linalg.vector_norm(rel, dim=-1).min(dim=1).values      # (nv, P)
perr = torch.linalg.vector_norm(pred - true, dim=-1)                 # (nv, P)
print("\n=== error vs gripper proximity (occlusion) ===")
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 1e9)]:
    m = (gdist >= lo) & (gdist < hi)
    if m.sum() > 50:
        tag = "held/occluded" if hi <= 5 else ("near" if hi <= 10 else "clear")
        print(f"  gripper {lo:>2}-{min(hi,999):>3} cm  n={int(m.sum()):>6}  "
              f"mean err {float(perr[m].mean()):>6.2f} cm   {tag}")

print("\n=== verdict ===")
if gain >= 0.30 and res["VISION"][1] >= 0.6:
    print("  Vision recovers cloth configuration. The factored design has a")
    print("  foundation; next unknown is dp/du, which needs a working grasp.")
elif gain >= 0.10:
    print("  Vision adds real signal but is not yet accurate enough to drive a")
    print("  controller. More poses, or a spatial-softmax keypoint head instead")
    print("  of a globally pooled vector.")
else:
    print("  Vision adds little over not seeing the cloth at all. A globally")
    print("  pooled encoder cannot localise; go to spatial-softmax heatmaps")
    print("  supervised on projected check-point pixels.")

if args.json_out:
    Path(args.json_out).write_text(json.dumps(
        {k: {"j_r2": v[0], "dist_r2": v[1], "pos_r2": v[2]} for k, v in res.items()}, indent=2))

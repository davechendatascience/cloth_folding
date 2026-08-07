"""Behaviour cloning on the LeHome demonstrations.

Stage 1 of the plan the measurements forced: on-policy RL from scratch is not
reachable here (1.40 policy steps/s at num_envs=1 -> ~83 days for 1e7 steps,
plus an exploration landscape where standing still beats exploring). BC gets a
policy into the right basin from 265k demonstration frames, and the damped-RL
machinery then applies as *finetuning* -- the regime where damping accelerates
convergence instead of suppressing discovery.

Deliberately trains the **same** ``VisionAttentionPolicy`` used by PPO. A
stronger BC baseline exists (LeRobot's ACT, with action chunking), but the
architecture would then not accept a per-step PPO update, and continuity across
the two stages is the whole point of this pipeline. ACT is worth running as a
separate baseline, not as this policy.

Loss is Gaussian negative log-likelihood over the policy's own action
distribution rather than plain MSE, so the learned ``log_std`` is calibrated to
demonstration variance. PPO inherits that spread as its initial exploration
scale, which is a far better starting point than an arbitrary constant.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import LeHomeDemoDataset, split_episodes
from ..policy.vision_attention_policy import VisionAttentionPolicy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Behaviour cloning for LeHome cloth folding")
    p.add_argument("--cache", required=True, help="preprocess.py output directory")
    p.add_argument("--out", default="runs/bc")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_batches", type=int, default=0, help="smoke-test cap per epoch")
    return p.parse_args(argv)


def trivial_baselines(cache: str, train_eps, val_eps) -> dict:
    """MSE of predictors that have learned nothing, on the validation episodes.

    Reporting raw MSE is misleading here. At 30 fps consecutive joint targets
    are nearly identical, so **persistence** (``a[t] = a[t-1]``) already scores
    ~0.0026 -- and a policy that learns only to echo its previous output would
    match that while folding nothing. The GRU can represent exactly that
    degenerate solution, so it is a live attractor, not a hypothetical.

    The number worth watching is therefore ``val_mse / persistence_mse``:
    below 1 means the policy has learned something beyond temporal
    autocorrelation; at or above 1 it has not, however small the raw MSE looks.
    """
    from pathlib import Path

    import numpy as np

    cache = Path(cache)
    st = np.load(cache / "state.npy")
    ac = np.load(cache / "action.npy")
    ep = np.load(cache / "episode.npy")
    m = np.isin(ep, val_eps)
    s, a, e = st[m], ac[m], ep[m]

    persist = a.copy()
    persist[1:] = a[:-1]
    for i in range(1, len(a)):  # never carry across an episode boundary
        if e[i] != e[i - 1]:
            persist[i] = s[i]

    mse = lambda p: float(((a - p) ** 2).mean())  # noqa: E731
    const = np.repeat(ac[np.isin(ep, train_eps)].mean(0)[None], len(a), 0)
    return {
        "constant": mse(const),
        "identity": mse(s),
        "persistence": mse(persist),
    }


def run_epoch(policy, loader, device, optimizer=None, max_grad_norm=1.0, limit=0):
    train = optimizer is not None
    policy.train(train)
    tot_nll = tot_mse = tot_n = 0.0
    for bi, (img, prop, act) in enumerate(loader):
        if limit and bi >= limit:
            break
        # (B,T,...) -> (T,B,...) : the policy's sequence convention
        img = img.to(device, non_blocking=True).transpose(0, 1)
        prop = prop.to(device, non_blocking=True).transpose(0, 1)
        act = act.to(device, non_blocking=True).transpose(0, 1)

        with torch.set_grad_enabled(train):
            mean, _, _ = policy.forward_sequence(img, prop)
            dist = policy.distribution(mean)
            # BC targets are raw joint targets; with squashing the network's
            # bounded output cannot represent them, so BC trains the
            # pre-squash mean directly against the demonstrated action.
            nll = -dist.log_prob(act).sum(-1).mean()
            mse = torch.nn.functional.mse_loss(mean, act)

        if train:
            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        tot_nll += float(nll.detach()) * img.shape[1]
        tot_mse += float(mse.detach()) * img.shape[1]
        tot_n += img.shape[1]
    n = max(tot_n, 1)
    return tot_nll / n, tot_mse / n


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU", flush=True)
        device = "cpu"

    train_eps, val_eps = split_episodes(args.cache, args.val_frac, args.seed)
    train_ds = LeHomeDemoDataset(args.cache, args.seq_len, episodes=train_eps)
    val_ds = LeHomeDemoDataset(args.cache, args.seq_len, episodes=val_eps)
    # Validation must use the *training* statistics, or the normalisation
    # itself leaks information about the held-out episodes.
    val_ds.state_mean, val_ds.state_std = train_ds.state_mean, train_ds.state_std

    print(f"[data] {train_ds.meta['n_episodes']} episodes, {train_ds.meta['n_frames']} frames")
    print(f"[data] train {len(train_eps)} eps / {len(train_ds)} windows | "
          f"val {len(val_eps)} eps / {len(val_ds)} windows")
    print(f"[data] images {train_ds.image_shape} proprio {train_ds.proprio_dim} "
          f"action {train_ds.action_dim}")

    base = trivial_baselines(args.cache, train_eps, val_eps)
    persist = base["persistence"]
    print(f"[baseline] val MSE of learned-nothing predictors: "
          f"constant={base['constant']:.5f} identity={base['identity']:.5f} "
          f"persistence={persist:.5f}")
    print(f"[baseline] target: val_mse/persistence < 1.0 (below 1 = learned "
          f"more than temporal autocorrelation)")

    dl = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=(device != "cpu"), drop_last=True, persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)

    policy = VisionAttentionPolicy(
        image_channels=train_ds.image_shape[0],
        proprio_dim=train_ds.proprio_dim,
        action_dim=train_ds.action_dim,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        squash=False,  # joint targets are unbounded; see run_epoch
    ).to(device)
    print(f"[policy] {sum(p.numel() for p in policy.parameters())/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_nll, tr_mse = run_epoch(policy, train_dl, device, opt, args.max_grad_norm, args.limit_batches)
        with torch.no_grad():
            va_nll, va_mse = run_epoch(policy, val_dl, device, None, limit=args.limit_batches)
        sched.step()

        ratio = va_mse / max(persist, 1e-12)
        rec = {"epoch": epoch + 1, "train_nll": tr_nll, "train_mse": tr_mse,
               "val_nll": va_nll, "val_mse": va_mse, "vs_persistence": ratio,
               "log_std": float(policy.log_std.mean()), "sec": time.time() - t0}
        history.append(rec)
        flag = "BEATS-PERSISTENCE" if ratio < 1.0 else "no better than persistence"
        print(f"[{epoch+1:3d}/{args.epochs}] train mse={tr_mse:.5f} | val mse={va_mse:.5f} | "
              f"vs_persistence={ratio:.3f} [{flag}] | log_std={rec['log_std']:+.2f} | "
              f"{rec['sec']:.1f}s", flush=True)

        if va_mse < best:
            best = va_mse
            torch.save({"policy": policy.state_dict(), "epoch": epoch + 1,
                        "val_mse": va_mse, "args": vars(args),
                        "state_mean": train_ds.state_mean.tolist(),
                        "state_std": train_ds.state_std.tolist()},
                       out / "best.pt")
    (out / "history.json").write_text(json.dumps({"baselines": base, "history": history}, indent=2))
    print(f"[done] best val mse={best:.5f} -> {out/'best.pt'}")
    return history


if __name__ == "__main__":
    main()

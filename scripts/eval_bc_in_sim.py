"""Roll a behaviour-cloned policy out in the real LeHome env and measure J.

Validation MSE cannot answer the only question that matters. A policy can match
demonstrated actions closely and still never fold -- imitation error compounds
over a 300-step rollout, and the metric the challenge scores is a boolean over
garment check-point distances, not action similarity.

So this reports what the challenge reports: final J, best J reached, and
LeHome's own success predicate (J == 0), plus the damping diagnostics the spec
cares about (monotonicity of J, EE speed).

Baselines are included because an absolute J number is uninterpretable on its
own: a frozen policy and a random policy bound what "doing nothing" and "moving
arbitrarily" score, and the BC policy has to beat both to mean anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cpu", help="simulator device")
p.add_argument("--policy_device", default="cuda")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--decimation", type=int, default=3, help="3 -> 30Hz, matching the demos")
p.add_argument("--baselines", action="store_true", default=True)
p.add_argument("--out", default=None, help="write measured baselines + BC result as JSON")
p.add_argument("--dataset", default=None,
               help="LeRobot dir with meta/garment_info.json. Supplies the\n                    per-episode garment pose. Without it the garment starts\n                    wherever reset leaves it and the result reflects a\n                    distribution mismatch rather than the policy.")
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--original_damping", action="store_true",
               help="Evaluate on LeHome's ORIGINAL under-damped joints, which is "
                    "the plant the demonstrations were recorded on. Our per-joint "
                    "critical damping is a different plant from the one BC learned, "
                    "so evaluating there understates the policy through a "
                    "train/eval mismatch rather than a policy failure.")
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    from lehome.real_damped_project.policy.vision_attention_policy import (  # noqa: E402
        VisionAttentionPolicy,
    )
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg,
        IsaacGarmentBackend,
    )

    ckpt = torch.load(args.ckpt, map_location=args.policy_device, weights_only=False)
    state_mean = torch.as_tensor(ckpt["state_mean"], device=args.policy_device)
    state_std = torch.as_tensor(ckpt["state_std"], device=args.policy_device)
    cargs = ckpt["args"]
    # A delta-target policy emits a[t]-s[t]; the env wants absolute joint
    # targets. Forgetting to add s[t] back would command near-zero joint
    # angles -- the arm would fold to its zero pose and the result would look
    # like a policy failure rather than a units error.
    delta_policy = bool(ckpt.get("delta_target", False))
    print(f"[ckpt] target = {'a[t]-s[t] (delta)' if delta_policy else 'a[t] (absolute)'}")
    print(f"[ckpt] epoch={ckpt['epoch']} val_mse={ckpt['val_mse']:.5f}")

    cfg = IsaacGarmentCfg(
        garment_name=args.garment, device=args.device, decimation=args.decimation
    )
    if args.original_damping:
        cfg.joint_damping = {}
        print("[cfg] ORIGINAL under-damped joints (the plant BC learned from)")
    else:
        print("[cfg] per-joint CRITICAL damping (differs from the demo plant)")
    backend = IsaacGarmentBackend(cfg)
    print(f"[env] dt={backend.dt:.4f}s ({1/backend.dt:.1f} Hz), garment={backend.garment_type}")

    policy = VisionAttentionPolicy(
        image_channels=backend.image_shape[0],
        proprio_dim=backend.proprio_dim,
        action_dim=12,
        feature_dim=cargs["feature_dim"],
        hidden_dim=cargs["hidden_dim"],
        squash=False,
        # Must match how the checkpoint was trained: a run with lambda_j > 0
        # carries j_head weights, and loading those into a policy built without
        # the head fails on unexpected keys. The head is unused at rollout
        # time, but it has to exist to load.
        predict_j=cargs.get("lambda_j", 0.0) > 0.0,
    ).to(args.policy_device).eval()
    policy.load_state_dict(ckpt["policy"])

    # Garment poses from the demonstrations, one per episode.
    #
    # Without this the env resets to a default/random garment pose, and the
    # policy is asked to fold a garment lying somewhere it never saw in
    # training. That is not a policy failure but a distribution mismatch, and
    # it is exactly the bug that made demo *replay* look like a 0.7% J
    # reduction when the true figure with matched poses was 90%. Any
    # closed-loop number measured without it is uninterpretable.
    poses = []
    if args.dataset:
        ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
        gnames = list(ginfo.keys())
        for e in range(args.episodes):
            gi, li = e // args.eps_per_garment, e % args.eps_per_garment
            if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
                poses.append(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
        print(f"[pose] {len(poses)} per-episode garment poses from {args.dataset}")
    if not poses:
        print("[pose] WARNING: no per-episode poses -- the garment starts wherever "
              "reset leaves it, which understates the policy through a "
              "distribution mismatch. Pass --dataset for a meaningful number.")

    @torch.no_grad()
    def rollout(mode: str, ep: int):
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        if ep < len(poses):
            backend.set_garment_pose(poses[ep])
            for _ in range(5):  # let the cloth settle at the new pose
                backend.simulate()
        h = policy.initial_hidden(1, args.policy_device)
        js, speeds = [], []
        q0 = backend.get_proprioception().clone()
        for t in range(args.steps):
            if mode == "bc":
                img = backend.render_cameras().to(args.policy_device)
                prop = (backend.get_proprioception().to(args.policy_device) - state_mean) / state_std
                mean, _, h, _ = policy(img, prop, h)
                act = mean.to(backend.device)
                if delta_policy:
                    act = act + backend.get_proprioception()
            elif mode == "frozen":
                act = q0.clone()
            else:  # random walk around the start pose
                act = q0 + torch.randn_like(q0) * 0.05

            backend.set_joint_targets(act)
            backend.simulate()
            js.append(float(backend.compute_cloth_error()))
            speeds.append(float(
                torch.linalg.vector_norm(backend.get_end_effector_velocities().flatten())
            ))
        term, _ = backend.check_done()
        # monotone violations: J rising by more than epsilon
        dj = np.diff(js)
        return {
            "mode": mode, "ep": ep,
            "J_start": js[0], "J_end": js[-1], "J_min": float(np.min(js)),
            "success": bool(term.any()),
            "rise_frac": float((dj > 1e-3).mean()),
            "ee_speed": float(np.mean(speeds)),
        }

    modes = ["bc"] + (["frozen", "random"] if args.baselines else [])
    rows = []
    for mode in modes:
        for ep in range(args.episodes if mode == "bc" else 1):
            r = rollout(mode, ep)
            rows.append(r)
            print(f"  [{r['mode']:6s} ep{r['ep']}] J {r['J_start']:7.3f} -> {r['J_end']:7.3f}  "
                  f"min={r['J_min']:7.3f}  success={r['success']}  "
                  f"J_rise={r['rise_frac']:.2f}  ee_speed={r['ee_speed']:.4f}", flush=True)

    print("\n=== summary ===")
    for mode in modes:
        sel = [r for r in rows if r["mode"] == mode]
        print(f"  {mode:6s}  J_end={np.mean([r['J_end'] for r in sel]):7.3f}  "
              f"J_min={np.mean([r['J_min'] for r in sel]):7.3f}  "
              f"success={sum(r['success'] for r in sel)}/{len(sel)}")
    bc = [r for r in rows if r["mode"] == "bc"]
    fr = [r for r in rows if r["mode"] == "frozen"]
    if fr:
        better = np.mean([r["J_min"] for r in bc]) < np.mean([r["J_min"] for r in fr])
        print(f"\n  BC beats frozen on J_min: {better}")
        if not better:
            print("  => BC has not learned anything useful yet.")
    if args.out:
        summary = {
            m: {
                "J_end": float(np.mean([r["J_end"] for r in rows if r["mode"] == m])),
                "J_min": float(np.mean([r["J_min"] for r in rows if r["mode"] == m])),
                "successes": int(sum(r["success"] for r in rows if r["mode"] == m)),
                "n": int(sum(1 for r in rows if r["mode"] == m)),
            }
            for m in modes
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "episodes": rows, "ckpt": args.ckpt,
             "garment": args.garment, "decimation": args.decimation,
             "original_damping": bool(args.original_damping)}, indent=2))
        print(f"  wrote {args.out}")

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

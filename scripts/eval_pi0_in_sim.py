"""Score a finetuned pi0 checkpoint on the real task: closed-loop J.

Training loss cannot answer whether the garment gets folded, and this project
has the sharpest possible demonstration of that -- the behaviour-cloned policies
reached val MSE 0.00234 and scored identically to a frozen arm in closed loop.
pi0's flow-matching loss is even less interpretable, so the only criterion that
means anything is LeHome's own predicate, J == 0, measured by rolling the policy
out in the simulator.

Pre-registered bars, from measurements already in the repo:

    frozen (do nothing)        J = 7.118      <- below this or it is not manipulating
    20% capture bar            J <= 5.695     <- "meaningful", on a MEAN not a best
    demo replay, matched pose  J = 0 in 27%   <- the demonstrated ceiling for this data
    SUCCESS                    J == 0

Two protocol requirements that are not optional, both learned expensively here:

* **Per-episode garment pose.** `reset()` randomises placement; replaying a
  demonstration without `set_garment_pose()` reduces J by 0.7% instead of 90%.
  Any closed-loop number measured without it reflects a distribution mismatch
  rather than the policy.
* **Baselines in the same run.** An absolute J is uninterpretable. frozen and
  random bound "did nothing" and "moved arbitrarily", and a policy has to beat
  both to mean anything.

Reports the MEAN over episodes, not the best. With a chaotic simulator (identical
replays of one episode gave final J of 1.770, 1.062 and 2.214) a best-of-N will
drift downward by luck alone -- that artefact produced a "BEATS baseline" verdict
for a policy that was doing nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="pi0 checkpoint dir (…/checkpoints/XXXXXX/pretrained_model)")
p.add_argument("--dataset", required=True, help="LeRobot dir with meta/garment_info.json")
p.add_argument("--garment", default="Top_Long_Seen_0")
p.add_argument("--device", default="cuda", help="simulator device")
p.add_argument("--policy_device", default="cuda")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--decimation", type=int, default=3, help="3 -> 30 Hz, matching the demos")
p.add_argument("--task", default="fold the garment on the table",
               help="language instruction; pi0 is a vision-LANGUAGE-action model and the "
                    "dataset carries this string")
p.add_argument("--eps_per_garment", type=int, default=25)
p.add_argument("--baselines", action="store_true", default=True)
p.add_argument("--out", default=None)
args = p.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
simulation_app = launcher.app

EXIT = 0
try:
    import torch  # noqa: E402
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    policy = PI0Policy.from_pretrained(args.ckpt).to(args.policy_device).eval()
    print(f"[pi0] loaded {args.ckpt}", flush=True)
    print(f"[pi0] n_action_steps={policy.config.n_action_steps} "
          f"chunk={policy.config.chunk_size}", flush=True)

    # ORIGINAL under-damped joints: the plant the demonstrations were recorded
    # on, and the one pi0 was finetuned against. Our critical damping is a
    # different plant -- replaying demos there degrades the fold (J_end 2.200 vs
    # 0.211), so evaluating there would understate the policy through a
    # train/eval mismatch rather than a policy failure.
    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name=args.garment, device=args.device,
        decimation=args.decimation, joint_damping={}))
    print(f"[env] dt={backend.dt:.4f}s garment={backend.garment_type}", flush=True)

    ginfo = json.loads((Path(args.dataset) / "meta" / "garment_info.json").read_text())
    gnames = list(ginfo.keys())
    poses = []
    for e in range(args.episodes):
        gi, li = e // args.eps_per_garment, e % args.eps_per_garment
        if gi < len(gnames) and str(li) in ginfo[gnames[gi]]:
            poses.append(ginfo[gnames[gi]][str(li)]["object_initial_pose"])
    print(f"[pose] {len(poses)} per-episode garment poses", flush=True)
    if len(poses) < args.episodes:
        print("[pose] WARNING: fewer poses than episodes; the remainder start "
              "wherever reset leaves them and are NOT comparable", flush=True)

    CAMS = (("top_rgb", "top_camera"), ("left_rgb", "left_camera"),
            ("right_rgb", "right_camera"))

    def observation():
        """Build pi0's expected batch from the live simulator.

        Reads the cameras at their NATIVE 480x640 rather than through
        `backend.render_cameras()`, which downsamples to an 84x84 square for the
        old from-scratch encoder. pi0 was finetuned on the dataset's 480x640
        frames, so feeding it 84x84 would degrade the visual input silently --
        the run would produce a number, and the number would mean nothing.

        Keys must match the dataset feature names the policy was finetuned on,
        for the same reason.
        """
        batch = {
            "observation.state": backend.get_proprioception()
                                 .to(args.policy_device).reshape(1, -1).float(),
            "task": [args.task],
        }
        for name, attr in CAMS:
            rgb = getattr(backend.env, attr).data.output["rgb"]      # (1, H, W, 3) uint8
            x = rgb.permute(0, 3, 1, 2).float()
            if x.max() > 1.5:
                x = x / 255.0
            batch[f"observation.images.{name}"] = x.to(args.policy_device)
        return batch

    @torch.no_grad()
    def rollout(mode: str, ep: int):
        backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
        if ep < len(poses):
            backend.set_garment_pose(poses[ep])
            for _ in range(5):
                backend.simulate()
        if hasattr(policy, "reset"):
            policy.reset()
        js, speeds = [], []
        q0 = backend.get_proprioception().clone()
        for t in range(args.steps):
            if mode == "pi0":
                act = policy.select_action(observation()).reshape(-1).to(backend.device)
            elif mode == "frozen":
                act = q0.clone().reshape(-1)
            else:
                act = (q0 + torch.randn_like(q0) * 0.05).reshape(-1)
            backend.set_joint_targets(act)
            backend.simulate()
            js.append(float(backend.compute_cloth_error()))
            speeds.append(float(torch.linalg.vector_norm(
                backend.get_end_effector_velocities().flatten())))
        term, _ = backend.check_done()
        dj = np.diff(js)
        return {"mode": mode, "ep": ep, "J_start": js[0], "J_end": js[-1],
                "J_min": float(np.min(js)), "success": bool(term.any()),
                "rise_frac": float((dj > 1e-3).mean()), "ee_speed": float(np.mean(speeds))}

    modes = ["pi0"] + (["frozen", "random"] if args.baselines else [])
    rows = []
    for mode in modes:
        n = args.episodes if mode == "pi0" else max(3, args.episodes // 3)
        for ep in range(n):
            r = rollout(mode, ep)
            rows.append(r)
            print(f"[{mode:<6} ep{ep}] J {r['J_start']:6.3f} -> {r['J_end']:6.3f}  "
                  f"min={r['J_min']:6.3f}  success={r['success']}  "
                  f"ee_speed={r['ee_speed']:.4f}", flush=True)

    print("\n=== summary (MEAN over episodes, not best) ===")
    summary = {}
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        Jm = np.array([r["J_min"] for r in rs])
        Je = np.array([r["J_end"] for r in rs])
        succ = sum(r["success"] for r in rs)
        summary[mode] = {"n": len(rs), "J_min_mean": float(Jm.mean()),
                         "J_min_sd": float(Jm.std()), "J_end_mean": float(Je.mean()),
                         "successes": succ, "success_rate": succ / max(len(rs), 1)}
        print(f"  {mode:<7} n={len(rs):>2}  J_min {Jm.mean():6.3f} +- {Jm.std():5.3f}  "
              f"J_end {Je.mean():6.3f}  success {succ}/{len(rs)}")

    FROZEN_REF, BAR, REPLAY_RATE = 7.118, 5.695, 0.27
    p0 = summary.get("pi0", {})
    jm = p0.get("J_min_mean", float("nan"))
    print("\n=== verdict against pre-registered bars ===")
    print(f"  frozen reference       7.118   | this run's frozen: "
          f"{summary.get('frozen', {}).get('J_min_mean', float('nan')):.3f}")
    print(f"  manipulating at all  : {'YES' if jm < FROZEN_REF else 'NO'}  (mean J_min {jm:.3f})")
    print(f"  meaningful (<= 5.695): {'YES' if jm <= BAR else 'NO'}")
    print(f"  success rate         : {p0.get('success_rate', 0):.0%}  "
          f"(demo replay ceiling {REPLAY_RATE:.0%})")
    if p0.get("success_rate", 0) > 0:
        print("  => the policy folds garments. That is LeHome's own predicate.")
    elif jm <= BAR:
        print("  => manipulating meaningfully but not completing folds.")
    elif jm < FROZEN_REF:
        print("  => moves the cloth, but barely distinguishable from doing nothing.")
    else:
        print("  => no better than a frozen arm. Loss value is irrelevant to this.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "rows": rows, "summary": summary}, indent=2))
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

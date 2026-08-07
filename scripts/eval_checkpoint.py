"""Measure whether a checkpoint actually reduces J, against an untrained control.

Answers the only question that matters for this design: does the learned policy
drive the Lyapunov functional down, and does it do so without ringing?
"""

from __future__ import annotations

import argparse

import torch

from lehome.real_damped_project.policy.vision_attention_policy import VisionAttentionPolicy
from lehome.real_damped_project.tasks.cfg import make_env


@torch.no_grad()
def rollout(env, policy, steps: int, deterministic: bool = True):
    obs = env.reset()
    hidden = policy.initial_hidden(env.num_envs, env.device)
    j0 = env.cloth_error().mean().item()

    j_traj, near_steps, violations, speeds = [], 0, 0, []
    for _ in range(steps):
        action, _, _, hidden, _ = policy.act(
            obs["images"], obs["proprio"], hidden, deterministic=deterministic
        )
        obs, _, term, trunc, extras = env.step(action)
        done = term | trunc
        hidden = hidden * (~done).view(1, -1, 1).to(hidden.dtype)
        log = extras["log"]
        j_traj.append(log["J"].mean().item())
        near_steps += int(log["near_mask"].sum().item())
        violations += int(log["mono_violation"].sum().item())
        speeds.append(log["ee_speed"].mean().item())

    return {
        "J_start": j0,
        "J_end": j_traj[-1],
        "J_min": min(j_traj),
        "J_mean": sum(j_traj) / len(j_traj),
        "ee_speed": sum(speeds) / len(speeds),
        "near_steps": near_steps,
        "mono_violation_rate": violations / max(near_steps, 1),
        "traj": j_traj,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    args = p.parse_args()

    env = make_env(num_envs=args.num_envs, sim_device=args.device, use_mock_backend=True)
    shapes = env.observation_shapes

    def fresh():
        return VisionAttentionPolicy(
            shapes["images"][0], shapes["proprio"][0], env.action_dim,
            feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        ).to(args.device).eval()

    trained = fresh()
    ckpt = torch.load(args.ckpt, map_location=args.device)
    trained.load_state_dict(ckpt["policy"])
    print(f"loaded {args.ckpt} @ iteration {ckpt.get('iteration')}")

    torch.manual_seed(0)
    control = fresh()

    for name, pol in (("untrained", control), ("trained", trained)):
        torch.manual_seed(0)
        r = rollout(env, pol, args.steps)
        print(
            f"\n[{name}]  J_start={r['J_start']:.4f}  J_end={r['J_end']:.4f}  "
            f"J_min={r['J_min']:.4f}  J_mean={r['J_mean']:.4f}"
            f"\n           ee_speed={r['ee_speed']:.4f}  near_goal_steps={r['near_steps']}  "
            f"mono_violation_rate={r['mono_violation_rate']:.3f}"
        )
        every = max(1, args.steps // 10)
        print("           J traj:", " ".join(f"{v:.3f}" for v in r["traj"][::every]))


if __name__ == "__main__":
    main()

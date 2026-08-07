"""Training entrypoint (spec Sec. 6).

On DGX Spark with the full stack installed::

    cd lehome-challenge
    source .venv/bin/activate
    ./third_party/IsaacLab/isaaclab.sh \
      -p source/lehome/real_damped_project/train/train_ppo_real_damped.py \
      -- --task LeHome-Fold-Garment-RealDamped-v0 \
         --num_envs 2048 --device cuda --headless

Without Isaac Lab, the same script runs against the mock damped cloth::

    python -m lehome.real_damped_project.train.train_ppo_real_damped \
      --num_envs 32 --max_iterations 50 --mock

``--mock`` is the default when LeHome is not importable, so the script is always
runnable and the RL machinery can be validated before simulator time is spent.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def _bootstrap_path() -> None:
    """Allow running as a plain script from a checkout (no install)."""
    here = os.path.dirname(os.path.abspath(__file__))
    source_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


_bootstrap_path()

from lehome.real_damped_project.policy.vision_attention_policy import (  # noqa: E402
    VisionAttentionPolicy,
)
from lehome.real_damped_project.tasks.cfg import TASK_NAME, make_env  # noqa: E402
from lehome.real_damped_project.train.ppo import DampedPPOAgent, PPOCfg  # noqa: E402
from lehome.real_damped_project.train.runner import Runner, RunnerCfg  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Damped visual RL for LeHome cloth folding")
    p.add_argument("--task", default=TASK_NAME)
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max_iterations", type=int, default=10_000)
    p.add_argument("--num_steps_per_env", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target_kl", type=float, default=0.01)
    p.add_argument("--prior_kl_coef", type=float, default=0.0)
    p.add_argument("--polyak_tau", type=float, default=0.0)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)

    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--spectral_norm", action="store_true",
                   help="enforce per-layer Lipschitz bounds (Sec. 3.3)")

    p.add_argument("--mock", action="store_true",
                   help="force the damped mass-spring backend (no Isaac Lab)")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--save_interval", type=int, default=200)
    return p.parse_args(argv)


def lehome_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("lehome.tasks") is not None


def main(argv=None) -> Runner:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU", flush=True)
        device = "cpu"

    use_mock = args.mock or not lehome_available()
    if use_mock and not args.mock:
        print(
            "[warn] LeHome/Isaac Lab not importable -> using the mock damped cloth backend. "
            "Results validate the RL pipeline, NOT cloth physics.",
            flush=True,
        )

    # --- environment (Sec. 6.1) ----------------------------------------------
    env = make_env(
        task_name=args.task,
        num_envs=args.num_envs,
        sim_device=device,
        rl_device=device,
        graphics_device_id=0,
        headless=args.headless,
        use_mock_backend=use_mock,
    )

    shapes = env.observation_shapes
    image_channels = shapes["images"][0]
    proprio_dim = shapes["proprio"][0]
    action_dim = env.action_dim
    print(
        f"[env] task={args.task} num_envs={env.num_envs} device={device} "
        f"images={shapes['images']} proprio={proprio_dim} action={action_dim}",
        flush=True,
    )

    # --- policy (Sec. 6.2) ----------------------------------------------------
    policy = VisionAttentionPolicy(
        image_channels=image_channels,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        spectral_norm=args.spectral_norm,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[policy] {n_params/1e6:.2f}M parameters", flush=True)

    # --- PPO ------------------------------------------------------------------
    ppo_cfg = PPOCfg(
        lr=args.lr,
        num_steps_per_env=args.num_steps_per_env,
        target_kl=args.target_kl,
        prior_kl_coef=args.prior_kl_coef,
        polyak_tau=args.polyak_tau,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
    )
    agent = DampedPPOAgent(policy, ppo_cfg, device=device)

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    runner = Runner(
        env,
        agent,
        RunnerCfg(
            max_iterations=args.max_iterations,
            log_dir=args.log_dir,
            save_interval=args.save_interval,
        ),
        device=device,
    )

    runner.train()
    return runner


if __name__ == "__main__":
    main()

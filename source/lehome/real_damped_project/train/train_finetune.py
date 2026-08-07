"""Damped-RL finetuning of a behaviour-cloned policy.

Stage 2. BC puts the policy in the right basin; this applies the spec's actual
contribution -- the Lyapunov functional J, the monotone-descent reward, and the
damping hierarchy -- where damping helps rather than hurts. Everything measured
this project says on-policy RL cannot *discover* folding here (1.40 policy
steps/s at num_envs=1, and standing still beats exploring), but refining a
policy that already folds is a different problem.

Three couplings that must hold exactly, each of which silently destroys the
initialisation if broken:

* **Action space.** The demonstrations are 12-D joint position targets, so the
  env runs ``action_mode="joint"``. A BC policy's weights mean nothing in the
  spec's 6-D Cartesian space.
* **Observation space.** ``proprio_matches_dataset`` gives the same 12-D joint
  vector BC saw, and the BC checkpoint's ``state_mean``/``state_std`` are
  reapplied here. Skipping the normalisation feeds the network inputs scaled
  differently from anything it was trained on.
* **No action squashing.** BC trained ``squash=False`` against raw joint
  targets; tanh-bounding them now would make the loaded weights meaningless.

The run refuses to start unless its :class:`RunContract` passes preflight --
measured baselines, verified reachability, a real success threshold, and an
unbuffered log. Those were exactly what was missing when "is it converging?"
turned out to be unanswerable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from ..policy.vision_attention_policy import VisionAttentionPolicy
from .ppo import DampedPPOAgent, PPOCfg
from .run_contract import RunContract, Verdict, Watchdog
from .runner import Runner, RunnerCfg


class ProprioNormalizer:
    """Applies the BC checkpoint's input statistics to live observations.

    Not cosmetic: BC learned on standardised proprioception, so feeding raw
    joint angles at finetuning time shifts every input by roughly a standard
    deviation per dimension and the loaded weights stop meaning what they meant.
    """

    def __init__(self, mean, std, device) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)

    def __call__(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(obs)
        out["proprio"] = (obs["proprio"] - self.mean) / self.std
        return out


class NormalizedEnv:
    """Thin pass-through that normalises proprio on reset/step."""

    def __init__(self, env, normalizer: ProprioNormalizer) -> None:
        self._env = env
        self._norm = normalizer

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._norm(self._env.reset())

    def step(self, actions):
        obs, rew, term, trunc, extras = self._env.step(actions)
        return self._norm(obs), rew, term, trunc, extras


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Damped RL finetuning of a BC policy")
    p.add_argument("--bc_ckpt", required=True)
    p.add_argument("--contract", required=True, help="pre-registered RunContract JSON")
    p.add_argument("--out", default="runs/finetune")
    p.add_argument("--garment", default="Top_Long_Seen_0")
    p.add_argument("--sim_device", default="cpu")
    p.add_argument("--policy_device", default="cuda")
    p.add_argument("--decimation", type=int, default=3)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--num_steps_per_env", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Small by design: this refines a working policy rather "
                        "than searching, so the update should be a damped, "
                        "near-non-expansive map (Sec. 3.4).")
    p.add_argument("--target_kl", type=float, default=0.005)
    p.add_argument("--prior_kl_coef", type=float, default=0.1,
                   help="Anchor to the BC policy. Without it, early noisy "
                        "advantages can walk the policy out of the basin BC "
                        "found, discarding the whole point of stage 1.")
    p.add_argument("--entropy_coef", type=float, default=0.0,
                   help="Zero by default: the BC log_std is already calibrated "
                        "to demonstration variance, and an entropy bonus would "
                        "inflate it back toward undirected exploration.")
    p.add_argument("--polyak_tau", type=float, default=0.05)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- the run must be legitimate before anything expensive happens ----
    contract = RunContract.load(args.contract)
    contract.require_ready()
    print(f"[contract] {contract.name} digest={contract.digest()} "
          f"metric={contract.primary_metric} must_beat={contract.must_beat_baseline}")
    print(f"[contract] baselines={contract.baselines} "
          f"success<{contract.success_threshold}")
    watchdog = Watchdog(contract)

    from ..tasks.cfg import RealDampedTaskCfg
    from ..tasks.isaac_garment_backend import IsaacGarmentCfg
    from ..tasks.lehome_fold_garment_real_damped_task import (
        LeHomeFoldGarmentRealDampedEnv,
    )

    ckpt = torch.load(args.bc_ckpt, map_location=args.policy_device, weights_only=False)
    bc_args = ckpt["args"]
    print(f"[bc] epoch={ckpt['epoch']} val_mse={ckpt['val_mse']:.5f}")

    cfg = RealDampedTaskCfg()
    cfg.use_mock_backend = False
    cfg.action_mode = "joint"
    cfg.num_envs = 1
    cfg.backend = IsaacGarmentCfg(
        garment_name=args.garment,
        device=args.sim_device,
        decimation=args.decimation,
        proprio_matches_dataset=True,
    )

    env = LeHomeFoldGarmentRealDampedEnv(cfg, device=args.policy_device)
    env = NormalizedEnv(
        env, ProprioNormalizer(ckpt["state_mean"], ckpt["state_std"], args.policy_device)
    )
    shapes = env.observation_shapes
    print(f"[env] images={shapes['images']} proprio={shapes['proprio']} "
          f"action={env.action_dim} mode={env.action_mode}")

    policy = VisionAttentionPolicy(
        image_channels=shapes["images"][0],
        proprio_dim=shapes["proprio"][0],
        action_dim=env.action_dim,
        feature_dim=bc_args["feature_dim"],
        hidden_dim=bc_args["hidden_dim"],
        squash=False,  # BC trained on raw joint targets
    ).to(args.policy_device)
    policy.load_state_dict(ckpt["policy"])
    print(f"[policy] loaded BC weights, log_std={float(policy.log_std.mean()):+.2f}")

    agent = DampedPPOAgent(
        policy,
        PPOCfg(
            lr=args.lr,
            num_steps_per_env=args.num_steps_per_env,
            target_kl=args.target_kl,
            prior_kl_coef=args.prior_kl_coef,
            entropy_coef=args.entropy_coef,
            polyak_tau=args.polyak_tau,
        ),
        device=args.policy_device,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract.save(out / "contract.json")
    runner = Runner(env, agent, RunnerCfg(max_iterations=args.iterations,
                                          log_dir=str(out)), device=args.policy_device)

    history = []
    for it in range(args.iterations):
        stats = runner.train(max_iterations=1)[-1]
        verdict = watchdog.update({contract.primary_metric: stats["J_mean"], **stats})
        history.append({**stats, "verdict": verdict.value})
        print(f"[{it+1:4d}] J={stats['J_mean']:8.4f} R={stats['reward_mean']:8.4f} "
              f"mono_viol={stats['mono_violation_rate']:.3f} kl={stats['kl']:.4f} "
              f"| {verdict.value} {watchdog.report()}", flush=True)

        if verdict in (Verdict.NAN, Verdict.DIVERGED):
            print(f"[abort] {verdict.value}: " + "; ".join(watchdog.alerts[-2:]), flush=True)
            break
        if verdict is Verdict.SUCCESS:
            print("[done] success threshold reached", flush=True)
            break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    runner.save(str(out / "final.pt"))
    print(f"[done] {watchdog.report()} -> {out}")
    return history


if __name__ == "__main__":
    main()

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
    p.add_argument("--reinit_actor", action="store_true",
                   help="Keep the BC encoder, reinitialise the action head. The BC "
                        "policy is measurably a BAD prior -- closed loop it scored "
                        "J_min 7.168 against a frozen arm's 7.118, i.e. worse than "
                        "not moving -- because it reproduces a pose-blind template. "
                        "Its encoder is worth keeping; its actor is not.")
    p.add_argument("--damping_gate", default="smooth",
                   choices=["always", "near", "smooth"],
                   help="Where r_vel/r_act apply. 'always' is the spec as written and "
                        "measurably creates a freeze basin: both terms vanish when "
                        "stationary, so doing nothing scored -538 against -664 for "
                        "exploring. The convergence argument only needs monotone "
                        "descent EVENTUALLY, so damping belongs near the goal.")
    p.add_argument("--lr_max", type=float, default=0.0,
                   help="Cap on the adaptive learning rate (0 = uncapped). An "
                        "adaptive schedule RAISES lr when KL comes in under target, "
                        "which is anti-damping in a chaotic reward landscape: "
                        "measured climbing 3e-5 -> 1.01e-4 while the policy overshot "
                        "a good region and lost it.")
    p.add_argument("--reanchor_on_best", action="store_true",
                   help="Re-anchor the KL prior to the current policy whenever the "
                        "running mean improves. This is damping on the POLICY "
                        "trajectory: resist moving away from the best solution found, "
                        "rather than from behaviour cloning (measured worse than a "
                        "frozen arm) or from nothing at all.")
    p.add_argument("--ckpt_every", type=int, default=25,
                   help="Save every N iterations. Without this a multi-hour run has "
                        "nothing to inspect until it ends and loses everything on a "
                        "crash -- the same gap that cost 270 poses in the perception "
                        "collector.")
    p.add_argument("--init_log_std", type=float, default=-1.0,
                   help="Exploration scale for a reinitialised actor. BC trains this "
                        "to -3.20 (sigma 0.041 rad) fitting a deterministic template; "
                        "inheriting that leaves nothing to explore with.")
    p.add_argument("--j_anneal", type=float, default=1.0,
                   help="Scale for damping_gate='smooth': damping reaches ~37% at J=this.")
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
    cfg.reward.damping_gate = args.damping_gate
    cfg.reward.j_anneal = args.j_anneal
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
        # A BC run with lambda_j > 0 carries j_head weights, and loading those
        # into a policy built without the head fails on unexpected keys. The
        # head is unused during RL, but it has to exist to load. (Same bug was
        # fixed in eval_bc_in_sim.py and not propagated here.)
        predict_j=bc_args.get("lambda_j", 0.0) > 0.0,
    ).to(args.policy_device)
    policy.load_state_dict(ckpt["policy"])
    if args.reinit_actor:
        # Transfer perception, discard behaviour. Measured: the BC actor weights
        # proprioception ~9x more than images and executes a template that does
        # not fold, so anchoring RL to it would hold the search inside a basin we
        # already know is bad.
        policy.policy_head.reset_parameters()
        torch.nn.init.orthogonal_(policy.policy_head.weight, gain=0.01)
        torch.nn.init.zeros_(policy.policy_head.bias)
        # Reset the exploration scale too. BC drives log_std down to -3.20
        # (sigma = 0.041 rad) because it is fitting a near-deterministic
        # template; a reinitialised actor inheriting that cannot search at all,
        # and exploration is exactly what the under-damped regime is meant to
        # buy. Restore the policy's own init value.
        with torch.no_grad():
            policy.log_std.fill_(args.init_log_std)
        print(f"[policy] actor head reinitialised (log_std -> {args.init_log_std}); "
              "encoder/attention retained from BC")
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
        prev_best_mean = watchdog.best_mean
        verdict = watchdog.update({contract.primary_metric: stats["J_mean"], **stats})
        if args.reanchor_on_best and watchdog.best_mean is not None \
                and (prev_best_mean is None or watchdog.best_mean < prev_best_mean):
            if runner.agent.refresh_prior():
                print(f"        re-anchored KL prior at mean={watchdog.best_mean:.4f}",
                      flush=True)
        if args.lr_max > 0.0:
            for g in runner.agent.optimizer.param_groups:
                g["lr"] = min(g["lr"], args.lr_max)
        history.append({**stats, "verdict": verdict.value})
        print(f"[{it+1:4d}] J={stats['J_mean']:8.4f} R={stats['reward_mean']:8.4f} "
              f"mono_viol={stats['mono_violation_rate']:.3f} kl={stats['kl']:.4f} "
              f"| {verdict.value} {watchdog.report()}", flush=True)
        # ge-terms are the collapse detector: if they dominate, the policy is
        # bunching the cloth rather than folding it.
        ge = sum(v for k, v in stats.items() if k.startswith("J_c") and "_ge_" in k)
        le = sum(v for k, v in stats.items() if k.startswith("J_c") and "_le_" in k)
        if ge or le:
            print(f"        J split: le={le:.3f} ge={ge:.3f} "
                  f"{'<- COLLAPSE (ge dominates)' if ge > 2 * max(le, 1e-6) else ''}",
                  flush=True)
        if args.ckpt_every and (it + 1) % args.ckpt_every == 0:
            runner.save(str(out / f"iter_{it+1:05d}.pt"))
            (out / "history.json").write_text(json.dumps(history, indent=2))

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

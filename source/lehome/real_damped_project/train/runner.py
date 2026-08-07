"""Rollout/update loop for the damped folding task (spec Sec. 6.2, ``Runner``).

Beyond the usual PPO bookkeeping this tracks the quantities the spec's
convergence argument is actually about:

* ``J_mean`` -- the Lyapunov functional itself.
* ``mono_violation_rate`` -- fraction of *near-goal* steps on which
  ``J`` increased by more than ``epsilon``. This is the direct empirical test of
  "eventually monotone non-increasing" (Sec. 2.3). If training is working, this
  should trend to ~0 even while ``J_mean`` is still falling.
* ``ee_speed`` -- whether the physical trajectories are actually damped.

Those three make it possible to tell *why* a run is failing: a falling J with a
high violation rate means the policy is folding by oscillating, which is exactly
what this design forbids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .ppo import DampedPPOAgent, PPOCfg, RolloutBuffer


@dataclass
class RunnerCfg:
    max_iterations: int = 10_000
    log_interval: int = 1
    save_interval: int = 200
    log_dir: Optional[str] = None


class Runner:
    """Drives environment rollouts and PPO updates."""

    def __init__(
        self,
        env,
        agent: DampedPPOAgent,
        cfg: RunnerCfg,
        device: torch.device | str = "cpu",
    ) -> None:
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = agent.policy

        shapes = env.observation_shapes
        self.buffer = RolloutBuffer(
            num_steps=agent.cfg.num_steps_per_env,
            num_envs=env.num_envs,
            image_shape=shapes["images"],
            proprio_dim=shapes["proprio"][0],
            action_dim=env.action_dim,
            hidden_dim=self.policy.hidden_dim,
            device=self.device,
        )

        self.obs = env.reset()
        self.hidden = self.policy.initial_hidden(env.num_envs, self.device)
        self.iteration = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def collect(self) -> Dict[str, float]:
        self.buffer.reset(self.hidden)
        j_sum, ee_speed_sum, n = 0.0, 0.0, 0
        near_steps, violations = 0, 0

        for _ in range(self.buffer.num_steps):
            images, proprio = self.obs["images"], self.obs["proprio"]
            # `action` is bounded and goes to the env; `raw` is the pre-squash
            # sample and is what the buffer must hold for an exact PPO ratio.
            action, raw, log_prob, value, new_hidden, _ = self.policy.act_raw(
                images, proprio, self.hidden
            )

            next_obs, reward, terminated, truncated, extras = self.env.step(action)
            done = terminated | truncated

            # Value of the true final state, for envs cut off by the time limit.
            bootstrap = torch.zeros_like(value)
            trunc_only = truncated & ~terminated
            if trunc_only.any():
                final = extras["final_observation"]
                _, final_value, _, _ = self.policy(
                    final["images"], final["proprio"], new_hidden
                )
                bootstrap = torch.where(trunc_only, final_value, bootstrap)

            self.buffer.add(
                images, proprio, raw, log_prob, value, reward, done,
                terminated=terminated, bootstrap_value=bootstrap,
            )

            # Recurrence must not cross an episode boundary.
            self.hidden = new_hidden * (~done).view(1, -1, 1).to(new_hidden.dtype)
            self.obs = next_obs

            log = extras["log"]
            j_sum += log["J"].mean().item()
            n += 1
            near_steps += int(log["near_mask"].sum().item())
            violations += int(log["mono_violation"].sum().item())
            ee_speed_sum += log["ee_speed"].mean().item()

        images, proprio = self.obs["images"], self.obs["proprio"]
        _, last_value, _, _ = self.policy(images, proprio, self.hidden)
        self.buffer.compute_returns(last_value, self.agent.cfg.gamma, self.agent.cfg.gae_lambda)

        return {
            "J_mean": j_sum / max(n, 1),
            "ee_speed": ee_speed_sum / max(n, 1),
            "mono_violation_rate": violations / max(near_steps, 1),
            "near_goal_frac": near_steps / max(n * self.env.num_envs, 1),
            "reward_mean": self.buffer.rewards.mean().item(),
        }

    # -------------------------------------------------------------------- train

    def train(self, max_iterations: Optional[int] = None) -> List[Dict[str, float]]:
        total = max_iterations if max_iterations is not None else self.cfg.max_iterations
        for _ in range(total):
            t0 = time.time()
            roll_stats = self.collect()
            update_stats = self.agent.update(self.buffer)
            self.iteration += 1

            stats = {**roll_stats, **update_stats, "iter": self.iteration, "time": time.time() - t0}
            self.history.append(stats)
            if self.cfg.log_interval and self.iteration % self.cfg.log_interval == 0:
                self._log(stats)
            if self.cfg.log_dir and self.cfg.save_interval and self.iteration % self.cfg.save_interval == 0:
                self.save(f"{self.cfg.log_dir}/ckpt_{self.iteration:06d}.pt")
        return self.history

    def _log(self, s: Dict[str, float]) -> None:
        print(
            f"[{int(s['iter']):5d}] J={s['J_mean']:8.4f}  R={s['reward_mean']:8.4f}  "
            f"mono_viol={s['mono_violation_rate']:.3f}  ee_speed={s['ee_speed']:.4f}  "
            f"kl={s['kl']:.4f}  lr={s['lr']:.2e}  clip={s['clip_frac']:.3f}  "
            f"{s['time']:.2f}s",
            flush=True,
        )

    # ---------------------------------------------------------------- checkpoint

    def save(self, path: str) -> None:
        torch.save(
            {
                "iteration": self.iteration,
                "policy": self.policy.state_dict(),
                "eval_policy": self.agent.eval_policy.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "lr": self.agent.lr,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.agent.optimizer.load_state_dict(ckpt["optimizer"])
        self.agent.lr = ckpt.get("lr", self.agent.lr)
        self.iteration = ckpt.get("iteration", 0)

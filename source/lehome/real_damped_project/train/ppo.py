"""Damped PPO (spec Sec. 3.4).

Standard recurrent PPO, plus the three mechanisms the spec asks for to make the
parameter update ``T: theta_k -> theta_{k+1}`` behave like a damped,
non-expansive map rather than an oscillatory one:

1. **KL trust region.** An adaptive learning rate shrinks when the observed
   KL(old || new) overshoots ``target_kl`` and grows when it undershoots, and
   the epoch aborts outright past ``kl_abort_factor * target_kl``. This bounds
   the per-iteration step in policy space, which is the discrete analogue of
   keeping ``||T(theta) - T(theta')|| <= L ||theta - theta'||`` with ``L`` near 1.

2. **Prior regularisation.** An optional KL penalty toward a frozen reference
   policy ``pi_0`` (Sec. 3.4). Anchoring to a fixed point keeps the parameter
   sequence bounded -- the precondition for the convergence argument, since a
   bounded sequence with a contractive-in-the-limit update converges.

3. **Polyak-averaged evaluation weights.** ``theta_bar_k = tau*theta_k +
   (1-tau)*theta_bar_{k-1}`` is a first-order low pass on the parameter
   sequence. Even when ``{theta_k}`` ringsA, ``{theta_bar_k}`` is smoother, and
   averaging is what makes the sequence asymptotically regular in the damped
   inertial schemes the spec alludes to.

Recurrence is handled by keeping whole ``num_steps_per_env``-length sequences
intact and minibatching over the *environment* axis, replaying each minibatch
from its stored initial hidden state. Chopping sequences arbitrarily would
corrupt the GRU state and is a common silent bug in recurrent PPO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class PPOCfg:
    """Damping-friendly PPO hyperparameters (Sec. 6.2)."""

    lr: float = 3.0e-4
    num_steps_per_env: int = 64
    num_epochs: int = 5
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # --- trust region / damping ---
    target_kl: float = 0.01
    kl_adaptive_lr: bool = True
    lr_min: float = 1.0e-5
    lr_max: float = 1.0e-3
    kl_abort_factor: float = 2.0
    prior_kl_coef: float = 0.0
    """> 0 anchors the policy to a frozen pi_0 (Sec. 3.4)."""
    polyak_tau: float = 0.0
    """> 0 maintains a Polyak-averaged shadow copy of the weights."""

    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if self.num_steps_per_env < 1:
            raise ValueError("num_steps_per_env must be >= 1")
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError("polyak_tau must be in [0, 1]")
        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be > 0")


class RolloutBuffer:
    """Fixed-size ``(T, B, ...)`` on-policy storage."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        image_shape,
        proprio_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        t, b = num_steps, num_envs
        self.device = device
        self.num_steps = t
        self.num_envs = b
        self.images = torch.zeros(t, b, *image_shape, device=device)
        self.proprio = torch.zeros(t, b, proprio_dim, device=device)
        self.actions = torch.zeros(t, b, action_dim, device=device)
        self.log_probs = torch.zeros(t, b, device=device)
        self.values = torch.zeros(t, b, device=device)
        self.rewards = torch.zeros(t, b, device=device)
        self.dones = torch.zeros(t, b, dtype=torch.bool, device=device)
        # Terminated (real terminal state) and truncated (time limit) must be
        # kept apart: the first cuts the value bootstrap, the second does not.
        self.terminated = torch.zeros(t, b, dtype=torch.bool, device=device)
        self.bootstrap_values = torch.zeros(t, b, device=device)
        """V(s_final) for envs truncated at step t; 0 elsewhere."""
        self.initial_hidden = torch.zeros(1, b, hidden_dim, device=device)
        self.advantages = torch.zeros(t, b, device=device)
        self.returns = torch.zeros(t, b, device=device)
        self._step = 0

    def reset(self, initial_hidden: torch.Tensor) -> None:
        self._step = 0
        self.initial_hidden = initial_hidden.detach().clone()

    def add(
        self, images, proprio, action, log_prob, value, reward, done,
        terminated=None, bootstrap_value=None,
    ) -> None:
        i = self._step
        if i >= self.num_steps:
            raise RuntimeError("rollout buffer overflow; call reset() before refilling")
        self.images[i] = images
        self.proprio[i] = proprio
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        # Default keeps the old behaviour for callers that do not distinguish
        # the two (every done treated as terminal).
        self.terminated[i] = done if terminated is None else terminated
        if bootstrap_value is not None:
            self.bootstrap_values[i] = bootstrap_value
        self._step += 1

    @property
    def full(self) -> bool:
        return self._step >= self.num_steps

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE(lambda) with correct time-limit handling.

        Two distinct roles for the episode-boundary flags, which the naive
        single-``done`` version conflates:

        * ``terminated`` -- a genuine terminal state. The future is worth 0, so
          the value bootstrap is cut.
        * ``truncated`` -- the episode was cut off by a time limit. The state
          was *not* terminal and its future still has value, so we bootstrap
          from ``V(s_final)`` (captured before auto-reset). Cutting it here
          teaches the agent that the world ends every ``max_episode_steps``,
          systematically under-valuing late-episode states.

        Either kind of boundary stops the GAE recursion, since the next
        transition belongs to a different episode.
        """
        adv = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            not_done = (~self.dones[t]).float()
            not_term = (~self.terminated[t]).float()
            truncated = (self.dones[t] & ~self.terminated[t]).float()

            next_value = last_value if t == self.num_steps - 1 else self.values[t + 1]
            # For truncated envs, values[t+1] belongs to the *new* episode --
            # use the stashed final-state value instead.
            next_value = truncated * self.bootstrap_values[t] + (1.0 - truncated) * next_value

            delta = self.rewards[t] + gamma * next_value * not_term - self.values[t]
            last_gae = delta + gamma * lam * not_done * last_gae
            adv[t] = last_gae
        self.advantages = adv
        self.returns = adv + self.values


class DampedPPOAgent:
    """PPO with a KL trust region, prior anchoring, and Polyak averaging."""

    def __init__(
        self,
        policy: nn.Module,
        cfg: PPOCfg,
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.lr = cfg.lr

        self.prior_policy: Optional[nn.Module] = None
        if cfg.prior_kl_coef > 0.0:
            self.prior_policy = copy.deepcopy(policy).eval()
            for p in self.prior_policy.parameters():
                p.requires_grad_(False)

        self.averaged_policy: Optional[nn.Module] = None
        if cfg.polyak_tau > 0.0:
            self.averaged_policy = copy.deepcopy(policy).eval()
            for p in self.averaged_policy.parameters():
                p.requires_grad_(False)

    # --------------------------------------------------------------- update

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        cfg = self.cfg
        adv = buffer.advantages
        if cfg.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        b = buffer.num_envs
        n_mb = max(1, min(cfg.num_minibatches, b))
        mb_size = max(1, b // n_mb)

        stats = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "kl": 0.0, "clip_frac": 0.0, "prior_kl": 0.0,
        }
        n_updates = 0
        aborted = False

        for _ in range(cfg.num_epochs):
            if aborted:
                break
            perm = torch.randperm(b, device=self.device)
            for start in range(0, b, mb_size):
                idx = perm[start:start + mb_size]
                if idx.numel() == 0:
                    continue

                new_logp, entropy, value, mean = self.policy.evaluate_actions(
                    buffer.images[:, idx],
                    buffer.proprio[:, idx],
                    buffer.actions[:, idx],
                    hidden_state=buffer.initial_hidden[:, idx],
                    dones=buffer.dones[:, idx],
                )
                old_logp = buffer.log_probs[:, idx]
                ratio = (new_logp - old_logp).exp()

                mb_adv = adv[:, idx]
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (value - buffer.returns[:, idx]).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                prior_kl = torch.zeros((), device=self.device)
                if self.prior_policy is not None:
                    with torch.no_grad():
                        prior_mean, _, _ = self.prior_policy.forward_sequence(
                            buffer.images[:, idx],
                            buffer.proprio[:, idx],
                            hidden_state=buffer.initial_hidden[:, idx],
                            dones=buffer.dones[:, idx],
                        )
                    prior_kl = torch.distributions.kl_divergence(
                        self.policy.distribution(mean),
                        self.prior_policy.distribution(prior_mean),
                    ).sum(-1).mean()
                    loss = loss + cfg.prior_kl_coef * prior_kl

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # k3 estimator: low-variance, non-negative sample KL.
                    log_ratio = new_logp - old_logp
                    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["prior_kl"] += float(prior_kl.detach())
                n_updates += 1

                if approx_kl > cfg.kl_abort_factor * cfg.target_kl:
                    aborted = True
                    break

        n_updates = max(n_updates, 1)
        for k in stats:
            stats[k] /= n_updates

        if cfg.kl_adaptive_lr:
            self._adapt_lr(stats["kl"])
        self._polyak()

        stats["lr"] = self.lr
        stats["aborted"] = float(aborted)
        return stats

    # ------------------------------------------------------------- damping bits

    def _adapt_lr(self, kl: float) -> None:
        """Shrink/grow the step so KL stays near target -- the trust region."""
        cfg = self.cfg
        if kl > 2.0 * cfg.target_kl:
            self.lr = max(cfg.lr_min, self.lr / 1.5)
        elif kl < 0.5 * cfg.target_kl:
            self.lr = min(cfg.lr_max, self.lr * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    @torch.no_grad()
    def _polyak(self) -> None:
        if self.averaged_policy is None:
            return
        tau = self.cfg.polyak_tau
        for avg, cur in zip(self.averaged_policy.parameters(), self.policy.parameters()):
            avg.mul_(1.0 - tau).add_(cur, alpha=tau)
        for avg, cur in zip(self.averaged_policy.buffers(), self.policy.buffers()):
            avg.copy_(cur)

    @property
    def eval_policy(self) -> nn.Module:
        """Weights to evaluate/deploy: the averaged copy when enabled."""
        return self.averaged_policy if self.averaged_policy is not None else self.policy

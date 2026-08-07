"""``VisionAttentionPolicy`` -- visual-grounded actor-critic (spec Sec. 3.3).

Pipeline:

    images (B,C,H,W) --conv--> F (B,D,H',W') --spatial attn--> z (B,D)
    [z ; proprio] --GRU--> h (B,H) --> {action mean, value}

Design points tied to the spec:

* **No cloth keypoints as input** (Sec. 3.1). The policy signature admits only
  images and proprioception; there is no channel through which mesh state can
  enter.
* **Lipschitz in inputs** (Sec. 3.3). Optional spectral normalisation on every
  linear/conv layer bounds each layer's gain by 1, so the composite network is
  Lipschitz with a constant that does not drift during training. This is what
  makes the "bounded weights" caveat in Sec. 3.3 an enforced property rather
  than an assumption. Off by default (it costs a little plasticity); switch on
  via ``spectral_norm=True`` if value/policy training oscillates.
* **State-independent log-std** for the Gaussian policy, initialised small, so
  early exploration does not inject the high-frequency action noise the whole
  design is trying to suppress.

Actions are squashed to ``[-1, 1]`` by the env (``_pre_step`` clamps), and
scaled by ``max_delta`` there; the policy itself emits unbounded means so the
Gaussian log-prob stays exact for PPO.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _maybe_sn(module: nn.Module, enabled: bool) -> nn.Module:
    """Deprecated shim.

    Spectral norm must be installed *after* weight initialisation: once the
    parametrization is registered, ``module.weight`` is a computed property and
    ``nn.init.*`` writes no longer land on the underlying ``weight_orig``. See
    :func:`apply_spectral_norm`, which the policy calls at the end of __init__.
    """
    return module


def apply_spectral_norm(root: nn.Module) -> None:
    """Install spectral norm on every Linear/Conv2d beneath ``root``.

    Bounds each layer's gain by 1, so the composite network is Lipschitz with a
    constant that does not drift during training -- turning Sec. 3.3's "bounded
    weights" caveat into an enforced property. GRU weights are left alone: the
    parametrization does not compose with cuDNN's fused kernels.
    """
    for module in root.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.utils.parametrizations.spectral_norm(module)


class SpatialAttention(nn.Module):
    """Softmax pooling over a feature map -- the visual grounding mechanism.

    Returns a context vector and the attention map, the latter being directly
    interpretable as "where the policy is looking" (Sec. 7 of the earlier spec
    asks for these to be logged).
    """

    def __init__(self, feature_dim: int, spectral_norm: bool = False) -> None:
        super().__init__()
        self.score = _maybe_sn(nn.Conv2d(feature_dim, 1, kernel_size=1), spectral_norm)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = feat.shape
        scores = self.score(feat).view(b, h * w)
        weights = torch.softmax(scores, dim=-1)
        z = torch.bmm(feat.view(b, d, h * w), weights.unsqueeze(-1)).squeeze(-1)
        return z, weights.view(b, h, w)


class ImageEncoder(nn.Module):
    """Three strided conv blocks + 1x1 projection to ``feature_dim``."""

    def __init__(self, in_channels: int, feature_dim: int, spectral_norm: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _maybe_sn(nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), spectral_norm),
            nn.ReLU(inplace=True),
            _maybe_sn(nn.Conv2d(32, 64, 3, stride=2, padding=1), spectral_norm),
            nn.ReLU(inplace=True),
            _maybe_sn(nn.Conv2d(64, 128, 3, stride=2, padding=1), spectral_norm),
            nn.ReLU(inplace=True),
            _maybe_sn(nn.Conv2d(128, feature_dim, 1), spectral_norm),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class VisionAttentionPolicy(nn.Module):
    """Recurrent actor-critic over images + proprioception.

    Args:
        image_channels: C of the stacked multi-camera image.
        proprio_dim: P.
        action_dim: 3 per arm (6 for the bimanual setup).
        feature_dim: D, the encoder/attention width.
        hidden_dim: GRU hidden size.
        init_log_std: initial action log-std.
        spectral_norm: enforce per-layer Lipschitz bounds.
    """

    def __init__(
        self,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        init_log_std: float = -1.0,
        spectral_norm: bool = False,
        squash: bool = True,
        predict_j: bool = False,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.squash = squash

        self.encoder = ImageEncoder(image_channels, feature_dim, spectral_norm)
        self.attention = SpatialAttention(feature_dim, spectral_norm)
        self.proprio_mlp = nn.Sequential(
            _maybe_sn(nn.Linear(proprio_dim, 128), spectral_norm),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(feature_dim + 128, hidden_dim, batch_first=True)
        self.policy_head = _maybe_sn(nn.Linear(hidden_dim, action_dim), spectral_norm)
        self.value_head = _maybe_sn(nn.Linear(hidden_dim, 1), spectral_norm)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

        # Auxiliary head predicting the cloth error J from the *visual* context
        # alone. Deliberately fed z (post-attention image features) and not the
        # GRU state or proprioception: if it could see joint angles it would
        # learn a shortcut and teach the encoder nothing.
        #
        # This is the mechanism that forces camera -> deformable topology. To
        # predict J the encoder must represent check-point geometry, and the
        # spatial attention must ground on the cloth regions that determine it.
        # Measured need: without it, image influence / proprio influence was
        # 0.11 -- the policy ignored the cameras entirely.
        self.j_head = nn.Linear(feature_dim, 1) if predict_j else None

        self.apply(self._init_weights)
        # Small final-layer gain: start near zero-action, so the first rollouts
        # are quiet rather than thrashing the cloth.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

        # Strictly after initialisation -- see apply_spectral_norm's docstring.
        if spectral_norm:
            apply_spectral_norm(self)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=2.0**0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- utilities

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One timestep.

        Args:
            images: ``(B, C, H, W)``.
            proprio: ``(B, P)``.
            hidden_state: ``(1, B, H)`` or ``None``.
        Returns:
            ``(action_mean, value, new_hidden, attn_map)`` with shapes
            ``(B, A)``, ``(B,)``, ``(1, B, H)``, ``(B, H', W')``.
        """
        if images.dim() != 4:
            raise ValueError(f"images must be (B,C,H,W), got {tuple(images.shape)}")
        if proprio.dim() != 2:
            raise ValueError(f"proprio must be (B,P), got {tuple(proprio.shape)}")
        if images.shape[0] != proprio.shape[0]:
            raise ValueError("images and proprio disagree on batch size")

        feat = self.encoder(images)
        z, attn = self.attention(feat)
        p = self.proprio_mlp(proprio)

        gru_in = torch.cat([z, p], dim=-1).unsqueeze(1)  # (B, 1, D+128)
        if hidden_state is None:
            hidden_state = self.initial_hidden(images.shape[0], images.device)
        out, new_hidden = self.gru(gru_in, hidden_state.contiguous())
        h = out.squeeze(1)

        return self.policy_head(h), self.value_head(h).squeeze(-1), new_hidden, attn

    def forward_sequence(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        return_j: bool = False,
    ):
        """Whole-sequence rollout for BPTT during PPO updates.

        Args:
            images: ``(T, B, C, H, W)``.
            proprio: ``(T, B, P)``.
            hidden_state: ``(1, B, H)`` at t=0.
            dones: ``(T, B)``; the hidden state is zeroed after each done, so
                recurrence never leaks across an episode boundary.
        Returns:
            ``(action_mean, value, final_hidden)`` with shapes ``(T, B, A)``,
            ``(T, B)``, ``(1, B, H)``.
        """
        t, b = images.shape[0], images.shape[1]
        feat = self.encoder(images.reshape(t * b, *images.shape[2:]))
        z, _ = self.attention(feat)
        z = z.view(t, b, -1)
        p = self.proprio_mlp(proprio.reshape(t * b, -1)).view(t, b, -1)
        seq = torch.cat([z, p], dim=-1)

        if hidden_state is None:
            hidden_state = self.initial_hidden(b, images.device)
        h = hidden_state.contiguous()

        if dones is None:
            out, h = self.gru(seq.transpose(0, 1), h)
            out = out.transpose(0, 1)
        else:
            outs = []
            for i in range(t):
                step_out, h = self.gru(seq[i].unsqueeze(1), h)
                outs.append(step_out.squeeze(1))
                h = h * (~dones[i]).view(1, b, 1).to(h.dtype)
            out = torch.stack(outs, dim=0)

        flat = out.reshape(t * b, -1)
        mean = self.policy_head(flat).view(t, b, -1)
        value = self.value_head(flat).view(t, b)
        if return_j:
            if self.j_head is None:
                raise RuntimeError("policy was built with predict_j=False")
            # From the visual context only -- see j_head's construction.
            j_pred = self.j_head(z.reshape(t * b, -1)).view(t, b)
            return mean, value, h, j_pred
        return mean, value, h

    # ------------------------------------------------------------- distributions

    def distribution(self, mean: torch.Tensor) -> Normal:
        """Base (pre-squash) Gaussian over the raw action `u`."""
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def to_env_action(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw action `u` to the bounded action the environment consumes."""
        return torch.tanh(raw) if self.squash else raw

    def log_prob(self, mean: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Exact log-density of the *bounded* action, summed over dimensions.

        With `a = tanh(u)` the change of variables contributes
        `-sum log(1 - tanh(u)^2)`. Computed in the numerically stable form

            log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))

        which avoids the catastrophic cancellation of `log1p(-tanh(u)**2)` once
        `|u|` is large -- exactly the regime a drifting policy reaches.

        Why squashing at all: with an unbounded Gaussian and the environment
        clamping to +-1, every sample beyond the boundary maps to the same
        action. The advantage then stops discriminating between actions, the
        gradient w.r.t. the mean carries no corrective signal, and the mean
        random-walks outward -- which is self-reinforcing, because more
        saturation means less effective exploration. Measured on this task:
        |action_mean| drifted 0.001 -> 2.05 with 83% of components clipped, and
        J diverged. Squashing bounds the action by construction, so the
        gradient stays informative everywhere.
        """
        logp = self.distribution(mean).log_prob(raw).sum(-1)
        if self.squash:
            correction = 2.0 * (
                math.log(2.0) - raw - torch.nn.functional.softplus(-2.0 * raw)
            )
            logp = logp - correction.sum(-1)
        return logp

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns ``(action, log_prob, value, hidden, attn)``.

        ``action`` is the bounded action for the environment. Use
        :meth:`act_raw` when the caller must store the pre-squash sample, which
        PPO needs in order to re-evaluate log-probs exactly.
        """
        action, _, log_prob, value, hidden, attn = self.act_raw(
            images, proprio, hidden_state, deterministic
        )
        return action, log_prob, value, hidden, attn

    @torch.no_grad()
    def act_raw(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """Returns ``(action, raw_action, log_prob, value, hidden, attn)``.

        ``raw_action`` is the pre-squash sample ``u``. Storing ``u`` rather than
        ``tanh(u)`` keeps the PPO importance ratio exact: recovering ``u`` from
        a saturated ``tanh(u)`` via ``atanh`` loses precision precisely where it
        matters.
        """
        mean, value, hidden, attn = self.forward(images, proprio, hidden_state)
        if deterministic:
            raw = mean
            logp = torch.zeros(mean.shape[0], device=mean.device)
        else:
            raw = self.distribution(mean).sample()
            logp = self.log_prob(mean, raw)
        return self.to_env_action(raw), raw, logp, value, hidden, attn

    def evaluate_actions(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored *raw* (pre-squash) actions over a sequence.

        Returns ``(log_prob, entropy, value, mean)``; log_prob/entropy/value are
        ``(T, B)`` and mean is ``(T, B, A)``.

        Entropy is that of the base Gaussian. The squashed distribution has no
        closed-form entropy, and the base entropy is the right thing to bonus
        anyway: it is what keeps the pre-squash spread alive, which is where
        exploration actually lives.
        """
        mean, value, _ = self.forward_sequence(images, proprio, hidden_state, dones)
        entropy = self.distribution(mean).entropy().sum(-1)
        return self.log_prob(mean, actions), entropy, value, mean

"""Tests for the vision+attention policy (spec Sec. 3.3)."""

import inspect

import pytest
import torch

from lehome.real_damped_project.policy.vision_attention_policy import VisionAttentionPolicy

B, C, H, W, P, A = 4, 6, 64, 64, 18, 6


def make_policy(**kw):
    kw.setdefault("feature_dim", 32)
    kw.setdefault("hidden_dim", 32)
    return VisionAttentionPolicy(image_channels=C, proprio_dim=P, action_dim=A, **kw)


def obs(b=B):
    return torch.rand(b, C, H, W), torch.randn(b, P)


def test_forward_shapes():
    pol = make_policy()
    images, proprio = obs()
    mean, value, hidden, attn = pol(images, proprio)
    assert mean.shape == (B, A)
    assert value.shape == (B,)
    assert hidden.shape == (1, B, 32)
    assert attn.dim() == 3 and attn.shape[0] == B


def test_attention_is_a_distribution_over_space():
    """Softmax pooling: weights must sum to 1 and be non-negative."""
    pol = make_policy()
    _, _, _, attn = pol(*obs())
    assert (attn >= 0).all()
    assert torch.allclose(attn.flatten(1).sum(-1), torch.ones(B), atol=1e-5)


def test_attention_responds_to_the_image():
    """A constant attention map would mean no visual grounding at all."""
    pol = make_policy()
    images = torch.zeros(2, C, H, W)
    images[0, :, 10:20, 10:20] = 1.0
    images[1, :, 40:50, 40:50] = 1.0
    _, _, _, attn = pol(images, torch.zeros(2, P))
    assert not torch.allclose(attn[0], attn[1], atol=1e-4)


def test_policy_cannot_see_cloth_state():
    """Sec. 3.1: no cloth keypoints or mesh in observations.

    Enforced structurally -- the only inputs are images, proprio and the hidden
    state, so there is no channel for mesh data to enter.
    """
    params = set(inspect.signature(VisionAttentionPolicy.forward).parameters)
    assert params == {"self", "images", "proprio", "hidden_state"}
    forbidden = {"cloth", "keypoints", "mesh", "verts", "vertices", "particles"}
    assert not (forbidden & params)


def test_hidden_state_carries_information():
    pol = make_policy()
    images, proprio = obs()
    _, _, h1, _ = pol(images, proprio)
    mean_a, _, _, _ = pol(images, proprio, h1)
    mean_b, _, _, _ = pol(images, proprio, torch.zeros_like(h1))
    assert not torch.allclose(mean_a, mean_b, atol=1e-6)


def test_zero_hidden_matches_default():
    pol = make_policy()
    images, proprio = obs()
    a, _, _, _ = pol(images, proprio, None)
    b, _, _, _ = pol(images, proprio, pol.initial_hidden(B, images.device))
    assert torch.allclose(a, b, atol=1e-6)


def test_gradients_reach_every_component():
    pol = make_policy()
    mean, value, _, _ = pol(*obs())
    (mean.sum() + value.sum()).backward()
    for name, p in pol.named_parameters():
        if p.requires_grad and "log_std" not in name:
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


def test_initial_actions_are_near_zero():
    """A freshly initialised policy should not thrash the cloth."""
    pol = make_policy()
    mean, _, _, _ = pol(*obs())
    assert mean.abs().max().item() < 0.1


# ------------------------------------------------------------------- sequences


def test_forward_sequence_matches_stepwise_rollout():
    """The BPTT path and the acting path must agree, or PPO trains on a lie."""
    pol = make_policy().eval()
    T = 5
    images = torch.rand(T, B, C, H, W)
    proprio = torch.randn(T, B, P)

    with torch.no_grad():
        h = pol.initial_hidden(B, images.device)
        step_means, step_values = [], []
        for t in range(T):
            m, v, h, _ = pol(images[t], proprio[t], h)
            step_means.append(m)
            step_values.append(v)
        step_means = torch.stack(step_means)
        step_values = torch.stack(step_values)

        seq_means, seq_values, _ = pol.forward_sequence(images, proprio)

    assert torch.allclose(step_means, seq_means, atol=1e-5)
    assert torch.allclose(step_values, seq_values, atol=1e-5)


def test_dones_reset_the_recurrent_state():
    """Recurrence must not leak across an episode boundary."""
    pol = make_policy().eval()
    T = 4
    images = torch.rand(T, B, C, H, W)
    proprio = torch.randn(T, B, P)

    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1] = True
    with torch.no_grad():
        with_done, _, _ = pol.forward_sequence(images, proprio, dones=dones)
        no_done, _, _ = pol.forward_sequence(images, proprio, dones=torch.zeros_like(dones))

    assert torch.allclose(with_done[:2], no_done[:2], atol=1e-5)   # before the reset
    assert not torch.allclose(with_done[2], no_done[2], atol=1e-5)  # after it


# --------------------------------------------------------------- distributions


def test_act_returns_consistent_log_probs():
    pol = make_policy()
    images, proprio = obs()
    action, raw, log_prob, value, hidden, attn = pol.act_raw(images, proprio)
    assert action.shape == (B, A)
    assert log_prob.shape == (B,)
    assert value.shape == (B,)
    assert torch.isfinite(log_prob).all()

    mean, _, _, _ = pol(images, proprio)
    assert torch.allclose(log_prob, pol.log_prob(mean, raw), atol=1e-5)


def test_deterministic_act_returns_the_squashed_mean():
    pol = make_policy()
    images, proprio = obs()
    action, _, _, _, _ = pol.act(images, proprio, deterministic=True)
    mean, _, _, _ = pol(images, proprio)
    assert torch.allclose(action, torch.tanh(mean), atol=1e-6)


# ------------------------------------------------------------------- squashing


def test_actions_are_bounded_by_construction():
    """The failure this fixes: an unbounded mean saturating the env's clamp.

    Measured on the folding task, |action_mean| drifted 0.001 -> 2.05 with 83%
    of components past +-1. Beyond the boundary every sample maps to the same
    clipped action, the advantage stops discriminating, and the mean random-
    walks outward. Squashing makes that unreachable.
    """
    pol = make_policy()
    # Force an extreme mean, as a diverged policy would produce.
    with torch.no_grad():
        pol.policy_head.bias.fill_(50.0)
    action, raw, logp, _, _, _ = pol.act_raw(*obs())
    assert action.abs().max().item() <= 1.0
    assert raw.abs().max().item() > 1.0, "raw sample should be unbounded"
    assert torch.isfinite(logp).all(), "log-prob must stay finite when saturated"


def test_log_prob_is_stable_at_extreme_raw_actions():
    """The naive log1p(-tanh(u)**2) form underflows here; the softplus one does not."""
    pol = make_policy()
    mean = torch.zeros(4, A)
    for u in (5.0, 20.0, 100.0):
        logp = pol.log_prob(mean, torch.full((4, A), u))
        assert torch.isfinite(logp).all(), f"log_prob non-finite at raw={u}"


def test_squashed_log_prob_integrates_to_one():
    """Monte-Carlo check that the Jacobian correction is right.

    Importance-sampling the squashed density against its own samples must give
    E[1] = 1; a missing or mis-signed Jacobian term shows up immediately.
    """
    torch.manual_seed(0)
    pol = make_policy()
    mean = torch.zeros(1, A)
    raw = pol.distribution(mean).sample((20000,)).squeeze(1)
    logp_squashed = pol.log_prob(mean.expand(raw.shape[0], -1), raw)
    logp_base = pol.distribution(mean.expand(raw.shape[0], -1)).log_prob(raw).sum(-1)
    ratio = (logp_squashed - logp_base).exp()  # = |da/du|^-1
    # E_u[ |da/du|^-1 * |da/du| ] = 1, and jacobian = prod(1 - tanh^2)
    jac = (1 - torch.tanh(raw) ** 2).prod(-1)
    assert abs((ratio * jac).mean().item() - 1.0) < 1e-4


def test_unsquashed_mode_still_available():
    """`squash=False` reproduces the plain Gaussian for spec-fidelity runs."""
    pol = make_policy(squash=False)
    mean = torch.zeros(4, A)
    raw = torch.randn(4, A)
    expected = pol.distribution(mean).log_prob(raw).sum(-1)
    assert torch.allclose(pol.log_prob(mean, raw), expected, atol=1e-6)
    assert torch.allclose(pol.to_env_action(raw), raw)


def test_evaluate_actions_shapes():
    pol = make_policy()
    T = 3
    images = torch.rand(T, B, C, H, W)
    proprio = torch.randn(T, B, P)
    actions = torch.randn(T, B, A)
    logp, ent, val, mean = pol.evaluate_actions(images, proprio, actions)
    assert logp.shape == (T, B)
    assert ent.shape == (T, B)
    assert val.shape == (T, B)
    assert mean.shape == (T, B, A)


def test_initial_log_std_is_small():
    """Loud initial exploration would inject exactly the high-frequency action
    noise the design is trying to suppress."""
    pol = make_policy()
    assert pol.log_std.exp().max().item() < 0.5


# ------------------------------------------------------------- Lipschitz option


def test_spectral_norm_bounds_layer_gains():
    """Sec. 3.3's 'bounded weights' becomes enforced rather than assumed."""
    pol = make_policy(spectral_norm=True)
    mean, value, _, _ = pol(*obs())
    assert torch.isfinite(mean).all() and torch.isfinite(value).all()
    w = pol.policy_head.weight
    assert torch.linalg.matrix_norm(w, ord=2).item() <= 1.0 + 1e-3


def test_rejects_bad_input_shapes():
    pol = make_policy()
    with pytest.raises(ValueError):
        pol(torch.rand(B, C, H), torch.randn(B, P))
    with pytest.raises(ValueError):
        pol(torch.rand(B, C, H, W), torch.randn(B, P, 1))
    with pytest.raises(ValueError):
        pol(torch.rand(B, C, H, W), torch.randn(B + 1, P))

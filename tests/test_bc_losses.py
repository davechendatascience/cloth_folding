"""Tests for the J-aware BC formulation.

The first BC attempt optimised action similarity with no reference to J, and
learned to ignore the cameras (image influence / proprio influence = 0.11)
while producing a policy indistinguishable from doing nothing. These pin the
three changes that connect the imitation loss to the Lyapunov framework.
"""

import numpy as np
import pytest
import torch

from lehome.real_damped_project.policy.vision_attention_policy import VisionAttentionPolicy
from lehome.real_damped_project.train.train_bc import lyapunov_weights


# ------------------------------------------------------- Lyapunov weighting


def test_descending_transitions_are_weighted_above_ascending():
    """The whole point: copy the parts of a demonstration where J fell."""
    dJ = torch.tensor([-1.0, 0.0, 1.0])
    w = lyapunov_weights(dJ, beta=1.0)
    assert w[0] > w[1] > w[2], f"weights not ordered by descent: {w}"


def test_weights_normalise_to_mean_one():
    """Keeps the loss scale independent of beta, so beta changes *which*
    transitions dominate without also changing the effective learning rate."""
    for beta in (0.1, 1.0, 10.0):
        w = lyapunov_weights(torch.randn(256), beta=beta)
        assert abs(float(w.mean()) - 1.0) < 1e-5


def test_small_beta_sharpens_the_preference():
    dJ = torch.tensor([-1.0, 0.0, 1.0])
    sharp = lyapunov_weights(dJ, beta=0.25)
    soft = lyapunov_weights(dJ, beta=4.0)
    assert sharp[0] / sharp[2] > soft[0] / soft[2]


def test_weights_are_clipped_against_outliers():
    """A single huge dJ must not swamp the batch."""
    dJ = torch.tensor([-100.0, 0.0, 0.0, 0.0])
    w = lyapunov_weights(dJ, beta=0.1, clip=5.0)
    assert torch.isfinite(w).all()
    assert float(w.max()) < 100.0


def test_zero_dJ_gives_uniform_weights():
    """No labels (all zeros) must degrade to plain BC, not to garbage."""
    w = lyapunov_weights(torch.zeros(64), beta=1.0)
    assert torch.allclose(w, torch.ones(64), atol=1e-6)


# ------------------------------------------------------------ delta target


def test_delta_target_defeats_the_proprio_shortcut():
    """a[t] is predictable from s[t]; a[t]-s[t] is not.

    This is the measured failure: because a[t] ~ s[t] at 30 fps, proprio
    predicts the absolute target through a near-identity map and captures
    almost all the loss, leaving no gradient pressure on the visual encoder.
    """
    rng = np.random.RandomState(0)
    s = np.cumsum(rng.randn(2000, 12) * 0.01, axis=0)      # smooth joint traj
    a = s + rng.randn(2000, 12) * 0.02                      # target near state

    # best linear predictor of the target from state alone, in each convention
    def residual(target):
        coef, *_ = np.linalg.lstsq(
            np.hstack([s, np.ones((len(s), 1))]), target, rcond=None
        )
        pred = np.hstack([s, np.ones((len(s), 1))]) @ coef
        return float(((target - pred) ** 2).mean() / target.var())

    abs_unexplained = residual(a)
    delta_unexplained = residual(a - s)
    assert abs_unexplained < 0.05, "proprio should explain the absolute target"
    assert delta_unexplained > 0.5, "proprio must NOT explain the delta target"


# --------------------------------------------------- auxiliary J prediction


def test_j_head_exists_only_when_requested():
    assert VisionAttentionPolicy(9, 12, 12, feature_dim=16, hidden_dim=16).j_head is None
    assert VisionAttentionPolicy(9, 12, 12, feature_dim=16, hidden_dim=16,
                                 predict_j=True).j_head is not None


def test_forward_sequence_returns_j_prediction():
    pol = VisionAttentionPolicy(9, 12, 12, feature_dim=16, hidden_dim=16, predict_j=True)
    img, prop = torch.rand(4, 2, 9, 84, 84), torch.randn(4, 2, 12)
    mean, value, h, j = pol.forward_sequence(img, prop, return_j=True)
    assert mean.shape == (4, 2, 12) and value.shape == (4, 2) and j.shape == (4, 2)


def test_requesting_j_without_the_head_is_an_error_not_a_silent_zero():
    pol = VisionAttentionPolicy(9, 12, 12, feature_dim=16, hidden_dim=16)
    with pytest.raises(RuntimeError, match="predict_j=False"):
        pol.forward_sequence(torch.rand(2, 1, 9, 84, 84), torch.randn(2, 1, 12), return_j=True)


def test_j_prediction_depends_on_images_not_proprioception():
    """The head is fed the visual context only.

    If it could see joint angles it would learn a shortcut and teach the
    encoder nothing -- which is precisely the failure this is meant to fix.
    """
    torch.manual_seed(0)
    pol = VisionAttentionPolicy(9, 12, 12, feature_dim=16, hidden_dim=16, predict_j=True).eval()
    img, prop = torch.rand(3, 2, 9, 84, 84), torch.randn(3, 2, 12)
    with torch.no_grad():
        _, _, _, j_base = pol.forward_sequence(img, prop, return_j=True)
        _, _, _, j_newprop = pol.forward_sequence(img, torch.randn_like(prop), return_j=True)
        _, _, _, j_newimg = pol.forward_sequence(torch.rand_like(img), prop, return_j=True)
    assert torch.allclose(j_base, j_newprop, atol=1e-6), "J head saw proprioception"
    assert not torch.allclose(j_base, j_newimg, atol=1e-4), "J head ignored the images"


def test_j_gradient_reaches_the_visual_encoder():
    """The mechanism that forces camera -> deformable topology grounding."""
    pol = VisionAttentionPolicy(9, 12, 12, feature_dim=16, hidden_dim=16, predict_j=True)
    img, prop = torch.rand(3, 2, 9, 84, 84), torch.randn(3, 2, 12)
    _, _, _, j = pol.forward_sequence(img, prop, return_j=True)
    j.pow(2).mean().backward()
    g = pol.encoder.net[0].weight.grad
    assert g is not None and float(g.abs().sum()) > 0, "no gradient into the image encoder"

"""Tests for damped PPO and the runner (spec Sec. 3.4, 6.2)."""

import copy

import pytest
import torch

from lehome.real_damped_project.policy.vision_attention_policy import VisionAttentionPolicy
from lehome.real_damped_project.tasks.cfg import make_env
from lehome.real_damped_project.train.ppo import DampedPPOAgent, PPOCfg, RolloutBuffer
from lehome.real_damped_project.train.runner import Runner, RunnerCfg

T, B, A, HID = 6, 4, 6, 16
IMG = (6, 16, 16)
P = 18


def make_buffer():
    return RolloutBuffer(T, B, IMG, P, A, HID, torch.device("cpu"))


def fill(buf):
    buf.reset(torch.zeros(1, B, HID))
    for _ in range(T):
        buf.add(
            torch.rand(B, *IMG), torch.randn(B, P), torch.randn(B, A),
            torch.randn(B), torch.randn(B), torch.randn(B),
            torch.zeros(B, dtype=torch.bool),
        )
    return buf


def make_agent(**kw):
    pol = VisionAttentionPolicy(IMG[0], P, A, feature_dim=16, hidden_dim=HID)
    return DampedPPOAgent(pol, PPOCfg(num_steps_per_env=T, num_minibatches=2, **kw))


# ------------------------------------------------------------------ the buffer


def test_buffer_fills_and_reports_full():
    buf = fill(make_buffer())
    assert buf.full
    with pytest.raises(RuntimeError, match="overflow"):
        buf.add(torch.rand(B, *IMG), torch.randn(B, P), torch.randn(B, A),
                torch.randn(B), torch.randn(B), torch.randn(B),
                torch.zeros(B, dtype=torch.bool))


def test_gae_matches_a_reference_implementation():
    buf = fill(make_buffer())
    gamma, lam = 0.99, 0.95
    last_value = torch.randn(B)
    buf.compute_returns(last_value, gamma, lam)

    expected = torch.zeros(T, B)
    gae = torch.zeros(B)
    for t in reversed(range(T)):
        nd = (~buf.dones[t]).float()
        nv = last_value if t == T - 1 else buf.values[t + 1]
        delta = buf.rewards[t] + gamma * nv * nd - buf.values[t]
        gae = delta + gamma * lam * nd * gae
        expected[t] = gae

    assert torch.allclose(buf.advantages, expected, atol=1e-6)
    assert torch.allclose(buf.returns, expected + buf.values, atol=1e-6)


def test_time_limit_truncation_bootstraps_instead_of_cutting():
    """A time limit is not a terminal state.

    Cutting the bootstrap at truncation teaches the agent the world ends every
    `max_episode_steps`, systematically under-valuing late-episode states. With
    a constant reward and V(s_final)=10, the truncated step's advantage must
    reflect that bootstrap rather than assuming zero future value.
    """
    buf = make_buffer()
    buf.reset(torch.zeros(1, B, HID))
    for t in range(T):
        trunc = torch.zeros(B, dtype=torch.bool)
        term = torch.zeros(B, dtype=torch.bool)
        boot = torch.zeros(B)
        if t == 2:
            trunc[:] = True          # time limit, NOT a terminal state
            boot[:] = 10.0
        buf.add(torch.rand(B, *IMG), torch.randn(B, P), torch.randn(B, A),
                torch.zeros(B), torch.zeros(B), torch.ones(B),
                trunc | term, terminated=term, bootstrap_value=boot)
    buf.compute_returns(torch.zeros(B), gamma=0.99, lam=0.95)

    # r + gamma * V(s_final) - V(s) = 1 + 0.99*10 - 0 = 10.9
    assert torch.allclose(buf.advantages[2], torch.full((B,), 10.9), atol=1e-5)


def test_real_termination_still_cuts_the_bootstrap():
    """Control for the test above: a genuine terminal state has no future."""
    buf = make_buffer()
    buf.reset(torch.zeros(1, B, HID))
    for t in range(T):
        term = torch.zeros(B, dtype=torch.bool)
        boot = torch.zeros(B)
        if t == 2:
            term[:] = True
            boot[:] = 10.0  # must be ignored -- the episode really ended
        buf.add(torch.rand(B, *IMG), torch.randn(B, P), torch.randn(B, A),
                torch.zeros(B), torch.zeros(B), torch.ones(B),
                term, terminated=term, bootstrap_value=boot)
    buf.compute_returns(torch.zeros(B), gamma=0.99, lam=0.95)
    assert torch.allclose(buf.advantages[2], torch.ones(B), atol=1e-6)


def test_gae_truncates_at_episode_boundaries():
    """Advantage must not propagate across a done flag."""
    buf = make_buffer()
    buf.reset(torch.zeros(1, B, HID))
    for t in range(T):
        done = torch.zeros(B, dtype=torch.bool)
        if t == 2:
            done[:] = True
        buf.add(torch.rand(B, *IMG), torch.randn(B, P), torch.randn(B, A),
                torch.zeros(B), torch.zeros(B), torch.ones(B), done)
    buf.compute_returns(torch.zeros(B), gamma=0.99, lam=0.95)
    # At t=2 the bootstrap is cut, so the advantage is exactly the immediate reward.
    assert torch.allclose(buf.advantages[2], torch.ones(B), atol=1e-6)


# ------------------------------------------------------------------- the update


def test_update_runs_and_changes_parameters():
    agent = make_agent()
    buf = fill(make_buffer())
    buf.compute_returns(torch.zeros(B), 0.99, 0.95)
    before = copy.deepcopy([p.detach().clone() for p in agent.policy.parameters()])

    stats = agent.update(buf)

    assert set(stats) >= {"policy_loss", "value_loss", "entropy", "kl", "clip_frac", "lr"}
    assert all(v == v for v in stats.values())  # no NaN
    after = list(agent.policy.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_update_keeps_parameters_finite():
    agent = make_agent()
    for _ in range(3):
        buf = fill(make_buffer())
        buf.compute_returns(torch.zeros(B), 0.99, 0.95)
        agent.update(buf)
    for name, p in agent.policy.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter {name}"


def test_approx_kl_is_nonnegative():
    """The k3 estimator is used precisely so KL cannot come out negative."""
    agent = make_agent()
    for _ in range(3):
        buf = fill(make_buffer())
        buf.compute_returns(torch.zeros(B), 0.99, 0.95)
        assert agent.update(buf)["kl"] >= -1e-9


# ----------------------------------------------------------- trust region / damping


def test_lr_shrinks_when_kl_overshoots():
    agent = make_agent(target_kl=1e-8, lr=1e-3)  # any real step overshoots
    buf = fill(make_buffer())
    buf.compute_returns(torch.zeros(B), 0.99, 0.95)
    before = agent.lr
    agent.update(buf)
    assert agent.lr < before


def test_lr_grows_when_kl_undershoots():
    agent = make_agent(target_kl=1e6, lr=1e-4)  # nothing can reach this KL
    buf = fill(make_buffer())
    buf.compute_returns(torch.zeros(B), 0.99, 0.95)
    before = agent.lr
    agent.update(buf)
    assert agent.lr > before


def test_lr_respects_its_bounds():
    agent = make_agent(target_kl=1e-12, lr=1e-3, lr_min=5e-4)
    for _ in range(20):
        buf = fill(make_buffer())
        buf.compute_returns(torch.zeros(B), 0.99, 0.95)
        agent.update(buf)
    assert agent.lr >= 5e-4 - 1e-12


def test_epoch_aborts_past_the_kl_ceiling():
    agent = make_agent(target_kl=1e-9, kl_abort_factor=1.0, lr=1e-2)
    buf = fill(make_buffer())
    buf.compute_returns(torch.zeros(B), 0.99, 0.95)
    assert agent.update(buf)["aborted"] == 1.0


def test_prior_regularisation_is_wired_up():
    agent = make_agent(prior_kl_coef=1.0)
    assert agent.prior_policy is not None
    assert not any(p.requires_grad for p in agent.prior_policy.parameters())

    buf = fill(make_buffer())
    buf.compute_returns(torch.zeros(B), 0.99, 0.95)
    stats = agent.update(buf)
    assert stats["prior_kl"] >= 0.0

    # The prior must stay frozen while the policy moves.
    before = [p.clone() for p in agent.prior_policy.parameters()]
    buf = fill(make_buffer())
    buf.compute_returns(torch.zeros(B), 0.99, 0.95)
    agent.update(buf)
    assert all(torch.equal(a, b) for a, b in zip(before, agent.prior_policy.parameters()))


def test_polyak_average_lags_the_live_policy():
    """The averaged weights are a low pass on the parameter sequence."""
    agent = make_agent(polyak_tau=0.1, lr=1e-2)
    assert agent.averaged_policy is not None
    initial = [p.clone() for p in agent.averaged_policy.parameters()]

    for _ in range(3):
        buf = fill(make_buffer())
        buf.compute_returns(torch.zeros(B), 0.99, 0.95)
        agent.update(buf)

    live = list(agent.policy.parameters())
    avg = list(agent.averaged_policy.parameters())
    moved_live = sum((a - b).abs().sum().item() for a, b in zip(live, initial))
    moved_avg = sum((a - b).abs().sum().item() for a, b in zip(avg, initial))
    assert moved_avg > 0, "averaged policy never updated"
    assert moved_avg < moved_live, "averaging did not damp the parameter trajectory"


def test_eval_policy_selection():
    assert make_agent(polyak_tau=0.0).eval_policy is not None
    a = make_agent(polyak_tau=0.1)
    assert a.eval_policy is a.averaged_policy


def test_rejects_bad_cfg():
    with pytest.raises(ValueError):
        PPOCfg(num_steps_per_env=0)
    with pytest.raises(ValueError):
        PPOCfg(polyak_tau=1.5)
    with pytest.raises(ValueError):
        PPOCfg(clip_ratio=0.0)


# ---------------------------------------------------------------------- runner


@pytest.fixture
def runner():
    env = make_env(num_envs=4, sim_device="cpu", use_mock_backend=True,
                   cfg_overrides={"image_res": 16})
    pol = VisionAttentionPolicy(
        env.observation_shapes["images"][0],
        env.observation_shapes["proprio"][0],
        env.action_dim,
        feature_dim=16, hidden_dim=HID,
    )
    agent = DampedPPOAgent(pol, PPOCfg(num_steps_per_env=8, num_minibatches=2))
    return Runner(env, agent, RunnerCfg(max_iterations=2, log_interval=0))


def test_runner_collect_produces_diagnostics(runner):
    stats = runner.collect()
    for key in ("J_mean", "ee_speed", "mono_violation_rate", "near_goal_frac", "reward_mean"):
        assert key in stats
        assert stats[key] == stats[key]
    assert 0.0 <= stats["mono_violation_rate"] <= 1.0
    assert 0.0 <= stats["near_goal_frac"] <= 1.0
    assert runner.buffer.full


def test_runner_trains_without_error(runner):
    history = runner.train(max_iterations=2)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(h["reward_mean"])) for h in history)
    for name, p in runner.policy.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter {name} after training"


def test_runner_checkpoint_roundtrip(runner, tmp_path):
    runner.train(max_iterations=1)
    path = str(tmp_path / "ckpt.pt")
    runner.save(path)
    before = [p.clone() for p in runner.policy.parameters()]

    runner.train(max_iterations=1)
    runner.load(path)
    after = list(runner.policy.parameters())
    assert all(torch.allclose(a, b) for a, b in zip(before, after))
    assert runner.iteration == 1

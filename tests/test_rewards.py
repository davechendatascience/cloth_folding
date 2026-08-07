"""Tests for the discrete Lyapunov descent reward (spec Sec. 2.3)."""

import pytest
import torch

from lehome.real_damped_project.tasks.rewards import LyapunovDescentReward, LyapunovRewardCfg

N = 4
A = 6


def make_reward(**kw):
    cfg = LyapunovRewardCfg(**kw)
    return LyapunovDescentReward(cfg, num_envs=N, action_dim=A), cfg


def zeros_vel():
    return torch.zeros(N, 2, 3)


def zeros_act():
    return torch.zeros(N, A)


# ------------------------------------------------------------------- task term


def test_task_term_is_negative_J():
    r, _ = make_reward(lambda_v=0.0, lambda_delta_a=0.0)
    j = torch.tensor([0.5, 1.0, 0.0, 2.0])
    reward, comps = r.compute(j, zeros_vel(), zeros_act())
    assert torch.allclose(comps["r_task"], -j)
    assert torch.allclose(reward, -j)  # first step: no r_mono


def test_first_step_of_an_episode_has_no_mono_term():
    """dJ across a reset is not a dynamical transition and must not be scored."""
    r, _ = make_reward(lambda_v=0.0, lambda_delta_a=0.0)
    _, comps = r.compute(torch.full((N,), 0.001), zeros_vel(), zeros_act())
    assert torch.allclose(comps["r_mono"], torch.zeros(N))
    assert not comps["near_mask"].any()


# ------------------------------------------------------------------- mono term


def test_error_increase_near_goal_is_penalised():
    r, cfg = make_reward(j_near=0.02, epsilon=1e-3, lambda_up=10.0, lambda_v=0.0, lambda_delta_a=0.0)
    r.compute(torch.full((N,), 0.010), zeros_vel(), zeros_act())     # J_t, near goal
    dj = 0.005
    _, comps = r.compute(torch.full((N,), 0.010 + dj), zeros_vel(), zeros_act())
    assert torch.allclose(comps["r_mono"], torch.full((N,), -10.0 * dj), atol=1e-6)
    assert comps["mono_violation"].all()


def test_error_decrease_near_goal_is_rewarded():
    """Default mode: the bonus is proportional to the progress made."""
    r, _ = make_reward(j_near=0.02, epsilon=1e-3, lambda_down=1.0, lambda_v=0.0, lambda_delta_a=0.0)
    r.compute(torch.full((N,), 0.015), zeros_vel(), zeros_act())
    _, comps = r.compute(torch.full((N,), 0.005), zeros_vel(), zeros_act())
    assert torch.allclose(comps["r_mono"], torch.full((N,), 0.010), atol=1e-6)  # -1.0 * (-0.01)


def test_constant_mode_reproduces_the_spec_literally():
    """mono_mode='constant' is Sec. 2.3 verbatim -- flat bonus, and exploitable."""
    with pytest.warns(UserWarning, match="oscillation profitable"):
        r, _ = make_reward(j_near=0.02, epsilon=1e-3, lambda_down=1.0,
                           lambda_v=0.0, lambda_delta_a=0.0, mono_mode="constant")
    r.compute(torch.full((N,), 0.015), zeros_vel(), zeros_act())
    _, comps = r.compute(torch.full((N,), 0.005), zeros_vel(), zeros_act())
    assert torch.allclose(comps["r_mono"], torch.ones(N))


def test_spec_constant_mode_is_exploitable_by_ringing():
    """Documents the flaw rather than hiding it.

    Under the spec's literal reward, oscillating around the goal out-earns
    holding still -- the opposite of the intended incentive. This test exists so
    that if anyone "fixes" the default back to constant, it fails loudly.
    """
    amp = 0.005
    kw = dict(j_near=0.05, epsilon=1e-3, lambda_up=10.0, lambda_down=1.0,
              lambda_v=0.0, lambda_delta_a=0.0, mono_mode="constant")
    base = 0.02

    with pytest.warns(UserWarning):
        ringing, _ = make_reward(**kw)
    ringing.compute(torch.full((N,), base), zeros_vel(), zeros_act())
    total_ring = 0.0
    for i in range(20):
        j = base - amp if i % 2 == 0 else base + amp
        rew, _ = ringing.compute(torch.full((N,), j), zeros_vel(), zeros_act())
        total_ring += rew.mean().item()

    with pytest.warns(UserWarning):
        holding, _ = make_reward(**kw)
    holding.compute(torch.full((N,), base), zeros_vel(), zeros_act())
    total_hold = sum(
        holding.compute(torch.full((N,), base), zeros_vel(), zeros_act())[0].mean().item()
        for _ in range(20)
    )
    assert total_ring > total_hold, "the spec's constant-bonus exploit has disappeared"


def test_dead_band_suppresses_noise():
    r, _ = make_reward(j_near=0.02, epsilon=1e-3, lambda_v=0.0, lambda_delta_a=0.0)
    r.compute(torch.full((N,), 0.010), zeros_vel(), zeros_act())
    _, comps = r.compute(torch.full((N,), 0.0105), zeros_vel(), zeros_act())  # dJ = 5e-4 < eps
    assert torch.allclose(comps["r_mono"], torch.zeros(N))


def test_mono_term_is_inactive_far_from_the_goal():
    """Sec. 2.3 only enforces monotonicity once J(x_t) < J_near."""
    r, _ = make_reward(j_near=0.02, lambda_v=0.0, lambda_delta_a=0.0)
    r.compute(torch.full((N,), 5.0), zeros_vel(), zeros_act())
    _, comps = r.compute(torch.full((N,), 8.0), zeros_vel(), zeros_act())
    assert torch.allclose(comps["r_mono"], torch.zeros(N))


def test_near_gate_prev_penalises_leaving_the_near_goal_region():
    """The point of gating on J(x_t): catching the step that rings *outward*.

    Gating on the new J cannot see this transition, which is precisely the
    oscillation the spec is trying to forbid.
    """
    common = dict(j_near=0.02, epsilon=1e-3, lambda_up=10.0, lambda_v=0.0, lambda_delta_a=0.0)
    j_t, j_next = 0.010, 0.050  # starts near goal, rings out of it

    r_prev, _ = make_reward(near_gate="prev", **common)
    r_prev.compute(torch.full((N,), j_t), zeros_vel(), zeros_act())
    _, c_prev = r_prev.compute(torch.full((N,), j_next), zeros_vel(), zeros_act())

    r_next, _ = make_reward(near_gate="next", **common)
    r_next.compute(torch.full((N,), j_t), zeros_vel(), zeros_act())
    _, c_next = r_next.compute(torch.full((N,), j_next), zeros_vel(), zeros_act())

    assert (c_prev["r_mono"] < 0).all(), "prev-gate must penalise ringing outward"
    assert torch.allclose(c_next["r_mono"], torch.zeros(N)), "next-gate is blind to it"


def test_ringing_is_net_unprofitable():
    """The economic content of lambda_up > lambda_down.

    A policy that oscillates J down-and-up around the goal must not out-earn one
    that simply holds still. This is what makes the monotone-descent incentive
    bite rather than being a free bonus farm.
    """
    amp = 0.005
    cfg_kw = dict(j_near=0.05, epsilon=1e-3, lambda_up=10.0, lambda_down=1.0,
                  lambda_v=0.0, lambda_delta_a=0.0)

    ringing, _ = make_reward(**cfg_kw)
    base = 0.02
    ringing.compute(torch.full((N,), base), zeros_vel(), zeros_act())
    total_ring = 0.0
    for i in range(20):
        j = base - amp if i % 2 == 0 else base + amp
        rew, _ = ringing.compute(torch.full((N,), j), zeros_vel(), zeros_act())
        total_ring += rew.mean().item()

    holding, _ = make_reward(**cfg_kw)
    holding.compute(torch.full((N,), base), zeros_vel(), zeros_act())
    total_hold = 0.0
    for _ in range(20):
        rew, _ = holding.compute(torch.full((N,), base), zeros_vel(), zeros_act())
        total_hold += rew.mean().item()

    assert total_ring < total_hold, "oscillating around the goal was profitable"


def test_warns_when_lambda_up_is_too_small():
    with pytest.warns(UserWarning, match="oscillating"):
        LyapunovRewardCfg(lambda_up=0.5, lambda_down=1.0)


# ---------------------------------------------------------------- damping terms


def test_velocity_penalty():
    r, _ = make_reward(lambda_v=0.1, lambda_delta_a=0.0)
    vel = torch.zeros(N, 2, 3)
    vel[:, 0, 0] = 3.0
    vel[:, 1, 1] = 4.0
    _, comps = r.compute(torch.zeros(N), vel, zeros_act())
    assert torch.allclose(comps["r_vel"], torch.full((N,), -0.5), atol=1e-6)  # -0.1 * 5


def test_action_penalty_uses_the_change_not_the_magnitude():
    """Sec. 2.3 defines r_act on ||a_t - a_{t-1}||.

    A constant non-zero action is smooth and must not be penalised; a reversal
    of the same magnitude must be.
    """
    r, _ = make_reward(lambda_v=0.0, lambda_delta_a=1.0)
    a = torch.full((N, A), 0.5)
    r.compute(torch.zeros(N), zeros_vel(), a)

    _, steady = r.compute(torch.zeros(N), zeros_vel(), a)
    assert torch.allclose(steady["r_act"], torch.zeros(N), atol=1e-6)

    _, flipped = r.compute(torch.zeros(N), zeros_vel(), -a)
    assert (flipped["r_act"] < -1.0).all()


# ---------------------------------------------------------------------- state


def test_reset_clears_prev_state_for_selected_envs():
    r, _ = make_reward(j_near=1.0, lambda_v=0.0, lambda_delta_a=0.0)
    r.compute(torch.full((N,), 0.5), zeros_vel(), torch.full((N, A), 0.3))

    r.reset(torch.tensor([0, 1]))
    assert not r.has_prev_J[:2].any()
    assert r.has_prev_J[2:].all()
    assert torch.allclose(r.prev_J[:2], torch.zeros(2))
    assert torch.allclose(r.prev_action[:2], torch.zeros(2, A))


def test_full_reset_clears_everything():
    r, _ = make_reward()
    r.compute(torch.full((N,), 0.5), zeros_vel(), torch.full((N, A), 0.3))
    r.reset()
    assert not r.has_prev_J.any()
    assert r.prev_J.abs().sum() == 0
    assert r.prev_action.abs().sum() == 0


def test_total_is_the_sum_of_components():
    r, _ = make_reward(lambda_v=0.1, lambda_delta_a=0.01, j_near=10.0)
    r.compute(torch.rand(N), torch.randn(N, 2, 3), torch.randn(N, A))
    reward, c = r.compute(torch.rand(N), torch.randn(N, 2, 3), torch.randn(N, A))
    assert torch.allclose(reward, c["r_task"] + c["r_mono"] + c["r_vel"] + c["r_act"], atol=1e-6)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        LyapunovRewardCfg(j_near=0.0)
    with pytest.raises(ValueError):
        LyapunovRewardCfg(epsilon=-1.0)
    with pytest.raises(ValueError):
        LyapunovRewardCfg(near_gate="nonsense")


class TestDampingGate:
    """Damping must not make standing still the best available action.

    Measured on the mock task with the spec's global damping: freeze scored
    -538 against -664 for exploring. Both damping terms are zero when the arm
    is stationary, so before the policy has found anything worth doing, the
    highest-return action is no action. The convergence argument only needs
    monotone descent *eventually*, so damping belongs in the near-goal band.
    """

    def _cfg(self, **kw):
        from lehome.real_damped_project.tasks.rewards import LyapunovRewardCfg
        return LyapunovRewardCfg(**kw)

    def _reward(self, cfg, n=1, adim=3):
        from lehome.real_damped_project.tasks.rewards import LyapunovDescentReward
        return LyapunovDescentReward(cfg, num_envs=n, action_dim=adim)

    def test_rejects_unknown_gate(self):
        with pytest.raises(ValueError, match="damping_gate"):
            self._cfg(damping_gate="sometimes")

    def test_rejects_nonpositive_j_anneal(self):
        with pytest.raises(ValueError, match="j_anneal"):
            self._cfg(damping_gate="smooth", j_anneal=0.0)

    def test_always_is_the_default_and_unchanged(self):
        """Backward compatibility: the spec-literal behaviour must be default."""
        assert self._cfg().damping_gate == "always"

    def test_near_gate_switches_damping_off_far_from_goal(self):
        cfg = self._cfg(damping_gate="near", j_near=0.02, lambda_v=1.0, lambda_delta_a=1.0)
        r = self._reward(cfg)
        far = torch.tensor([5.0])          # nowhere near folded
        vel = torch.ones(1, 1, 3) * 3.0
        act = torch.ones(1, 3) * 2.0
        r.compute(far, vel, act)            # first call seeds prev_J
        _, comp = r.compute(far, vel, act)
        assert float(comp["r_vel"]) == pytest.approx(0.0), \
            "damping applied far from the goal re-creates the freeze basin"
        assert float(comp["r_act"]) == pytest.approx(0.0)
        assert float(comp["damping_gate"]) == pytest.approx(0.0)

    def test_always_gate_still_damps_far_from_goal(self):
        """The contrast case: this is the behaviour that created the freeze basin."""
        cfg = self._cfg(damping_gate="always", lambda_v=1.0, lambda_delta_a=1.0)
        r = self._reward(cfg)
        far = torch.tensor([5.0])
        vel = torch.ones(1, 1, 3) * 3.0
        act = torch.ones(1, 3) * 2.0
        r.compute(far, vel, act)
        _, comp = r.compute(far, vel, act)
        assert float(comp["r_vel"]) < 0.0, "spec-literal damping should penalise motion anywhere"

    def test_smooth_gate_is_monotone_in_J(self):
        """Damping must strengthen as the garment approaches folded, never weaken."""
        cfg = self._cfg(damping_gate="smooth", j_anneal=1.0, lambda_v=1.0)
        gates = []
        for j_val in (4.0, 2.0, 1.0, 0.25, 0.0):
            g = float(torch.exp(-torch.tensor(j_val) / cfg.j_anneal))
            gates.append(g)
        assert gates == sorted(gates), "gate must increase as J decreases"
        assert gates[-1] == pytest.approx(1.0), "full damping at J=0"
        assert gates[0] < 0.05, "damping essentially off when far from the goal"

    def test_smooth_gate_reaches_1_over_e_at_j_anneal(self):
        cfg = self._cfg(damping_gate="smooth", j_anneal=2.0)
        g = float(torch.exp(-torch.tensor(2.0) / cfg.j_anneal))
        assert g == pytest.approx(0.3679, abs=1e-3)

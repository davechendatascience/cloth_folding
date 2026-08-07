"""Tests for the damped impedance controller (spec Sec. 2.2, 3.2)."""

import math

import pytest
import torch

from lehome.real_damped_project.control.impedance_controller import DampedImpedanceController


def make_ctrl(**kw):
    kw.setdefault("smoothing", 1.0)  # disable the low pass unless a test wants it
    return DampedImpedanceController(**kw)


# ----------------------------------------------------------------- gain synthesis


def test_critical_damping_formula():
    """D = 2 zeta sqrt(K M)."""
    c = make_ctrl(damping_ratio=1.0, stiffness=200.0, ee_mass=2.0)
    assert c.damping == pytest.approx(2.0 * math.sqrt(200.0 * 2.0))
    assert c.natural_frequency == pytest.approx(math.sqrt(100.0))


def test_overdamped_scales_damping():
    a = make_ctrl(damping_ratio=1.0, stiffness=100.0)
    b = make_ctrl(damping_ratio=2.0, stiffness=100.0)
    assert b.damping == pytest.approx(2.0 * a.damping)


def test_underdamped_is_rejected():
    """zeta < 1 contradicts the non-ringing requirement, so it is a hard error."""
    with pytest.raises(ValueError, match="damping_ratio"):
        make_ctrl(damping_ratio=0.7)


@pytest.mark.parametrize("bad", [{"stiffness": 0.0}, {"max_delta": -1.0},
                                 {"dt": 0.0}, {"ee_mass": 0.0}, {"smoothing": 0.0},
                                 {"smoothing": 1.5}])
def test_rejects_invalid_params(bad):
    with pytest.raises(ValueError):
        make_ctrl(**bad)


def test_sampling_stability_check():
    """A stiff EE at a coarse dt should be flagged, a soft one should not."""
    assert make_ctrl(stiffness=200.0, ee_mass=1.0, dt=1 / 60).is_sampling_stable()
    assert not make_ctrl(stiffness=50_000.0, ee_mass=1.0, dt=1 / 60).is_sampling_stable()


# --------------------------------------------------------------------- integrator


def test_reset_initialises_command_to_current_pose():
    c = make_ctrl()
    x = torch.randn(4, 2, 3)
    c.reset(x)
    assert torch.allclose(c.x_cmd, x)


def test_step_integrates_deltas():
    c = make_ctrl(max_delta=0.02)
    x = torch.zeros(3, 2, 3)
    c.reset(x)
    d = torch.full((3, 2, 3), 0.01)
    out = c.step(x, d)
    assert torch.allclose(out, torch.full((3, 2, 3), 0.01), atol=1e-6)
    out = c.step(x, d)
    assert torch.allclose(out, torch.full((3, 2, 3), 0.02), atol=1e-6)


def test_deltas_are_clipped():
    c = make_ctrl(max_delta=0.02)
    x = torch.zeros(2, 2, 3)
    c.reset(x)
    out = c.step(x, torch.full((2, 2, 3), 5.0))
    assert torch.allclose(out, torch.full((2, 2, 3), 0.02), atol=1e-6)


def test_ee_speed_is_bounded_by_max_delta_over_dt():
    """The clip is what bounds commanded EE speed."""
    dt, max_delta = 1 / 60, 0.02
    c = make_ctrl(max_delta=max_delta, dt=dt)
    x = torch.zeros(1, 2, 3)
    c.reset(x)
    prev = c.x_cmd.clone()
    for _ in range(20):
        cur = c.step(x, torch.randn(1, 2, 3) * 10.0)
        speed = (cur - prev).abs().max().item() / dt
        assert speed <= max_delta / dt + 1e-6
        prev = cur.clone()


# ------------------------------------------------------------------- Lipschitz


def test_lipschitz_in_the_action():
    """||x_cmd(u) - x_cmd(v)|| <= L ||u - v||  (spec Sec. 3.2)."""
    for smoothing in (1.0, 0.5, 0.2):
        x = torch.zeros(64, 2, 3)
        for _ in range(30):
            u = torch.randn(64, 2, 3) * 0.03
            v = torch.randn(64, 2, 3) * 0.03

            cu = make_ctrl(smoothing=smoothing)
            cu.reset(x)
            cv = make_ctrl(smoothing=smoothing)
            cv.reset(x)

            out_u = cu.step(x, u)
            out_v = cv.step(x, v)
            lhs = torch.linalg.vector_norm((out_u - out_v).flatten(1), dim=-1)
            rhs = torch.linalg.vector_norm((u - v).flatten(1), dim=-1)
            L = cu.lipschitz_constant
            assert (lhs <= L * rhs + 1e-6).all(), f"Lipschitz violated at smoothing={smoothing}"


def test_smoothing_attenuates_high_frequency_commands():
    """The low pass must reduce command-path ringing -- that is its whole job."""
    x = torch.zeros(1, 2, 3)
    alternating = [((-1.0) ** i) * 0.02 for i in range(40)]

    def total_variation(smoothing):
        c = make_ctrl(smoothing=smoothing)
        c.reset(x)
        pos, tv = [], 0.0
        for a in alternating:
            pos.append(c.step(x, torch.full((1, 2, 3), a)).clone())
        for i in range(1, len(pos)):
            tv += (pos[i] - pos[i - 1]).abs().sum().item()
        return tv

    assert total_variation(0.25) < total_variation(1.0)


def test_anti_windup_bounds_the_command_lead():
    """The command is an integrator; unbounded lead causes a limit cycle.

    With a stalled plant (x_current fixed) and a policy pushing hard, the naive
    integrator runs away. Measured on the folding task before the leash: a
    0.30 m move drove x_cmd 0.31 m *past* a 0.15 m goal, leaving the EE ringing
    at ~0.09 m amplitude forever. The leash bounds the accumulated lead.
    """
    c = make_ctrl(max_delta=0.02, max_lead=0.05)
    x = torch.zeros(2, 2, 3)
    c.reset(x)
    for _ in range(200):
        out = c.step(x, torch.full((2, 2, 3), 0.02))  # push hard, plant stalled
    lead = torch.linalg.vector_norm(out - x, dim=-1)
    assert (lead <= 0.05 + 1e-4).all(), f"lead {lead.max().item()} exceeded max_lead"


def test_anti_windup_still_respects_the_speed_bound():
    """The leash must not yank the command faster than max_delta per step."""
    c = make_ctrl(max_delta=0.02, max_lead=0.05)
    x = torch.zeros(1, 2, 3)
    c.reset(x)
    prev = c.x_cmd.clone()
    for _ in range(50):
        cur = c.step(x, torch.randn(1, 2, 3) * 10.0)
        assert (cur - prev).abs().max().item() <= 0.02 + 1e-6
        prev = cur.clone()


def test_anti_windup_can_be_disabled():
    """max_lead=None restores the spec's plain integrator."""
    c = make_ctrl(max_delta=0.02, max_lead=None)
    x = torch.zeros(1, 2, 3)
    c.reset(x)
    for _ in range(100):
        out = c.step(x, torch.full((1, 2, 3), 0.02))
    assert out.abs().max().item() > 0.5, "unbounded integrator should run away"


def test_workspace_bounds_are_respected():
    lo = torch.full((3,), -0.05)
    hi = torch.full((3,), 0.05)
    c = make_ctrl(max_delta=0.02, workspace_bounds=(lo, hi))
    x = torch.zeros(2, 2, 3)
    c.reset(x)
    for _ in range(20):
        out = c.step(x, torch.full((2, 2, 3), 0.02))
    assert (out <= 0.05 + 1e-6).all()


# -------------------------------------------------------------- partial resets


def test_partial_reset_only_touches_selected_envs():
    c = make_ctrl(max_delta=0.02)
    x = torch.zeros(4, 2, 3)
    c.reset(x)
    c.step(x, torch.full((4, 2, 3), 0.02))
    before = c.x_cmd.clone()

    ids = torch.tensor([1, 3])
    c.reset(torch.full((4, 2, 3), 9.0), env_ids=ids)

    assert torch.allclose(c.x_cmd[ids], torch.full((2, 2, 3), 9.0))
    keep = torch.tensor([0, 2])
    assert torch.allclose(c.x_cmd[keep], before[keep])


def test_partial_reset_clears_the_smoothing_state():
    """A stale filter state would leak velocity across an episode boundary."""
    c = make_ctrl(smoothing=0.5, max_delta=0.02)
    x = torch.zeros(2, 2, 3)
    c.reset(x)
    for _ in range(5):
        c.step(x, torch.full((2, 2, 3), 0.02))

    ids = torch.tensor([0])
    c.reset(torch.zeros(2, 2, 3), env_ids=ids)
    out = c.step(x, torch.zeros(2, 2, 3))
    # env 0 was reset: zero command in => no motion.
    assert out[0].abs().max().item() == pytest.approx(0.0, abs=1e-7)
    # env 1 still coasts on its filter state.
    assert out[1].abs().max().item() > 0.0


def test_shape_change_without_reset_is_an_error():
    c = make_ctrl()
    c.reset(torch.zeros(4, 2, 3))
    with pytest.raises(ValueError, match="reset"):
        c.step(torch.zeros(8, 2, 3), torch.zeros(8, 2, 3))

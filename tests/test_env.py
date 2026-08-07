"""Tests for the damped folding env and its mock backend (spec Sec. 4, 5)."""

import pytest
import torch

from lehome.real_damped_project.tasks.backend import MockClothCfg, MockFoldGarmentBackend
from lehome.real_damped_project.tasks.cfg import (
    TASK_NAME,
    build_lehome_real_damped_cfg,
    list_tasks,
    make_env,
)


@pytest.fixture
def env():
    return make_env(num_envs=4, sim_device="cpu", use_mock_backend=True)


# ------------------------------------------------------------------ registration


def test_task_is_registered():
    assert TASK_NAME in list_tasks()


def test_cfg_builder_matches_the_spec_defaults():
    """Sec. 5.1's named hyperparameters."""
    cfg = build_lehome_real_damped_cfg()
    assert cfg.dt == pytest.approx(1 / 60)
    assert cfg.damping_ratio >= 1.0
    assert cfg.stiffness == 200.0
    assert cfg.max_delta == 0.02
    assert cfg.reward.j_near == 0.02
    assert cfg.reward.epsilon == pytest.approx(1e-3)
    assert cfg.reward.lambda_up == 10.0
    assert cfg.reward.lambda_down == 1.0
    assert cfg.reward.lambda_v == 0.1
    assert cfg.reward.lambda_delta_a == 0.01


def test_cfg_sync_propagates_to_the_backend():
    cfg = build_lehome_real_damped_cfg()
    cfg.num_envs = 7
    cfg.stiffness = 123.0
    cfg.sync()
    assert cfg.backend.num_envs == 7
    assert cfg.backend.ee_stiffness == 123.0


def test_make_env_rejects_unknown_task():
    with pytest.raises(KeyError):
        make_env(task_name="Nope-v0", num_envs=2, sim_device="cpu")


# ------------------------------------------------------------------ observations


def test_observation_shapes(env):
    obs = env.reset()
    shapes = env.observation_shapes
    assert set(obs) == {"images", "proprio"}
    assert obs["images"].shape == (4, *shapes["images"])
    assert obs["proprio"].shape == (4, shapes["proprio"][0])
    assert torch.isfinite(obs["images"]).all()
    assert torch.isfinite(obs["proprio"]).all()


def test_observations_exclude_cloth_state(env):
    """Sec. 3.1: cloth mesh is reward-only.

    The proprio vector is EE pos/vel/cmd; assert its width matches exactly that,
    leaving no room for smuggled cloth information.
    """
    obs = env.reset()
    assert obs["proprio"].shape[1] == 18  # 3 * (2 arms * 3 dof)
    n_verts = env.backend.cfg.num_verts
    assert obs["proprio"].shape[1] < n_verts * 3


def test_images_are_bounded(env):
    obs = env.reset()
    assert (obs["images"] >= 0).all() and (obs["images"] <= 1.0 + 1e-5).all()


# ------------------------------------------------------------------------ step


def test_step_returns_well_formed_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, extras = env.step(torch.zeros(4, env.action_dim))
    assert reward.shape == (4,)
    assert terminated.shape == (4,) and terminated.dtype == torch.bool
    assert truncated.shape == (4,) and truncated.dtype == torch.bool
    assert torch.isfinite(reward).all()
    assert "log" in extras and "J" in extras["log"]


def test_rollout_is_numerically_stable(env):
    """Random actions for a full episode must not blow the cloth up."""
    env.reset()
    for _ in range(120):
        a = torch.rand(4, env.action_dim) * 2 - 1
        obs, reward, term, trunc, _ = env.step(a)
        assert torch.isfinite(reward).all()
        assert torch.isfinite(obs["images"]).all()
        assert torch.isfinite(env.backend.verts).all()
        assert env.backend.verts.abs().max().item() < 10.0, "cloth exploded"


def test_actions_are_clamped(env):
    """Out-of-range actions must not translate into unbounded EE motion."""
    env.reset()
    before = env.backend.ee_cmd.clone()
    env.step(torch.full((4, env.action_dim), 1e6))
    moved = (env.backend.ee_cmd - before).abs().max().item()
    assert moved <= env.cfg.max_delta + 1e-6


def test_rejects_wrong_action_shape(env):
    env.reset()
    with pytest.raises(ValueError):
        env.step(torch.zeros(4, env.action_dim + 1))


# --------------------------------------------------------------------- resets


def test_auto_reset_on_truncation():
    env = make_env(num_envs=2, sim_device="cpu", use_mock_backend=True)
    env.cfg.backend.max_episode_steps = 5
    env.backend.cfg.max_episode_steps = 5
    env.reset()
    for _ in range(5):
        _, _, _, truncated, _ = env.step(torch.zeros(2, env.action_dim))
    assert truncated.all()
    assert (env.backend.episode_step == 0).all(), "episode counter not reset"


def test_reset_clears_reward_memory(env):
    env.reset()
    env.step(torch.zeros(4, env.action_dim))
    assert env.reward_fn.has_prev_J.all()
    env._reset_idx(torch.tensor([0, 2]))
    assert not env.reward_fn.has_prev_J[[0, 2]].any()
    assert env.reward_fn.has_prev_J[[1, 3]].all()


def test_reset_randomises_initial_state():
    env = make_env(num_envs=8, sim_device="cpu", use_mock_backend=True)
    env.reset()
    v = env.backend.verts
    assert not torch.allclose(v[0], v[1], atol=1e-6), "all envs identical -- no randomisation"


# -------------------------------------------------------------- mock physics


def test_mock_cloth_is_dissipative():
    """Released from a perturbed state with no driving, kinetic energy must decay.

    This is the discrete stand-in for the contractive-semigroup requirement of
    Sec. 2.2: the flow must be strongly dissipative.
    """
    cfg = MockClothCfg(num_envs=2, gravity=0.0)
    b = MockFoldGarmentBackend(cfg)
    b.vel += torch.randn_like(b.vel) * 0.1
    b.ee_cmd = b.ee_pos.clone()

    energies = []
    for _ in range(40):
        b.simulate()
        energies.append((b.vel**2).sum().item())

    assert energies[-1] < energies[0] * 0.5, f"energy did not decay: {energies[0]} -> {energies[-1]}"
    assert all(e == e for e in energies)  # no NaN


def test_higher_rayleigh_damping_decays_faster():
    def final_energy(alpha):
        cfg = MockClothCfg(num_envs=1, gravity=0.0, rayleigh_alpha=alpha, seed=1)
        b = MockFoldGarmentBackend(cfg)
        torch.manual_seed(0)
        b.vel += torch.full_like(b.vel, 0.1)
        b.ee_cmd = b.ee_pos.clone()
        for _ in range(30):
            b.simulate()
        return (b.vel**2).sum().item()

    assert final_energy(8.0) < final_energy(1.0)


def test_ee_impedance_does_not_overshoot_when_critically_damped():
    """zeta = 1 must produce a monotone approach to the setpoint (Sec. 2.2)."""
    cfg = MockClothCfg(num_envs=1, ee_damping_ratio=1.0, gravity=0.0)
    b = MockFoldGarmentBackend(cfg)
    start = b.ee_pos.clone()
    target = start.clone()
    target[..., 0] += 0.05
    b.set_end_effector_targets(target)

    errs = []
    for _ in range(200):
        b.simulate()
        errs.append((b.ee_pos - target)[0, 0, 0].item())

    assert abs(errs[-1]) < 1e-3, "EE never converged to the setpoint"
    # The EE starts 0.05 m *below* the setpoint, so err rises from -0.05 to 0.
    # Overshoot means crossing past it, i.e. err going positive.
    assert max(errs) < 1e-4, f"overshoot detected: max error {max(errs)}"


def test_underdamped_backend_overshoots():
    """Control test: the overshoot check above can actually fail."""
    cfg = MockClothCfg(num_envs=1, ee_damping_ratio=0.15, gravity=0.0)
    b = MockFoldGarmentBackend(cfg)
    target = b.ee_pos.clone()
    target[..., 0] += 0.05
    b.set_end_effector_targets(target)
    errs = []
    for _ in range(200):
        b.simulate()
        errs.append((b.ee_pos - target)[0, 0, 0].item())
    assert max(errs) > 1e-3, "expected overshoot from an under-damped EE"


def test_grippers_track_the_commanded_pose():
    cfg = MockClothCfg(num_envs=1, gravity=0.0)
    b = MockFoldGarmentBackend(cfg)
    target = b.ee_pos.clone()
    target[..., 2] += 0.05
    b.set_end_effector_targets(target)
    for _ in range(100):
        b.simulate()
    assert torch.allclose(b.ee_pos, target, atol=5e-3)
    gripped = b.verts[:, b._grip_idx, :]
    assert torch.allclose(gripped, b.ee_pos, atol=1e-5), "gripped vertices did not follow the EEs"


def test_J_decreases_when_the_cloth_is_folded_by_hand():
    """Sanity check that the reward actually points at folding.

    Interpolating the mesh toward the folded target must monotonically reduce J;
    if it did not, no policy could learn to fold.
    """
    cfg = MockClothCfg(num_envs=1)
    b = MockFoldGarmentBackend(cfg)
    start = b.verts.clone()
    target = b.target_verts.unsqueeze(0)

    prev = float("inf")
    for t in torch.linspace(0, 1, 11):
        b.verts = (1 - t) * start + t * target
        j = b.compute_cloth_error().item()
        assert j < prev + 1e-6, f"J increased while interpolating toward the goal at t={t}"
        prev = j
    assert prev < cfg.success_threshold


# --------------------------------------------------- real-backend config guards


def test_sync_does_not_apply_mock_fields_to_a_real_backend():
    """sync() pushed cloth/impedance params that IsaacGarmentCfg does not have,
    which would raise AttributeError on the first real-backend launch."""
    from dataclasses import dataclass

    @dataclass
    class FakeIsaacCfg:  # stands in for IsaacGarmentCfg without needing Isaac
        garment_name: str = "Top_Long_Seen_0"

    cfg = build_lehome_real_damped_cfg()
    cfg.use_mock_backend = False
    cfg.num_envs = 1
    cfg.backend = FakeIsaacCfg()
    cfg.sync()  # must not touch rayleigh_alpha / rows / functional
    assert not hasattr(cfg.backend, "rayleigh_alpha")


def test_sync_rejects_multiple_envs_on_the_real_backend():
    """The LeHome scene uses absolute prim paths and single-prim particle APIs,
    so num_envs>1 silently shares one garment between all envs."""
    from dataclasses import dataclass

    @dataclass
    class FakeIsaacCfg:
        garment_name: str = "Top_Long_Seen_0"

    cfg = build_lehome_real_damped_cfg()
    cfg.use_mock_backend = False
    cfg.backend = FakeIsaacCfg()
    cfg.num_envs = 8
    with pytest.raises(ValueError, match="not authored for cloning"):
        cfg.sync()


def test_action_mode_defaults_to_cartesian_and_accepts_joint():
    cfg = build_lehome_real_damped_cfg()
    assert cfg.action_mode == "cartesian"
    cfg.action_mode = "joint"
    env = make_env(num_envs=2, sim_device="cpu", use_mock_backend=True,
                   cfg_overrides={"action_mode": "joint"})
    assert env.action_dim == 12
    assert env.action_mode == "joint"


def test_unknown_action_mode_is_rejected():
    with pytest.raises(ValueError, match="action_mode"):
        make_env(num_envs=2, sim_device="cpu", use_mock_backend=True,
                 cfg_overrides={"action_mode": "wiggle"})

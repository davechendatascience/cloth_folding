"""Tests for per-actuator-group damping resolution.

Pure logic, no Isaac needed. Pins the bug that broke the first adapter run:
Isaac Lab resolves dict-valued gains with strict=True, so handing the full
arm-joint dict to the SO101's *gripper* actuator group raises
"Not all regular expressions are matched" rather than being ignored.
"""

import pytest

from lehome.real_damped_project.tasks.isaac_garment_backend import (
    MEASURED_CRITICAL_DAMPING,
    actuator_overrides,
)

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = ["gripper"]


def test_arm_group_gets_only_arm_joints():
    k, d = actuator_overrides(ARM, 17.8, 0.60, MEASURED_CRITICAL_DAMPING)
    assert isinstance(d, dict)
    assert set(d) <= set(ARM), "leaked a joint this group does not own"
    assert d["shoulder_lift"] == pytest.approx(1.283)
    assert k == 17.8


def test_gripper_group_is_left_untouched():
    """The regression: passing arm joints to the gripper group is a hard error."""
    k, d = actuator_overrides(GRIPPER, 17.8, 0.60, MEASURED_CRITICAL_DAMPING)
    assert d == 0.60, "gripper damping should be unchanged, not a dict of arm joints"
    assert not isinstance(d, dict)


def test_unmeasured_joint_is_omitted_not_defaulted():
    """wrist_roll showed no overshoot, so it must keep its existing value."""
    _, d = actuator_overrides(ARM, 17.8, 0.60, MEASURED_CRITICAL_DAMPING)
    assert "wrist_roll" not in d


def test_every_key_matches_a_joint_in_the_group():
    """The exact precondition Isaac Lab enforces with strict=True."""
    import re

    for names in (ARM, GRIPPER):
        _, d = actuator_overrides(names, 17.8, 0.60, MEASURED_CRITICAL_DAMPING)
        if isinstance(d, dict):
            for key in d:
                assert any(re.fullmatch(e, key) for e in names), f"{key} matches nothing"


def test_stiffness_scale_preserves_damping_ratio():
    """zeta = D / (2 sqrt(K J)); scaling D by sqrt(K-factor) holds it fixed."""
    k0, d0 = actuator_overrides(ARM, 17.8, 0.60, MEASURED_CRITICAL_DAMPING, 1.0)
    k1, d1 = actuator_overrides(ARM, 17.8, 0.60, MEASURED_CRITICAL_DAMPING, 4.0)
    assert k1 == pytest.approx(4 * k0)
    for j in d0:
        assert d1[j] == pytest.approx(2.0 * d0[j])  # sqrt(4) = 2
        zeta0 = d0[j] / (k0**0.5)
        zeta1 = d1[j] / (k1**0.5)
        assert zeta1 == pytest.approx(zeta0)


def test_stiffness_scale_applies_without_overrides():
    k, d = actuator_overrides(GRIPPER, 17.8, 0.60, {}, 4.0)
    assert k == pytest.approx(71.2)
    assert d == pytest.approx(1.2)


def test_empty_overrides_are_a_noop():
    k, d = actuator_overrides(ARM, 17.8, 0.60, {}, 1.0)
    assert (k, d) == (17.8, 0.60)
    k, d = actuator_overrides(ARM, 17.8, 0.60, None, 1.0)
    assert (k, d) == (17.8, 0.60)


def test_every_measured_joint_is_under_damped_at_the_configured_gain():
    """All measured D_crit exceed LeHome's D=0.60, i.e. every joint has zeta<1.

    wrist_flex is only marginally so (0.633 -> zeta=0.95), which is why it shows
    almost no overshoot; the shoulder joints are the real offenders.
    """
    for joint, d_crit in MEASURED_CRITICAL_DAMPING.items():
        assert d_crit > 0.60, f"{joint} would be over-damped at the configured D"
    assert MEASURED_CRITICAL_DAMPING["wrist_flex"] < 0.70, "wrist_flex is near-critical"
    assert MEASURED_CRITICAL_DAMPING["shoulder_lift"] > 1.2, "shoulder_lift is the worst"


# Measured pairs from scripts/measure_joint_damping.py: (zeta_from_overshoot, omega_n)
_MEASURED = {
    "shoulder_pan": (0.567, 33.23),
    "shoulder_lift": (0.529, 27.76),
    "elbow_flex": (0.649, 32.80),
    "wrist_flex": (0.933, 56.24),
}


@pytest.mark.parametrize("joint", sorted(_MEASURED))
def test_second_order_model_self_consistency(joint):
    """How well does a single-DOF 2nd-order model actually describe each joint?

    By construction zeta = D/D_crit exactly, since D_crit = 2K/omega_n and
    zeta = D*omega_n/(2K). But zeta is estimated from the *overshoot ratio* and
    omega_n from the *peak spacing* -- two independent features of the response.
    A gap between them measures how badly the single-DOF fit is failing (joint
    coupling, discrete-time actuator behaviour, configuration-dependent inertia).

    This test documents the residual rather than asserting it is zero: it bounds
    how much confidence the D_crit numbers deserve.
    """
    zeta_overshoot, _ = _MEASURED[joint]
    zeta_implied = 0.60 / MEASURED_CRITICAL_DAMPING[joint]
    rel = abs(zeta_overshoot - zeta_implied) / zeta_overshoot
    assert rel < 0.25, (
        f"{joint}: zeta from overshoot={zeta_overshoot:.3f} vs from period="
        f"{zeta_implied:.3f} ({rel:.0%} apart) -- single-DOF fit has broken down"
    )

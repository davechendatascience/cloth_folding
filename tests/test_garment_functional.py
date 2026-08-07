"""Tests for the continuous J built on LeHome's real success metric.

The critical property: J's zero set must coincide *exactly* with
``success_checker_garment_fold``. If they diverge, the policy optimises
something the challenge does not score.
"""

import math

import pytest
import torch

from lehome.real_damped_project.math.garment_functional import (
    GARMENT_CONDITIONS,
    GarmentFoldFunctional,
    GarmentFunctionalCfg,
)

# Real values from Assets/.../Top_Long/Top_Long_Seen_0/TCLC_002_obj_exp.json
TOP_LONG_THRESH = [16.0, 19.0, 16.0, 22.0, 20.0]
SCALE = 0.45
SCALED = [d * SCALE for d in TOP_LONG_THRESH]


def fn(garment_type="top-long-sleeve", thresholds=None, **kw):
    return GarmentFoldFunctional(
        garment_type, thresholds or SCALED, GarmentFunctionalCfg(**kw)
    )


def points_from(dists, garment_type="top-long-sleeve"):
    """Build 6 check-points realising prescribed pairwise distances.

    Places each referenced pair along its own axis offset so the distances are
    independent -- enough to exercise the condition logic.
    """
    p = torch.zeros(1, 6, 3)
    for (i, j, _, slot), d in zip(GARMENT_CONDITIONS[garment_type], dists):
        p[0, j] = p[0, i] + torch.tensor([d, 0.0, 0.0])
    return p


# --------------------------------------------------------------- zero set == G


def test_J_is_zero_exactly_when_all_conditions_pass():
    f = fn()
    # le conditions satisfied (small), ge conditions satisfied (large).
    p = torch.zeros(1, 6, 3)
    p[0, 4] = torch.tensor([1.0, 0.0, 0.0])    # d(0,4)=1  <= 7.2   ok
    p[0, 3] = torch.tensor([0.0, 1.0, 0.0])
    p[0, 2] = torch.tensor([0.0, 1.0, 0.0])    # d(2,3)=0  <= 8.55  ok
    p[0, 1] = torch.tensor([0.0, 0.0, 50.0])   # d(0,1)=50 >= 9.9   ok
    p[0, 5] = torch.tensor([1.0, 0.0, 50.0])   # d(1,5)=1  <= 7.2   ok
                                               # d(4,5)=50 >= 9.0   ok
    j = f(p)
    assert j.shape == (1,)
    assert j.item() == pytest.approx(0.0, abs=1e-9)
    assert bool(f.is_success(p))


def test_J_is_positive_when_any_single_condition_fails():
    """Every condition must be able to keep J off zero on its own."""
    f = fn()
    for k, (i, j_, cmp_, slot) in enumerate(f.conditions):
        p = torch.zeros(1, 6, 3)
        # Start from a configuration that satisfies everything.
        p[0, 1] = torch.tensor([0.0, 0.0, 50.0])
        p[0, 5] = torch.tensor([1.0, 0.0, 50.0])
        p[0, 4] = torch.tensor([1.0, 0.0, 0.0])
        base = f(p)
        assert base.item() == pytest.approx(0.0, abs=1e-9), "setup should satisfy all"

        # Break exactly this condition.
        t = SCALED[slot]
        offset = (t + 10.0) if cmp_ == "le" else 0.0
        p[0, j_] = p[0, i] + torch.tensor([offset, 0.0, 0.0])
        assert f(p).item() > 0.0, f"condition {k+1} ({cmp_} {i},{j_}) did not raise J"


def test_J_is_nonnegative_on_random_configurations():
    f = fn()
    p = torch.randn(64, 6, 3) * 30.0
    j = f(p)
    assert j.shape == (64,)
    assert (j >= 0).all()
    assert torch.isfinite(j).all()


# ------------------------------------------------------------------ continuity


def test_J_is_continuous_and_piecewise_linear_in_distance():
    """J must respond smoothly, so dJ measures progress rather than quantisation."""
    f = fn()
    prev_j, prev_d = None, None
    for d in [30.0, 25.0, 20.0, 15.0, 10.0]:
        p = torch.zeros(1, 6, 3)
        p[0, 1] = torch.tensor([0.0, 0.0, 50.0])
        p[0, 5] = torch.tensor([1.0, 0.0, 50.0])
        p[0, 4] = torch.tensor([d, 0.0, 0.0])   # violates cond1 (d(0,4) <= 7.2)
        j = f(p).item()
        if prev_j is not None:
            # relu margin => J drops exactly 1:1 with distance, scaled.
            assert j < prev_j
            assert (prev_j - j) == pytest.approx((prev_d - d) / f.cfg.scale, rel=1e-5)
        prev_j, prev_d = j, d


def test_J_saturates_at_zero_not_below():
    """Overshooting a satisfied condition must not earn negative error."""
    f = fn()
    p = torch.zeros(1, 6, 3)
    p[0, 1] = torch.tensor([0.0, 0.0, 500.0])
    p[0, 5] = torch.tensor([0.0, 0.0, 500.0])
    assert f(p).item() >= 0.0


# ------------------------------------------------------------- garment variants


@pytest.mark.parametrize("gtype,n_cond", [
    ("top-long-sleeve", 5), ("top-short-sleeve", 5),
    ("long-pant", 4), ("short-pant", 4),
])
def test_all_garment_types_supported(gtype, n_cond):
    """Condition counts must match the checkers in success_checker_chanllege.py."""
    f = GarmentFoldFunctional(gtype, [10.0] * 6)
    assert f.num_conditions == n_cond
    assert f.required_points == 6
    assert f(torch.randn(2, 6, 3) * 5).shape == (2,)


def test_unknown_garment_type_rejected():
    with pytest.raises(ValueError, match="unknown garment_type"):
        GarmentFoldFunctional("dress", [10.0] * 6)


def test_missing_thresholds_rejected():
    with pytest.raises(ValueError, match="thresholds"):
        GarmentFoldFunctional("top-long-sleeve", [10.0, 10.0])


# ------------------------------------------------------------------- mechanics


def test_components_report_each_condition():
    f = fn()
    j, comps = f(torch.randn(1, 6, 3) * 20, return_components=True)
    assert comps["J"].shape == (1,)
    assert sum(1 for k in comps if k.endswith("_dist")) == f.num_conditions
    violations = [v for k, v in comps.items() if not k.endswith("_dist") and k != "J"]
    assert len(violations) == f.num_conditions
    assert all((v >= 0).all() for v in violations)


def test_weights_are_applied():
    p = torch.zeros(1, 6, 3)
    p[0, 4] = torch.tensor([100.0, 0.0, 0.0])  # big violation on condition 1
    base = fn()(p).item()
    weighted = fn(weights=[2.0, 1.0, 1.0, 1.0, 1.0])(p).item()
    assert weighted > base


def test_scale_normalises():
    p = torch.zeros(1, 6, 3)
    p[0, 4] = torch.tensor([100.0, 0.0, 0.0])
    assert fn(scale=1.0)(p).item() == pytest.approx(10.0 * fn(scale=10.0)(p).item(), rel=1e-5)


def test_batching_matches_single_evaluation():
    f = fn()
    p = torch.randn(5, 6, 3) * 20
    batched = f(p)
    for i in range(5):
        assert batched[i].item() == pytest.approx(f(p[i : i + 1]).item(), rel=1e-6)


def test_matches_lehome_checker_on_random_configurations():
    """Differential test against a transcription of check_top_sleeve.

    Guards the thing that actually matters: J == 0 iff LeHome says success.
    """
    f = fn()
    torch.manual_seed(0)
    agree = 0
    for _ in range(300):
        p = torch.randn(1, 6, 3) * 8.0
        d = lambda a, b: torch.linalg.vector_norm(p[0, a] - p[0, b]).item()
        lehome_success = (
            d(0, 4) <= SCALED[0] and d(2, 3) <= SCALED[1] and d(1, 5) <= SCALED[2]
            and d(0, 1) >= SCALED[3] and d(4, 5) >= SCALED[4]
        )
        assert bool(f.is_success(p)) == lehome_success
        agree += 1
    assert agree == 300

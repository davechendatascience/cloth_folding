"""Tests for J -- the properties the whole convergence argument rests on."""

import pytest
import torch

from lehome.real_damped_project.math.cloth_functional import (
    ClothErrorFunctional,
    ClothFunctionalCfg,
    make_flat_cloth,
    make_folded_target,
)

ROWS = COLS = 9


def make_fn(**kw):
    cfg = ClothFunctionalCfg(grid_shape=(ROWS, COLS), grid_bounds=(-0.35, 0.35, -0.35, 0.35), **kw)
    target = make_folded_target(ROWS, COLS)
    return ClothErrorFunctional(cfg, target), target


# --------------------------------------------------------------- core properties


def test_J_is_zero_at_the_goal():
    """J(x) = 0 iff x in G (spec Sec. 0.1)."""
    fn, target = make_fn()
    j = fn(target.unsqueeze(0))
    assert j.shape == (1,)
    assert j.item() == pytest.approx(0.0, abs=1e-6)


def test_J_is_nonnegative_on_random_states():
    fn, _ = make_fn()
    verts = torch.randn(16, ROWS * COLS, 3) * 0.2
    j = fn(verts)
    assert j.shape == (16,)
    assert (j >= 0).all()
    assert torch.isfinite(j).all()


def test_J_is_positive_away_from_the_goal():
    fn, target = make_fn()
    flat = make_flat_cloth(ROWS, COLS)
    assert fn(flat.unsqueeze(0)).item() > 0.1


def test_J_is_continuous_in_the_vertices():
    """Small perturbations produce small changes -- required for J continuous.

    A hard-binned occupancy mask would fail this: a vertex crossing a cell
    boundary would step-change the IoU.
    """
    fn, target = make_fn()
    base = target.unsqueeze(0)
    j0 = fn(base).item()
    prev = None
    for scale in [1e-2, 1e-3, 1e-4, 1e-5]:
        pert = base + torch.randn_like(base) * scale
        dj = abs(fn(pert).item() - j0)
        if prev is not None:
            assert dj < prev + 1e-6, "J did not shrink as the perturbation shrank"
        prev = dj
    assert prev < 1e-3


def test_J_increases_with_misalignment():
    """Monotone in the defect (spec Sec. 2.1)."""
    fn, target = make_fn()
    prev = -1.0
    for shift in [0.0, 0.01, 0.03, 0.06, 0.10]:
        verts = target.clone().unsqueeze(0)
        verts[..., 0] += shift
        j = fn(verts).item()
        assert j > prev, f"J not increasing at shift={shift}"
        prev = j


def test_J_is_differentiable():
    """Needed if J is ever used for analysis/gradient diagnostics."""
    fn, target = make_fn()
    verts = (target.unsqueeze(0) + 0.02).requires_grad_(True)
    fn(verts).sum().backward()
    assert verts.grad is not None
    assert torch.isfinite(verts.grad).all()
    assert verts.grad.abs().sum() > 0


# ------------------------------------------------------------------- components


def test_components_are_individually_nonnegative_and_vanish_at_goal():
    fn, target = make_fn()
    j, comps = fn(target.unsqueeze(0), return_components=True)
    for name, value in comps.items():
        assert (value >= -1e-6).all(), f"{name} negative"
        assert value.abs().item() < 1e-5, f"{name} nonzero at the goal: {value.item()}"


def test_wrinkle_term_is_relative_to_the_target_curvature():
    """A flat sheet must not score better on wrinkle than the folded target.

    This is the reason the term is |R(x) - R(target)| rather than R(x): a folded
    garment has curvature at the fold, so an absolute-roughness penalty would
    place J's zero at the unfolded sheet.
    """
    fn, target = make_fn()
    flat = make_flat_cloth(ROWS, COLS).unsqueeze(0)
    _, flat_comps = fn(flat, return_components=True)
    _, tgt_comps = fn(target.unsqueeze(0), return_components=True)
    assert tgt_comps["wrinkle"].item() == pytest.approx(0.0, abs=1e-6)
    assert flat_comps["wrinkle"].item() > 0.0


def test_soft_occupancy_is_bounded():
    fn, target = make_fn()
    occ = fn.soft_occupancy(target.unsqueeze(0))
    assert occ.shape == (1, 64, 64)
    assert (occ >= 0).all() and (occ <= 1.0 + 1e-6).all()
    assert occ.max() > 0.5, "target cloth left essentially no footprint on the grid"


def test_iou_term_responds_to_overlap():
    fn, target = make_fn()
    disjoint = target.clone().unsqueeze(0)
    disjoint[..., 0] += 1.0  # translate far outside the grid
    _, comps = fn(disjoint, return_components=True)
    assert comps["iou"].item() > 0.9


def test_edge_gap_scales_linearly_with_displacement():
    fn, target = make_fn(lambda_iou=0.0, lambda_wrinkle=0.0, edge_scale=0.1)
    a = target.clone().unsqueeze(0)
    a[..., 0] += 0.01
    b = target.clone().unsqueeze(0)
    b[..., 0] += 0.02
    _, ca = fn(a, return_components=True)
    _, cb = fn(b, return_components=True)
    assert cb["edge_gap"].item() == pytest.approx(2 * ca["edge_gap"].item(), rel=1e-4)


# ------------------------------------------------------------------- weights/cfg


def test_weights_scale_terms():
    _, target = make_fn()
    verts = make_flat_cloth(ROWS, COLS).unsqueeze(0)
    f1, _ = make_fn(lambda_iou=1.0, lambda_edge=0.0, lambda_wrinkle=0.0)
    f2, _ = make_fn(lambda_iou=2.0, lambda_edge=0.0, lambda_wrinkle=0.0)
    assert f2(verts).item() == pytest.approx(2 * f1(verts).item(), rel=1e-5)


def test_batching_matches_single_evaluation():
    fn, target = make_fn()
    verts = torch.stack([make_flat_cloth(ROWS, COLS), target, target + 0.01])
    batched = fn(verts)
    for i in range(3):
        assert batched[i].item() == pytest.approx(fn(verts[i : i + 1]).item(), rel=1e-5)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        ClothFunctionalCfg(splat_sigma=0.0)      # J would be discontinuous
    with pytest.raises(ValueError):
        ClothFunctionalCfg(lambda_iou=-1.0)      # J could go negative
    with pytest.raises(ValueError):
        ClothFunctionalCfg(grid_bounds=(1.0, 0.0, 0.0, 1.0))


def test_rejects_bad_shapes():
    fn, target = make_fn()
    with pytest.raises(ValueError):
        fn(torch.randn(4, ROWS * COLS, 2))
    with pytest.raises(ValueError):
        fn(torch.randn(4, ROWS * COLS, 3), target_verts=torch.randn(4, 10, 3))


def test_folded_target_geometry():
    """The target really is the left half reflected onto the right half."""
    t = make_folded_target(ROWS, COLS, size=(0.3, 0.3), thickness=0.004).view(ROWS, COLS, 3)
    assert (t[..., 0] >= -1e-6).all(), "no vertex should remain left of the fold line"
    left_half_z = t[:, : COLS // 2, 2]
    assert torch.allclose(left_half_z, torch.full_like(left_half_z, 0.004), atol=1e-6)

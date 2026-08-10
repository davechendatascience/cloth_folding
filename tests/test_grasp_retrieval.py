"""Tests for retrieval-based grasp posing.

The point of retrieval is that a DLS solver picks whatever null-space solution
it likes -- measured 30 degrees from the demonstrators' wrist posture, with
double the variance -- while the demonstrations carry configurations that
provably hold cloth. These tests cover the properties that make retrieval
trustworthy: it stays on the demonstrated manifold, it abstains outside it, and
it never mixes the two arms' disjoint workspaces.
"""

import numpy as np
import pytest

from lehome.real_damped_project.control.grasp_retrieval import (
    GraspRetriever, GraspRetrievalCfg)


def _library(tmp_path, n=300, seed=0):
    """A synthetic library on a known smooth map, so predictions are checkable."""
    rng = np.random.RandomState(seed)
    out = tmp_path / "lib.npz"
    # arm0 occupies x<-10, arm1 x>7 -- the real arms' disjoint workspaces
    pos0 = np.stack([rng.uniform(-45, -12, n), rng.uniform(-20, 8, n),
                     rng.uniform(58, 78, n)], axis=-1)
    pos1 = np.stack([rng.uniform(9, 42, n), rng.uniform(-20, 8, n),
                     rng.uniform(58, 78, n)], axis=-1)
    pos = np.vstack([pos0, pos1])
    arm = np.array([0] * n + [1] * n)
    # joints as a smooth linear function of position, so a local linear fit
    # should recover it nearly exactly
    M = rng.RandomState if False else rng
    A = M.uniform(-0.02, 0.02, size=(3, 12))
    q = pos @ A + 0.1
    grip = np.where(arm == 0, -0.14, -0.15) + rng.normal(0, 1e-3, len(arm))
    for i, a in enumerate(arm):
        q[i, 5 if a == 0 else 11] = grip[i]
    np.savez(out, q=q, ee_pos=pos, ee_quat=np.tile([1, 0, 0, 0], (len(pos), 1)),
             cp_pos=pos, arm=arm, grip_val=grip,
             cp_idx=np.zeros(len(pos), int), episode=np.zeros(len(pos), int),
             frame=np.arange(len(pos)))
    return out


def test_recovers_a_smooth_map(tmp_path):
    """On a linear ground-truth map the local linear fit should be near-exact."""
    r = GraspRetriever(_library(tmp_path))
    target = np.array([-25.0, -5.0, 68.0])
    res = r.query(target, arm=0)
    assert res is not None
    q, grip, dist = res
    assert dist < 12.0
    # compare against a direct fit of the same library restricted to arm0
    idx = r._by_arm[0]
    A = np.hstack([r.ee_pos[idx], np.ones((len(idx), 1))])
    W = np.linalg.lstsq(A, r.q[idx], rcond=None)[0]
    expect = np.hstack([target, 1.0]) @ W
    arm_ids = [0, 1, 2, 3, 4]
    assert np.abs(q[arm_ids] - expect[arm_ids]).max() < 0.05


def test_abstains_outside_the_manifold(tmp_path):
    """A confident answer with no supporting demonstration is worse than none."""
    r = GraspRetriever(_library(tmp_path))
    assert r.query(np.array([0.0, 0.0, 200.0]), arm=0) is None
    assert r.query(np.array([500.0, 0.0, 68.0]), arm=0) is None


def test_never_mixes_arms(tmp_path):
    """The workspaces are disjoint; interpolating across the gap is meaningless."""
    r = GraspRetriever(_library(tmp_path))
    # a point deep in arm1 territory must not be answerable by arm0
    assert r.query(np.array([35.0, -5.0, 68.0]), arm=0) is None
    assert r.query(np.array([35.0, -5.0, 68.0]), arm=1) is not None


def test_picks_the_reaching_arm(tmp_path):
    r = GraspRetriever(_library(tmp_path))
    assert r.arm_for(np.array([-30.0, -5.0, 68.0])) == 0
    assert r.arm_for(np.array([30.0, -5.0, 68.0])) == 1


def test_gripper_value_comes_from_demonstrations(tmp_path):
    """We had been guessing which pole of the bimodal demo distribution grips.
    Retrieval must take it from the frames that provably held cloth."""
    r = GraspRetriever(_library(tmp_path))
    q, grip, _ = r.query(np.array([-25.0, -5.0, 68.0]), arm=0)
    assert grip == pytest.approx(-0.14, abs=5e-3)
    assert q[5] == pytest.approx(grip)


def test_leave_one_out_is_reported(tmp_path):
    """If position does not determine posture, retrieval cannot deliver posture."""
    r = GraspRetriever(_library(tmp_path))
    st = r.loo_error(arm=0, n=40)
    assert st and st["mean_rad"] < 0.05


def test_empty_library_rejected(tmp_path):
    out = tmp_path / "empty.npz"
    np.savez(out, q=np.zeros((0, 12)), ee_pos=np.zeros((0, 3)),
             ee_quat=np.zeros((0, 4)), cp_pos=np.zeros((0, 3)),
             arm=np.zeros(0, int), grip_val=np.zeros(0),
             cp_idx=np.zeros(0, int), episode=np.zeros(0, int), frame=np.zeros(0, int))
    with pytest.raises(ValueError, match="empty grasp library"):
        GraspRetriever(out)


def test_coverage_reports_reachability(tmp_path):
    r = GraspRetriever(_library(tmp_path))
    inside = np.array([[-25.0, -5.0, 68.0], [-30.0, 0.0, 70.0]])
    outside = np.array([[-25.0, -5.0, 300.0], [-999.0, 0.0, 70.0]])
    assert r.coverage(inside, arm=0) == 1.0
    assert r.coverage(outside, arm=0) == 0.0

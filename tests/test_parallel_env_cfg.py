"""Tests for the parallel-env config rewriting.

Isaac-free: these check the path surgery that decides whether Isaac Lab's
cloner replicates anything at all. LeHome's env silently produces N empty env
origins when the paths are wrong, so a mistake here looks like working training
on identical data rather than an error.
"""

from dataclasses import dataclass, field

import pytest

from lehome.real_damped_project.tasks.parallel_garment_env import (
    GARMENT_SUBPATH,
    build_parallel_cfg,
)


@dataclass
class FakeAsset:
    prim_path: str


@dataclass
class FakeScene:
    num_envs: int = 1
    env_spacing: float = 4.0
    replicate_physics: bool = True


@dataclass
class FakeCfg:
    """Mirrors the fields of GarmentEnvCfg that the rewrite touches."""

    scene: FakeScene = field(default_factory=FakeScene)
    left_robot: FakeAsset = field(default_factory=lambda: FakeAsset("/World/Robot/Left_Robot"))
    right_robot: FakeAsset = field(default_factory=lambda: FakeAsset("/World/Robot/Right_Robot"))
    top_camera: FakeAsset = field(
        default_factory=lambda: FakeAsset("/World/Robot/Right_Robot/base/top_camera")
    )
    left_wrist: FakeAsset = field(
        default_factory=lambda: FakeAsset("/World/Robot/Left_Robot/gripper/left_wrist_camera")
    )
    right_wrist: FakeAsset = field(
        default_factory=lambda: FakeAsset("/World/Robot/Right_Robot/gripper/right_wrist_camera")
    )


def test_robots_move_into_the_env_namespace():
    """The cloner only replicates prims under /World/envs/env_.*/."""
    cfg = build_parallel_cfg(FakeCfg(), num_envs=8)
    assert cfg.left_robot.prim_path == "/World/envs/env_.*/Robot/Left_Robot"
    assert cfg.right_robot.prim_path == "/World/envs/env_.*/Robot/Right_Robot"


def test_cameras_follow_their_robots():
    """Wrist/top cameras are children of the arms, so they must move too."""
    cfg = build_parallel_cfg(FakeCfg(), num_envs=4)
    for path in (cfg.top_camera.prim_path, cfg.left_wrist.prim_path, cfg.right_wrist.prim_path):
        assert path.startswith("/World/envs/env_.*/Robot/"), path


def test_num_envs_and_spacing_are_applied():
    cfg = build_parallel_cfg(FakeCfg(), num_envs=16, env_spacing=2.5)
    assert cfg.scene.num_envs == 16
    assert cfg.scene.env_spacing == 2.5
    assert cfg.scene.replicate_physics is True


def test_rewrite_is_idempotent():
    """Applying it twice must not produce /World/envs/env_.*/Robot/envs/..."""
    cfg = build_parallel_cfg(build_parallel_cfg(FakeCfg(), 4), 4)
    assert cfg.left_robot.prim_path == "/World/envs/env_.*/Robot/Left_Robot"
    assert cfg.left_robot.prim_path.count("envs") == 1


def test_only_the_robot_prefix_is_rewritten():
    """The bedroom scene and light stay global on purpose -- static shared
    geometry, replicating it would multiply cost for no benefit."""
    cfg = FakeCfg()
    cfg.left_robot = FakeAsset("/World/Scene/NotARobot")
    out = build_parallel_cfg(cfg, num_envs=4)
    assert out.left_robot.prim_path == "/World/Scene/NotARobot"


def test_batched_view_expression_matches_the_cloned_paths():
    """The ClothPrim regex must match what cloning actually produces."""
    import re

    expr = f"/World/envs/env_.*/{GARMENT_SUBPATH}"
    for i in (0, 1, 7, 63):
        assert re.fullmatch(expr, f"/World/envs/env_{i}/{GARMENT_SUBPATH}"), i
    # and must not match the pre-clone absolute path
    assert not re.fullmatch(expr, f"/World/Object/{GARMENT_SUBPATH}")


def test_missing_camera_fields_are_tolerated():
    """Not every cfg variant defines all three cameras."""
    cfg = FakeCfg()
    delattr(cfg, "left_wrist")
    build_parallel_cfg(cfg, num_envs=2)  # must not raise

"""Task configuration, registration, and ``make_env`` (spec Sec. 5).

Provides:

* :class:`RealDampedTaskCfg` -- the flat config the env reads.
* :func:`build_lehome_real_damped_cfg` -- the spec's config builder, registered
  under ``"LeHome-Fold-Garment-RealDamped-v0"``.
* :func:`register_task` / :func:`make_env` -- a minimal local registry. When
  Isaac Lab is installed the same builder is additionally handed to
  ``gymnasium.register`` so ``isaaclab_tasks`` can find it; when it is not, the
  local registry keeps the training script runnable against the mock backend.

Damping parameters are chosen so all three levels of Sec. 1 are dissipative:

* **Cloth** -- Rayleigh ``(alpha, beta)`` sized so the sheet's oscillation decays
  within a fraction of a second rather than ringing across the episode.
* **Robot** -- ``zeta = 1.0`` exactly critical, with ``K`` low enough that
  ``omega_n * dt`` stays well inside the sampling-stability margin.
* **Policy** -- ``action_smoothing`` low-passes the command path, and
  ``lambda_delta_a`` penalises jerk in the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..math.cloth_functional import ClothFunctionalCfg
from .backend import MockClothCfg
from .rewards import LyapunovRewardCfg

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_task(name: str, env_cls: type) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Decorator binding a task name to ``(env_cls, cfg_builder)``."""

    def _decorator(builder: Callable[[], Any]) -> Callable[[], Any]:
        _TASK_REGISTRY[name] = {"env_cls": env_cls, "builder": builder}
        _try_register_with_isaaclab(name, env_cls, builder)
        return builder

    return _decorator


def _try_register_with_isaaclab(name: str, env_cls: type, builder: Callable[[], Any]) -> None:
    """Mirror the registration into gymnasium if Isaac Lab is present."""
    try:  # pragma: no cover - only exercised with Isaac Lab installed
        import gymnasium as gym
    except ImportError:
        return
    if name in gym.registry:
        return
    gym.register(
        id=name,
        entry_point=f"{env_cls.__module__}:{env_cls.__qualname__}",
        disable_env_checker=True,
        kwargs={"cfg_builder": builder},
    )


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RealDampedTaskCfg:
    """Everything the damped task needs (Sec. 5.1)."""

    # --- simulation ---
    dt: float = 1.0 / 60.0
    num_envs: int = 8
    device: str = "cuda"

    # --- robot impedance (Sec. 2.2) ---
    damping_ratio: float = 1.0
    """zeta >= 1: critically or over-damped."""
    stiffness: float = 200.0
    ee_mass: float = 1.0
    max_delta: float = 0.02
    """Metres per control step, per axis."""
    action_smoothing: float = 0.5
    """Low-pass beta on the command path; also the controller's Lipschitz const."""

    # --- cloth (Newton Rayleigh damping, Sec. 2.2) ---
    cloth_rayleigh_alpha: float = 4.0
    cloth_rayleigh_beta: float = 0.005
    """Small by necessity: see MockClothCfg.rayleigh_beta for the explicit-
    integration stability bound. Newton/PhysX solve damping implicitly and
    tolerate far larger values, so this is a mock-backend limit, not a physical
    statement about how damped the cloth should be."""
    cloth_stiffness: float = 60.0

    # --- cameras ---
    image_res: int = 64
    num_cameras: int = 2

    # --- reward (Sec. 2.3 / 5.1) ---
    reward: LyapunovRewardCfg = field(default_factory=LyapunovRewardCfg)

    # --- backend ---
    use_mock_backend: bool = True
    backend: MockClothCfg = field(default_factory=MockClothCfg)

    def sync(self) -> "RealDampedTaskCfg":
        """Propagate the top-level knobs into the backend config."""
        b = self.backend
        b.num_envs = self.num_envs
        b.dt = self.dt
        b.image_res = self.image_res
        b.rayleigh_alpha = self.cloth_rayleigh_alpha
        b.rayleigh_beta = self.cloth_rayleigh_beta
        b.stiffness = self.cloth_stiffness
        b.ee_stiffness = self.stiffness
        b.ee_mass = self.ee_mass
        b.ee_damping_ratio = self.damping_ratio
        if b.functional.grid_shape is None:
            b.functional.grid_shape = (b.rows, b.cols)
        b.success_threshold = self.reward.j_near
        return self


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

# Imported here (not at module top) to avoid a circular import: the task module
# imports backend/rewards, and this module imports the task class.
from .lehome_fold_garment_real_damped_task import LeHomeFoldGarmentRealDampedEnv  # noqa: E402

TASK_NAME = "LeHome-Fold-Garment-RealDamped-v0"


@register_task(name=TASK_NAME, env_cls=LeHomeFoldGarmentRealDampedEnv)
def build_lehome_real_damped_cfg() -> RealDampedTaskCfg:
    """Default config for the real-analysis-guided damped folding task."""
    cfg = RealDampedTaskCfg()

    cfg.dt = 1.0 / 60.0
    cfg.damping_ratio = 1.0
    cfg.stiffness = 200.0
    cfg.max_delta = 0.02
    cfg.action_smoothing = 0.5

    cfg.cloth_rayleigh_alpha = 4.0
    cfg.cloth_rayleigh_beta = 0.005

    cfg.reward = LyapunovRewardCfg(
        j_near=0.02,
        epsilon=1.0e-3,
        lambda_up=10.0,
        lambda_down=1.0,
        lambda_v=0.1,
        lambda_delta_a=0.01,
        near_gate="prev",
    )

    cfg.backend = MockClothCfg(
        rows=9,
        cols=9,
        size=(0.30, 0.30),
        max_episode_steps=200,
        functional=ClothFunctionalCfg(
            lambda_iou=1.0,
            lambda_edge=1.0,
            lambda_wrinkle=0.1,
            grid_res=64,
            grid_bounds=(-0.35, 0.35, -0.35, 0.35),
            splat_sigma=0.015,
            grid_shape=(9, 9),
        ),
    )
    return cfg.sync()


def make_env(
    task_name: str = TASK_NAME,
    num_envs: int = 8,
    sim_device: str = "cuda",
    rl_device: Optional[str] = None,
    graphics_device_id: int = 0,
    headless: bool = True,
    use_mock_backend: Optional[bool] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> LeHomeFoldGarmentRealDampedEnv:
    """Build a registered task (Sec. 6.1).

    ``graphics_device_id`` and ``headless`` are accepted to match the spec's
    call signature; they are forwarded to the Isaac backend and ignored by the
    mock, which never opens a renderer.
    """
    if task_name not in _TASK_REGISTRY:
        raise KeyError(f"unknown task {task_name!r}; registered: {list_tasks()}")

    entry = _TASK_REGISTRY[task_name]
    cfg = entry["builder"]()
    cfg.num_envs = num_envs
    cfg.device = sim_device
    if use_mock_backend is not None:
        cfg.use_mock_backend = use_mock_backend
    for key, value in (cfg_overrides or {}).items():
        if not hasattr(cfg, key):
            raise AttributeError(f"cfg has no attribute {key!r}")
        setattr(cfg, key, value)
    cfg.sync()

    if not cfg.use_mock_backend:  # pragma: no cover - requires Isaac Lab
        cfg.headless = headless
        cfg.graphics_device_id = graphics_device_id

    return entry["env_cls"](cfg, device=rl_device or sim_device)

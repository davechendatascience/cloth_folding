"""Task, reward, backend, and registration."""
from .backend import LeHomeBackend, MockClothCfg, MockFoldGarmentBackend
from .cfg import TASK_NAME, build_lehome_real_damped_cfg, make_env, register_task
from .lehome_fold_garment_real_damped_task import LeHomeFoldGarmentRealDampedEnv
from .rewards import LyapunovDescentReward, LyapunovRewardCfg

__all__ = [
    "LeHomeBackend",
    "MockClothCfg",
    "MockFoldGarmentBackend",
    "LeHomeFoldGarmentRealDampedEnv",
    "LyapunovDescentReward",
    "LyapunovRewardCfg",
    "TASK_NAME",
    "build_lehome_real_damped_cfg",
    "make_env",
    "register_task",
]

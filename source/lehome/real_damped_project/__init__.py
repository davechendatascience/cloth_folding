"""Real-analysis-guided damped visual RL for LeHome cloth folding.

Implements ``real_analysis_damped_cloth_folding_rl_spec.md``. Importing this
package registers ``LeHome-Fold-Garment-RealDamped-v0``.

Submodules are imported lazily so that ``import lehome.real_damped_project``
does not pull in torch until something is actually used.
"""

from __future__ import annotations

__all__ = [
    "ClothErrorFunctional",
    "ClothFunctionalCfg",
    "DampedImpedanceController",
    "LyapunovDescentReward",
    "LyapunovRewardCfg",
    "LeHomeFoldGarmentRealDampedEnv",
    "VisionAttentionPolicy",
    "build_lehome_real_damped_cfg",
    "make_env",
    "TASK_NAME",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in ("ClothErrorFunctional", "ClothFunctionalCfg"):
        from .math import cloth_functional as m

        return getattr(m, name)
    if name == "DampedImpedanceController":
        from .control.impedance_controller import DampedImpedanceController

        return DampedImpedanceController
    if name in ("LyapunovDescentReward", "LyapunovRewardCfg"):
        from .tasks import rewards as m

        return getattr(m, name)
    if name == "VisionAttentionPolicy":
        from .policy.vision_attention_policy import VisionAttentionPolicy

        return VisionAttentionPolicy
    if name in ("build_lehome_real_damped_cfg", "make_env", "TASK_NAME"):
        from .tasks import cfg as m

        return getattr(m, name)
    if name == "LeHomeFoldGarmentRealDampedEnv":
        from .tasks.lehome_fold_garment_real_damped_task import LeHomeFoldGarmentRealDampedEnv

        return LeHomeFoldGarmentRealDampedEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

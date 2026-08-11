"""A 'policy' that plays back a recorded episode's actions. LEHOME_REPLAY_EPISODE=0.

Purpose: get *rendered frames* for states we have ground truth for.

Demo replay folds the garment, so the environment is sound; but replay runs
through `dataset_sim.py`, which cannot save video. The eval path can. Driving the
eval with recorded actions therefore produces a rollout that follows the
demonstration exactly **and** writes the observation video, so frame k of our
render can be compared against frame k of the recording.

Any divergence is purely rendering: same actions, same garment, same start pose.
That isolates whether our images differ by camera geometry, garment scale, or
reset configuration -- the last unknown blocking a trustworthy evaluation, given
that two independent policies (ours and a 74.5% challenge winner) both imitate
demonstrations correctly on dataset frames and both no-op on ours.

Observations are ignored by construction: this is open-loop playback, and that is
the point.
"""

from __future__ import annotations

import os

import numpy as np

from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.registry import PolicyRegistry


class DemoReplayPolicy(BasePolicy):
    def __init__(self, device: str = "cuda", **kw):
        super().__init__()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = os.environ.get(
            "LEHOME_REPLAY_ROOT",
            "/home/edge-host/Documents/GitHub/lehome-challenge/Datasets/example/top_long_merged")
        ep = int(os.environ.get("LEHOME_REPLAY_EPISODE", "0"))
        ds = LeRobotDataset("lehome/dataset_challenge_merged", root=root,
                            revision="main", video_backend="pyav")
        a = int(ds.meta.episodes["dataset_from_index"][ep])
        b = int(ds.meta.episodes["dataset_to_index"][ep])
        self.actions = np.stack([ds[i]["action"].numpy() for i in range(a, b)])
        self.i = 0
        print(f"[replay_policy] episode {ep}: {len(self.actions)} recorded actions "
              f"(dataset frames {a}..{b - 1})", flush=True)

    def reset(self):
        self.i = 0

    def select_action(self, observation):
        # Hold the final pose if the episode outlasts the recording.
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return np.asarray(a, dtype=np.float32).reshape(-1)


def install() -> bool:
    if not os.environ.get("LEHOME_REPLAY_EPISODE"):
        return False
    PolicyRegistry._registry["lerobot"] = DemoReplayPolicy
    print("[replay_policy] ACTIVE - 'lerobot' now DemoReplayPolicy", flush=True)
    return True

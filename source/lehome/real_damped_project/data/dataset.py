"""Sequence dataset over the preprocessed LeHome demonstration cache.

Serves ``(images, proprio, action)`` windows in exactly the form the policy
consumes at rollout time, so a behaviour-cloned network can be dropped straight
into the RL runner:

* images  ``(T, 9, H, W)`` float32 in [0,1] -- same channel order and scaling
  as ``IsaacGarmentBackend.render_cameras()``
* proprio ``(T, 12)``                       -- same as ``get_proprioception()``
  under ``proprio_matches_dataset``
* action  ``(T, 12)``                       -- joint position targets

Windows never straddle an episode boundary. This matters more than it looks:
the policy is recurrent, so a window spanning two episodes would train the GRU
to carry state across a teleport, which is the same error the RL side guards
against with per-episode hidden-state resets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LeHomeDemoDataset(Dataset):
    """Windows of demonstration frames from a :mod:`.preprocess` cache.

    Args:
        cache_dir: output of ``preprocess.py``.
        seq_len: window length ``T``. 1 gives plain per-frame BC.
        normalize_proprio: standardise the 12-D state with dataset statistics.
            Strongly recommended -- joint angles span very different ranges and
            an unnormalised input makes the encoder waste capacity on scale.
        episodes: optional subset of episode indices (for train/val splits).
    """

    def __init__(
        self,
        cache_dir: str,
        seq_len: int = 16,
        normalize_proprio: bool = True,
        episodes: Optional[np.ndarray] = None,
        delta_target: bool = True,
    ) -> None:
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "meta.json").read_text())
        self.meta = meta
        self.seq_len = int(seq_len)

        c, h, w = meta["image_shape"]
        self.image_shape = (c, h, w)
        n = meta["n_frames"]
        self._images = np.memmap(
            cache_dir / "images.u8", dtype=np.uint8, mode="r", shape=(n, c, h, w)
        )
        self.state = np.load(cache_dir / "state.npy")
        self.action = np.load(cache_dir / "action.npy")
        self.episode = np.load(cache_dir / "episode.npy")

        # --- delta target -------------------------------------------------
        # Predicting the absolute joint target a[t] is what broke the first
        # attempt. Because a[t] ~ s[t] at 30 fps, proprioception predicts the
        # label through a near-identity map and captures nearly all of the
        # loss; the cameras are needed only for the residual, so the gradient
        # never pressures the visual encoder. Measured consequence: image
        # influence / proprio influence = 0.11, and the policy did not move.
        #
        # With the target a[t] - s[t], proprioception cannot produce the label
        # and persistence (a[t] = a[t-1]) stops being a competitive predictor,
        # so *where the cloth is* becomes the only remaining signal.
        self.delta_target = bool(delta_target)
        self.target = (self.action - self.state) if self.delta_target else self.action

        # --- Lyapunov labels (optional) -----------------------------------
        # J(x_t) per frame, produced by replaying demonstrations in the
        # simulator (scripts/label_demos_with_J.py). Absent until that pass has
        # been run; the trainer falls back to unweighted BC.
        self.J = None
        self.dJ = None
        jf = cache_dir / "J.npy"
        if jf.exists():
            self.J = np.load(jf).astype(np.float32)
            dJ = np.zeros_like(self.J)
            dJ[:-1] = self.J[1:] - self.J[:-1]
            # dJ is undefined across an episode boundary -- the cloth teleports.
            dJ[np.asarray(self.episode[:-1] != self.episode[1:]).nonzero()[0]] = 0.0
            dJ[-1] = 0.0
            self.dJ = dJ

        # Per-dimension statistics over the *training* frames only.
        self.normalize_proprio = normalize_proprio
        sel = np.isin(self.episode, episodes) if episodes is not None else slice(None)
        self.state_mean = self.state[sel].mean(0).astype(np.float32)
        self.state_std = (self.state[sel].std(0) + 1e-6).astype(np.float32)

        # Valid window starts: seq_len frames entirely inside one episode.
        starts = []
        ep = self.episode
        i = 0
        while i < len(ep):
            j = i
            while j + 1 < len(ep) and ep[j + 1] == ep[i]:
                j += 1
            if episodes is None or ep[i] in episodes:
                for s in range(i, j - self.seq_len + 2):
                    starts.append(s)
            i = j + 1
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def proprio_dim(self) -> int:
        return int(self.state.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    @property
    def has_lyapunov_labels(self) -> bool:
        return self.J is not None

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        e = s + self.seq_len
        img = torch.from_numpy(np.asarray(self._images[s:e], dtype=np.float32) / 255.0)
        p = self.state[s:e]
        if self.normalize_proprio:
            p = (p - self.state_mean) / self.state_std

        t = torch.from_numpy(np.ascontiguousarray(self.target[s:e], dtype=np.float32))
        out = [img, torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)), t]
        if self.J is not None:
            out.append(torch.from_numpy(np.ascontiguousarray(self.J[s:e], dtype=np.float32)))
            out.append(torch.from_numpy(np.ascontiguousarray(self.dJ[s:e], dtype=np.float32)))
        else:
            z = torch.zeros(self.seq_len, dtype=torch.float32)
            out.extend([z, z])
        return tuple(out)


def split_episodes(cache_dir: str, val_frac: float = 0.1, seed: int = 0):
    """Deterministic episode-level train/val split.

    Splitting by *episode* rather than by frame is essential: neighbouring
    frames are nearly identical, so a frame-level split leaks almost the whole
    validation set into training and reports a meaninglessly low val loss.
    """
    episode = np.load(Path(cache_dir) / "episode.npy")
    eps = np.unique(episode)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(eps)
    n_val = max(1, int(round(len(eps) * val_frac)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])

"""Decode a LeHome LeRobot-v3 dataset into a memmapped tensor cache.

Random access into AV1 mp4 is slow (every read seeks and re-decodes a GOP),
which would make the dataloader the bottleneck. Sequential decode is fast
though -- measured **1567 fps** on this box -- so we decode each video exactly
once, in order, and write frames into a uint8 memmap at the policy's input
resolution. Training then reads by index at memory-bandwidth speed.

Layout produced in ``out_dir``:

    images.u8      memmap (N, 3*C, H, W) uint8   -- cameras stacked on channels
    state.npy      (N, 12) float32               -- observation.state
    action.npy     (N, 12) float32               -- action
    episode.npy    (N,)    int64                 -- episode index per frame
    frame.npy      (N,)    int64                 -- frame index within episode
    meta.json      shapes, fps, camera order, source

Channel order matches ``IsaacGarmentBackend.render_cameras()`` exactly
(top, left, right), so a policy trained here consumes the simulator's
observations unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Order is load-bearing: it must match IsaacGarmentBackend.render_cameras().
CAMERA_KEYS = (
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
)


def _resize_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour box resize, HWC uint8 -> (size, size, 3) uint8.

    Deliberately dependency-free: cv2/PIL are not in this venv, and bilinear
    quality is irrelevant at this downsampling factor for a control policy.
    """
    h, w = frame.shape[:2]
    ys = (np.arange(size) * (h / size)).astype(np.int32)
    xs = (np.arange(size) * (w / size)).astype(np.int32)
    return frame[ys][:, xs]


def preprocess(dataset_dir: str, out_dir: str, size: int = 84, limit_episodes: int | None = None):
    import av
    import pyarrow.parquet as pq

    dataset_dir = Path(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    ep_files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    ep = pq.read_table(ep_files).to_pydict()
    n_ep = len(ep["episode_index"])
    if limit_episodes:
        n_ep = min(n_ep, limit_episodes)
    print(f"[preprocess] {dataset_dir.name}: {n_ep} episodes, fps={fps}")

    # ---- frame tables -----------------------------------------------------
    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    cols = ["observation.state", "action", "episode_index", "frame_index"]
    tab = pq.read_table(data_files, columns=cols).to_pydict()

    keep = np.array(tab["episode_index"]) < n_ep
    state = np.asarray(tab["observation.state"], dtype=object)[keep]
    action = np.asarray(tab["action"], dtype=object)[keep]
    state = np.stack([np.asarray(r, dtype=np.float32) for r in state])
    action = np.stack([np.asarray(r, dtype=np.float32) for r in action])
    episode = np.asarray(tab["episode_index"], dtype=np.int64)[keep]
    frame = np.asarray(tab["frame_index"], dtype=np.int64)[keep]
    n = len(state)
    print(f"[preprocess] {n} frames, state{state.shape} action{action.shape}")

    np.save(out / "state.npy", state)
    np.save(out / "action.npy", action)
    np.save(out / "episode.npy", episode)
    np.save(out / "frame.npy", frame)

    # global row for (episode, frame_index)
    row_of = {}
    for i in range(n):
        row_of[(int(episode[i]), int(frame[i]))] = i

    n_cam = len(CAMERA_KEYS)
    images = np.memmap(out / "images.u8", dtype=np.uint8, mode="w+", shape=(n, n_cam * 3, size, size))

    # ---- decode each camera, sequentially per video file ------------------
    for cam_i, key in enumerate(CAMERA_KEYS):
        # group episodes by the video file they live in
        by_file: dict[tuple[int, int], list[int]] = {}
        for e in range(n_ep):
            c = int(ep[f"videos/{key}/chunk_index"][e])
            f = int(ep[f"videos/{key}/file_index"][e])
            by_file.setdefault((c, f), []).append(e)

        written = 0
        for (c, f), eps in sorted(by_file.items()):
            path = dataset_dir / "videos" / key / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
            # frame-in-file -> global row, for every episode in this file
            target: dict[int, int] = {}
            for e in eps:
                start = int(round(float(ep[f"videos/{key}/from_timestamp"][e]) * fps))
                length = int(ep["length"][e])
                for k in range(length):
                    r = row_of.get((e, k))
                    if r is not None:
                        target[start + k] = r
            if not target:
                continue

            last = max(target)
            container = av.open(str(path))
            idx = 0
            for pic in container.decode(video=0):
                r = target.get(idx)
                if r is not None:
                    arr = pic.to_ndarray(format="rgb24")
                    images[r, cam_i * 3 : (cam_i + 1) * 3] = _resize_uint8(arr, size).transpose(2, 0, 1)
                    written += 1
                idx += 1
                if idx > last:
                    break
            container.close()
        print(f"[preprocess]   {key}: {written}/{n} frames")

    images.flush()
    (out / "meta.json").write_text(json.dumps({
        "n_frames": int(n),
        "n_episodes": int(n_ep),
        "image_shape": [n_cam * 3, size, size],
        "cameras": list(CAMERA_KEYS),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "fps": fps,
        "source": str(dataset_dir),
    }, indent=2))
    print(f"[preprocess] wrote {out}  ({os.path.getsize(out/'images.u8')/1e9:.2f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--limit_episodes", type=int, default=None)
    a = p.parse_args()
    preprocess(a.dataset, a.out, a.size, a.limit_episodes)

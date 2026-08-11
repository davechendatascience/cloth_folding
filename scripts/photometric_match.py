"""Match eval camera statistics to the training distribution. LEHOME_PHOTOMATCH=1.

The simulator on this machine renders about half as bright as the data the
policy was trained on:

    dataset  observation.images.top_rgb : mean 0.774  (208.6/255)
    eval     observation.images.top_rgb : mean 0.414  (105.6/255)

That is a train/eval distribution shift large enough on its own to explain a
policy that scores 0/24 while its training loss falls to 0.042.

The cause is the renderer, not the scene. `rtx/nrd/PackForNRD.cs.hlsl` fails to
compile on GB10 (1,895 times in a single episode under `rendering_mode=quality`),
and the recording machine evidently had a working RTX path. Two things were
tried and rejected:

* `rendering_mode=performance` removes every shader error but only lifts the
  mean 105.6 -> 129.8.
* Raising the dome light 1200 -> 2400 lifts it 129.8 -> 139.1. The renderer
  tonemaps hard, so 2x the light buys 7% of pixel value; reaching 208.6 that way
  would blow out highlights and change the image rather than match it. (Enabling
  `light_randomization` makes it *worse*: it samples `color` absolutely from
  [0.0, 0.2] against a 0.75 default, so it darkens more than the 3500-5000
  intensity brightens.)

Since the discrepancy is a global photometric transform, correcting it on the
observation is both cheaper and more faithful than retraining: per channel,

    x' = (x - mean_eval) / std_eval * std_train + mean_train

with the eval statistics computed per frame (they drift as the scene changes)
and the training statistics fixed, measured over 40 random dataset frames.

This is a workaround for a renderer we cannot reproduce, not a correctness fix.
The robust alternative is to retrain with photometric augmentation so the policy
stops caring about absolute brightness -- worth doing if this proves brittle,
but it costs a full training run and this does not.
"""

from __future__ import annotations

import os

import numpy as np

from scripts.eval_policy.registry import PolicyRegistry

# Per-channel (R, G, B) statistics of the training data, over 40 random frames.
TRAIN_STATS = {
    "top_rgb":   ((0.8014, 0.7820, 0.7379), (0.1667, 0.2022, 0.2435)),
    "left_rgb":  ((0.7493, 0.7107, 0.6679), (0.1964, 0.2462, 0.2752)),
    "right_rgb": ((0.7527, 0.7217, 0.6825), (0.1952, 0.2399, 0.2775)),
}


def _match(img: np.ndarray, key: str) -> np.ndarray:
    """Rescale an (H, W, 3) uint8 or float image to the training statistics."""
    stats = None
    for name, s in TRAIN_STATS.items():
        if name in key:
            stats = s
            break
    if stats is None:
        return img

    was_uint8 = img.dtype == np.uint8
    x = img.astype(np.float32) / (255.0 if was_uint8 else 1.0)
    tm, ts = np.asarray(stats[0], np.float32), np.asarray(stats[1], np.float32)

    if x.ndim == 3 and x.shape[-1] == 3:
        # Match LUMINANCE only, applying one affine to all three channels, so
        # chroma ratios survive. Per-channel matching (the previous version)
        # corrects brightness but rewrites hue: it drove bluish-pixel fraction
        # 41.3% -> 11.0% and rendered the denim garment violet. A policy
        # conditioned on colour is then looking at the wrong object.
        lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        tl = float(np.dot(np.asarray(stats[0], np.float32),
                          [0.2126, 0.7152, 0.0722]))
        tsl = float(np.dot(np.asarray(stats[1], np.float32),
                           [0.2126, 0.7152, 0.0722]))
        gain = tsl / (lum.std() + 1e-6)
        bias = tl - gain * lum.mean()
        x = x * gain + bias
    else:  # unexpected layout -- fall back to a scalar match, never crash an eval
        x = (x - x.mean()) / (x.std() + 1e-6) * float(ts.mean()) + float(tm.mean())

    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8) if was_uint8 else x


def install() -> bool:
    if not os.environ.get("LEHOME_PHOTOMATCH"):
        return False

    cls = PolicyRegistry._registry["lerobot"]
    orig = cls.select_action

    def wrapped(self, observation):
        obs = dict(observation)
        for k, v in observation.items():
            if "images" in k:
                obs[k] = _match(np.asarray(v), k)
        return orig(self, obs)

    cls.select_action = wrapped
    print("[photometric_match] ACTIVE - eval images rescaled to training statistics",
          flush=True)
    return True

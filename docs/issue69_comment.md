# Draft comment for lehome-official/lehome-challenge#69

Seeing the same failure on different hardware, with some measurements that may
narrow it down.

## Environment

| | |
|---|---|
| GPU | NVIDIA GB10 (DGX Spark), sm_121, **aarch64** |
| Isaac Sim | 5.1.0.0 |
| IsaacLab | `lehome-official/IsaacLab` @ `69f6fa54` |
| torch | 2.7.0+cu128, Python 3.11 |
| dataset | `Datasets/example/top_long_merged`, garment `Top_Long_Seen_0` |

Same symptom as the original report: arms move, garment does not, episodes run to
`max_steps`, videos land in `failure/`, checker reports failure.

## The measurement that seems most useful

Replaying the **recorded actions** from the dataset (feeding
`episode 0`'s recorded `action` values straight into `env.step`) *does* fold the
garment. Trained policies do not. Same environment, same garment, same episode.

Check-point displacement, measured after the garment has settled (the reset drop
from z=0.73 to z=0.53 must be excluded or it swamps everything):

| driver | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| **replayed recorded actions** | 1.1 | 0.6 | 0.7 | 0.2 | **29.8** | **26.6** |
| SmolVLA finetuned from `smolvla_base`, 15k steps | 0.7 | 0.5 | 0.5 | 0.3 | 1.1 | 0.4 |
| `Papercold/lehome-smolvla-top-long` (30k steps, community) | 0.7 | 0.6 | 0.4 | 4.0 | 1.1 | 0.5 |
| `IliaLarchenko/lehome_sim` (1st place, pi0.5, reported 74.5% on long tops) | 0.6 | 0.6 | 0.5 | 6.3 | 1.1 | 0.5 |

Three independently trained policies, two architectures, one of them the released
winning checkpoint — all produce essentially no manipulation, while open-loop
replay works. That is what makes me think this is environmental rather than a
training problem.

Segmented garment footprint over an episode tells the same story
(Otsu threshold, robot masked out):

```
                  step 0%   36%    72%   100%
recorded dataset    20.5%  15.3%  11.3%   6.3%     <- folds
our sim, replayed   37.1%  38.5%  38.1%  38.5%     <- flat
```

## The mechanism I can see

Logging the distance from each gripper body to the nearest of the 14,544 cloth
particles, every step, while replaying recorded actions (965 ticks):

```
                 min     p05   median   steps within 0.09 m
left  gripper   0.065   0.068   0.110      156/965
right gripper   0.101   0.103   0.146        0/965
```

The gripper *body* origin sits ~9 cm from the actual contact point, so the left
hand reaches the fabric and **the right hand never does — 0 of 965 steps**, ~1 cm
short at its closest. A bimanual fold with one hand always off the cloth would
explain both the replay variability and why a closed-loop policy gets no useful
feedback.

Is the recorded dataset expected to reproduce the fold in the current sim? If the
recorded joint trajectories are supposed to land both grippers on the garment and
here one of them consistently misses, that would point at a robot/base/URDF
offset rather than anything policy-side.

## Framing question

Our top camera renders the garment noticeably larger than the recorded frames do:
37.1% of frame area against the dataset's 20.5% (measured over three episodes:
20.5 / 20.5 / 24.5), with the garment clipped at the frame edges. Camera geometry
reads as configured — world position (-0.015, 0.190, 1.060), garment centroid
(-0.008, 0.056, 0.530), distance 0.547 m, HFOV 67.2 deg — so the config is being
applied; the recorded data just implies a camera roughly 0.27 m further back.
Is a framing difference between the released dataset and the current eval config
expected?

## Ruled out here

In case it saves anyone time — none of these changed the outcome:

* `--rendering_mode` performance / balanced / quality
* `SimulationCfg.use_fabric` True and False
* **`TiledCamera` -> `Camera`** (the fix from #59). Byte-identical output for us,
  and we never saw the `unsupported toolchain` / warp-compile error, so #59 does
  not appear to apply to GB10 even though it is also Blackwell
* camera registration (cameras are in `scene.sensors` and do update)
* geometry writeback — `get_current_mesh_points()` and
  `_cloth_prim_view.get_world_positions()` agree exactly, and both move
* action units — the policy returns absolute joint targets, matching the dataset
* running the winner's own fork of this repo for env + eval code

The render is faithful to the physics: the rendered garment and its 3-D bbox
agree with each other. The cloth genuinely is not being manipulated.

## Two small bugs found on the way

1. `success_checker_garment_fold` is decorated `@step_interval(interval=50)`,
   which returns a literal `False` on 49 of 50 calls where callers expect the
   result dict, and its `call_count` lives in the decorator closure so it never
   resets between episodes or garment switches. The evaluated step therefore
   drifts arbitrarily through a run and a transient success can be missed. Our
   episodes logged 13 checks over 600 steps before we bypassed it, 315 after.

2. `/isaaclab/cameras_enabled` is commented out in `tiled_camera.py` in the
   IsaacLab fork but still enforced in `camera.py`. So `--enable_cameras` never
   actually reaches the setting, and swapping to `Camera` per #59 fails with
   "A camera was spawned without the --enable_cameras flag" until that guard is
   disabled too.

Happy to share the probe scripts or run further diagnostics if useful.

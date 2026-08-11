# Pipeline defects and corrections

Written 2026-08-11, after a SmolVLA finetune reached step ~14000 with loss 0.042
and scored 0/24 on LeHome's official success predicate at every checkpoint.

Investigating *why* turned up a train/eval mismatch that makes every evaluation
number in this project's SmolVLA thread uninterpretable, plus a list of my own
analysis errors that produced three successive wrong diagnoses before the right
one. Both are recorded here, because the errors were as expensive as the bugs.

Ordered by what blocks progress. Items are marked FIXED / OPEN.

---

## 1. BLOCKING — eval renders ~2x darker than the training data (OPEN)

| | mean px | p95 | max |
|---|---|---|---|
| demo `observation.images.top_rgb` | 208.6 | 225 | 239 |
| eval `observation.images.top_rgb` | 105.6 | 140 | 173 |

The policy is fed images roughly half as bright as everything it trained on, and
which never reach the brightness the demos routinely hit. For a vision-language
policy this is sufficient on its own to explain 0/24 success, an untouched
garment, and demo-scale arm motion arriving nowhere.

**Suspected cause:** 1,895 occurrences of

```
[Error] [rtx.denoising.plugin] Failed to compile compute shader: rtx/nrd/PackForNRD.cs.hlsl
```

NRD (NVIDIA Real-time Denoisers) never initialises on this GB10. It is the *only*
shader that fails — no geometry, load, or render errors of any other kind — and
the dataset was recorded on hardware where it compiled.

**Do not tune, retrain, or re-evaluate anything until this is resolved.** Every
eval number currently in `runs/smolvla_eval/` should be treated as void.

## 2. Possible residual garment scale / framing difference (OPEN)

After matching brightness statistics to the demos, bright(table) is 55% against
the demos' 68.5%. Much smaller than the brightness effect and possibly an
artefact of the correction. Re-check only after #1.

---

## Environment defects (all FIXED unless noted)

| defect | symptom | fix |
|---|---|---|
| `warp-lang` unpinned by IsaacLab | 1.16.0 shadows Isaac's bundled 1.8.2; no `warp.types.array` | pin 1.8.1 |
| torchvision 0.24.0 vs torch 2.7.0 | `operator torchvision::nms does not exist` | 0.22.0, `--no-deps` |
| `python -m scripts.eval` on aarch64 | forced `spawn` + `isaacsim`'s import-time Process = unbounded recursion | keep `fork` (`scripts/lehome_eval.py`) |
| sklearn bundled libgomp | `cannot allocate memory in static TLS block` | preload it too |
| `LD_PRELOAD` | Isaac matches the literal `/lib/...`, not the `/usr/lib/...` symlink | use the literal path |
| CWD | garment USDs resolve relative to the working directory | run from the lehome-challenge checkout |
| their documented `--device cpu` | crashes after policy load | use `cuda` (headless, so their GUI rationale does not apply) |
| their `configs/train_smolvla.yaml` | no `pretrained_path`; `load_vlm_weights` defaults False, so it trains from scratch | point at `lerobot/smolvla_base` |
| **transformers on the openpi fork (4.53.3)** | installed for pi0, which was abandoned; SmolVLA now trains under it | **OPEN** — no version gate and it loads fine, but it is an unintended deviation from the baseline environment |

---

## My analysis errors

Recorded because the same failure mode recurred: concluding from a measurement
whose dynamic range or sensitivity I had not checked against a control.

1. **"The robot isn't rendered into the observations."** Wrong — it is, at 4-5%
   of frame, the same as the demos. I inferred this from a 104 rad arm sweep
   producing 0.10 grey of image change, without asking whether the *brightness*
   was comparable.
2. **"Garment position is never randomised."** Wrong. `soft_reset_pos_range` has
   min == max in the *common* config, but the per-garment JSON overrides it with
   x over 12 cm and y over 6 cm. This materially weakens the phase-leakage
   argument I built on it.
3. **"Camera framing is the root cause."** Mostly wrong. My composition metric
   thresholded on `>200` for "bright table", so a dark render scored 0% table and
   73% cloth — it was measuring brightness, not framing.
4. **Dismissed the shader errors as harmless** ("image quality, not geometry").
   For a vision policy, image quality *is* the input. The user flagged these
   first; they appear to be the root cause.
5. **Presented a J checkpoint curve as progress** (4.90 -> 3.59, "3.9 standard
   errors"). It was driven by two discontinuous single-vertex jumps of 21.7 cm
   and 9.0 cm, in episodes where five of six check-points moved under 2.4 cm.
6. **Cross-referenced a video from one run with a joint log from another**, on a
   simulator already documented as chaotic. Produced two successive wrong
   diagnoses. Same checkpoint, garment and seed gave J_final 4.25 and 6.87.
7. **Frame-mean pixel difference** to conclude "nothing moves" — insensitive to
   localised motion. The correct test (diff against a no-op control) took one
   command and settled it immediately.
8. **Built the vision ablation before validating the observations.** Wasted; it
   would have compared a broken input against a differently-broken input.
9. **`parse_eval.py` subtracted "Unseen" from "Seen"** though the sets never
   overlap (`Top_Long_Unseen_0` contains no capital-S "Seen"), reporting 0/16
   where the truth was 0/20.
10. **Bypassed the eval chain's mutex**, running two Isaac instances at once.

---

## Open methodological gaps

- **No validation split.** All 250 episodes are in training (`dataset.episodes:
  None`), so the loss is a train loss. Mitigated only by the schedule being
  ~11.6 epochs rather than hundreds.
- **10 of 12 eval garments are in the training set.** `Seen_0..9` train; eval
  adds `Unseen_0/1` and pools all twelve into one number, so the headline rate is
  83% in-distribution. Report Seen and Unseen separately, always with
  denominators — at `--num_episodes 2` the unseen half is 4 episodes.
- **Large eval run-to-run variance.** J_final 4.25 vs 6.87 on an identical
  configuration. n=24 is underpowered for the effect sizes previously quoted.

---

## What is NOT broken

Training reads the dataset directly and is unaffected by the render defect:
SmolVLA loads `smolvla_base` (450M total / 100M trainable, VLM frozen), loss
falls 0.768 -> 0.042 over 14000 steps with grad norm decaying smoothly, and
`data_s` ~0.01 confirms the input pipeline keeps up. The dataset itself is
sound. The defect is confined to the evaluation render path.

---

## Fix log

### #1 render mismatch — FIXED, and it did not help (2026-08-11)

Two changes, both eval-side, neither requiring a retrain:

* `--rendering_mode performance` — eliminates **all** 1,895
  `PackForNRD.cs.hlsl` failures (`quality` 1895, `balanced` 185, `performance` 0)
  and lifts mean pixel 105.6 -> 129.8.
* `scripts/photometric_match.py` (`LEHOME_PHOTOMATCH=1`) — per-channel rescale of
  each observation to the training statistics, measured over 40 dataset frames.
  Lifts 129.8 -> 197.0 against the dataset's 208.6.

Rejected along the way: raising the dome light 1200 -> 2400 moves the mean only
129.8 -> 139.1 (the renderer tonemaps hard, so 2x light buys 7% of pixel value),
and enabling `light_randomization` makes it *worse* — it samples `color`
absolutely from [0.0, 0.2] against a 0.75 default, darkening more than the
3500-5000 intensity brightens. That is a defect in their config.

**Result on checkpoint 12500, one episode:**

| | J start -> min -> final | max check-point displacement |
|---|---|---|
| no-op | 7.36 -> 7.36 -> 7.43 | — |
| policy, broken render | 7.35 -> 7.33 -> 6.87 | 8.04 cm (single-vertex jump) |
| policy, render fixed | 7.35 -> 7.33 -> 7.42 | 1.23 cm |

With correct imagery the policy is **indistinguishable from doing nothing**. The
render defect was real and is fixed; it was not the reason the policy fails.
n=1, and run-to-run variance here is large, so this needs the full 12-garment
protocol before it is more than indicative.

**What this rules out:** train/eval appearance shift as the explanation.
**What it does not rule out:** the policy simply has not learned the task at
12500 steps (4.8 epochs), or `train_expert_only: True` leaving the frozen VLM
features insufficient.

### #1 follow-ups: what the fixes ruled out (2026-08-11)

Three experiments, each ruling out a hypothesis:

| experiment | result | rules out |
|---|---|---|
| **demo replay** through this eval env | J 7.27 -> **0.00, Success** | environment, initial state, physics, success predicate, render fix |
| **offline action check**, 120 demo frames | policy MSE 0.005083 vs persistence 0.044366 — **beats it 8.7x** | undertrained / model choice / data volume |
| **`n_action_steps` 50 -> 10** (60 observations per episode, not 12) | J 7.35 -> 7.33, displacement <= 1.01 cm | open-loop horizon as the sole cause |

So the environment works, the policy imitates well, and the two together do not.

**Lead:** comparing the policy's joint distribution against demo episode 0, the
**left arm never extends**:

| joint | demo mean | demo range | policy mean | policy range | delta |
|---|---|---|---|---|---|
| L_lift | -0.627 | [-1.726, 1.264] | -1.420 | [-1.815, 0.201] | **-0.793** |
| L_elbow | 0.598 | [-1.607, 1.547] | 1.380 | [-0.192, 1.630] | **+0.782** |
| R_elbow | 0.684 | [-1.405, 1.559] | 0.579 | [-0.554, 1.584] | -0.105 |

~45 degrees of offset on both left joints, in the directions that keep the arm
folded, while the right arm tracks the demonstrations to ~0.1 rad. A bimanual
fold with one arm parked cannot succeed, and replay -- which uses both -- does.

Caveats: n=1 episode against n=1 demo episode, on a simulator with large
run-to-run variance. Confirm across episodes before acting on it. Candidate
causes to separate: a left/right observation or camera mismatch at eval, versus
the policy genuinely having learned this asymmetry (checkable offline, by
measuring per-joint error on left vs right across many demo frames -- the
offline check already reports per-joint MSE and did **not** show the left arm
as anomalous, which points at eval rather than training).

### #2 The winner's policy also no-ops here — the sim observation is the fault (2026-08-11)

Ran the LeHome Challenge 2026 winning policy (Larchenko, 1st of 62, 74.5% on
long tops) through *our* harness. Getting there required:

* a separate `jaxenv` (JAX 0.5.3 + openpi), **CPU-only torch** so JAX owns the GPU
* `restore_params(..., dtype=jnp.float32)` and `pi_modified_config.dtype="float32"`
  — with the shipped `bfloat16` the server dies on GB10 with
  `Unsupported conversion from bf16 to f16 / LLVM ERROR`, an XLA crash on sm_121
* `scripts/remote_ws_policy.py` — their `serve.py` speaks a stateless WebSocket
  `infer_chunk` protocol and expects LeHome's native observation keys, so our env
  output forwards nearly unchanged (images base64, `next_initial_actions` echoed
  back as the rolling inpaint anchor). Their served config is
  **`execute=5`** — re-planning every 5 steps, against our 50.

**Their policy on a real demo frame** (dataset image, episode 0 frame 100):

```
state       [ 0.227, -1.633,  1.538, ...  -0.123, -0.375, -0.491, ...]
action[0]   [ 0.154, -1.622,  1.412, ...   0.355, -1.562,  1.297, ...]
demo action [ 0.150, -1.623,  1.395, ...   0.253, -1.414,  1.185, ...]
```

Absolute actions (not deltas), reproducing the demonstration to ~3 decimals on
the left arm. Protocol, units and adapter are all correct.

**Their policy in our simulator:** J 7.35 -> 7.26 -> 7.37, max check-point
displacement 1.13 cm, `Return=106.41` — the *identical* return produced by the
no-op control and by our SmolVLA. Its own success head read 0.88-0.95 throughout,
so the policy believed it was folding.

**Conclusion.** Two independent policies — ours and a competition winner — both
imitate demonstrations correctly on dataset images and both collapse to a no-op
on our simulator images. The policy is no longer a plausible common cause. The
fault is in **what our evaluation feeds the policy**.

This retires the compounding-error explanation from the previous section: a
policy that scores 74.5% on this benchmark does not drift to a standstill.

Still-unexplained observation difference, and the next thing to pin down: the
demo top camera shows a flat garment over wide white margins, while our sim
frame shows the garment filling the view. Earlier I attributed this to my
brightness threshold and dropped it; with the policy ruled out it is the leading
candidate and needs a proper same-scale comparison of a dataset frame against a
sim frame.

### #3 Replacing the eval chain does not help — it is the render (2026-08-11)

Cloned the winner's own `lehome-challenge` fork and ran **their** env + eval code,
against **their** policy, on this machine: `Return=105.53`, no fold. Same failure
as our chain.

Their fork does contain two things worth taking:

* **`success_checker_garment_fold` is defective in the official repo.** Their
  comment: its `@step_interval(50)` decorator "returns a literal False on 49 of
  50 calls and leaks its module-global counter across episodes". They bypass it
  and call `check_top_sleeve` / `check_pant_*` directly. This explains why our
  per-episode success checks appeared only ~12-36 times over 600 steps at
  inconsistent phases, and means a transient success can be missed outright.
* `apply_camera_overrides` — camera resolution/depth are expected to vary
  (`LEHOME_TOP_CAMERA_WIDTH`, `LEHOME_NO_DEPTH`).

Their `visual_augmentation.py` is *training-time* domain randomisation for RL
rollouts (garment recolour, camera pos/rot/focal jitter, dome-light rotation),
i.e. how their policy tolerates visual variation — not an observation fix. Their
sim was never broken, so their code contains no repair for ours.

**Elimination table**

| component | verdict |
|---|---|
| our eval chain | not the cause — theirs fails identically |
| their eval chain | not a fix |
| policy | not the cause — two independent policies, one a 74.5% winner |
| action plumbing, units, absolute-vs-delta | verified correct |
| env, physics, success predicate, initial robot pose | verified — replay folds, J 7.27 -> 0.00 |
| **rendered observation** | **the cause** |

Demo replay succeeds *because* it is open-loop and never reads an image.
Everything that reads an image fails.

**The visual difference, measured.** Dataset frame 0 vs our sim frame 0, same
garment, brightness-matched:

* demo: flat spread T-shirt, both sleeves visible, ~40% of frame, wide white margins
* sim: garment fills the frame, sleeves running off the edges, bunched

roughly 2x larger linearly. Candidate causes, in order: camera FOV/aspect or
position, garment `scale` (per-garment JSON says 0.45, the common config 0.4),
or the reset not laying the garment flat.

**Also note:** `scripts/photometric_match.py` fixes brightness but distorts hue —
bluish-pixel fraction goes 41.3% raw -> 11.0% matched and the denim renders
violet. Per-channel mean/std matching is the wrong transform; match luminance
only, or fix the renderer instead.

### #4 Success checker repaired (2026-08-11) — FIXED

`scripts/fix_success_checker.py`, installed by both launchers before
`lehome.tasks.bedroom` is imported (so the `from ... import` binding picks up the
repaired reference).

The upstream defect, in eight lines:

```python
def step_interval(interval=50):
    def decorator(func):
        call_count = 0                       # closure state, created ONCE
        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % interval == 0:
                return func(*args, **kwargs) # a result dict
            else:
                return False                 # a bool
```

* returns a bare `False` where callers expect the dict — `_check_success` guards
  with `isinstance(result, dict)`, so 49 of 50 calls silently read as "not
  successful" rather than "not evaluated"
* `call_count` never resets across episodes **or garment switches**, so which
  step of an episode gets evaluated drifts arbitrarily through a run

Repair unwraps the decorator and evaluates every call (`LEHOME_CHECK_INTERVAL`
can re-throttle, but returns the cached last dict rather than `False`, with
counter and cache reset per episode).

**Verified** on a demo replay: **315 checks per episode, up from 13.**

That run reported 0 successes, and it was a genuine near-miss rather than a
detection regression: J 7.15 -> **min 0.118** -> final 0.15, with 48 steps below
0.5. An earlier identical replay reached exactly 0.00 and succeeded.

**This is worth carrying:** perfect action reproduction lands within ~1.2 cm of
the threshold and still fails on physics variance alone. The ~27% replay rate is
not sloppiness in the demonstrations -- the task is marginal at these thresholds,
and any success rate measured here carries that noise floor underneath it.

---

## ROOT CAUSE (2026-08-11): our cameras do not render the garment's motion

Drove the eval with **recorded demonstration actions** (`scripts/replay_policy.py`)
so the rollout follows episode 0 exactly, and captured the observation video.
Same actions, same garment, same start pose as the recording — therefore any
difference is purely rendering.

| | pixels ever changing >25 | drift f0 -> end |
|---|---|---|
| **demo recording** | **53.85%** | 11.83 |
| our render, `rendering_mode=performance` | 6.51% | 3.13 |
| our render, `rendering_mode=quality` (their default) | 0.35% | 6.05 |

The recording shows over half the frame changing as the garment folds. Ours shows
6.5%, or 0.35% under the shipped default. Frame-to-frame the sim video is noise
around a nearly fixed scene: `|f_k - f_0|` reaches 3.15 by frame 28 and stays
flat for all 315 frames while consecutive frames differ by 2.63.

Meanwhile the physics is fine — that same trajectory reaches J = 0.118.

**So the simulation advances and the cameras do not follow it.** This is the
single defect underneath everything:

* image-conditioned policies are shown a near-static scene and cannot act
* demo replay succeeds because it is open-loop and never reads an image
* our SmolVLA and the 74.5% challenge winner fail identically, on our eval chain
  and on the winner's own

`performance` mode is ~19x better than `quality` here (6.51% vs 0.35%) and was a
real improvement, but it is an order of magnitude short of the 53.85% the
recording shows. Likely the same GB10/sm_121 RTX problem that prevents
`rtx/nrd/PackForNRD.cs.hlsl` from compiling.

**Next:** find why camera output lags the physics. Candidates, cheapest first —
`sim.render_interval` vs `decimation` binding (a known trap in this repo: setting
`decimation` does not update `render_interval`, which is bound at class definition
time); whether `env.step()` actually triggers a camera update on this build; and
whether `use_fabric=False` in `SimulationCfg` starves the render of updated
transforms. Until this is closed, no closed-loop number here is meaningful.

---

## CORRECTION (2026-08-11): it is not the renderer, and not the camera

Two measurements overturned the "render is broken" conclusion above. Both used
`scripts/replay_policy.py` to drive the eval with **recorded episode-0 actions**,
so the trajectory matches the recording exactly.

### Camera geometry is correct (`scripts/probe_camera_geometry.py`)

```
camera world pos (-0.015, 0.190, 1.060)   garment centroid (-0.008, 0.056, 0.530)
distance 0.547 m   HFOV 67.2 / VFOV 52.9  ->  visible extent 0.726 x 0.545 m
garment 0.519 x 0.346 m  ->  predicted ~45% of frame area
```

### Garment extent over time, Otsu-segmented, robot masked out

```
step%  |  DEMO area    w    h  |   SIM area    w    h
    0% |     20.5%   639  367  |     37.1%   639  401
   36% |     15.3%   598  378  |     38.5%   639  401
   72% |     11.3%   413  381  |     38.1%   639  401
  100% |      6.3%   503  388  |     38.5%   639  401
       |  swing 14.3 pp, 226px |  swing 1.6 pp, 0px
```

**The recorded garment folds** -- its footprint drops to a third. **Ours does not
move at all.** And our 3D bbox barely moves either (y-extent 0.346 -> 0.294 m), so
the image and the geometry agree: **the render is faithful to the physics.**

### The camera is not the discrepancy either

The winner's fork does not modify `garment_bi_cfg_v2.py` -- same camera config as
ours -- and their sim-round augmentation defaults are neutral
(`top_camera_pos_offset (0,0,0)`, `rot_offset (0,0,0)`, `focal_scale 1.0`). The
large documented offsets (`[0, 0.25, 0.15]`, `-29 deg`) belong to
`replay_real_in_sim.py`, which aligns the sim camera to their **real robot**.
They scored 74.5% on our camera configuration.

An apparent 1.9x area difference is explained by **fill density, not scale**:
demo 20.5% within a 639x367 bbox (27% fill) vs ours 37.1% within 639x401 (44%
fill). Same footprint, ours denser -- a bunched garment versus a flat-spread
shirt whose thin sleeves let the table show through.

### Where this leaves it

Eliminated: renderer, `rendering_mode`, `use_fabric`, camera placement/FOV,
garment scale, geometry writeback (`MESH == VIEW`), camera registration, policy,
eval chain, action units, success predicate.

Remaining: **the garment starts in a different configuration and recorded
actions do not fold it here.** A physics / initial-state question. Consistent
with the 27% replay rate, J stalling at 0.118, and a 74.5% policy no-opping --
it acts on cloth that does not respond as its training data did.

**Process note.** I reversed twice here (render broken -> camera too close -> ne
ither). Each reversal came from a threshold-based image metric quoted before it
was validated against a control. The measurements that held up were the ones
with a reference beside them: replayed actions vs recording, MESH vs VIEW, our
config vs the winner's.

---

## Two defects, separated (2026-08-11)

`scripts/shift_top_camera.py` (`LEHOME_CAM_BACK=<m>`) moves the top camera back
along its garment->camera axis, patched into
`scripts.utils.common.stabilize_garment_after_reset` -- the first moment both the
camera and a settled garment exist. Replayed recorded episode-0 actions with
+0.27 m (0.548 -> 0.818 m):

### 1. Framing — FIXED

```
garment area in frame     before 37.1%  (clipped top and both sides)
                          after  17.2%
                          DEMO   20.5%   (20.5 / 20.5 / 24.5 over three episodes)
```

The camera was ~0.27 m too close. Slightly overshot: +0.19 m lands on 20.5%.

### 2. Motion — NOT fixed

```
pixels ever changing >25
  vs frame 0 (includes the one-off camera jump)   50.61%   <- artefact
  vs frame 5 (post-shift only)                     2.23%   <- real
  DEMO                                            53.85%
```

Garment area holds at 17.1-17.2% for all 315 steps. **Recorded actions still do
not fold it.** Measuring against frame 0 would have reported a spurious 50.61%
and looked like a fix -- the control was to re-baseline after the shift.

### Unresolved tension

The challenge winner used the **unmodified** camera config and scored 74.5%; its
sim-round augmentation defaults are neutral. So either their training absorbed
the same framing gap (they jitter camera position, rotation and focal length), or
something in our setup differs from the standard configuration. Do not treat the
0.27 m offset as "the fix" until that is explained -- it corrects a measured
symptom, and the winner apparently did not need it.

### Next

Physics, not graphics: why does replaying the exact recorded joint trajectory
not reproduce the recorded fold? Note the task tolerates 7.2-9.9 cm on
check-point distances, so this is not a precision problem -- the cloth is barely
being displaced at all.

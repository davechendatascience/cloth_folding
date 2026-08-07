# Levers

Running list of knobs that materially change throughput, learnability, or
correctness on this project. **Check this before optimising anything** — most
entries here were rediscovered the expensive way at least once.

Status key: **measured** = we have numbers · **tried** = attempted, outcome noted ·
**untried** = plausible, unquantified · **blocked** = needs work elsewhere first

---

## The pipeline these levers act on

```
   camera  ──▶  spatial attention  ──▶  reasoning  ──▶  joint manipulation  ──▶  cloth
     │              │                      │                 │                     │
  is the        does it ground         does it map        do the commands      does J fall
  cloth         on the cloth?          state → plan?      move the cloth?      monotonically?
  visible?
```

Everything in this project is a link in that chain, and **a break anywhere makes
every downstream measurement meaningless**. So diagnose by localising the break
before tuning anything — three of our expensive detours were tuning a link that
was fine while a different one was severed.

| link | how to measure it | status |
|---|---|---|
| **camera** | is cloth state recoverable from the observation at all? | **OK-ish.** 3× RGB at 84×84. Depth (the direct topology signal) is *not* in the merged dataset. |
| **spatial attention** | attention entropy vs uniform | **OK.** 2.753 vs 4.796 uniform, peak weight 0.25 vs 0.008. It attends somewhere structured. |
| **attention → reasoning** | perturb images vs proprio, compare \|Δaction\| | **SEVERED.** image/proprio influence = **0.0445** on `runs/bc_top_long/best.pt`, measured by `scripts/measure_attribution.py`. The policy ignores what the attention computed. |
| **reasoning → joints** | closed-loop Cartesian accuracy | **OK.** 8/8 moves to 0.0000 m residual via DLS differential IK. |
| **joints → cloth** | does replay reach J = 0? | **OK.** 2/3 episodes to J = 0.000; ~90% mean reduction. |
| **cloth → J monotone** | `mono_violation_rate` near goal | **untested** — never reached the near-goal band. |

**The demonstrations are ~69% a fixed script (2026-08-07). This dominates
everything below.** Measured offline on the cache, no training required:

| predictor | sees | held-out R² |
|---|---|---|
| phase only — mean trajectory vs normalised time | **nothing** | **0.688** |
| ridge on 5 frames of proprio, 1-step | proprio | 0.862 |
| ridge on 5 frames of proprio, 80-step chunk | proprio | 0.725 |

Cross-episode trajectory std 0.231 vs within-episode 0.469 (ratio 0.49) — every
episode traces close to the same arm motion.

A predictor knowing *only how far through the episode it is* explains 69% of
held-out action variance. Proprioception adds ~0.04 on top. **No loss function
can create a need for vision in data where vision is not needed** — which is
why the delta target, the ΔJ weighting and the `Ĵ` head all failed to move
attribution, and why neither BC run beat persistence.

It also matches the replay result independently: 33/85 episodes reach J=0
(**39%**), which is what a largely open-loop script scores — it works when the
garment starts favourably and fails otherwise.

**Consequences:**

- **Action chunking will not fix attribution.** The proprio shortcut does not
  die at any chunkable horizon (R² 0.725 at k=80, ~2.7 s). This was proposed as
  the principled fix on the theory that the shortcut was *temporal*; it is
  *structural*. Cost of finding out: two minutes offline, versus a training run.
- **Residual BC is the loss-side fix.** Subtract the phase-conditioned mean
  trajectory and train on the residual — the same move as the delta target, one
  level up. The delta target killed the static proprio shortcut; this kills the
  template shortcut. The residual 31% is where garment-specific adaptation lives
  and the only part vision can explain.
- **BC has a low ceiling here by construction** (~"reproduce the mean
  trajectory"). Visual adaptation has to be *discovered*, not imitated, which is
  a much stronger argument for the RL stage than throughput ever was.

---

**The whole failure is one link.** Camera, attention, joint control and physics
all check out; the policy simply doesn't *use* the visual features. That is why
the fix is a loss change (delta target + `Ĵ` auxiliary) rather than a bigger
encoder or better IK — capacity and control were never the constraint.

*(Superseded in part by the stereotypy finding above: the link is severed, but
the cause is upstream of the loss — the data does not require vision.)*

**The "0.11" figure was never reproducible (2026-08-07).** It came from an
ad-hoc check whose perturbation scheme was not recorded. `measure_attribution.py`
scores the *same checkpoint* at **0.0445**, because the ratio depends entirely on
how the two modalities are perturbed — this script scales each by its own std,
so the comparison is scale-free; an absolute-epsilon version would mostly report
that images and joint angles live on different scales.

Consequence: the pre-registered pass criterion "attribution ≫ 0.11" was not
checkable as written. **Use 0.0445 as the baseline and compare with the same
script**, via `--baseline`. A diagnostic number is only a threshold if the code
that produced it still exists.

Levers by link:

- **camera** — resolution, camera count, depth (needs `dataset_challenge`, not the
  merged set), pretrained encoder
- **spatial attention** — the `Ĵ(o)` auxiliary head is the direct supervision:
  predicting J *requires* representing check-point geometry
- **attention → reasoning** — delta target (kills the proprio bypass), action
  chunking (kills the recurrent bypass), multimodal head (stops mode-averaging
  collapsing motion to zero)
- **reasoning → joints** — action space (joint vs Cartesian), DLS λ, plant damping
- **joints → cloth** — `joint_damping` (10× on replay fidelity), decimation,
  solver iterations
- **cloth → J** — λ_v / λ_Δa annealing, `r_mono` gating, ΔJ weighting

---

## Throughput

| lever | status | effect | notes |
|---|---|---|---|
| **`num_envs` > 1** | **works (2026-08-07)** | throughput scaling unmeasured | Verified at N=4: `(4, 14544, 3)`, all finite, centroid spacing 2.9915 m vs `env_spacing=3.0`. Needed four things: robot/camera paths under `/World/envs/env_.*/`, the garment created in `env_0` so cloning replicates it, a `ClothPrim` **view** aimed at `Garment/mesh`, and **`replicate_physics=False`** (the actual blocker). See [Parallel envs](#parallel-envs). **Not yet usable for training** — per-env garment pose reset is missing, see below. |
| GPU vs CPU sim | **measured** | **2.1×** (82 → 39 s/episode) | Requires `PYTORCH_JIT=0` on GB10. Only 2× because the workload is 1 env × 14.7k particles — too small and too serial for a GPU. |
| CPU sharding (no GPU job) | **measured** | ~4× at 4 shards | 6 shards thrashed the box (load 95, 109/121 GB). 4 is the safe ceiling; each Isaac instance is ~15 GB under load, not the 7 GB measured at startup. |
| CPU shards **alongside** a GPU job | **measured → NO GAIN** | 1.49 vs 1.50 ep/min | Exactly a wash. The GPU labeller is not GPU-bound: it needs a CPU core to drive the Python step loop and marshal particle reads, so 3 CPU shards starve it by the same amount they contribute (GPU 1.50 → 0.51 ep/min; shards add 0.98). **A high idle-core count is not spare capacity if one of those cores is load-bearing for the accelerator job.** Reverted to GPU alone. |
| skip rendering when images unused | **measured, works** | **~1.35×** on the labeller | Two halves, both required: `AppLauncher(enable_cameras=False)` **and** `skip_images=True` so `_get_observations` stops reading camera buffers. Disabling the renderer alone crashes — illegal GPU access in a sensor buffer read, which poisons the CUDA context and then reports itself as a `GpuParticleClothView` failure, pointing at the cloth instead of the cameras. Verified label-identical: cam-vs-no-cam ΔJ noise (SNR 3.65) is *lower* than cam-vs-cam run-to-run noise (SNR 2.69). |
| ~~`render_interval` very high~~ | **tried → HANGS** | none | `render_interval=100000` hangs the env (3 min, 0 episodes, GPU 0%, stray `zenity`). Sensors still exist and Isaac Lab's step path waits on renders that never arrive. Superseded by the row above, which removes the work instead of deferring it. |
| avoid per-step GPU→host sync | **untried** | unknown | `compute_cloth_error()` copies particle positions to host every step, stalling the pipeline. Could batch or keep on device. |
| `decimation` | **measured** | linear in physics cost | 3 = 30 Hz (matches demo fps, required for BC). 20 = 4.5 Hz (one action per plant settling time, for from-scratch RL). |
| `stiffness_scale` | **untried** | settling ∝ 1/√K | 4× stiffness halves settling → allows ~half the decimation. Costs contact realism: a stiffer arm pushes harder on the cloth. |
| solver iteration count | **untried** | ~linear | `solver_position_iteration_count=16`. Iterations are serial, so this is the main serial-depth knob. Changes physics fidelity. |
| BC dataloader | **measured** | GPU at 43% | Input-bound: 4 workers at ~64% CPU each. More workers, or ship uint8 and convert on GPU (4× less to transfer). |
| image resolution / camera count | **untried** | — | 84×84×3 cameras now. Depth would need the non-merged dataset. |

## Learnability

| lever | status | effect | notes |
|---|---|---|---|
| **delta target `a[t]−s[t]`** | **measured** | necessary, insufficient | Absolute targets let proprio shortcut the loss. Kills the *static* shortcut only; the template shortcut survives (see stereotypy above). |
| ΔJ weighting `exp(−ΔJ/β)` | **measured → no effect on attribution** | — | β must come from the data: dJ std is 0.0537, so β=0.5 gives weight p95/p5 = 1.35 (**unweighted BC wearing a Lyapunov label**) and β=0.1 gives 4.53. Even correctly tuned it reweights *which transitions* to imitate and cannot change *which input channel* the actor reads. |
| auxiliary `Ĵ(o)` head | **measured → insufficient** | — | Pressures the **encoder** (reads `z`, visual context only) but nothing penalises the **actor** for weighting proprio ≫ z. Right instinct, wrong half of the break: the failure is at *fusion*. J is also only a scalar summary — the check-point positions are the full-rank version of the same signal. |
| **vision → deformable config (factored)** | **untried, best candidate** | — | Ask vision for the *cloth configuration*, not the action: `vision → check-point positions → task-space target → DLS IK`. Sidesteps stereotypy entirely, because the template lives in the *execution* and the variation lives in the *plan*. Gives **direct supervision** instead of a gradient filtered through action imitation. `check_point_positions_cm()` already reads them cheaply, and `compute_cloth_error()` is literally `functional(check_point_positions_cm())` — so labelling already computes and discards them. |
| pretrained visual encoder | **untried** | likely large | 0.63M conv from scratch on 83k frames. DINOv2/DINOv3/SigLIP all cached locally. |
| action chunking | **untried** | likely large | Closes the *recurrent* shortcut the delta target leaves open (a GRU can echo its own previous delta; persistence-of-delta still scores 0.00219). Also the ACT default. |
| multimodal head (diffusion / MoG) | **untried** | unknown | A unimodal Gaussian averages demonstration modes; averaging "reach left" and "reach right" gives "don't move", which is what we measured (ee_speed 0.07). Diffusion breaks PPO finetuning (no log-prob); a mixture keeps it. |
| more garment types | **untried** | 83k → 265k frames | The other three types are downloaded. If val loss is flat and attribution is fine, this is the lever. |

## Damping (per [[damping-as-search-allocation]])

| lever | status | effect | notes |
|---|---|---|---|
| joint damping ζ | **measured** | 10× on demo replay | Demos were recorded on LeHome's under-damped plant (ζ≈0.53–0.65). Critical damping degrades replay: J_end 2.200 vs 0.211. **Use `joint_damping={}` for anything touching demonstrations.** |
| λ_v, λ_Δa (reward) | **measured** | creates a freeze basin | Both zero when stationary, so "do nothing" beats exploring (−538 vs −664). Should be annealed with J, not applied globally — the theorem only needs *eventual* monotonicity. |
| KL trust region / prior anchor | **implemented** | — | Keeps finetuning from leaving the BC basin. |
| DLS λ (IK) | **implemented** | — | Damping at the kinematic layer; keeps the solution bounded near singularities. |

## BC runs so far — neither beat a predictor that ignores the observation

| run | train eps | loss | best `val_mse/persistence` |
|---|---|---|---|
| `bc_top_long` | 225 | delta target, unweighted | **1.10** |
| `bc_j` | 84 | delta + β=0.1 + λ_j=0.2 | **1.176** (converged by epoch 13) |

Below 1.0 means the policy learned more than temporal autocorrelation. **Neither
run got there.** `bc_j` converged hard: val flat at ~0.00235 from epoch 13 while
train fell to 0.00079 — a 3× gap, overfitting 84 episodes.

The two are **confounded by training-set size** (84 vs 225 episodes, because
`--beta/--lambda_j` restrict to J-labelled episodes as a side effect). `--require_labels`
exists to run the unweighted control on the identical 84 episodes; without it,
"J weighting made things worse" is not a supportable claim.

Read alongside the stereotypy finding, though, the more likely reading is that
**no loss change can succeed here** — the data does not require vision, so
attribution cannot be fixed from the loss side.

## Correctness gotchas that look like performance problems

- **`PYTORCH_JIT=0`** required for GPU sim on GB10. Isaac Lab's `math_utils.normalize` is TorchScript; nvrtc in CUDA 12.8 rejects sm_121. Precompiled kernels (matmul/conv) work fine, so a naive GPU smoke test passes while Isaac fails.
- **Setting `cfg.decimation` does NOT update `cfg.sim.render_interval`.** `GarmentEnvCfg` binds `render_interval=decimation` at class-definition time, where the class default is `decimation = 1`. Assigning `cfg.decimation = 3` afterwards leaves `render_interval = 1`, so the env renders **3× per policy step**. The backend sets it explicitly; ad-hoc scripts usually don't. This silently inflated the entire first throughput sweep — a measured "3.44× from disabling rendering" was really ~1.35× against what production does. **Any timing measured with a hand-built cfg must have `sim.render_interval` set explicitly, or it is not measuring the production path.**
- **Kit never exits cleanly.** It ignores SIGTERM and `app.close()` can itself hang (measured once at 26 min, 154% CPU). Every Isaac script must arm a `threading.Timer` → `os._exit` **before** calling `close()`. All active scripts do; `launch_probe.py` does not (unused). A "hanging" Kit is usually a *finished* job stuck on exit, so check episode progress and file mtimes before assuming the run itself is stuck.
- **`merge_J_shards.py` is destructive.** It globs only `J_shard*.npy`, excludes `J.npy` from its inputs, and writes to `J.npy` — so it overwrites the main labels with shard data. It also races a running labeller. Do not run it as-is.
- **Garment pose must be set per episode** when replaying demos. Random reset → 0.7% J reduction; matched pose → 90%. Looks exactly like env infidelity.
- **`joint_damping={}`** for demo-related work (above).
- **Proprio normalisation** must carry from BC to finetuning, or the loaded weights see inputs at a different scale.
- **`weights_only=False`** when loading our checkpoints under torch ≥2.6.
- **Baselines must match the target.** With delta targets, "do nothing" is the zero vector, not `s[t]`.

---

## Parallel envs

The blocked lever above, written out because it is the one worth doing.

**Why it's blocked.** `garment_bi_cfg_v2` places everything at absolute paths:
`/World/Robot/{Left,Right}_Robot`, `/World/Object/{garment}`, `/World/Scene`,
`/World/Light`. Isaac Lab's cloner only replicates prims under
`/World/envs/env_.*/`, so `num_envs=8` yields 8 empty env origins sharing one
garment and one pair of arms. `replicate_physics=True` has nothing to act on.
Separately, `SingleClothPrim` / `SingleParticleSystem` wrap one prim and
`get_object_particle_position()` reads one cloth, so per-env J is not even
expressible.

**What it would take** — scoped 2026-08-07, and it is more tractable than the
first read suggested:

1. Move robot + camera prim paths to `/World/envs/env_.*/Robot/...` in
   `garment_bi_cfg_v2`, so Isaac Lab's cloner replicates them.
2. **Create one `GarmentObject` per env in a loop.** `GarmentObject.__init__`
   already takes an arbitrary `prim_path` — LeHome merely hardcodes
   `/World/Object/{name}` at the call site. So this needs no new Isaac API,
   just `/World/envs/env_{i}/Garment`.
3. **Keep the `Single*` wrappers, hold a list of N of them**, and stack their
   particle reads into `(num_envs, P, 3)`. A true batched cloth view would be
   better, but N host-side reads per step is functionally correct and adequate
   at modest N — and it avoids depending on APIs that may not exist.
4. `GarmentFoldFunctional` is already batched over envs, so J is free.
5. Keep the bedroom scene and light global (static geometry, no need to
   replicate).

**Measured constraint (2026-08-07): cloths must be created before `sim.reset()`.**
Adding a `GarmentObject` to a *running* sim fails with

```
ClothPrim.initialize() → self._count = self._physics_view.count
AttributeError: 'NoneType' object has no attribute 'count'
```

`_physics_view` is None because the physics tensor view is built once at
`sim.reset()` and a later-created cloth is never registered with it. This is a
**lifecycle constraint, not a particle-system limit** — so the plan is "create N
garments inside `_setup_scene()`", which is exactly where LeHome already creates
its one. It does *not* block parallel envs.

Getting here took three wrong guesses (bad prim path → particle systems can't
replicate → fundamentally impossible) across two full env builds, because the
probe swallowed the traceback into a one-line message. **Get the traceback
before forming a hypothesis** — the answer was in the stack the whole time.

**`replicate_physics` must be False (2026-08-07).** This is the actual blocker,
found after four failed builds. With `replicate_physics=True`, PhysX replicates
env_0's physics structure instead of parsing each env's prims — which covers
articulations and rigid bodies but *not* particle cloths. The signature is a
stage where everything looks right and the physics disagrees:

```
/World/envs/env_{0..3}/Garment/mesh   EXISTS       (all four, IsInstanceable=False)
cloth_view.count                       1           (PhysX parsed one cloth)
particle_positions()                   (1, 14544, 3)
```

**Diagnosing this took five attempts, four of which were guesses made without
looking at the stage.** In order: nested prim path → garment outside the env
namespace → view aimed at the parent instead of `Garment/mesh` → `copy_from_source`
inheritance (a good story, and wrong — the docstring says clones "mirror
env_0's changes", which would explain a shared cloth perfectly, but a stage dump
showed the prims were real and independent). One `is_prim_path_valid` loop plus
`GetChildren()` would have separated "not cloned" from "cloned but not parsed"
at attempt one, and every subsequent hypothesis was unnecessary.

This is the **same lesson already recorded two sections above** for the
multi-cloth work. Both times the cheap observation was available from the start
and skipped in favour of a plausible mechanism. **When a symptom admits two
categories of cause, measure which category first — do not pick one and start
fixing.**

**The static scene is global, so envs are not equivalent (2026-08-07).** This is
the current blocker, and it was nearly missed. `garment_bi_v2._setup_scene`
spawns the bedroom once:

```python
cfg.func("/World/Scene", cfg, translation=(0.0, 0.0, 0.0), ...)
```

`/World/Scene` sits outside `/World/envs/env_.*/`, so the cloner never
replicates it. env_0's garment rests on the bedroom furniture; every other env
is 3 m away with nothing beneath it. Measured settled centroid z after per-env
posing: **env_0 = 0.5349, envs 1-3 = 0.2136 / 0.2121 / 0.2050.**

Keeping the scene global was a deliberate choice in `build_parallel_cfg`
("static shared geometry, replicating it would multiply cost for no benefit").
That reasoning is right for rendering and wrong for physics: the garment *rests
on* that geometry.

**`test_per_env_pose.py` passed while this was true**, reporting "parallel envs
are usable". It asked whether poses were *independent* and never whether envs
were *interchangeable* — so N independent, correctly-posed, physically-unequal
envs sailed through. The equivalence check (all envs must settle to the same
height, spread < 0.05 m) now exists precisely because the earlier criteria
could not fail on this.

**Generalises past this bug:** for parallel envs, `distinct` and `equivalent`
are two separate properties and both need asserting. The distinctness test was
written to catch silent aliasing and did; it had nothing to say about silent
divergence.

**Options**, unresolved: spawn the bedroom under `/World/envs/env_0/Scene` so it
clones (N copies of full bedroom geometry — expensive, and the env origins are a
centred grid so the garment poses shift); or spawn only the support surface per
env; or accept N=1 for demo-matched work and use parallel envs only where the
furniture is irrelevant. Note `filter_collisions(global_prim_paths=["/World/Scene"])`
assumes the global scene and must change with it.

---

**Per-env garment pose — solved (2026-08-07).** `set_garment_poses()` mirrors
LeHome's identity → initial-points → target-pose sequence per env, using the
batched view for particles and a `SingleXFormPrim` per env for the pose.
Verified at N=4: inter-env xy offset error 0.0128 m; re-posing env 1 alone moved
it 0.7353 m while envs 0/2/3 moved ≤ 0.0218 m.

The problem it fixed, measured at N=4 before the method existed:

| env | z |
|---|---|
| 0 | **0.5292** |
| 1 | 0.2029 |
| 2 | 0.1970 |
| 3 | 0.2018 |

Envs 1–3 agree to ~0.006 m — identical initial conditions, differing only by
float nondeterminism, exactly as expected. **env_0 sits 0.33 m higher**, because
LeHome's reset logic acts on `self.object`, the single `SingleClothPrim` that
wraps env_0 alone. The clones are never posed.

This is the parallel-env form of the bug that already cost us once: garment pose
must be set per episode, or demo replay drops from 90% J reduction to 0.7%
(§ "Correctness gotchas"). Until per-env pose reset exists, `num_envs > 1` gives
N independent cloths that all start from the *wrong* configuration, which is
worse than useless for labelling and for RL resets. **Do not train on this yet.**

**Remaining unknowns.** Whether `replicate_physics=False` costs enough start-up
time or memory at large N to cap `num_envs` below the useful range; how
throughput actually scales with N (the 10–60× estimate is still unmeasured);
and whether O(N) host-side particle reads per step become the new bottleneck.

**Why it is worth it.** It is the only lever that fixes the GPU utilisation
problem at its root (62% compute / 0% memory bandwidth = not enough parallel
work), and it is simultaneously the thing that makes on-policy RL feasible:
1.4 policy steps/s × 64 envs changes an 83-day budget into something closer to
a day.

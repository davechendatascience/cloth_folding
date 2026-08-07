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
| **attention → reasoning** | perturb images vs proprio, compare \|Δaction\| | **SEVERED.** image/proprio influence = **0.11**. The policy ignores what the attention computed. |
| **reasoning → joints** | closed-loop Cartesian accuracy | **OK.** 8/8 moves to 0.0000 m residual via DLS differential IK. |
| **joints → cloth** | does replay reach J = 0? | **OK.** 2/3 episodes to J = 0.000; ~90% mean reduction. |
| **cloth → J monotone** | `mono_violation_rate` near goal | **untested** — never reached the near-goal band. |

**The whole failure is one link.** Camera, attention, joint control and physics
all check out; the policy simply doesn't *use* the visual features. That is why
the fix is a loss change (delta target + `Ĵ` auxiliary) rather than a bigger
encoder or better IK — capacity and control were never the constraint.

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
| **`num_envs` > 1** | **blocked** | potentially 10–60× | The single biggest lever and the reason the GPU is idle. LeHome authors every prim at an absolute path (`/World/Robot/...`, `/World/Object/...`), so Isaac Lab's cloner has nothing to replicate, and `SingleClothPrim`/`SingleParticleSystem` wrap exactly one prim. Needs scene re-authoring under `/World/envs/env_.*/` + a batched cloth view. See [Parallel envs](#parallel-envs). |
| GPU vs CPU sim | **measured** | **2.1×** (82 → 39 s/episode) | Requires `PYTORCH_JIT=0` on GB10. Only 2× because the workload is 1 env × 14.7k particles — too small and too serial for a GPU. |
| CPU sharding (no GPU job) | **measured** | ~4× at 4 shards | 6 shards thrashed the box (load 95, 109/121 GB). 4 is the safe ceiling; each Isaac instance is ~15 GB under load, not the 7 GB measured at startup. |
| CPU shards **alongside** a GPU job | **measured → NO GAIN** | 1.49 vs 1.50 ep/min | Exactly a wash. The GPU labeller is not GPU-bound: it needs a CPU core to drive the Python step loop and marshal particle reads, so 3 CPU shards starve it by the same amount they contribute (GPU 1.50 → 0.51 ep/min; shards add 0.98). **A high idle-core count is not spare capacity if one of those cores is load-bearing for the accelerator job.** Reverted to GPU alone. |
| skip rendering when images unused | **tried → HANGS** | none; reverted | `render_interval=100000` hangs the env: 3 min, 0 episodes, GPU 0%, plus a stray `zenity` dialog. The `TiledCamera` sensors exist regardless of the interval and Isaac Lab's step path waits on renders that never arrive. Moderate values (10–30) untested and might capture most of the benefit. Reverted to rendering once per policy step (39 s/episode, GPU 62–70%). |
| avoid per-step GPU→host sync | **untried** | unknown | `compute_cloth_error()` copies particle positions to host every step, stalling the pipeline. Could batch or keep on device. |
| `decimation` | **measured** | linear in physics cost | 3 = 30 Hz (matches demo fps, required for BC). 20 = 4.5 Hz (one action per plant settling time, for from-scratch RL). |
| `stiffness_scale` | **untried** | settling ∝ 1/√K | 4× stiffness halves settling → allows ~half the decimation. Costs contact realism: a stiffer arm pushes harder on the cloth. |
| solver iteration count | **untried** | ~linear | `solver_position_iteration_count=16`. Iterations are serial, so this is the main serial-depth knob. Changes physics fidelity. |
| BC dataloader | **measured** | GPU at 43% | Input-bound: 4 workers at ~64% CPU each. More workers, or ship uint8 and convert on GPU (4× less to transfer). |
| image resolution / camera count | **untried** | — | 84×84×3 cameras now. Depth would need the non-merged dataset. |

## Learnability

| lever | status | effect | notes |
|---|---|---|---|
| **delta target `a[t]−s[t]`** | **measured** | decisive | Absolute targets let proprio shortcut the loss (linear fit leaves <5% unexplained vs >50% for deltas). This is why the first BC ignored the cameras (attribution 0.11). |
| ΔJ weighting `exp(−ΔJ/β)` | **implemented** | untested | AWR. Imitate the descending segments; discount the demonstrator's wobbles. Needs J labels. |
| auxiliary `Ĵ(o)` head | **implemented** | untested | Forces camera → deformable topology. Fed visual context only, or it shortcuts. Needs J labels. |
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

## Correctness gotchas that look like performance problems

- **`PYTORCH_JIT=0`** required for GPU sim on GB10. Isaac Lab's `math_utils.normalize` is TorchScript; nvrtc in CUDA 12.8 rejects sm_121. Precompiled kernels (matmul/conv) work fine, so a naive GPU smoke test passes while Isaac fails.
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

**Remaining unknowns.** Whether one PhysX particle *system* can host N cloths or
each env needs its own; and whether O(N) host-side particle reads per step
become the new bottleneck, which would then justify a proper batched view.

**Why it is worth it.** It is the only lever that fixes the GPU utilisation
problem at its root (62% compute / 0% memory bandwidth = not enough parallel
work), and it is simultaneously the thing that makes on-policy RL feasible:
1.4 policy steps/s × 64 envs changes an 83-day budget into something closer to
a day.

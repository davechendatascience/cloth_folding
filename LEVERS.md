# Levers

Running list of knobs that materially change throughput, learnability, or
correctness on this project. **Check this before optimising anything** — most
entries here were rediscovered the expensive way at least once.

Status key: **measured** = we have numbers · **tried** = attempted, outcome noted ·
**untried** = plausible, unquantified · **blocked** = needs work elsewhere first

---

## Throughput

| lever | status | effect | notes |
|---|---|---|---|
| **`num_envs` > 1** | **blocked** | potentially 10–60× | The single biggest lever and the reason the GPU is idle. LeHome authors every prim at an absolute path (`/World/Robot/...`, `/World/Object/...`), so Isaac Lab's cloner has nothing to replicate, and `SingleClothPrim`/`SingleParticleSystem` wrap exactly one prim. Needs scene re-authoring under `/World/envs/env_.*/` + a batched cloth view. See [Parallel envs](#parallel-envs). |
| GPU vs CPU sim | **measured** | **2.1×** (82 → 39 s/episode) | Requires `PYTORCH_JIT=0` on GB10. Only 2× because the workload is 1 env × 14.7k particles — too small and too serial for a GPU. |
| CPU sharding | **measured** | ~4× at 4 shards | 6 shards thrashed the box (load 95, 109/121 GB). 4 is the safe ceiling; each Isaac instance is ~15 GB under load, not the 7 GB measured at startup. |
| skip rendering when images unused | **untried** | unknown, likely large | Labelling renders three 480×640 cameras every step and reads none. GPU shows 62% compute / **0% memory bandwidth** — launch-overhead bound, so removing whole render passes should help more than physics tuning. `IsaacGarmentCfg.render_interval`. |
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

**What it would take.**
1. Move robot + camera prim paths under `/World/envs/env_.*/`.
2. Create the garment per env (or clone it) rather than once at `/World/Object/`.
3. Replace the `Single*` particle wrappers with a batched cloth view so particle
   positions come back as `(num_envs, N, 3)`.
4. Vectorise `GarmentFoldFunctional` over the env dimension — already written
   batched, so this part is free.
5. Keep the bedroom scene global (it is static) to avoid replicating geometry.

**Why it is worth it.** It is the only lever that fixes the GPU utilisation
problem at its root (62% compute / 0% memory bandwidth = not enough parallel
work), and it is simultaneously the thing that makes on-policy RL feasible:
1.4 policy steps/s × 64 envs changes an 83-day budget into something closer to
a day.

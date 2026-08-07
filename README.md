# Real-Analysis-Guided Damped Visual RL for Cloth Folding

Implementation of `real_analysis_damped_cloth_folding_rl_spec.md` against the
LeHome Challenge 2026 garment-folding task, on DGX Spark (GB10).

The design treats cloth + robot + policy + RL updates as one dynamical system,
and uses a Lyapunov-like functional `J` plus damping at every level to force
non-oscillatory convergence to a folded configuration.

---

## Status

| Component | State |
|---|---|
| Cloth error functional `J` (mock) | done, 21 tests |
| **Garment functional `J` (real LeHome metric)** | done, 16 tests |
| Damped impedance controller | done, 20 tests |
| Lyapunov descent reward | done, 22 tests |
| Vision + attention policy | done, 21 tests |
| Damped PPO + runner | done, 21 tests |
| Task env + mock backend | done, 24 tests |
| Isaac/LeHome backend adapter | done, **closed-loop verified 8/8 to 0.0000 m** |
| Run contracts + watchdogs | done, 25 tests |
| BC pipeline (preprocess/dataset/train) | done, **two runs completed — both failed, see below** |
| Parallel envs (`num_envs > 1`) | works at N=4; **not training-ready** (static scene is global, envs are not physically equivalent) |
| Damped-RL finetuning entrypoint | written, not yet run |

**198/198 tests pass.** Isaac Sim launches headless on aarch64, the real garment
env builds and steps, `J` computes on real particle data, and **replayed
demonstrations reach `J = 0`** (see below).

**Behaviour cloning has now been run and evaluated closed-loop. It does not
work**, and the reason is a property of the demonstrations rather than of the
loss — see [Why behaviour cloning fails here](#why-behaviour-cloning-fails-here).
Do not read the mechanisms below as achieving their goals; several are measured
to be insufficient, and the measurements are recorded alongside them.

## Reachability: the real task is solvable here

`scripts/check_reachability.py` replays demonstrated joint actions through
`IsaacGarmentBackend` at 30 Hz with our critical damping and our J:

| ep | J | reduction | success |
|---|---|---|---|
| 0 | 7.086 → 2.200 | 69% | ✗ |
| 1 | 7.004 → **0.000** | 100% | ✓ |
| 2 | 7.486 → **0.000** | 100% | ✓ |

Random-pose control: **0.7%**. So the env reproduces the demonstrations, `J = 0`
is attainable, and critical damping does not prevent folding. This is exactly
what the mock failed (oracle best 1.51 against a 0.02 threshold).

**At scale the success rate is 27%, not the 2/3 this three-episode sample
suggests.** The figure fell every time the sample grew:

| episodes labelled | replay reaches `J = 0` |
|---|---|
| 3 | 67% |
| 85 | 39% |
| **150** | **27%** (41/150, median best `J` 0.656) |

Reachability is still demonstrated — 41 episodes genuinely fold, and `J = 0` is
attainable with critical damping — but "the demonstrations succeed" is a much
weaker claim than the three-episode table implied. It is consistent with a
largely open-loop script that works when the garment happens to start
favourably, which is exactly what the stereotypy measurement below shows.

**Treat the three-episode row above as an illustration, not a result.**

Two protocol requirements, both load-bearing: `meta/garment_info.json` gives a
per-episode `object_initial_pose`, and `reset()` randomises placement — replay
without `set_garment_pose()` yields 0.7% and looks precisely like env
infidelity. Merged datasets concatenate per-garment sets, so global episode `e`
maps to garment `e // 25`, local `e % 25`.

---

## Quick start

```bash
# project venv (CUDA 13, for the pure-torch package + tests)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  torch==2.13.0+cu130 torchvision==0.28.0+cu130
uv pip install --python .venv/bin/python -e . --no-deps pytest

.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m lehome.real_damped_project.train.train_ppo_real_damped \
  --mock --num_envs 256 --num_steps_per_env 32 --max_iterations 400 --device cuda
```

Anything touching Isaac Sim uses **lehome-challenge's venv** instead, and must
set `LD_PRELOAD` + `OMNI_KIT_ACCEPT_EULA` (see [Isaac Sim operational
notes](#isaac-sim-operational-notes)).

---

## Two functionals

`J` must be continuous, non-negative, and zero exactly on the goal set. There
are two implementations:

* **`math/cloth_functional.py`** — soft-IoU + edge gap + wrinkle, for the mock
  backend. The wrinkle term is `|R(x) − R(x_target)|`, *relative* to the target:
  a folded garment is not flat, so penalising absolute roughness would put `J`'s
  zero at an unfolded sheet.
* **`math/garment_functional.py`** — **the real one.** LeHome scores folding
  with a boolean over 4–5 pairwise check-point distances. This sums the *margin
  violations* of those same conditions, so `J = 0` ⟺ `success_checker_garment_fold`
  returns True. The goal set is *identical* to the official one, not merely
  correlated with it. Verified against a transcription of their checker over 300
  random configurations.

Live reading on `Top_Long_Seen_0` at reset (real particle data, 14746 particles):

```
thresholds (cm): [7.2, 8.55, 7.2, 9.9, 9.0]
  cond1 d(0,4)=25.86 ≤ 7.20  → violation 18.66
  cond2 d(2,3)=44.85 ≤ 8.55  → violation 36.30
  cond3 d(1,5)=27.25 ≤ 7.20  → violation 20.05
  cond4 d(0,1)=17.34 ≥ 9.90  → violation  0.00
  cond5 d(4,5)=13.71 ≥ 9.00  → violation  0.00
J = 7.50
```

Note conditions 4–5 start satisfied — they only break under crumpling. All the
signal is in the three "bring together" conditions, which argues for weighting
rather than treating all conditions equally (`GarmentFunctionalCfg.weights`).

---

## Measured joint damping

`scripts/measure_joint_damping.py`, step response at q=0, K=17.8, D=0.60:

| joint | overshoot | ζ | ω_n | J (kg·m²) | D_crit | settle |
|---|---|---|---|---|---|---|
| shoulder_pan | 11.5% | 0.567 | 33.23 | 0.0161 | 1.071 | 15 |
| shoulder_lift | 14.1% | 0.529 | 27.76 | 0.0231 | 1.283 | 27 |
| elbow_flex | 6.8% | 0.649 | 32.80 | 0.0165 | 1.085 | 17 |
| wrist_flex | 0.03% | 0.933 | 56.24 | 0.0056 | 0.633 | 11 |
| wrist_roll | 0.00% | ≥1 | — | — | — | 12 |

**mean ζ = 0.670 — under-damped**, contradicting the spec's ζ ≥ 1.

The defect is *structural*: gains are uniform but inertia varies 4×, and
ζ ∝ 1/√J, so proximal joints ring while distal ones are fine. A single global
`D` cannot fix it — `D=1.283` would drive `wrist_flex` to ζ=2.03. Per-joint
`D_j = 2K/ω_n,j` is in `MEASURED_CRITICAL_DAMPING`.

This matters because **plant overshoot makes `J(x_t)` non-monotone regardless of
the policy**, undermining the spec's convergence argument below the level the
reward can reach.

`decimation=1` (90 Hz) means the policy issues up to **27 commands before the
plant responds**. Adapter default is 20 (≈4.5 Hz); `stiffness_scale` buys speed
back since settling ∝ 1/√K.

*Caveat: J is configuration-dependent, measured at q=0 with other joints held,
single-DOF fit ignores coupling. Re-measure at a folding pose before relying on
these quantitatively.*

---

## Findings that changed the design

### Why behaviour cloning fails here

Two BC runs, both evaluated closed-loop in the real env with matched per-episode
garment poses on the demonstration plant:

| run | train eps | loss | `val_mse/persistence` | closed-loop |
|---|---|---|---|---|
| `bc_top_long` | 225 | delta target | 1.10 | — |
| `bc_j` | 84 | delta + `β=0.1` + `λ_j=0.2` | 1.176 | **0/3, indistinguishable from frozen** |

```
bc      J_end=7.647  J_min=7.168  success=0/3
frozen  J_end=7.602  J_min=7.118  success=0/1
random  J_end=7.716  J_min=7.122  success=0/1
```

Neither run beat a predictor that ignores the observation entirely. Note the
baselines are what make this readable: BC alone shows `J` rising 7.11 → 7.70,
which looks like the policy dragging the cloth, but frozen rises 7.12 → 7.60
with *zero* arm motion. Most of that is the cloth settling. BC is inert, not
destructive.

**The cause is the data, not the loss.** Measured offline on the cache:

| predictor | sees | held-out R² |
|---|---|---|
| phase only — mean trajectory vs normalised time | **nothing** | **0.688** |
| ridge on 5 frames of proprio, 1-step | proprio | 0.862 |
| ridge on 5 frames of proprio, 80-step chunk | proprio | 0.725 |

A predictor knowing only how far through the episode it is explains 69% of
held-out action variance. **No loss function can create a need for vision in
data where vision is not needed**, which is why the delta target, the ΔJ
weighting and the `Ĵ` head all left image attribution at ~0.11.

This also refutes action chunking as a fix: proprio still predicts at R² 0.725
across 80 steps, so there is no chunkable horizon where the shortcut dies.

**What the diagnostics actually show.** Perturbing each input and measuring
`|Δaction|` (`scripts/measure_attribution.py`):

```
proprio influence : 0.5969     <- dominates
hidden influence  : 0.1173
image influence   : 0.0661
J_hat R² (vision only): 0.87
```

The encoder reads cloth state well — `Ĵ` from the attention-pooled visual
context alone reaches R² 0.87. **The representation exists and is not routed.**
`λ_j` supervises the encoder through a head wired to `z`; nothing in that
gradient reaches the fusion weights the actor uses. In control terms BC learned
the *feedforward* (the phase template) and never learned the *feedback* (the
pose-dependent correction).

Residual BC — subtracting the phase template so only the pose-dependent part
remains — was the natural next fix. The residual is 87% of the template by RMS,
but it is **not predictable from the initial garment pose** (held-out R² −0.042)
and has no persistent within-episode structure (half-to-half corr −0.006). It
looks like demonstrator jitter.

**Consequence:** BC has a low ceiling here by construction, and visual
adaptation has to be *discovered* rather than imitated. That is a far stronger
argument for the damped-RL stage than throughput ever was — RL's objective is
`J`, which depends on cloth configuration, so vision becomes load-bearing
because the objective makes it so.

One caveat for that stage: **the BC actor is a bad prior** (worse than frozen on
`J_min`), so `prior_kl_coef` should not anchor to it. The encoder transfers; the
actor should not.

### The spec's reward is exploitable
§2.3 pairs a *proportional* ascent penalty with a *constant* descent bonus, so
oscillating with amplitude `a` nets `λ_down − λ_up·a` — **positive for any
`a < λ_down/λ_up`** (0.1 with the spec's defaults). Measured: ringing scored
**+8.6** vs **−0.4** for holding still. Default is now `mono_mode="proportional"`;
`"constant"` reproduces the spec verbatim and warns.

### Damping suppresses exploration
`−λ_v‖v‖` and `−λ_Δa‖Δa‖` are both zero when stationary, so **freezing beats
every exploratory policy** (−538 vs −541/−664). The theorem only needs
*eventually* monotone `J` — a tail property — so applying these globally is
strictly stronger than required, and it is paid for during discovery. Proposed
fix: anneal with `J`, mirroring the gating `r_mono` already has.

### LeHome's IK solver disagrees with the simulator
At q≈0 the URDF places the gripper 0.452 m from base; the sim's `gripper` body
is at 0.386 m. Distances-from-base are rotation-invariant, so this is a genuine
kinematic mismatch. **A commanded 5 cm move drove the arm 0.394 m.** Use
`DifferentialIKController(ik_method="dls")` on the simulator's own Jacobian
instead — consistent by construction, and DLS adds damping at the kinematic
layer, a fourth level the spec never names.

### Bugs found in this implementation
* Rayleigh damping sign error — the stiffness-proportional term was *injecting*
  energy (diverged at only ~1.5×/substep, so it read as a CFL problem).
* GAE conflated `terminated` with `truncated`, so every time-limit was scored as
  a real terminal state with zero future value.
* Unbounded Gaussian mean against the env's ±1 clamp: the mean drifted to 2.05
  with 83% of components saturated, the advantage stopped discriminating, and
  the policy random-walked outward. Fixed with a tanh-squashed Gaussian.
* Command-path integrator wind-up produced a sustained limit cycle
  (`x_cmd` ran 0.31 past a 0.15 goal). Fixed with a rate-limited leash.

### Low loss can mean nothing without a baseline
BC's validation MSE read as excellent (0.00283, RMSE ≈ 3°) for two epochs while
the policy was **worse than repeating its previous action** (persistence
scores 0.00256). At 30 fps consecutive joint targets are nearly identical, so
the metric is dominated by temporal autocorrelation — and a GRU can represent
that degenerate solution exactly. The trainer now reports
`val_mse / persistence` with an explicit `BEATS-PERSISTENCE` flag.

### On-policy RL from scratch is not reachable on this hardware
Measured throughput at `num_envs=1`: **1.40 policy steps/s**, 3.2× slower than
realtime, with physics 77% and `J` 22% of each step (rendering is 0.9% —
the opposite of what I expected). That is **83 days for 10⁷ steps**, against an
exploration landscape where freezing beats exploring. Hence BC first, damped RL
as finetuning.

### The mock cloth cannot fold — treat it as CI only
An oracle driving both grippers exactly onto their folded-target positions
reaches only `J = 1.51` against a 0.02 threshold, because the un-gripped
vertices end 0.204 m away. The mass-spring sheet has no table contact, friction,
self-collision, or bending stiffness, so moving two corners just stretches it.
**All mock training runs were optimising an unreachable objective.** The mock is
still useful as fast CI for the RL machinery; it is not a folding testbed.

---

## Isaac Sim operational notes

* **Kit never exits on its own.** Non-daemon threads mean a raised exception or
  interrupt leaves the process at ~300% CPU forever, and **it ignores SIGTERM** —
  `kill -9` is required. Always launch via `tasks/isaac_app.py`, which closes in
  a `finally` and `os._exit`s.
* `OMNI_KIT_ACCEPT_EULA=YES` or Kit blocks on a prompt and dies with
  *"Unable to bootstrap inner kit kernel: EOF when reading a line"*.
* `LD_PRELOAD` needs system libgomp + torch's bundled one; Isaac Lab prints the
  path but it goes **stale after any torch reinstall**.
* `env.initialize_obs()` must run after `gym.make(...).unwrapped`, before
  stepping — `DirectRLEnv` never calls it and `GarmentObject.reset()` raises
  without it.
* Cold start ~55 s, warm ~14 s; garment env build ~60–70 s on top.

---

## lehome-challenge patches

Four dependency defects, all silent. Patched in their `pyproject.toml`:

| defect | symptom |
|---|---|
| `pinocchio` → nose plugin | **all IK dead**; the real library is `pin` |
| torch/torchvision cu128 pair | resolves to **CPU**, GPU unused |
| `open3d==0.19.0` | no ARM64 wheel → `0.18.0` |
| `required-environments` x86_64 | blocks aarch64 despite ARM wheels existing |

Also: `isaaclab.sh -i none` silently upgrades torch (reverting the CUDA fix) and
its core install fails on `flatdict` needing `pkg_resources` under uv build
isolation.

`torch 2.7.0+cu128` **works on GB10** (sm_120 cubins run on sm_121). CUDA-13
wheels are impossible here: `isaacsim_core` pins `torch==2.7.0` exactly and
cu130 starts at 2.9.0.

---

## Layout

```
source/lehome/real_damped_project/
  math/cloth_functional.py          J for the mock backend
  math/garment_functional.py        J for the real LeHome success metric
  math/functional_design.md         design notes for J and the damping hierarchy
  control/impedance_controller.py   clipped/low-passed integrator + anti-windup
  tasks/rewards.py                  discrete Lyapunov descent reward
  tasks/backend.py                  LeHomeBackend protocol + mock cloth
  tasks/isaac_garment_backend.py    real GarmentEnv adapter (DLS differential IK)
  tasks/isaac_app.py                guaranteed-teardown Isaac launcher
  tasks/lehome_fold_garment_real_damped_task.py
  tasks/cfg.py                      config builder + registration + make_env
  policy/vision_attention_policy.py conv → spatial attention → GRU → actor/critic
  train/ppo.py                      PPO + KL trust region + prior anchor + Polyak
  train/runner.py                   rollout loop + convergence diagnostics
scripts/
  launch_probe.py                   verify Isaac + garment env + J
  verify_fk_ik.py                   FK/IK vs the simulator
  measure_joint_damping.py          step-response ζ measurement
  test_adapter_closed_loop.py       Cartesian accuracy of the adapter
```

The package is symlinked into `lehome-challenge/source/lehome/lehome/` so it
imports as `lehome.real_damped_project`, matching the spec's intended layout.

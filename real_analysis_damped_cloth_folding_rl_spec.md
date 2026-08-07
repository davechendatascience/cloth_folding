# Project Spec: Real-Analysis-Guided Damped Visual RL for Cloth Folding on DGX Spark

## 0. Mathematical Design Principle

We design the entire system — cloth + robot + policy + RL updates — as a **dynamical system** over a metric space, and use real-analysis ideas (monotone sequences, contractive maps, Lyapunov-like functionals) to enforce **damped, non‑oscillatory convergence** to folded cloth configurations.

### 0.1. State Space and Error Functional

- Let \(X\) be a metric space of cloth+robot states with metric \(d: X \times X \to \mathbb{R}_{\ge 0}\).
  - Practically, \(X\) can be the space of internal LeHome cloth meshes plus robot joint states.
  - For design, we consider a lower-dimensional descriptor \(x_t \in X\) (cloth configuration + gripper poses).

- Define a **goal set** \(G \subset X\):
  - \(G\) = set of all "folded" cloth states for a given garment type.

- Define a **cloth error functional** \(J : X \to \mathbb{R}_{\ge 0}\):
  - \(J(x)\) quantifies distance from folded goal (task error).
  - Properties:
    - \(J\) is continuous.
    - \(J(x) = 0\) if and only if \(x \in G\).
    - \(J(x)\) increases with misalignment, wrinkles, and poor overlap.

We want the closed-loop system (including RL) to produce trajectories \(\{x_t\}_{t\ge0}\) such that:

- \(J(x_t)\) is **eventually monotone non-increasing**.
- \(J(x_t)\) converges to 0, and hence \(x_t\) converges to a folded configuration in \(G\).

This guides the RL pipeline.

---

## 1. System & Environment Setup (DGX Spark + LeHome + Isaac Lab)

### 1.1. Core Stack

- Hardware: **NVIDIA DGX Spark** (multi-GPU box).
- Simulation & RL framework: **Isaac Lab** with GPU-native physics and visual RL.
- Cloth environment: **LeHome Fold Garment** environment (SO‑ARM101 bimanual robot + garment assets).
- Cloth physics: **Newton cloth** solver integrated with Isaac Lab, configured for strong damping.
- Programming language: Python 3.11.

### 1.2. Repository and Package Layout

Assume `lehome-challenge` + pinned IsaacLab are installed.

Add a new package:

```text
source/lehome/real_damped_project/
  __init__.py
  math/
    functional_design.md   # optional notes for J, metrics
  tasks/
    lehome_fold_garment_real_damped_task.py
  control/
    impedance_controller.py
  policy/
    vision_attention_policy.py
  train/
    train_ppo_real_damped.py
```

This package implements the real-analysis-guided RL pipeline.

---

## 2. Real-Analysis-Guided RL Formulation

We treat the RL pipeline itself as an iterative scheme over sequences:

- **State sequence**: \(x_{t+1} = F(x_t; \theta)\), where \(\theta\) are policy parameters.
- **Parameter sequence**: \(\theta_{k+1} = T(\theta_k)\) via RL updates.

The design constraints:

1. The **physical dynamics** (Newton cloth + robot control) are **dissipative**, making \(F\) nearly contractive in \(d\).
2. The **policy** is trained so that \(J(x_t)\) is eventually monotone non-increasing.
3. The **RL updates** form a **damped, non-expansive map** \(T\) so parameters converge without oscillation.

### 2.1. Cloth Functional \(J\): Lyapunov-like Objective

Define \(J(x)\) as:

\[
J(x) = \lambda_1 \cdot (1 - \text{IOU}(x, x_{\text{target}}))
      + \lambda_2 \cdot \text{edge
gap}(x)
      + \lambda_3 \cdot \text{wrinkle
global}(x),
\]

where:

- `IOU` measures overlap between cloth mask and target folded template.
- `edge_gap` measures misalignment of garment edges (distance between corresponding edges).
- `wrinkle_global` measures total curvature/roughness of the cloth (e.g., sum of squared deviations from flatness).

Properties:

- \(J : X \to \mathbb{R}_{\ge 0}\) is continuous.
- \(J(x) = 0\) if cloth exactly matches target fold.
- As \(J(x_t)\) decreases and approaches 0, the cloth is closer to \(G\).

### 2.2. Damped Physics and Control: Contractive Semigroup

- Configure Newton cloth with **Rayleigh damping** parameters \(\alpha, \beta\) giving a linear damping matrix \(D = \alpha M + \beta K\), where \(M\) is mass and \(K\) stiffness.
- Choose \(\alpha, \beta\) so that cloth oscillations decay rapidly and the cloth flow \(\Phi_t\) is strongly dissipative:

  - For small time steps, \(d(\Phi_t(x), \Phi_t(y)) \le \alpha(t) d(x,y)\) with \(\alpha(t) < 1\) for some \(t > 0\).

- Configure robot control as **critically/over-damped impedance**:

  - EE dynamics:

    \[
    M \ddot x + D \dot x + K (x - x_{\text{cmd}}) = 0,
    \]

  - Damping ratio \(\zeta \ge 1\) (critical or over-damped).

This ensures the combined closed-loop map \(F\) does not induce high-frequency oscillations in \(x_t\).

### 2.3. Reward as Discrete Lyapunov Descent

The RL reward is a direct encoding of the real-analysis conditions on \(J\):

- **Task term (Lyapunov descent):**

  \[
  r_{\text{task}}(t) = -J(x_t).
  \]

- **Monotone convergence term (near goal):**

  - Define threshold \(J_{\text{near}} > 0\) and tolerance \(\epsilon > 0\).
  - If \(J(x_t) < J_{\text{near}}\), enforce eventual monotonicity:

    - If \(J(x_{t+1}) > J(x_t) + \epsilon\), i.e., error increases near goal, heavily penalize:

      \[
      r_{\text{mono}}(t) = -\lambda_{\text{up}} (J(x_{t+1}) - J(x_t)).
      \]

    - If \(J(x_{t+1}) < J(x_t) - \epsilon\), i.e., error decreases significantly, reward:

      \[
      r_{\text{mono}}(t) = \lambda_{\text{down}}.
      \]

    - Otherwise, \(r_{\text{mono}}(t) = 0\).

This makes the sequence \(\{J(x_t)\}\) **eventually monotone non-increasing**, so by real analysis it converges to some \(J_\infty \ge 0\). By design of \(J\), we want \(J_\infty \approx 0\).

- **Velocity and action damping terms:**

  - EE velocity norm:

    \[
    r_{\text{vel}}(t) = -\lambda_v \|\dot x_{\text{ee}}(t)\|.
    \]

  - Action change norm:

    \[
    r_{\text{act}}(t) = -\lambda_{\Delta a} \|a_t - a_{t-1}\|.
    \]

These encourage smooth, damped trajectories in both physical and action spaces.

Total reward:

\[
r_t = r_{\text{task}}(t) + r_{\text{mono}}(t) + r_{\text{vel}}(t) + r_{\text{act}}(t).
\]

---

## 3. RL Pipeline Design (Real-Analysis-Guided)

### 3.1. Observation and Action Spaces

- **Observations (policy input):**
  - Images: multi-camera RGB(-D) from LeHome environment, stacked into `(C, H, W)`.
  - Proprioception: robot joint positions, gripper states, EE poses.
  - **No cloth keypoints or mesh** in observations; these are used only for \(J(x_t)\) in the reward.

- **Actions:**
  - Continuous Cartesian deltas for each arm:
    - `action_dim = 6` → 3D delta for left EE + 3D delta for right EE.
  - These deltas are bounded and passed through the **damped impedance controller**.

### 3.2. Damped Impedance Controller (Interface)

Class: `DampedImpedanceController` (as in previous spec), but now clearly tied to the real-analysis view:

- It enforces that the effective map from actions to EE pose increments is **bounded and Lipschitz**:

  - For action deltas \(u\) and \(v\):

    \[
    \|x_{\text{cmd}}(u) - x_{\text{cmd}}(v)\| \le L \|u - v\|.
    \]

- This supports the contraction properties of the closed-loop map \(F\).

### 3.3. Vision + Attention Policy

Policy: `VisionAttentionPolicy`, same structural components:

1. **Image encoder:** maps images to feature map \(F \in \mathbb{R}^{D \times H' \times W'}\).
2. **Spatial attention:** produces context vector \(z \in \mathbb{R}^D\), representing **visual grounding** on relevant cloth regions.
3. **Recurrent head:** GRU/Transformer mapping \([z; p_t]\) to hidden state \(h_t\).
4. **Policy head:** maps \(h_t\) to action deltas \(a_t\).
5. **Value head:** maps \(h_t\) to scalar value \(V(x_t)\).

This architecture is **Lipschitz in inputs** under standard assumptions (bounded weights), supporting convergence of value/policy training.

### 3.4. RL Algorithm: Damped PPO / AWR

Use PPO or AWR with damping-inspired constraints:

- **Learning rate and trust region:**
  - Small step sizes; restrict KL divergence between old/new policies to prevent oscillatory updates.

- **Action smoothing:**
  - Include action-change penalty in reward.
  - Optionally, use a prior \(\pi_0\) and regularize \(\pi_\theta\) towards \(\pi_0\) to keep updates non-expansive.

The RL update \(T\) should be designed so that parameter sequence \(\{\theta_k\}\) is **bounded and asymptotically convergent**, analogous to damped inertial schemes studied in real analysis and optimization. This reduces ringing in both parameter and action spaces.

---

## 4. Environment Class: `LeHomeFoldGarmentRealDampedEnv`

File: `tasks/lehome_fold_garment_real_damped_task.py`

### 4.1. Responsibilities

- Wrap LeHome’s Fold Garment environment with:
  - Newton cloth solver + damping.
  - Damped impedance controller for SO‑ARM101.
  - Real-analysis-guided reward based on \(J\), monotonicity, and smoothness.

### 4.2. Core Methods

Implement methods:

- `__init__(cfg, sim_device, rl_device, graphics_device_id, headless)`:
  - Instantiate base LeHome environment.
  - Configure Newton damping parameters.
  - Instantiate `DampedImpedanceController`.
  - Initialize reward parameters: \(J_{\text{near}}, \epsilon, \lambda_{\text{up}}, \lambda_{\text{down}}, \lambda_v, \lambda_{\Delta a}\).

- `_reset_idx(env_ids)`:
  - Reset LeHome env for selected envs.
  - Get initial EE positions and call `controller.reset(x_init)`.
  - Reset internal tracking (`prev_J`, `prev_action`).

- `_pre_step(actions)`:
  - Map actions to EE deltas; pass through controller.
  - Apply commanded EE positions to LeHome robot control.
  - Store actions as `prev_action`.

- `_post_step()`:
  - Collect images and proprio observations.
  - Compute cloth error \(J(x_t)\) via LeHome’s internal success metric.
  - Compute rewards as per Section 2.3:
    - Task term: \(r_{\text{task}}\).
    - Monotone convergence term: \(r_{\text{mono}}\) using \(\Delta J = J(x_{t+1}) - J(x_t)\).
    - Velocity and action damping terms: \(r_{\text{vel}}, r_{\text{act}}\).
  - Aggregate reward \(r_t\).
  - Compute done flags based on LeHome’s logic.
  - Return `(obs, rewards, dones)`.

This environment implements the **discrete Lyapunov descent** notion directly.

---

## 5. Task Configuration and Registration

### 5.1. `build_lehome_real_damped_cfg()`

Define a config builder that:

- Sets simulation parameters (`dt`, physics solver, number of envs).
- Configures Newton cloth damping (\(\alpha, \beta\)) for strong dissipation.
- Configures SO‑ARM101 controller with impedance parameters (\(D, K\)) giving critical/over-damped behavior.
- Configures camera layout and image resolution.
- Sets reward hyperparameters:
  - `J_near` (threshold for monotonicity enforcement),
  - `epsilon` (error change tolerance),
  - `lambda_up`, `lambda_down`, `lambda_v`, `lambda_delta_a`.

Register task:

```python
@register_task(name="LeHome-Fold-Garment-RealDamped-v0", env_cls=LeHomeFoldGarmentRealDampedEnv)
def build_lehome_real_damped_cfg():
    cfg = RLTaskEnvCfg()
    # fill cfg with physics, robot, camera, reward parameters
    return cfg
```

---

## 6. Training Script: `train_ppo_real_damped.py`

### 6.1. CLI and Environment Creation

The script should:

1. Parse arguments:
   - `--task` (default `"LeHome-Fold-Garment-RealDamped-v0"`).
   - `--num_envs` (e.g., `1024` or `2048`).
   - `--device` (default `"cuda"`).
   - `--headless` flag.

2. Create environment:

   ```python
   env = make_env(
       task_name=args.task,
       num_envs=args.num_envs,
       sim_device=args.device,
       rl_device=args.device,
       graphics_device_id=0,
       headless=args.headless,
   )
   ```

### 6.2. Policy and PPO Initialization

3. Extract observation and action spaces:

   - `obs_space = env.observation_space` → dict with `"images"`, `"proprio"`.
   - `act_space = env.action_space`.

4. Instantiate `VisionAttentionPolicy` with:

   - `image_channels = obs_space["images"].shape[0]`.
   - `proprio_dim = obs_space["proprio"].shape[0]`.
   - `action_dim = act_space.shape[0]`.

5. Instantiate PPO agent (`PPOAgent`) with damping-friendly hyperparameters:

   - Small learning rate (e.g., `3e-4`).
   - Value coefficient `0.5`, entropy coefficient `0.01`.
   - `num_steps_per_env` (e.g., 64).
   - `max_iterations` (~1e4–2e4).
   - Optionally, use KL-based trust region to control update magnitude.

6. Train:

   ```python
   runner = Runner(env, agent)
   runner.train()
   ```

### 6.3. Running on DGX Spark

Invoke on DGX Spark via Isaac Lab wrapper:

```bash
cd lehome-challenge
source .venv/bin/activate

./third_party/IsaacLab/isaaclab.sh \
  -p source/lehome/real_damped_project/train/train_ppo_real_damped.py \
  -- --task LeHome-Fold-Garment-RealDamped-v0 \
     --num_envs 2048 \
     --device cuda \
     --headless
```

This runs large-batch visual RL with explicit Lyapunov-based reward and damping.

---

## 7. Summary of Real-Analysis Integration

- **Lyapunov-like functional** \(J\):
  - Encodes cloth folding quality.
  - Used in reward to enforce eventual monotone descent.

- **Dissipative physics and control**:
  - Cloth and robot dynamics are chosen so that the closed-loop map \(F\) is nearly contractive.

- **Damped RL updates**:
  - PPO/AWR with trust regions and action smoothing to avoid oscillatory parameter updates.

- **Convergence behavior**:
  - Near goal, \(\{J(x_t)\}\) is monotone non-increasing and bounded below.
  - By real analysis, \(J(x_t)\) converges, and via \(J\)'s design this implies convergence to a folded state.

This spec connects your real-analysis intuition directly to the RL pipeline choices, giving Claude (or any engineer) a mathematically principled blueprint for implementing a damped visual-grounded cloth folding system on DGX Spark.
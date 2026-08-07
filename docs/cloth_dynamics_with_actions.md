# Cloth State as a *Controlled* Dynamical System

Written 2026-08-07, after measuring that our cloth model is predictable but
barely controllable. The conclusion is that we asked a model to relearn things
we already know exactly, and that the fix is structural rather than a matter of
more data or a bigger network.

---

## 1. What we measured

Ground-truth check-point positions from demo replay, 9 episodes / 2795 frames,
ridge regression predicting `Δp` over a horizon `h`. Baselines matter more than
the model here, so all of them are reported.

| horizon | best baseline | model | improvement |
|---|---|---|---|
| 1 step (33 ms) | velocity extrap. 0.0174 | **0.0139** | **+20.4%** |
| 5 steps (167 ms) | phase clock 0.7976 | **0.4014** | **+49.7%** |
| 20 steps | phase clock 2.224 | 2.973 | −33.6% |
| 60 steps | phase clock 3.358 | 4.345 | −29.4% |

**Cloth state is predictable at short horizons, and genuinely so** — it beats the
phase-only predictor, which is the confound that invalidated three earlier
results in this project. The long-horizon failure is expected and not fatal:
MPC re-plans every step.

Then the ablation that matters:

| feature set | MSE (h=1) | MSE (h=5) |
|---|---|---|
| cloth only `(p, ṗ)` | 0.01420 | 0.47426 |
| + arm pose `(…, s)` | 0.01394 | 0.40932 |
| + action `(…, a)` | **0.01386** | **0.39973** |
| action only `(s, a, a−s)` | 0.04130 | 0.67105 |

**The commanded action contributes 2.4% at h=1.** At h=5 the 15.7% total gain is
almost entirely *arm pose* (13.7%); the command adds ~0.6% on top.

So the cloth is predictable from its **own momentum and settling** — passive
dynamics. That is not what control needs. Control needs `∂p/∂u` to be
substantial, and here it is nearly flat.

**Predictable ≠ controllable.**

---

## 2. Diagnosis: we asked the model to rebuild what we already have

The features were 12 joint angles and 12 joint position targets. For a model to
turn those into a cloth prediction, it must implicitly learn:

1. **forward kinematics** — where the gripper is, from joint angles
2. **contact geometry** — whether the gripper is touching cloth, and where
3. **grasp state** — whether cloth is actually held
4. **assignment** — which cloth region a given arm motion affects

We know all four exactly:

| quantity | how we already have it |
|---|---|
| gripper pose | verified FK/IK, 8/8 Cartesian moves to 0.0000 m |
| contact / proximity | gripper position vs check-point positions, both exact |
| grasp state | we *command* the gripper; it is not a latent variable |
| affected region | nearest cloth particles to the gripper, computable |

Handing the model raw joint vectors and asking it to rediscover a kinematic
chain we have in closed form is the definition of rebuilding from scratch. It
also explains the 2.4%: at 30 fps the commanded action is nearly identical to
the current joint state, so as a *feature* it carries almost no information
beyond arm pose — while as a *physical cause* it carries everything.

The information is not missing from the data. It is in the wrong coordinates.

---

## 3. What the literature does instead

The consistent pattern across recent work is that **the action is applied at the
grasped point, in cloth space** — not fed in as robot commands.

- **[GraphGarment](https://arxiv.org/html/2503.05817)** builds a graph whose
  nodes are categorised as `garment_left_grasped`, `garment_right_grasped`,
  `main body`, and **`action nodes`** — the end-effector is an explicit node in
  the cloth graph. Edges connect the grasped regions and action nodes and
  deliberately *exclude* the main body. 10,000 transition pairs per garment,
  single-step prediction, quasi-static.
- **[Mesh-based Dynamics with Occlusion Reasoning](https://arxiv.org/pdf/2206.02881)**
  states it plainly: *"the action is encoded to the dynamics model by directly
  modifying the position and velocity of the grasped point on the cloth."*
- **[MPC for Dynamic Cloth Manipulation](https://arxiv.org/pdf/2209.05798)** uses
  *displacement of the grasped corner* as the control input, and simplifies
  further by fixing end-effector orientation and controlling two adjacent mesh
  points.
- **[Koopman-operator MPC](https://arxiv.org/pdf/2605.18373)** goes further and
  fits a *linear* operator in a lifted space, precisely because the underlying
  structure is exploitable rather than arbitrary.

Note what this buys. If the action *is* the grasped point's displacement, then
for the grasped node `∂p/∂u ≈ I` — the hardest term becomes the identity, and
the model only has to predict how the *rest* of the cloth responds. Our
formulation asked it to learn that identity map through a kinematic chain first.

Also note the data scale: ~10k transitions per garment, with randomised actions.
We have 2795 frames of stereotyped demonstration.

---

## 4. Prerequisites to inject

Ordered by expected value, all of them things we possess exactly and are
currently making the model infer.

**4.1 Action in cloth space, not joint space.**
Replace `(s, a)` with the end-effector **position and displacement**, computed
by our verified FK. Additionally provide the displacement of the *grasped
particle* when grasping. This is the single largest change.

**4.2 Explicit contact / grasp features.**
- gripper-to-check-point distances (6 scalars per arm)
- a binary grasp indicator per arm (commanded, therefore known)
- nearest-particle index and its offset from the gripper

Contact is where all the causality lives; making the model detect it from joint
angles wastes capacity on a solved problem.

**4.3 Mesh topology.**
The garment has 14,544 particles with known connectivity. The 6 check-points are
an arbitrary subsample chosen by LeHome for *scoring*, and there is no reason
`p_{t+1}` should be Markovian in 6 points — the cloth's actual state is the mesh.
This is the argument for a graph model over a flat regressor, and for a dense
field representation over keypoints (see `LEVERS.md`).

**4.4 Measured structural facts.**
- **Planarity**: dropping `z` changes the pairwise distances J consumes by 0.207 cm
  mean / 0.899 cm p95, a 2.53% relative error, corr 0.996. The task is effectively
  2-D, which shrinks both the perception and dynamics problems.
- **Momentum**: velocity is the dominant predictor at short horizons. The state
  is `(p, ṗ)`; a first-order state loses to plain extrapolation.

**4.5 The objective, in closed form.**
`J = f(p)` is a *definition* — margin violations over five pairwise distances —
and is differentiable, verified nonzero on all 6 check-points. `∂J/∂p` never
needs to be learned. Any architecture that regresses J from pixels instead of
computing it from an estimated configuration is discarding an exact function in
favour of an approximation.

---

## 5. Proposed model

```
state    x_t = (mesh positions, velocities)        ← known in sim, estimated from vision later
action   u_t = grasped-point displacement          ← in cloth space, not joint space
                + grasp indicator
dynamics x_{t+1} = x_t + GNN(x_t, u_t)             ← residual, mesh-structured
objective J = f(readout(x))                        ← exact, differentiable
control  u* = argmin J(x_{t+1})  via ∂J/∂p · ∂p/∂u ← MPC, one step, re-planned
```

Residual formulation (`predict Δ`, not absolute) matters because the identity
map dominates at 33 ms and should not consume capacity.

Start with the smallest thing that tests the claim: a **linear or MLP model on
grasp-centric features** — gripper displacement, grasp flag, per-check-point
offsets from the gripper. If `∂p/∂u` does not appear even in that formulation,
a GNN will not rescue it, and the problem is the data rather than the model.

---

## 6. Where damping enters

Damping is usually treated as a control parameter. Here it is a property of the
**data-generating process**, and it is upstream of everything above.

An under-damped plant answers one action with ringing and overshoot, so a single
`u` maps to a *distribution* of outcomes and the model must learn the transient.
Critically damped, the same action settles smoothly to a predictable
configuration and the map is nearly static — which is exactly the quasi-static
assumption the entire pick-and-place literature relies on without stating it.

Consequences:

1. **The forward model becomes learnable** from fewer samples, because the
   target has lower variance.
2. **The observer's prediction step becomes reliable**, which is what lets state
   estimates survive occlusion.
3. **Quasi-static primitives become valid**, so pick-and-place is well-posed.

And note the asymmetry we are now free to exploit: BC had to use the demo plant
(under-damped, ζ≈0.53–0.65) because it was imitating trajectories recorded there.
Once we generate our own data, critical damping is free and strictly better.

This is testable and should be tested: collect identical perturbations under both
plants, fit the same model, compare held-out error. If critical damping does not
produce a materially cleaner model, the argument is wrong.

---

## 7. What to measure first

In order, cheapest and most decisive first.

1. **Re-run the ablation with grasp-centric features** on the data we already
   have. No new collection. If `∂p/∂u` rises from 2.4% to something substantial
   purely by changing coordinates, that confirms the diagnosis in section 2 for
   the cost of one script.
2. **Solve grasping.** LeHome uses *no particle attachment* — holding cloth
   relies entirely on material properties (`adhesion 0.1`, `friction 0.5`,
   `particle_friction_scale 0.6`). A hand-written approach achieved |Δp| of
   0.16–1.27 cm, i.e. no grasp. The demonstrations *do* grasp, so the honest
   route is to extract successful grasp approaches from demo replay rather than
   hand-tune.
3. **Randomised action collection**, both damping modes.
4. **Re-run the ablation** on that data. If the action still contributes ~2%
   with varied gripping actions, the cloth is not controllable at this state
   representation — a genuine finding about the task, not our method.

---

## 8. Risks

- **6 check-points may not be a Markovian state.** `p_{t+1}` plausibly depends on
  the full mesh, not 6 samples of it. The short-horizon result (+20%, +50%)
  suggests they are adequate at small `h`, but this is the assumption most
  likely to break, and it argues for the mesh/field representation.
- **Grasping is unsolved and now on the critical path**, not optional. The whole
  plan is contingent on it.
- **Demo data cannot establish controllability.** Actions are stereotyped and
  confounded with phase, so "the arm caused the cloth to move" and "both follow
  the same script" are not separable. This is not fixable with a better model.
- **Sim-to-real is out of scope here** but GraphGarment found it large enough to
  need a dedicated residual model; anything we conclude is a statement about
  this simulator.

---

## Sources

- [GraphGarment: Learning Garment Dynamics for Bimanual Cloth Manipulation](https://arxiv.org/html/2503.05817)
- [Mesh-based Dynamics with Occlusion Reasoning for Cloth Manipulation](https://arxiv.org/pdf/2206.02881)
- [MPC for Dynamic Cloth Manipulation: Parameter Learning and Experimental Validation](https://arxiv.org/pdf/2209.05798)
- [Dynamic Robotic Cloth Folding with Koopman Operator-Based MPC](https://arxiv.org/pdf/2605.18373)
- [FoldNet: Keypoint-Driven Asset and Demonstration Synthesis](https://arxiv.org/abs/2505.09109)
- [QDP: Quasi-Static and Dynamic Manipulation Primitives](https://arxiv.org/abs/2303.13320)
- [Deep learning-based robotic cloth manipulation: systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12921407/)
- [Learning Latent Graph Dynamics for Visual Manipulation of Deformable Objects](https://arxiv.org/pdf/2104.12149)

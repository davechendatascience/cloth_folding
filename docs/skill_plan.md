# Plan: Skill Learning, Not Cloth Physics

Written 2026-08-10. This supersedes the behaviour-cloning direction. It is a
restart on approach, not on findings — the measurements below are what justify
abandoning the previous thread, and several of them constrain the new one.

---

## 1. Why the previous direction is abandoned

Three behaviour-cloning designs, all evaluated closed-loop with matched garment
poses on the demonstration plant:

| run | loss | `val_mse/persistence` | closed loop |
|---|---|---|---|
| `bc_top_long` | delta target | 1.10 | — |
| `bc_j` | + ΔJ weighting + `Ĵ` head | 1.176 | 0/3, = frozen |
| `bc_residual` | + phase template removed | 6.95 | 0/3, = frozen |

None beat a predictor that ignores the observation. The cause is not the loss:

```
phase only  -> action    R² 0.688     (a predictor that sees NOTHING)
proprio     -> action    R² 0.725
phase only  -> J         R² 0.810
vision      -> J         R² 0.863     (adds 0.053; within-band R² NEGATIVE)
```

**The demonstrations are ~69% a fixed script**, so no loss function can create a
need for vision in data that does not require it. Confirmed behaviourally: under
randomised texture and lighting the policy's closed-loop J moved by 0.009, while
frozen's own run-to-run noise was 0.086. It is invariant to appearance because it
does not use appearance.

Abandoned: BC on joint trajectories, the `Ĵ` auxiliary head, residual BC,
frame-level `p_{t+1}` regression as the primary thread.

---

## 2. What carries forward

| asset | status |
|---|---|
| `J = f(p)` — margin violations over check-point distances | exact, differentiable, verified `J=0` ⟺ LeHome's success predicate over 300 configs |
| DLS IK | 8/8 Cartesian moves to 0.0000 m |
| **gripper body → contact offset ≈ 9 cm** | calibrated from 4,926 demo frames where cloth tracks a gripper |
| translation invariance of cloth dynamics | +14.9% at h=20; **rotation is −30%**, so the group is translation in (x,y), not SE(2) |
| contact structure as a feature | `prox(d)·Δee` took the action's contribution 1.5% → 10.3% |
| measurement discipline | phase controls, trivial baselines, pose-level splits |

That last row is the one to keep hardest. Three separate headline numbers in this
project collapsed once a predictor that sees nothing was run beside them.

**Cautions carried forward:** replay is chaotic (identical replays of episode 0
gave final J of 1.770, 1.062, 2.214), so single-rollout outcomes mean little; and
the 27% replay-success figure is a *fidelity* statement, not a claim that 73% of
demonstrations fail.

---

## 3. The organising idea

Do not identify cloth physics. Identify **actions whose outcome is invariant to
cloth micro-state**.

Two cloth states are equivalent if the same skill has the same outcome
distribution from both — an action-conditioned abstraction:

```
z(s₁) = z(s₂)  ⟹  P(z′ | s₁, u) ≈ P(z′ | s₂, u)
```

This is why the field uses quasi-static pick-and-place primitives rather than
continuous control: the primitive absorbs the variation. A skill *is* an action
whose result does not depend on wrinkle-level detail.

**Where damping enters, and it is load-bearing.** An under-damped plant rings, so
the same primitive from the same state lands differently — outcome labels become
noisy and the invariance is masked by *execution* variance. Critically damped, the
primitive settles repeatably, so measured variance reflects **cloth-state
sensitivity** rather than controller sensitivity. Damping is what makes invariance
measurable at all. We are free to use it here because we generate this data;
only demo imitation was forced onto the under-damped plant.

---

## 4. First experiment: invariance profiling

Before building a skill library, **measure which candidate actions are actually
invariant.** For each primitive, execute from many randomised cloth states and
measure the spread of its outcome.

- **low variance** → a genuine skill; usable open-loop, needs little perception
- **high variance** → cloth state matters here; this is where perception is
  required, and where a closed-loop controller earns its cost

The output is a ranked list that decides the skill library empirically, and it
simultaneously tells us *where* perception is needed — instead of assuming it is
needed everywhere, which is the assumption that cost us the BC thread.

Outcomes are recorded per skill, all computed from privileged state, none
requiring material parameters:

```
grasp_held_frac      fraction of the move during which cloth tracked the gripper
held_at_end          did the hold survive
cp_displacement_cm   how far the targeted point actually moved
delta_J              did the action make task progress
```

**Controls.** A primitive can look invariant by doing nothing. So the profile
reports displacement alongside variance, and a "no-op" primitive is included as
the floor. An action with low outcome variance *and* near-zero displacement is
not a skill; it is inactivity.

---

## 5. Sequence after profiling

1. **Invariance profile** → skill library, and a map of where perception matters
2. **Outcome model** `P(outcome | o, u)` for the high-variance skills only
3. **Composition** — sequence skills to drive `J` down, using `∂J/∂p` to choose
   which point to move where
4. **Perception** — only for the quantities step 2 shows are needed

Data comes from replay wherever possible. Only 10 of 250 LeHome episodes have
been replayed with cloth-state labels; those 10 alone yielded 4,926 frames of
real grasp events. Generate new data only for gaps the demonstrations cannot
cover — chiefly action diversity and grasp *failures*, since demos are
stereotyped and mostly succeed.

---

## 6. Risks

- **Grasping may remain unreliable.** LeHome uses no particle attachment; holding
  depends entirely on `adhesion 0.1` / `friction 0.5`. If no primitive achieves a
  stable hold, the library reduces to pushing and dragging, and the fold task may
  be out of reach in this simulator.
- **6 check-points may be too coarse** to distinguish states that behave
  differently under the same skill — the bisimulation criterion could fail at
  this resolution.
- **Invariance measured in sim is a statement about this simulator.**
- **Low-variance may mean inactive.** Guarded by reporting displacement, but
  worth restating because it is the easiest way to fool this experiment.

# Design notes for `J`, the metric, and the convergence argument

Notes backing `cloth_functional.py` and `tasks/rewards.py`. Spec §0.1, §2.1–2.3.

## 1. What the convergence argument actually needs

The claim we want is:

> Near the goal, `{J(x_t)}` is monotone non-increasing and bounded below,
> therefore it converges (monotone convergence theorem); by design of `J`, the
> limit is 0 and hence `x_t → G`.

That argument needs four things, and each is a design obligation:

| Requirement | Where it is discharged |
|---|---|
| `J ≥ 0` | every term is a norm, an absolute value, or `1 − IoU ∈ [0,1]`; `forward()` clamps at 0 against fp round-off |
| `J` continuous | soft (Gaussian-splatted) occupancy instead of hard binning; see §2 |
| `J(x) = 0 ⟺ x ∈ G` | each term vanishes exactly at the target; see §3 for the wrinkle subtlety |
| eventual monotonicity | not a property of `J` at all — it must be *induced* by the reward; see §4 |

The fourth is the one that is easy to get wrong, because `J` being a perfectly
good Lyapunov *candidate* says nothing about whether the learned policy makes it
descend. That is entirely the reward's job.

## 2. Why the occupancy mask is soft

`IoU` needs a cloth mask. The obvious implementation buckets each vertex into a
grid cell and counts. That map is **piecewise constant**: a vertex crossing a
cell boundary changes `J` by a jump. `J` is then discontinuous, `ΔJ` is
dominated by quantisation rather than by physical progress, and the `ε`
dead-band in `r_mono` becomes meaningless — it cannot distinguish "the cloth
moved a little" from "a vertex changed cell".

So each vertex is splatted as an isotropic Gaussian of width `splat_sigma`, and
density is mapped to occupancy by `1 − exp(−ρ)` (smooth, monotone, saturating
at 1). `splat_sigma` is the knob trading smoothness against spatial resolution:
larger `σ` gives a smaller Lipschitz constant for `J` and a blurrier IoU.
`test_J_is_continuous_in_the_vertices` checks that shrinking a perturbation
shrinks `ΔJ`, which the hard-binned version would fail.

Implementation note: the splat is computed separably —
`(B,N,R)` in x and y, contracted by `einsum("bnx,bny->bxy")` — so cost is
`O(B·N·R²)` flops with no `B·N·R²` intermediate.

## 3. Why the wrinkle term is *relative*

The natural reading of "wrinkle_global = total curvature" is
`R(x) = mean ‖Δx‖²` over the discrete Laplacian. But a **folded garment is not
flat**: it has curvature along the fold line. Penalising absolute roughness
would make the flat, unfolded sheet the minimiser of that term, putting `J`'s
zero somewhere that is not the goal and breaking `J(x) = 0 ⟺ x ∈ G`.

So the term is `|R(x) − R(x_target)|`: still continuous, still `≥ 0`, and zero
exactly when the cloth's curvature matches the target's.
`test_wrinkle_term_is_relative_to_the_target_curvature` locks this in.

## 4. The reward is where monotonicity is enforced — and the spec's version leaks

Spec §2.3:

```
r_mono = −λ_up·ΔJ    if J(x_t) < J_near and ΔJ >  ε
       = +λ_down      if J(x_t) < J_near and ΔJ < −ε
       = 0            otherwise
```

The ascent penalty is **proportional** to `ΔJ`; the descent bonus is
**constant**. Consider a policy oscillating with amplitude `a > ε` near the
goal. Per full cycle (one down step, one up step):

```
gain = λ_down − λ_up·a
```

which is **positive** whenever `a < λ_down/λ_up`. With the spec's defaults
(`λ_down = 1`, `λ_up = 10`) the break-even amplitude is `0.1` — five times the
width of the entire near-goal band `J_near = 0.02`. So *every* oscillation the
gate can even see is profitable. A return-maximiser learns to ring forever,
which is the exact failure the design was built to exclude.

Fix (`mono_mode="proportional"`, the default): make the bonus proportional too,
`r_mono = −λ_down·ΔJ`. A cycle then nets `(λ_down − λ_up)·a`, negative whenever
`λ_up > λ_down` — preserving the spec's intent (descent rewarded, ascent
punished harder) while closing the leak. `mono_mode="constant"` reproduces the
spec literally and warns.

An alternative worth considering if ringing persists: a **ratchet**, awarding
the bonus only when `J` sets a new running minimum. The running minimum is
monotone by construction, so this encodes "eventually monotone non-increasing"
directly rather than incentivising it. It is a larger departure from the spec
and is not implemented.

### Which `J` gates the near-goal region

§2.3 says "If `J(x_t) < J_near`" — the gate is on the **previous** `J`. The
earlier spec revision gated on the new one. This matters: gating on the new `J`
cannot penalise the step that rings *out* of the near-goal region (that step
ends with `J > J_near`, so the gate is closed exactly when the violation
happens). `near_gate="prev"` is the default;
`test_near_gate_prev_penalises_leaving_the_near_goal_region` demonstrates the
difference.

### Episode boundaries

`ΔJ` across a reset is not a dynamical transition — the cloth teleports. The
reward tracks `has_prev_J` per env and contributes no `r_mono` on the first step
of an episode. Without this, every reset would register as a large spurious
`ΔJ`.

## 5. Damping: what is actually contractive

The spec asks for the closed-loop map `F` to be nearly contractive. Three
separate mechanisms, none of which alone is sufficient:

1. **Cloth.** Rayleigh damping `D = αM + βK`. In the mock, `α` acts on velocity
   directly and `β` through the graph Laplacian of velocity. *Sign convention is
   load-bearing*: the operator must be `+Lv` with `(Lv)_i = Σ_j (v_i − v_j)`.
   With the sign flipped the "damping" term becomes an energy source; it
   diverges at only ~1.5×/sub-step, which reads like an ordinary CFL problem
   rather than a sign error. This bug was in the first draft and is now pinned
   by `test_mock_cloth_is_dissipative`.

2. **Robot.** EE impedance `M ẍ + D ẋ + K(x − x_cmd) = 0` with
   `D = 2ζ√(KM)`, `ζ ≥ 1`. `ζ < 1` is rejected at construction.
   Critical damping gives a monotone step response — no crossing of the
   setpoint — which is asserted directly, with an under-damped control case to
   prove the assertion can fail.

   Note `ζ ≥ 1` is necessary but not sufficient *in discrete time*: the
   integrator must also resolve the dynamics. `is_sampling_stable()` requires
   ≳10 samples per natural period; otherwise the sampled loop can ring even
   though the continuous system cannot.

3. **Command path.** Clip to `±max_delta` (projection onto a box, 1-Lipschitz),
   then a first-order low pass (β-Lipschitz), then integrate (1-Lipschitz),
   then optionally project onto the workspace box (1-Lipschitz). The
   composition gives `‖x_cmd(u) − x_cmd(v)‖ ≤ β‖u − v‖`, which is §3.2's
   requirement with an explicit constant.

Explicit integration of the mock imposes its own stability bounds
(`h·β·k·λ_max(L)/m < 2`, with `λ_max ≈ 8` for the 5-point stencil). These cap
`β` near 0.01 — far below the intuitive 0.08 — so `_required_substeps()`
computes the bound and raises the sub-step count rather than letting it surface
as a NaN. Newton and PhysX solve damping implicitly and have no such cap; this
is a mock limitation, not a statement about how damped the cloth should be.

## 5b. Where damping belongs: measured, not assumed

The spec says "damp everything." Measurement says damping should **increase
inward and decrease outward**, for a reason that is not aesthetic.

**Plant damping funds search; policy damping spends it.** If the plant rings
with settling window `N`, the effect of action `a_t` persists `N` steps, so the
advantage estimate must span that window and its variance grows with `N`.
Minimising `N` is minimising the price of a sample. Damping the *policy*, by
contrast, directly narrows exploration. Same word, opposite sign.

Measured on the real SO101 (`scripts/measure_joint_damping.py`, q=0, K=17.8,
D=0.60):

| joint | ζ | ω_n | J | D_crit | settle |
|---|---|---|---|---|---|
| shoulder_pan | 0.567 | 33.23 | 0.0161 | 1.071 | 15 |
| shoulder_lift | 0.529 | 27.76 | 0.0231 | 1.283 | 27 |
| elbow_flex | 0.649 | 32.80 | 0.0165 | 1.085 | 17 |
| wrist_flex | 0.933 | 56.24 | 0.0056 | 0.633 | 11 |
| wrist_roll | ≥1 | — | — | — | 12 |

Three things follow.

1. **ζ ≈ 0.67 mean — the plant is under-damped**, contradicting Sec. 2.2 at the
   innermost layer. This is not cosmetic: if the *plant* overshoots, `J(x_t)` is
   non-monotone no matter what the policy does, so the monotone-convergence
   argument fails below the level `r_mono` can reach. Plant damping is a
   *precondition* for the reward's monotonicity term to mean anything.

2. **The defect is structural.** Gains are uniform but inertia varies 4×, and
   ζ ∝ 1/√J, so proximal joints ring while distal ones are fine — and the
   ringing sits precisely in the joints with the most authority over EE
   position. A single global `D` cannot fix it: `D=1.283` sets shoulder_lift to
   critical but drives wrist_flex to ζ=2.03. Per-joint `D_j = 2K/ω_n,j`.

3. **Time-scale separation is violated.** At `decimation=1` (90 Hz) the policy
   issues up to 27 commands before the plant responds to the first, so an
   action's effect is unobservable within its own step. Cascading layers at
   comparable time constants also raises the system order and destroys the
   per-layer damping guarantee — which is exactly how the command integrator
   produced a limit cycle (§5). Settling ∝ 1/√K, so stiffness is the lever if
   the required decimation is too slow.

Caveat on precision: ζ (from overshoot ratio) and ω_n (from peak spacing) are
independent estimates, and they disagree by 12–15% for shoulder_lift and
elbow_flex — the two joints most coupled in the vertical plane, where the
single-DOF fit genuinely breaks down. Those `D_crit` values carry that
uncertainty. They are worth applying anyway: 15% off still moves ζ from 0.53 to
~0.85–1.15.

### Corollary: ζ=1 is not chosen for speed

The time-optimal damping for settling *to a 2% band* is ζ ≈ 0.7, not 1.
Critical damping is the fastest approach **with no overshoot**, which is a
different objective. Here the second is what matters, because non-monotonicity
of the state trajectory is what corrupts `J`. So ζ=1 buys monotonicity at a
small cost in settling time — a deliberate trade, not a free lunch.

## 6. Damping the parameter sequence

`{θ_k}` gets the same treatment as `{x_t}`:

- **KL trust region** — adaptive lr plus a hard epoch abort bounds the step in
  policy space, the discrete analogue of keeping `T` non-expansive.
- **Prior anchoring** — optional KL penalty toward a frozen `π_0` keeps the
  sequence bounded, which is the precondition for the convergence argument.
- **Polyak averaging** — `θ̄_k = τθ_k + (1−τ)θ̄_{k−1}` is a first-order low pass
  on the parameter sequence; even when `{θ_k}` rings, `{θ̄_k}` is smoother.

## 7. What to watch during training

`Runner.collect()` reports the quantities the argument is actually about:

- `J_mean` — the functional itself.
- `mono_violation_rate` — fraction of *near-goal* steps where `J` rose by more
  than `ε`. The direct empirical test of "eventually monotone non-increasing";
  should trend to ~0 while `J_mean` is still falling.
- `ee_speed` — whether trajectories are physically damped.

The diagnostic that matters: **falling `J` with a high violation rate means the
policy is folding by oscillating**, which is what this whole design forbids.
Note `mono_violation_rate` is only meaningful once `near_goal_frac > 0`; far
from the goal the gate never opens and the rate is trivially 0.

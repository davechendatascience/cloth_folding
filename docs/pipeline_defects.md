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

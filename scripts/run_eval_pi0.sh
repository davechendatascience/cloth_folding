#!/usr/bin/env bash
# Launch scripts/eval_pi0_in_sim.py with the environment Isaac actually needs.
#
# Four things here are load-bearing and each one was a separate failed launch:
#
#   1. INTERPRETER. pi0 lives in the project venv, `isaaclab` does not. Only
#      lehome-challenge/.venv has both (after the repair recorded in LEVERS.md).
#   2. CWD. LeHome resolves garment USDs relative to the working directory, so
#      running from the cloth_folding checkout looks for Assets/ that are not
#      there and dies with FileNotFoundError after a ~60 s Isaac start-up.
#   3. LD_PRELOAD. Isaac Lab checks for the literal path `/lib/aarch64-...`,
#      not the `/usr/lib/...` symlink to the same file, and refuses to boot
#      otherwise. torch's bundled libgomp goes second.
#   4. PYTHONPATH. Points at this repo's source/ so the symlinked package
#      resolves here rather than to a stale copy.
#
# Usage:  scripts/run_eval_pi0.sh <step> [extra args to eval_pi0_in_sim.py...]
set -euo pipefail

REPO=/home/edge-host/Documents/GitHub/cloth_folding
CHAL=/home/edge-host/Documents/GitHub/lehome-challenge
VENV=$CHAL/.venv
DATA=$CHAL/Datasets/example/top_long_merged
RUN=${RUN:-pi0_final}

STEP=$(printf "%06d" "${1:?usage: run_eval_pi0.sh <step> [args...]}")
shift || true
CKPT=$REPO/runs/$RUN/checkpoints/$STEP/pretrained_model

[ -f "$CKPT/model.safetensors" ] || { echo "no checkpoint at $CKPT" >&2; exit 2; }

cd "$CHAL"
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:$VENV/lib/python3.11/site-packages/torch/lib/libgomp.so.1" \
PYTHONPATH="$REPO/source" \
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_JIT=0 \
exec "$VENV/bin/python" "$REPO/scripts/eval_pi0_in_sim.py" \
  --ckpt "$CKPT" --dataset "$DATA" \
  --out "$REPO/runs/eval_${RUN}_${STEP}.json" "$@"

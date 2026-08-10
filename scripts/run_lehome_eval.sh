#!/usr/bin/env bash
# Run LeHome's official evaluation on a lerobot checkpoint, reporting their
# success predicate (`env._get_success()`) -- the metric that is comparable to
# published baseline numbers, unlike any training loss.
#
# Usage:  scripts/run_lehome_eval.sh <checkpoint_dir> [extra args...]
#   e.g.  scripts/run_lehome_eval.sh runs/smolvla_top_long/checkpoints/002500/pretrained_model
#
# Environment requirements are the same ones documented in LEVERS.md: Isaac
# matches LD_PRELOAD by literal path, sklearn's bundled libgomp must be
# preloaded too or it dies on static TLS, and CWD must be the lehome-challenge
# checkout because garment USDs resolve relative to it.
#
# The simulator runs on CPU (`--device cpu`, as their docs recommend) so it does
# not contend with a training job for the GPU; the policy still loads to CUDA.
set -euo pipefail

REPO=/home/edge-host/Documents/GitHub/cloth_folding
CHAL=/home/edge-host/Documents/GitHub/lehome-challenge
VENV=$CHAL/.venv
SKGOMP=$(echo "$VENV"/lib/python3.11/site-packages/scikit_learn.libs/libgomp-*.so.1.0.0)

CKPT=${1:?usage: run_lehome_eval.sh <checkpoint_dir> [args...]}
shift || true
[ -f "$CKPT/model.safetensors" ] || { echo "no checkpoint at $CKPT" >&2; exit 2; }

cd "$CHAL"
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:$VENV/lib/python3.11/site-packages/torch/lib/libgomp.so.1:$SKGOMP" \
PYTHONPATH="$CHAL" \
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_JIT=0 \
exec "$VENV/bin/python" "$REPO/scripts/lehome_eval.py" \
  --policy_type lerobot \
  --policy_path "$CKPT" \
  --garment_type top_long \
  --dataset_root "$CHAL/Datasets/example/top_long_merged" \
  --enable_cameras --headless --device cuda \
  --task_description "fold the garment on the table" \
  "$@"

#!/bin/bash
set -euo pipefail

# Optional: activate ROCm virtualenv if present
if [ -d "ROCm" ] && [ -f "ROCm/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ROCm/bin/activate
fi

# Target primary AMD GPU
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# PyTorch ROCm allocator tuning for 20GB VRAM
# - expandable_segments helps reduce fragmentation on long runs
# - garbage_collection_threshold keeps more blocks around to avoid thrash
# - max_split_size_mb prevents over-fragmentation with large tensors
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-"garbage_collection_threshold:0.6,expandable_segments:True,max_split_size_mb:256"}

# Do NOT override GFX version on supported drivers; uncomment only if your stack needs it
# export HSA_OVERRIDE_GFX_VERSION=10_3_0

# Make sure we don't accidentally disable GPU
unset POKERAI_DISABLE_GPU || true

ulimit -n 4096 || true

exec python train.py "$@"


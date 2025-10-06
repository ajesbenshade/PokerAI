#!/bin/bash

# Force activation of ROCm virtual environment
source ROCm/bin/activate

# Set environment variables for ROCm
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256
export HSA_OVERRIDE_GFX_VERSION=10_3_0

# Set ulimit for file descriptors
ulimit -n 4096

# Run the training script with all arguments passed through
python train.py "$@"

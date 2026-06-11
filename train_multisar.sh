#!/bin/bash
# Script to train Multi-SAR to Optical model with 2 GPUs

# Activate conda environment
source /mnt/ssd1tb/code/chipk/anaconda3/etc/profile.d/conda.sh
conda activate bbdm

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Training with Multi-SAR config using DDP (2 GPUs)
cd /mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main

# Use torchrun for distributed training with 2 GPUs
torchrun --nproc_per_node=2 --master_port=29500 main.py \
    --cfg_path configs/multisar_opera_hls.yaml \
    --save_dir experiments/multisar_opera_hls

# Notes:
# - The model uses Temporal Transformer to fuse N SAR images (2-8) into single feature map
# - Training with batch size 4 per GPU (total 8 effective batch size with accumulation)
# - Uses LPIPS loss + MSE loss + Confidence prediction (C-DiffSET)
# - Input: Multiple SAR images [B, N, 3, 256, 256]
# - Output: Single optical RGB image [B, 3, 256, 256]
# - Distributed training across 2 GPUs using PyTorch DDP
# - Using light temporal transformer to reduce memory usage

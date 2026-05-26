#!/bin/bash

# =============================================================================
# Emotion-Aware GRPO Training Script for Qwen2.5-VL
# Based on VideoChat-R1 framework
#
# IMPORTANT: Update the following paths before running:
#   - MODEL_NAME: Path to your pretrained/checkpoint model
#   - OUTDIR: Output directory for training artifacts
#   - Train/eval data paths: JSON files with training/evaluation data
#   - video_folder: Directory containing images/videos
#   - torchrun script path: grpo_emotion.py location
# =============================================================================

MODEL_NAME="path/to/your/lora-model"
export PYTHONPATH=src:$PYTHONPATH
export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR="path/to/your/output"

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=7
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If your dataset is mixed with images and videos, you need to use zero2.

DISTRIBUTED_ARGS="
    --nproc_per_node 7 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port 25808 \
"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# master_port 25808
torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 25909 \
    src/open_r1/grpo_emotion.py \
    --deepspeed training_scripts/zero3.json \
    --output_dir $OUTDIR \
    --model_name_or_path   $MODEL_NAME \
    --train_data_path "path/to/your/train.json" \
    --eval_data_path "path/to/your/eval.json" \
    --video_folder path/to/your/images \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation None \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 3000 \
    --save_total_limit 1 \
    --use_vllm true \
    --save_only_model false
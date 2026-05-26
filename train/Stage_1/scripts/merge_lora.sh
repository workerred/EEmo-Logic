#!/bin/bash

# =============================================================================
# PATHS TO UPDATE BEFORE RUNNING
# =============================================================================
# 1. MODEL_NAME: HuggingFace model ID or local path of the base model
# 2. --model-path: path to your LoRA checkpoint directory
# 3. --save-model-path: path where merged model will be saved
# =============================================================================

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path path/to/your/checkpoint \
    --model-base $MODEL_NAME  \
    --save-model-path path/to/your/merged-model \
    --safe-serialization
#!/bin/bash

clear
# LLM Qwen LoRA Fine-tuning Script
# Based on the unified fine-tuning architecture

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

# Default dataset (can be overridden)
DATASET_NAME=${1:-boolq}

echo "Running LLM ${DATASET_NAME} with Qwen + Fat LoRA"

python ./src/run_experiment.py \
    --dataset ${DATASET_NAME} \
    --model Qwen/Qwen3-8B \
    --padding_side left \
    --optimizer adamw \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --max_train_steps 2000 \
    --max_seq_length 512 \
    --logging_steps 1 \
    --ft_strategy FatLoRA \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --quantization_bit 8 \
    --dtype bfloat16 \
    --use_fast_tokenizer \
    --max_eval_samples 1000 \
    --max_fat_steps 1 \
    --fat_step 5 \
    --eval_steps 250 \
    --wandb \

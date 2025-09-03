#!/bin/bash

clear
# LLM Qwen LoRA Fine-tuning Script
# Based on the unified fine-tuning architecture

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

# Default dataset (can be overridden)
DATASET_NAME=${1:-mathqa}

echo "Running LLM ${DATASET_NAME} with Qwen + LoRA"

python ./src/run_experiment.py \
    --dataset ${DATASET_NAME} \
    --model Qwen/Qwen2-7B \
    --padding_side left \
    --optimizer adamw \
    --batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --max_train_steps 10 \
    --max_seq_length 512 \
    --logging_steps 1 \
    --ft_strategy LoRA \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --quantization_bit 4 \
    --dtype bfloat16 \
    --gradient_accumulation 4 \
    --use_fast_tokenizer \
    --wandb \

# Alternative models:
# Qwen/Qwen2-7B
# meta-llama/Llama-2-7b-hf (requires HF_AUTH_TOKEN)
# meta-llama/Llama-3.1-8B (requires HF_AUTH_TOKEN)

# Available LLM datasets:
# gsm8k aqua commonsensqa boolq addsub multiarith singleeq
# strategyqa svamp bigbench_date object_tracking coin_flip last_letters mathqa
#
# Usage examples:
# ./llm_qwen_lora.sh gsm8k
# ./llm_qwen_lora.sh aqua
# ./llm_qwen_lora.sh commonsensqa

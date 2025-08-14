#!/bin/bash

clear
# LLM Qwen LoRA Fine-tuning Script
# Based on the unified fine-tuning architecture

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Default dataset (can be overridden)
DATASET_NAME=${1:-mathqa}

echo "Running LLM ${DATASET_NAME} with Qwen + Weight LoRA"

python ./src/run_experiment.py \
    --dataset ${DATASET_NAME} \
    --model Qwen/Qwen2-7B \
    --padding_side left \
    --optimizer weight_adamw \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --max_train_steps 1000 \
    --max_seq_length 512 \
    --logging_steps 1 \
    --ft_strategy WeightLoRA \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --quantization_bit 4 \
    --dtype bfloat16 \
    --use_fast_tokenizer \
    --max_eval_samples 100 \
    --max_fat_steps 5 \
    --K 60 \
    --wandb \

# Alternative models:
# Qwen/Qwen2-7B
# meta-llama/Llama-2-7b-hf (requires HF_AUTH_TOKEN=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)
# meta-llama/Llama-3.1-8B (requires HF_AUTH_TOKEN=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)

# Available LLM datasets:
# gsm8k aqua commonsensqa boolq addsub multiarith singleeq
# strategyqa svamp bigbench_date object_tracking coin_flip last_letters mathqa
#
# Usage examples:
# ./llm_qwen_lora.sh gsm8k
# ./llm_qwen_lora.sh aqua
# ./llm_qwen_lora.sh commonsensqa

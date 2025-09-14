#!/bin/bash

clear
# LLM Qwen LoRA Fine-tuning Script
# Available datasets: gsm8k, boolq, mathqa, hella_swag, arc_challenge

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

# Default dataset (can be overridden)
DATASET_NAME=${1:-gsm8k}

echo "Running LLM ${DATASET_NAME} with Qwen + LoRA"

#for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3
for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3
do
    python ./src/run_experiment.py \
        --dataset ${DATASET_NAME} \
        --model Qwen/Qwen2-7B \
        --padding_side left \
        --optimizer mikola_drop_soap \
        --batch_size 8 \
        --gradient_accumulation_steps 1 \
        --lr $lr \
        --weight_decay 1e-5 \
        --lr_scheduler_type linear \
        --warmup_steps 100 \
        --max_train_steps 2000 \
        --max_seq_length 1024 \
        --logging_steps 1 \
        --ft_strategy Full \
        --dtype bfloat16 \
        --use_fast_tokenizer \
        --max_eval_samples 100 \
        --wandb
done

# Alternative models:
# Qwen/Qwen2-7B
# meta-llama/Llama-2-7b-hf (requires HF_AUTH_TOKEN=)
# meta-llama/Llama-3.1-8B (requires HF_AUTH_TOKEN=)

# Available LLM datasets:
# gsm8k aqua commonsensqa boolq addsub multiarith singleeq
# strategyqa svamp bigbench_date object_tracking coin_flip last_letters mathqa
#
# Usage examples:
# ./llm_qwen_lora.sh gsm8k
# ./llm_qwen_lora.sh aqua
# ./llm_qwen_lora.sh commonsensqa

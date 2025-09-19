#!/bin/bash

clear
echo "============================================"
echo "Starting LLM Optimizer Comparison: TAIA vs Muon vs AdamW"
echo "Running on Llama 2 7B with GSM8K and AQUA datasets"
echo "============================================"

export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2-7B"
PADDING_SIDE="left"
BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
WEIGHT_DECAY=1e-4
LR_SCHEDULER_TYPE="linear"
WARMUP_RATIO=0.1
MAX_TRAIN_STEPS=1000
MAX_EVAL_SAMPLES=200
MAX_SEQ_LENGTH=512
LOGGING_STEPS=10
FT_STRATEGY="full"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
QUANTIZATION_BIT=4
DTYPE="bfloat16"

seeds=(42 123 456)

datasets=(boolq hella_swag arc_challenge)

learning_rates=(2e-4 1e-4 5e-5 3e-5)

echo "Datasets: ${datasets[@]}"
echo "Model: $MODEL"
echo "Max Train Steps: $MAX_TRAIN_STEPS"
echo "Seeds: ${seeds[@]}"
echo "Learning rates to test: ${learning_rates[@]}"
echo "Fine-tuning Strategy: $FT_STRATEGY (LoRA)"
echo "============================================"

run_experiment() {
    local dataset=$1
    local optimizer=$2
    local lr=$3
    local seed=$4
    local extra_args=$5
    
    echo ""
    echo ">>> Running $optimizer on $dataset with lr=$lr, seed=$seed <<<"
    
    python ./src/run_experiment.py \
        --dataset $dataset \
        --model $MODEL \
        --padding_side $PADDING_SIDE \
        --optimizer $optimizer \
        --batch_size $BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --lr $lr \
        --weight_decay $WEIGHT_DECAY \
        --lr_scheduler_type $LR_SCHEDULER_TYPE \
        --warmup_ratio $WARMUP_RATIO \
        --max_train_steps $MAX_TRAIN_STEPS \
        --max_eval_samples $MAX_EVAL_SAMPLES \
        --max_seq_length $MAX_SEQ_LENGTH \
        --logging_steps $LOGGING_STEPS \
        --ft_strategy $FT_STRATEGY \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --quantization_bit $QUANTIZATION_BIT \
        --dtype $DTYPE \
        --seed $seed \
        --use_fast_tokenizer \
        --wandb \
        --run_prefix "llm_comp_${optimizer}_${dataset}_seed${seed}" \
        $extra_args
    
    echo ">>> Completed $optimizer on $dataset with lr=$lr, seed=$seed <<<"
}

for dataset in "${datasets[@]}"
do
    echo ""
    echo "########################################"
    echo "STARTING DATASET: $dataset"
    echo "########################################"
    
    for lr in "${learning_rates[@]}"
    do
        echo ""
        echo "========================================"
        echo "Dataset: $dataset - Learning Rate: $lr"
        echo "========================================"
        
        for seed in "${seeds[@]}"
        do
            echo ""
            echo "----------------------------------------"
            echo "Dataset: $dataset - LR: $lr - Seed: $seed"
            echo "----------------------------------------"
            
            echo ""
            echo "1/3. Testing AdamW..."
            run_experiment "$dataset" "adamw" $lr $seed ""
            
            echo ""
            echo "2/3. Testing Muon..."
            muon_args="--momentum 0.9 --ns_steps 10"
            run_experiment "$dataset" "muon" $lr $seed "$muon_args"
            
            echo ""
            echo "3/3. Testing TAIA (Madam)..."
            taia_args="--lmo spectral --precondition_type adam --momentum 0.9 --ns_steps 10"
            run_experiment "$dataset" "taia" $lr $seed "$taia_args"
            
            echo ""
            echo "----------------------------------------"
            echo "Completed all optimizers for $dataset with lr=$lr, seed=$seed"
            echo "----------------------------------------"
        done
        
        echo ""
        echo "========================================"
        echo "Completed all seeds for $dataset with lr=$lr"
        echo "========================================"
    done
    
    echo ""
    echo "########################################"
    echo "COMPLETED DATASET: $dataset"
    echo "########################################"
done

echo ""
echo "============================================"
echo "ALL LLM EXPERIMENTS COMPLETED!"
echo "============================================"
echo "Results should be available in wandb"
echo "============================================"
echo ""
echo "Summary:"
echo "- Model: $MODEL"
echo "- Datasets: ${datasets[@]}"
echo "- Optimizers: AdamW, Muon, TAIA (Madam)"
echo "- Learning rates: ${learning_rates[@]}"
echo "- Seeds: ${seeds[@]}"
echo "- Training steps: $MAX_TRAIN_STEPS"
echo "- Fine-tuning: LoRA (r=$LORA_R, alpha=$LORA_ALPHA)"
echo "============================================"

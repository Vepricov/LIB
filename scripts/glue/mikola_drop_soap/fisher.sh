clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb

#for lr in 1e-4 5e-4 1e-3 5e-3 1e-2
# 5e-5 8e-5 1e-4 2e-4 1e-4 5e-4 1e-3
for lr in 1e-4
do
    CUDA_VISIBLE_DEVICES=2 python ./src/run_experiment.py \
        --dataset sst2 \
        --model microsoft/deberta-v3-base \
        --optimizer soap \
        --init kron \
        --batch_size 16 \
        --gradient_accumulation_steps 1 \
        --lr $lr \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --max_steps 512 \
        --eval_strategy steps \
        --eval_steps 512 \
        --save_strategy no \
        --lora_r 4 \
        --ft_strategy LoRA \
        --dtype bfloat16 \
        --report_fisher_diff \
        --wandb
done

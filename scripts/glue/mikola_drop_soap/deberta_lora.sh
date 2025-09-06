clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb

for lr in 1e-4 5e-4 1e-3
do
    CUDA_VISIBLE_DEVICES=6 python ./src/run_experiment.py \
        --dataset qqp \
        --model microsoft/deberta-v3-base \
        --optimizer muon \
        --init eps \
        --batch_size 16 \
        --gradient_accumulation_steps 2 \
        --lr $lr \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --num_train_epochs 3 \
        --eval_strategy epoch \
        --save_strategy no \
        --ft_strategy LoRA \
        --dtype bfloat16 \
        --lora_r 4 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --wandb
done

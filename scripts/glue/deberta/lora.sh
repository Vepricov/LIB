clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb
CUDA_VISIBLE_DEVICES=7 python ./src/run_experiment.py \
    --dataset sst2 \
    --model microsoft/deberta-v3-base \
    --optimizer muon \
    --batch_size 32 \
    --gradient_accumulation_steps 6 \
    --lr 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --max_train_steps 5000 \
    --eval_strategy epoch \
    --save_strategy no \
    --ft_strategy LoRA \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 \
    --wandb \

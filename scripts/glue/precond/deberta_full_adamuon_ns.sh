clear

datasets=(cola mnli mrpc qnli qqp rte sst2 stsb)
lrs=(3e-4 1e-3 2e-5)

for dataset in "${datasets[@]}"; do
    for lr in "${lrs[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python ./src/run_experiment.py \
            --dataset $dataset \
            --model distilbert/distilbert-base-uncased \
            --optimizer adamuon \
            --ns_steps 6 \
            --batch_size 32 \
            --gradient_accumulation_steps 2 \
            --lr $lr \
            --weight_decay 0.1 \
            --lr_scheduler_type linear \
            --warmup_ratio 0.1 \
            --max_train_steps 10000 \
            --eval_strategy epoch \
            --save_strategy no \
            --ft_strategy Full \
            --wandb
    done
done
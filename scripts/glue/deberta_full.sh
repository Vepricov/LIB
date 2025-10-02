clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb mrpc

#for lr in 1e-4 5e-4 1e-3 5e-3 1e-2
# 5e-5 8e-5 1e-4 2e-4 1e-4 5e-4 1e-3
for lr in 3e-5 5e-5 7e-5
do
    CUDA_VISIBLE_DEVICES=4 python ./src/run_experiment.py \
        --dataset cola \
        --model microsoft/deberta-v3-base \
        --adam_rank_one \
        --optimizer dykaf \
        --init kron \
        --batch_size 16 \
        --gradient_accumulation_steps 2 \
        --lr $lr \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --update_freq 10 \
        --num_train_epochs 10 \
        --eval_strategy epoch \
        --save_strategy no \
        --ft_strategy Full \
        --dtype bfloat16 \
        --wandb # --adam_rank_one \
done

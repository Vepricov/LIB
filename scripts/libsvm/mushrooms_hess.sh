#!/usr/bin/env bash
clear
for optimizer in dykaf soap
do
    for n_samples in 4000
    do
        export CUDA_VISIBLE_DEVICES=7
        python ./src/run_experiment.py \
            --dataset mushrooms \
            --eval_runs 1 \
            --batch_size 100 \
            --n_epoches_train 5 \
            --optimizer $optimizer \
            --hidden_dim 0 \
            --init kron \
            --lr 5e-2 \
            --no_bias \
            --update_freq 10 \
            --n_samples $n_samples \
            --report_fisher_diff \
            --tune --n_epoches_tune 5 --tune_runs 40 \
            --wandb \
            # --tune --n_epoches_tune 5 --tune_runs 40 \ # --use_old_tune_params \  \ --adam_rank_one \ --tune --n_epoches_tune 5 --tune_runs 40 \
    done
done

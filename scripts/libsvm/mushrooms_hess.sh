#!/usr/bin/env bash
clear
for optimizer in mikola_drop_soap
do
    for n_samples in 6000
    do
        export CUDA_VISIBLE_DEVICES=3
        python ./src/run_experiment.py \
            --dataset mushrooms \
            --eval_runs 1 \
            --n_epoches_train 5 \
            --n_epoches_tune 5 \
            --tune_runs 40 \
            --optimizer $optimizer \
            --adam_rank_one \
            --hidden_dim 0 \
            --init kron \
            --no_bias \
            --update_freq 10 \
            --tune \
            --n_samples $n_samples \
            --report_fisher_diff \
            --wandb # --use_old_tune_params \ --tune \
    done
done

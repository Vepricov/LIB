#!/usr/bin/env bash
clear
for uf in 10
do
    export CUDA_VISIBLE_DEVICES=3
    python ./src/run_experiment.py \
        --dataset mushrooms \
        --eval_runs 1 \
        --n_epoches_train 5 \
        --tune_runs 40 \
        --optimizer mikola_drop_soap \
        --hidden_dim 0 \
        --init eps \
        --no_bias \
        --eps 1e-40 \
        --update_freq $uf \
        --use_old_tune_params \
        --n_samples 1300 \
        --report_fisher_diff \
        --wandb # --use_old_tune_params \ --tune \
done

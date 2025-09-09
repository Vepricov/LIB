#!/usr/bin/env bash
clear
for uf in 10
do
    export CUDA_VISIBLE_DEVICES=5
    python ./src/run_experiment.py \
        --dataset cifar10 \
        --optimizer mikola_drop_soap \
        --eval_runs 1 \
        --init eps \
        --n_epoches_train 20 \
        --lr 1e-3 \
        --tune_runs 20 \
        --n_epoches_tune 1 \
        --report_fisher_diff \
        --wandb \
        --update_freq $uf \
        # --use_old_tune_params --momentum 0.95 --weight_decay 1e-5 --tune --report_fisher_diff \\\
done

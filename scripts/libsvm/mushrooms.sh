#!/usr/bin/env bash
clear
for optimizer in mikola_drop_soap_old
do
    export CUDA_VISIBLE_DEVICES=3
    python ./src/run_experiment.py \
        --dataset mushrooms \
        --eval_runs 1 \
        --n_epoches_train 5 \
        --tune_runs 40 \
        --optimizer $optimizer \
        --hidden_dim 10 \
        --no_bias \
        --use_old_tune_params \
        --weight_init uniform \
        --momentum 0.9 \
        --eps 1e-40 \
        --report_fisher_diff \
        --wandb \
        # --wandb # --use_old_tune_params \ --tune \
done

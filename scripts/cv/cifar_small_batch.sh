#!/usr/bin/env bash
clear
export CUDA_VISIBLE_DEVICES=7
python ./src/run_experiment.py \
    --dataset cifar10 --batch_size 32 \
    --eval_runs 1 \
    --n_epochs_train 20 \
    --optimizer tensor_dykaf \
    --verbose \
    --use_old_tune_params \
    --wandb \

# --use_old_tune_params --tune --momentum 0.95 --weight_decay 1e-5 -
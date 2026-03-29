#!/usr/bin/env bash
clear
export CUDA_VISIBLE_DEVICES=1
python ./src/run_experiment.py \
    --dataset cifar10 \
    --eval_runs 1 \
    --n_epochs_train 20 \
    --lr 0.0001 \
    --optimizer tensor_dykaf \
    --verbose \
    --wandb

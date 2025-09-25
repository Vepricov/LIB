clear

# datasets=(cola mnli mrpc qnli qqp rte sst2 stsb)
datasets=(cola rte)
lrs=(3e-4 1e-3 2e-5)

for dataset in "${datasets[@]}"; do
  for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python ./src/run_experiment.py \
      --dataset $dataset \
      --model distilbert/distilbert-base-uncased \
      --optimizer muon \
      --batch_size 16 \
      --gradient_accumulation_steps 2 \
      --lr $lr \
      --lr_scheduler_type linear \
      --warmup_ratio 0.1 \
      --max_train_steps 10000 \
      --eval_strategy epoch \
      --save_strategy no \
      --ft_strategy LoRA \
      --dtype bfloat16 \
      --lora_r 4 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --wandb
  done
  done

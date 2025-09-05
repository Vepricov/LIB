# 🚀 Optimization Library (LIB)

This library provides a comprehensive framework for experimenting with various optimization algorithms across different machine learning tasks. The library supports multiple datasets and models, with a special focus on optimization strategies.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Available Datasets](#available-datasets)
- [Argument System](#argument-system)
- [Available Optimizers](#available-optimizers)
- [Arguments Reference](#arguments-reference)
  - [Core Arguments](#core-arguments)
  - [Optimizer Arguments](#optimizer-arguments)
  - [LIBSVM Arguments](#libsvm-arguments)
  - [Computer Vision Arguments](#computer-vision-arguments)
  - [Fine-Tuning Arguments](#fine-tuning-arguments)
- [Scripts](#scripts)
- [Examples](#examples)

## 🔍 Overview

This library allows researchers and practitioners to:
- 📊 Benchmark various optimization algorithms on standard datasets
- 🔄 Experiment with parameter-efficient fine-tuning strategies
- 📈 Compare performance across different tasks and models
- 🧩 Easily extend the framework with custom optimizers and models

## 🛠️ Setup

### Environment Setup

You can create the required environment using the `venv` and `requirements.txt` files:

```bash
python -m venv optim_venv
source optim_venv/bin/activate
pip install -r requirements.txt
```

This will install all necessary dependencies including PyTorch, transformers, and other required libraries.

## 📁 Project Structure

The project is organized into several key directories:

- `src/` - Core source code
  - `config.py` - Main configuration parser
  - `libsvm/` - LIBSVM datasets and models
  - `cv/` - Computer Vision datasets and models
  - `fine_tuning/` - Fine-tuning strategies for pre-trained models
  - `optimizers/` - Implementation of various optimization algorithms
- `scripts/` - Ready-to-use scripts for running experiments
- `data/` - Default location for datasets
- `notebooks/` - Example notebooks

## 📊 Available Datasets

The library supports the following dataset categories:

### 📑 LIBSVM Datasets
Standard datasets for binary and multi-class classification:
- mushrooms
- binary
- and other standard LIBSVM datasets

### 🖼️ Computer Vision (CV) Datasets
Image classification datasets:
- cifar10
- and other CV datasets

### 🔤 Fine-Tuning Datasets
Datasets for natural language tasks:

#### GLUE Datasets
- cola
- mnli
- mrpc
- qnli
- qqp
- rte
- sst2
- stsb
- wnli

#### 🤖 LLM Datasets
- Various datasets for large language model fine-tuning

## ⚙️ Argument System

The library uses a hierarchical argument system:

1. **Base Arguments** (`config.py`): Core arguments applicable to all experiments
2. **Task-Specific Arguments**: Extended arguments for specific tasks
   - LIBSVM Arguments (`libsvm/config_libsvm.py`)
   - Computer Vision Arguments (`cv/config_cv.py`)
   - Fine-Tuning Arguments (`fine_tuning/config_ft.py`)

Arguments are processed hierarchically. When running an experiment:
1. Base arguments are loaded first
2. Based on the selected dataset, task-specific arguments are added
3. If a configuration file is specified with `--config_name`, its values override defaults

## 🧮 Available Optimizers

Optimizers are implemented as individual Python files. The library currently supports:

- `adamw` - AdamW optimizer with weight decay
- `soap` - SOAP (Second Order Approximation) optimizer
- `shampoo` - Shampoo optimizer for efficient second-order optimization
- `sgd` - Stochastic Gradient Descent
- `muon` - MUON optimizer

## 📝 Arguments Reference

### 🔧 Core Arguments

#### Problem Arguments
- `--dataset`: Dataset name (required)
- `--config_name`: Name of the configuration file for your problem (optional)
- `--optimizer`: Name of the optimizer to use (choices: adamw, soap, shampoo, sgd, adam-sania, muon)

#### Training Arguments
- `--batch_size` / `--per_device_train_batch_size`: Batch size for training (default: 8)
- `--n_epoches_train`: How many epochs to train (default: 1)
- `--eval_runs`: Number of re-training model with different seeds (default: 1)
- `--dtype`: Default type for torch (default: None)
- `--use_old_tune_params`: Use already tuned parameters (flag)

#### Wandb Arguments
- `--wandb`: Enable Weights & Biases logging (flag)
- `--run_prefix`: Run prefix for the experiment run name
- `--wandb_project`: W&B project name (default: "OPTIM_TEST")
- `--verbose`: Print training results in the terminal (flag)
- `--seed`: Random seed (default: 18)

#### Saving Paths
- `--results_path`: Path to save the results of the experiment (default: "results_raw")
- `--data_path`: Path to save the datasets (default: "data")

#### Optimizer Arguments
- `--lr` / `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay` / `-wd`: Weight decay (default: 1e-5)

##### For Adam-based Optimizers (adamw, adam-sania, etc.)
- `--beta1`: First momentum (default: 0.9)
- `--beta2`: Second momentum (default: 0.999)
- `--eps`: Epsilon for Adam (default: 1e-8)

##### For Momentum-based Optimizers (shampoo, sgd, muon)
- `--momentum`: First momentum (default: 0.9)

##### For SOAP Optimizer
- `--shampoo_beta`: Momentum for SOAP. If -1, equals to beta2 (default: -1)

##### For Shampoo, SOAP, Diag-HVP
- `--update_freq`: Frequency to update Q for Shampoo and SOAP (default: 1)

##### For MUON
- `--ns_steps`: Number of the NS steps algorithm (default: 10)
- `--adamw_lr`: Learning rate for Adam in MUON (default: None)

### 📊 LIBSVM Arguments

#### Dataset Arguments
- `--scale`: Use or not scaling for the LIBSVM datasets (flag)
- `--scale_bound`: Scaling ~exp[U(-scale_bound, scale_bound)] (default: 20)
- `--rotate`: Use or not rotating for the LIBSVM datasets (flag)

#### Model Arguments
- `--model`: Model name (default: "linear-classifier", choices: ["linear-classifier"])
- `--hidden_dim`: Hidden dimension of linear classifier (default: 10)
- `--no_bias`: No bias in the FCL of the linear classifier (flag)
- `--weight_init`: Initial weights of the linear classifier (default: "uniform", choices: ["zeroes", "uniform", "bad_scaled", "ones", "zero/uniform"])

#### Training Arguments
The LIBSVM tasks set the following defaults:
- `batch_size = 128`
- `n_epoches_train = 2`
- `eval_runs = 3`
- `dtype = "float64"`

#### Tuning Arguments
- `--tune`: Tune parameters with Optuna (flag)
- `--n_epoches_tune`: How many epochs to tune with Optuna (default: 1)
- `--tune_runs`: Number of Optuna steps (default: 20)
- `--tune_path`: Path to save the tuned parameters (default: "tuned_params")

### 🖼️ Computer Vision Arguments

#### Dataset Arguments
- `--not_augment`: Disable data augmentation (flag)

#### Model Arguments
- `--model`: Model name (default: "resnet20", choices: ["resnet20", "resnet32", "resnet44", "resnet56"])

#### Training Arguments
The CV tasks set the following defaults:
- `batch_size = 64`
- `n_epoches_train = 10`
- `eval_runs = 5`

#### Tuning Arguments
- `--tune`: Tune parameters with Optuna (flag)
- `--n_epoches_tune`: How many epochs to tune with Optuna (default: 5)
- `--tune_runs`: Number of Optuna steps (default: 100)
- `--tune_path`: Path to save the tuned parameters (default: "tuned_params")

### 🔄 Fine-Tuning Arguments

#### Dataset Arguments
- `--dataset_config`: Dataset config name
- `--dataset_path`: Path to dataset for LLM tasks
- `--max_seq_length`: Maximum total input sequence length after tokenization (default: 128)
- `--pad_to_max_length`: Pad all samples to max_seq_length (flag, default: True)
- `--max_train_samples`: Truncate number of training examples
- `--max_eval_samples` / `--max_val_samples`: Truncate number of validation examples
- `--max_test_samples`: Truncate number of test examples
- `--train_file`: CSV or JSON file containing training data
- `--validation_file`: CSV or JSON file containing validation data
- `--test_file`: CSV or JSON file containing test data
- `--preprocessing_num_workers` / `--workers`: Number of processes for preprocessing
- `--overwrite_cache`: Overwrite cached training and evaluation data (flag)

#### Model Arguments
- `--model`: Path to pretrained model or HuggingFace model identifier
- `--config`: Pretrained config name or path
- `--cache_dir`: Where to store downloaded pretrained models
- `--tokenizer`: Pretrained tokenizer name or path
- `--padding_side`: Padding side for tokenization (default: "right", choices: ["left", "right"])
- `--use_fast_tokenizer`: Use fast tokenizer (flag, default: True)
- `--model_revision`: Specific model version (default: "main")
- `--use_auth_token`: Use token from transformers-cli login (flag)
- `--quant_bit` / `--quantization_bit`: Number of bits for quantization

#### Training Arguments
- `--do_not_train`: Skip training (flag)
- `--do_not_eval`: Skip validation (flag)
- `--do_predict`: Do prediction (flag)
- `--eval_batch_size` / `--per_device_eval_batch_size`: Batch size for evaluation (default: 32)
- `--max_steps_train` / `--max_train_steps` / `--max_steps`: Maximum training steps (default: -1)
- `--lr_scheduler_type`: Scheduler for optimizer (default: "linear")
- `--grad_acc_steps` / `--gradient_accumulation_steps` / `--gradient_accumulation`: Gradient accumulation steps (default: 6)
- `--warmup_steps`: Number of warmup steps (default: 100)
- `--warmup_ratio`: Ratio of total steps for warmup (default: 0.1)
- `--eval_strategy` / `--evaluation_strategy`: Strategy to evaluate model (default: "epoch")
- `--eval_steps`: Steps between evaluations when eval_strategy="steps"
- `--logging_steps`: How often to print train loss (default: 1)
- `--save_strategy`: Strategy to save checkpoints (default: "no")
- `--save_steps`: Steps between saves when save_strategy="steps" (default: 500)
- `--save_every`: Save model every N steps (default: 500)

#### PEFT Arguments
- `--ft_strategy`: PEFT strategy to use (default: "LoRA")
- `--lora_r`: Rank for LoRA adapters (default: 8)
- `--lora_alpha`: Scaling of LoRA adapters (default: 32)
- `--lora_dropout`: Dropout of LoRA adapters (default: 0.05)

Fine-tuning tasks set the following defaults:
- `batch_size = 8`
- `n_epoches_train = 3`
- `eval_runs = 1`
- `dtype = "float16"`

## 📜 Scripts

The `scripts/` directory contains ready-to-use scripts for running common experiments. Make the scripts executable before using them:

```bash
chmod +x ./scripts/**/*.sh
```

### GLUE Scripts

#### DeBERTa Scripts
Located at `scripts/glue/deberta/`

- **lora.sh**: Fine-tunes Microsoft DeBERTa-v3-base on GLUE tasks using LoRA
  ```bash
  ./scripts/glue/deberta/lora.sh
  ```
#### Llama3 Scripts
Located at `scripts/glue/llama3/`

- **lora.sh**: Fine-tunes Meta-Llama-3.1-8B on GLUE tasks using LoRA
  ```bash
  ./scripts/glue/llama3/lora.sh
  ```

### LLM Scripts

Located at `scripts/llm/`

- **qwen.sh**: Fine-tunes Qwen2-7B model on various LLM tasks using LoRA
  ```bash
  ./scripts/llm/qwen.sh [dataset_name]
  ```
  Supported dataset names include: gsm8k, aqua, commonsensqa, boolq, mathqa, and more.

  Example: `./scripts/llm/qwen.sh gsm8k`

## 🧪 Examples

The main entry point for running experiments is `src/run_experiment.py`. Here are some examples of how to use it:

### Basic Example with Command Line Arguments

```bash
python ./src/run_experiment.py \
    --dataset mushrooms \
    --optimizer adamw \
    --lr 0.001 \
    --weight_decay 0.01 \
    --seed 42 \
    --verbose
```

### Using JSON Configuration Files

You can use JSON configuration files to set multiple parameters at once. For example, using `libsvm/configs/basic.json`:

```bash
python ./src/run_experiment.py \
    --dataset mushrooms \
    --optimizer adamw \
    --config_name basic
```

The configuration file `basic.json` contains:
```json
{
    "batch_size": 128,
    "n_epoches_train": 2,
    "eval_runs": 3,
    "n_epoches_tune": 1,
    "tune_runs": 20,
    "dtype": "float32"
}
```

### Fine-tuning a BERT Model on GLUE Tasks with LoRA

```bash
python ./src/run_experiment.py \
    --dataset sst2 \
    --model bert-base-uncased \
    --optimizer adamw \
    --ft_strategy LoRA \
    --lora_r 16 \
    --batch_size 16 \
    --eval_strategy steps \
    --eval_steps 100 \
    --wandb
```

### Training a ResNet on CIFAR-10 with Shampoo Optimizer

```bash
python ./src/run_experiment.py \
    --dataset cifar10 \
    --model resnet56 \
    --optimizer shampoo \
    --update_freq 10 \
    --n_epoches_train 20 \
    --wandb
```

### Using Scripts with Different Parameters

You can modify and run the provided scripts with custom parameters:

```bash
# First make scripts executable
chmod +x ./scripts/**/*.sh

# Run GLUE fine-tuning with DeBERTa
CUDA_VISIBLE_DEVICES=0 ./scripts/glue/deberta/lora.sh

# Run LLM fine-tuning with Qwen on the gsm8k dataset
./scripts/llm/qwen.sh gsm8k
```

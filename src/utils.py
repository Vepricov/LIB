import random
import torch
import os
import peft
import numpy as np


def set_global_seed(seed=18):
    def seed_worker(worker_seed):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return g, seed_worker


def print_trainable_params(model, verbose=True):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for param in model.buffers():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    train_proportion = 100 * trainable_params / all_param

    if verbose:
        print(
            f"trainable params: {trainable_params / 1_000_000:.2f}M || all params: {all_param / 1_000_000_000:.2f}B || trainable%: {train_proportion:.2f}%"
        )

    return all_param, trainable_params, 100 * trainable_params / all_param


def count_atapters(model, peft_type):
    if peft_type in ["LoRA", "ADALoRA", "DoRA", "rsLoRA", "WeightLoRA", "RandLoRA"]:
        adapter_name = "lora_A"
    elif peft_type == "LoKR":
        adapter_name = "lokr_w1"
    elif peft_type == "LoHA":
        adapter_name = "hada_w1_a"
    elif peft_type == "VERA":
        adapter_name = "vera_lambda_b"
    elif peft_type == "Full":
        adapter_name = None
    else:
        raise ValueError(f"Wrong peft_type: {peft_type}")

    num_adapters = None
    if adapter_name is not None:
        num_adapters = 0
        for name, param in model.named_parameters():
            if adapter_name in name and param.requires_grad:
                num_adapters += 1

    return num_adapters


def get_run_name(args, parser, tuning=False):
    key_args = ["optimizer", "model", "dataset"]
    ignore_args = [
        "verbose",
        "seed",
        "run_prefix",
        "wandb_project",
        "results_path",
        "tune_path",
        "data_path",
        "wandb",
        "tune",
        "use_old_tune_params",
        "eval_runs",
        "augment",
        "run_name",
    ]
    ignore_args_tuning = [
        "lr",
        "weight_decay",
        "rotate",
        "scale_bound",
        "weight_init",
        "ns_steps",
        "report_fisher_diff",
        "n_epoches_train",
        "n_samples",
    ]
    # Get the default values
    defaults = vars(parser.parse_args([]))

    # Generate the prefix with key arguments
    if not tuning:
        print("~~~~~~~~~~~~~~~ KEY ARGUMENTS ~~~~~~~~~~~~~~~")
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if not tuning:
                print(f"{key:<20} {value}")
            # if value != defaults[key]:
            prefix_parts.append(f"{value}")

    prefix = "_".join(prefix_parts)

    # Generate the rest of the string with non-default arguments
    if not tuning:
        print("~~~~~~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~~~~~~")
    non_default_parts = []
    for key, value in vars(args).items():
        if not tuning:
            print(f"{key:<40} {value}")
        if key in ignore_args:
            continue
        if key in ignore_args_tuning and tuning:
            continue
        if key not in key_args:
            if defaults[key] != value:
                if type(value) == bool:
                    non_default_parts.append(f"{key}")
                else:
                    non_default_parts.append(f"{key}-{value}")

    non_default_string = "__".join(non_default_parts)

    if args.run_prefix is not None and not tuning:
        prefix = args.run_prefix + "__" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


def get_peft_arguments(args):
    if args.ft_strategy.lower() == "lora":
        peft_args = peft.LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.ft_strategy.lower() == "lokr":
        peft_args = peft.LoKrConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.ft_strategy.lower() == "loha":
        peft_args = peft.LoHaConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.ft_strategy.lower() == "vera":
        peft_args = peft.VeraConfig(r=args.lora_r, vera_dropout=args.lora_dropout)
    elif args.ft_strategy.lower() == "adalora":
        peft_args = peft.AdaLoraConfig(
            target_r=args.lora_r,
        )
    elif args.ft_strategy.lower() == "dora":
        peft_args = peft.LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_dora=True,
        )
    elif args.ft_strategy.lower() == "rslora":
        peft_args = peft.LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=True,
        )
    elif args.ft_strategy.lower() == "weightlora":
        peft_args = peft.WeightLoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.ft_strategy.lower() == "full":
        peft_args = peft.LoraConfig(
            target_modules=None,
        )
    else:
        raise ValueError(f"Incorrect FT type {args.ft_strategy}!")

    if "deberta" in args.model.lower():
        peft_args.target_modules = [
            "query_proj",
            "key_proj",
            "value_proj",
            # "intermediate.dense",
            # "output.dense",
        ]
    elif "bart" in args.model.lower():
        peft_args.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        ]
    elif "llama" in args.model.lower():
        peft_args.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lm_head",
        ]
    elif "qwen" in args.model.lower():
        peft_args.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            "down_proj",
            # "lm_head",
        ]
    elif "distilbert-base" in args.model.lower():
        peft_args.target_modules = [
            "q_lin",
            "k_lin",
            "v_lin",
        ]
    elif "flan" in args.model.lower() or "t5" in args.model.lower():
        peft_args.target_modules = [
            "v",
            "o",
            "q",
            "o",
            "wi",
        ]
    else:
        raise ValueError(f"Pass target_modules to your model {args.model}")
    return peft_args


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)

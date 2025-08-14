import gc
import json
import torch
import wandb
from transformers import Trainer, TrainingArguments
from utils_glue import glue_preprocess
import peft
import warnings
import utils

DATASETS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

warnings.filterwarnings("ignore")

import utils
from optimizers.main import get_optimizer


def main(args):
    ################# Model, Tokenizer and Dataset Downloading #################
    (
        train_dataset,
        eval_dataset,
        test_dataset,
        datasets,
        data_collator,
        compute_metrics,
        model,
        tokenizer,
    ) = glue_preprocess(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################### PEFT Adapters ##############################
    all_params_before_peft, _, _ = utils.print_trainable_params(model, verbose=False)
    peft_args = utils.get_peft_arguments(args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)
    num_peft_adapters = utils.count_atapters(model, args.ft_strategy)
    args.label_names = ["labels"]  # peft and compute_metrics() problem
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    if args.use_old_tune_params:
        f_name = "./glue_experiment/tuned_params.json"
        with open(f_name) as f:
            tuned_params = json.load(f)
        # [TODO] add more tuned params
        lr = tuned_params[args.model][args.dataset][args.ft_strategy]["lr"]
        args.lr = lr
    optimizer = get_optimizer(args, model)
    ############################### Wandb Saves ################################
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            tags=[args.model, args.dataset, args.optimizer],
            name=args.run_name,
            config=args,
        )
        params_info_dict = {
            "num_peft_adapters": num_peft_adapters,
            "all_params_before_peft": all_params_before_peft,
        }
        (
            params_info_dict["all_params"],
            params_info_dict["trainable_params"],
            params_info_dict["train_proportion"],
        ) = utils.print_trainable_params(model)
        params_info_dict["peft_params"] = (
            params_info_dict["all_params"] - all_params_before_peft
        )
        params_info_dict["peft_proportion"] = (
            params_info_dict["peft_params"] / params_info_dict["all_params"] * 100
        )
        wandb.config.update(params_info_dict, allow_val_change=True)
    ############################# Training #####################################
    training_args = TrainingArguments(
        output_dir=f"./src/fine_tuning/glue/{args.results_path}/{args.run_name}",
        do_train=not args.do_not_train,
        do_eval=not args.do_not_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=(
            args.eval_batch_size if args.eval_batch_size else args.batch_size
        ),
        gradient_accumulation_steps=args.grad_acc_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        num_train_epochs=args.n_epoches_train,
        max_steps=args.max_steps_train,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_steps else args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_dir=f"./src/fine_tuning/glue/{args.results_path}/{args.run_name}",
        run_name=args.run_name,
        report_to=["wandb"] if args.wandb else ["none"],
        label_names=["labels"],  # peft and compute_metrics() problem
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if not args.do_not_train else None,
        eval_dataset=eval_dataset if not args.do_not_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler],
    )

    if not args.do_not_train:
        for i, seed in enumerate(range(args.eval_runs)):
            print(f"~~~~~~~~~~~~~~~ TRAIN RUN {i+1}/{args.eval_runs} ~~~~~~~~~~~~~~~")
            args.seed = seed
            trainer.seed = seed
            utils.set_global_seed(seed)
            train_result = trainer.train()
            train_metrics = train_result.metrics
            max_train_samples = (
                args.max_train_samples
                if args.max_train_samples is not None
                else len(train_dataset)
            )
            train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            train_metrics["train_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
            train_metrics["train_runtime"] /= 60
            if args.ft_strategy == "WeightLoRA":
                remain_adapters = utils.count_remain_adapters(args, model)
            train_metrics = train_metrics | remain_adapters
            trainer.save_model()

            trainer.log_metrics("train", train_metrics)
            trainer.save_metrics("train", train_metrics)
            trainer.save_state()

        if args.wandb:
            wandb.config.update(train_metrics, allow_val_change=True)
    ################################ Evaluation ################################
    if not args.do_not_eval:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        print(f"~~~~~~~~~~~~~~~ EVALUATION ~~~~~~~~~~~~~~~")
        tasks = [args.dataset]
        eval_datasets = [eval_dataset]
        if args.dataset == "mnli":
            tasks.append("mnli-mm")
            validation_mismatched = datasets["validation_mismatched"]
            if args.max_eval_samples is not None:
                validation_mismatched = validation_mismatched.select(
                    range(args.max_eval_samples)
                )
            eval_datasets.append(validation_mismatched)

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = (
                args.max_eval_samples
                if args.max_eval_samples is not None
                else len(eval_dataset)
            )
            eval_metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            trainer.log_metrics("Eval_%s" % task, eval_metrics)
            trainer.save_metrics("Eval_%s" % task, eval_metrics)

        if "eval_runtime" in eval_metrics.keys():
            eval_metrics["eval_runtime"] /= 60
        if args.wandb:
            wandb.config.update(eval_metrics, allow_val_change=True)
    ################################# Testing ##################################
    if args.do_predict:
        print(f"~~~~~~~~~~~~~~~ PREDICTIONS ~~~~~~~~~~~~~~~")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [args.dataset]
        test_datasets = [eval_dataset]
        if args.dataset == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["validation_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            max_samples = (
                args.max_eval_samples
                if args.max_eval_samples is not None
                else len(test_dataset)
            )
            metrics["test_samples"] = min(max_samples, len(test_dataset))
            trainer.log_metrics("Test_%s" % task, metrics)
            trainer.save_metrics("Test_%s" % task, metrics)
            if args.wandb:
                wandb.config.update(metrics, allow_val_change=True)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(None)

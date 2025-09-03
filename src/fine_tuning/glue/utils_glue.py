import numpy as np
import logging
import torch
from datasets import load_dataset, load_metric
from peft import prepare_model_for_kbit_training
from transformers import (
    PretrainedConfig,
    EvalPrediction,
    DataCollatorWithPadding,
    default_data_collator,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def glue_preprocess(args):
    # Downloading and loading a dataset from the hub.
    datasets = load_dataset("glue", args.dataset)

    is_regression = args.dataset == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.dataset]

    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    config = AutoConfig.from_pretrained(
        args.config if args.config else args.model,
        num_labels=num_labels,
        finetuning_task=args.dataset,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.model,
        cache_dir=args.cache_dir,
        use_fast=False,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    if "llama" in args.model:
        if torch.cuda.get_device_capability()[0] >= 8:
            attn_implementation = "flash_attention_2"
            torch_dtype = torch.bfloat16
        else:
            attn_implementation = "eager"
            torch_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:  # for deberta
        attn_implementation = "eager"
        torch_dtype = torch.float32
        bnb_config = None
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        from_tf=bool(".ckpt" in args.model),
        config=config,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    if args.model in ["meta-llama/Meta-Llama-3.1-8B"]:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.dataset is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    datasets = datasets.map(
        preprocess_function, batched=True, load_from_cache_file=not args.overwrite_cache
    )
    train_dataset = None
    if not args.do_not_train:
        if "train" not in datasets:
            raise ValueError("Training requires a train dataset")
        train_dataset = datasets["train"]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

    eval_dataset = None
    if not args.do_not_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("Evaluation requires a validation dataset")
        eval_dataset = datasets[
            "validation_matched" if args.dataset == "mnli" else "validation"
        ]
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    test_dataset = None
    if args.do_predict or args.dataset is not None or args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if args.dataset == "mnli" else "test"]
        if args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(args.max_test_samples))

    # Get the metric function
    metric = load_metric("glue", args.dataset)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if args.dataset is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.dtype == "bfloat16":
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return (
        train_dataset,
        eval_dataset,
        test_dataset,
        datasets,
        data_collator,
        compute_metrics,
        model,
        tokenizer,
    )

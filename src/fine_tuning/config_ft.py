from termcolor import colored

def set_arguments_ft(parser):
    ### Dataset Arguments
    parser.add_argument(
        "--dataset_config",
        default=None,
        type=str,
        help="Dataset config name",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        help="Path to dataset for LLM tasks",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        default=True,
        help="Whether to pad all samples to `max_seq_length`.",
    )
    parser.add_argument(
        "--max_train_samples",
        default=None,
        type=int,
        help="Truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        "--max_val_samples",
        default=None,
        type=int,
        help="Truncate the number of validation examples to this value if set.",
    )
    parser.add_argument(
        "--max_test_samples",
        default=None,
        type=int,
        help="Truncate the number of test examples to this value if set.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        default=None,
        type=str,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="A csv or a json file containing the test data.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        "--workers",
        default=None,
        type=int,
        help="The number of processes to use for the preprocessing",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        default=False,
        help="Overwrite the cached training and evaluation data",
    )

    # SQuAD specific
    if False:  # [TODO] fix squad
        parser.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks.",
        )
        parser.add_argument(
            "--version_2_with_negative",
            action="store_true",
            help="If true, some of the examples do not have an answer.",
        )
        parser.add_argument(
            "--null_score_diff_threshold",
            default=0.0,
            type=float,
            help="The threshold used to select the null answer",
        )
        parser.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate when looking for an answer.",
        )
        parser.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help="The maximum length of an answer that can be generated.",
        )

    # NLG specific
    if False:  # [TODO] add nlg
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length for source text after tokenization.",
        )
        parser.add_argument(
            "--max_target_length",
            default=128,
            type=int,
            help="The maximum total sequence length for target text after tokenization.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=None,
            type=int,
            help="The maximum total sequence length for validation target text after tokenization.",
        )
        parser.add_argument(
            "--source_prefix",
            default="",
            type=str,
            help="A prefix to add before every source text (useful for T5 models).",
        )
        parser.add_argument(
            "--text_column",
            default=None,
            type=str,
            help="The name of the column in the datasets containing the full texts (for summarization).",
        )
        parser.add_argument(
            "--summary_column",
            default=None,
            type=str,
            help="The name of the column in the datasets containing the summaries (for summarization).",
        )
        parser.add_argument(
            "--ignore_pad_token_for_loss",
            action="store_true",
            default=True,
            help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
        )
        parser.add_argument(
            "--num_beams",
            default=None,
            type=int,
            help="Number of beams to use for evaluation.",
        )
        parser.add_argument(
            "--predict_with_generate",
            action="store_true",
            default=True,
            help="Whether to use generate to calculate generative metrics.",
        )

    ### Model Arguments
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="Pretrained config name or path if not the same as model",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model",
    )
    parser.add_argument(
        "--padding_side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Padding side for tokenization.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        default=False,
        help="Will use the token generated when running `transformers-cli login`",
    )
    parser.add_argument(
        "--quant_bit",
        "--quantization_bit",
        default=None,
        type=int,
        help="The number of bits to quantize the model to. If None, the model will not be quantized.",
    )

    ### Training Arguments
    parser.add_argument(
        "--do_not_train", action="store_true", default=False, help="Do training or not"
    )
    parser.add_argument(
        "--do_not_eval", action="store_true", default=False, help="Do validation or not"
    )
    parser.add_argument(
        "--do_predict", action="store_true", default=False, help="Do prediction or not"
    )
    parser.add_argument(
        "--eval_batch_size",
        "--per_device_eval_batch_size",
        default=32,
        type=int,
        help="Batch size for evaluation. If None, then it equals to batch_size.",
    )
    parser.add_argument(
        "--max_steps_train",
        "--max_train_steps",
        "--max_steps",
        default=-1,
        type=int,
        help="Maximum number of training steps (overrides num n_epoches_train)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        type=str,
        help="Scheduler for optimizer",
    )
    parser.add_argument(
        "--grad_acc_steps",
        "--gradient_accumulation_steps",
        "--gradient_accumulation",
        default=6,
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Number of warmup steps for learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Ratio of total training steps for warmup",
    )
    parser.add_argument(
        "--logging_steps", default=1, type=int, help="How often print train loss"
    )
    parser.add_argument(
        "--eval_strategy",
        "--save_strategy",
        "--val_strategy",
        "--evaluation_strategy",
        default="no",
        type=str,
        help="Strategy to save model checkpoints",
    )
    parser.add_argument(
        "--eval_steps",
        "--save_steps",
        "--val_steps",
        default=None,
        type=int,
        help="Number of steps between saves (if save_strategy==steps)",
    )
    parser.add_argument(
        "--metric_for_best_model",
        default="loss",
        type=str,
        choices=["loss", "accuracy", "f1", "precision", "recall"],
        help="Metric to use for best model selection",
    )

    ### PEFT Arguments
    # parser.add_argument(
    #     "--ft_strategy", default="LoRA", type=str, help="What PEFT strategy to use"
    # )
    parser.add_argument(
        "--lora_r",
        default=8,
        type=int,
        help="Rank for LoRA and LoRA-like PEFT adapters",
    )
    parser.add_argument(
        "--lora_alpha",
        default=32,
        type=int,
        help="Scaling of LoRA and LoRA-like PEFT adapters",
    )
    parser.add_argument(
        "--lora_dropout",
        default=0.05,
        type=float,
        help="Dropout of LoRA and LoRA-like PEFT adapters",
    )

    ### Override some default values from the main parser
    parser.set_defaults(batch_size=8, n_epoches_train=3, eval_runs=1, dtype="float16")

    return parser

def print_warnings_ft(args):
    if args.eval_steps is not None and args.eval_strategy != "steps":
        print(
            colored(
                "~~~~~~~~~~~~~~~ WARNING: EVAL STRATEGY SET TO STEPS ~~~~~~~~~~~~~~~",
                "yellow",
            )
        )
        line = f"you pass eval_steps={args.eval_steps} (!= None as in the defaults), so we set eval_strategy=steps"
        print(colored(line, "yellow"))
        args.eval_strategy = "steps"

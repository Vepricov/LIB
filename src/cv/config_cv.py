def set_arguments_cv(parser):
    ### Dataset Arguments
    parser.add_argument(
        "--not_augment", action="store_true", help="To not use the augmentation"
    )

    ### Model Arguments
    parser.add_argument(
        "--model",
        default="resnet20",
        choices=["resnet20", "resnet32", "resnet44", "resnet56"],
        help="Model name",
    )

    ### Training Arguments (change defaults)
    parser.set_defaults(batch_size=64, n_epochs_train=10, eval_runs=5)

    ### Tuning Arguments
    parser.add_argument("--tune", action="store_true", help="Tune params")
    parser.add_argument(
        "--n_epoches_tune",
        default=5,
        type=int,
        help="How many epochs to tune with optuna",
    )
    parser.add_argument(
        "--tune_runs", default=100, type=int, help="Number of optuna steps"
    )
    parser.add_argument(
        "--tune_path", default="tuned_params", help="Path to save the tuned params"
    )

    return parser

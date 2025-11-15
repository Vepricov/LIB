from argparse import ArgumentParser
from termcolor import colored
import os, json

from libsvm.config_libsvm import set_arguments_libsvm
from libsvm.main_libsvm import DATASETS as LIBSVM_DATASETS

from cv.config_cv import set_arguments_cv
from cv.main_cv import DATASETS as CV_DATASETS

from llm.config_llm import set_arguments_llm
from llm.main_llm import DATASETS as LLM_DATASETS
from llm.problems_llm import SQUAD_DATASETS, NLG_DATASETS


def parse_args():
    parser1 = ArgumentParser(description="Main Experiment")

    ##### Problem Arguments #####
    parser1.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Dataset name",
    )
    parser1.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Name of the configuration file for your problem. If None, we will use default settings from the file set_arguments_{problem}.py",
    )
    parser1.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Name of the optimizer to use",
        choices=[
            "adamw",
            "soap",
            "shampoo",
            "sgd",
            "adam-sania",
            "muon",
        ],
    )
    args1, _ = parser1.parse_known_args()
    parser = parser1

    ### Training Arguments
    parser.add_argument(
        "--batch_size",
        "--per_device_train_batch_size",
        default=8,
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--n_epochs_train",
        default=1,
        type=int,
        help="How many epochs to train"
    )
    parser.add_argument(
        "--eval_runs",
        default=1,
        type=int,
        help="Number of re-training model with different seeds",
    )
    parser.add_argument(
        "--dtype", default=None, type=str, help="Default type for torch"
    )
    parser.add_argument(
        "--use_old_tune_params", action="store_true", help="Use already tuned params"
    )

    ##### Wandb Arguments #####
    parser.add_argument("--wandb", action="store_true", help="To use wandb")
    parser.add_argument(
        "--run_prefix", default=None, help="Run prefix for the experiment run name"
    )
    parser.add_argument("--wandb_project", default="OPTIM_TEST")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="To print training resuls in the terminal",
    )
    parser.add_argument("--seed", default=18, type=int)

    ##### Saving Paths #####
    parser.add_argument(
        "--results_path",
        default="results_raw",
        help="Path to save the results of the experiment",
    )
    parser.add_argument("--data_path", default="data", help="Path to save the datasets")

    ##### Otimizer Arguments #####
    parser.add_argument(
        "--lr", "--learning rate", default=1e-4, type=float, help="learning rate"
    )  # tuneed param
    parser.add_argument(
        "--weight_decay", "-wd", default=1e-5, type=float, help="weight decay"
    )  # tuneed param
    if args1.optimizer not in ["shampoo", "sgd"]:
        parser.add_argument("--beta1", default=0.9, type=float, help="First momentum")
        parser.add_argument(
            "--beta2", default=0.999, type=float, help="Second momentum"
        )
        parser.add_argument("--eps", default=1e-8, type=float, help="Epsilon for Adam")

    if args1.optimizer in ["shampoo", "sgd", "muon"]:
        parser.add_argument(
            "--momentum", default=0.9, type=float, help="First momentum"
        )

    if args1.optimizer in ["soap"]:
        parser.add_argument(
            "--shampoo_beta",
            default=-1,
            type=float,
            help="momentum for SOAP. if -1, the equals to beta2",
        )
    if args1.optimizer in ["shampoo", "soap", "diag-hvp"]:
        parser.add_argument(
            "--update_freq",
            default=1,
            type=int,
            help="Freqiensy to update Q for Shampoo and SOAP",
        )
    if args1.optimizer in ["muon"]:
        parser.add_argument(
            "--ns_steps", default=10, type=int, help="Number of the NS steps algo"
        )
        parser.add_argument(
            "--adamw_lr", default=None, type=float, help="lr for adam in "
        )

    ##### Problem Specific Arguments #####
    dataset = args1.dataset.lower()
    if dataset in LLM_DATASETS:
        problem = "llm"
        parser = set_arguments_llm(parser, dataset in NLG_DATASETS, dataset in SQUAD_DATASETS)
    elif dataset in LIBSVM_DATASETS:
        problem = "libsvm"
        parser = set_arguments_libsvm(parser)
    elif dataset in CV_DATASETS:
        problem = "cv"
        parser = set_arguments_cv(parser)
    else:
        raise ValueError(
            f"""
            Unknown dataset: {args1.dataset}.
            Possible datasets are:
            LLM: {LLM_DATASETS}
            LIBSVM: {LIBSVM_DATASETS}
            CV: {CV_DATASETS}
            """
        )

    args, unparsed_args = parser.parse_known_args()

    ##### Warnings #####
    if args1.config_name is not None:
        path = f"./src/{problem}/configs/{args1.config_name}.json"
        if os.path.exists(path):
            print(colored("~~~~~~~~~~~~~~~ CONFIG FILE FOUND ~~~~~~~~~~~~~~~", "green"))
            line = f"Configuration file found at: {path}"
            print(colored(line, "green"))
            with open(path) as f:
                params = json.load(f)
                for key in params.keys():
                    setattr(args, key, params[key])
        else:
            print(
                colored(
                    "~~~~~~~~~~~~~~~ WARNING: CONFIG FILE NOT FOUND ~~~~~~~~~~~~~~~",
                    "red",
                )
            )
            line = f"Path {path} does not exist. Using default configuration."
            print(colored(line, "red"))

    if len(unparsed_args) > 0:
        print(colored("~~~~~~~~~~~~~~~ WARNING: UNPARSED ARGS ~~~~~~~~~~~~~~~", "red"))
        line = "You pass unrecognized arguments:"
        print(colored(line, "red"), end="")
        for arg in unparsed_args:
            if "--" in arg:
                print(colored(f"\n{arg}", "red"), end=" ")
            else:
                print(colored(arg, "red"), end="")
        print()
    if not args.wandb and not args.verbose:
        print(colored("~~~~~~~~~~~~~~~ WARNING: NO VERBOSE ~~~~~~~~~~~~~~~", "yellow"))
        line = "wandb and verbose set to False, so we set verbose to True"
        print(colored(line, "yellow"))
        args.verbose = True

    return args, parser

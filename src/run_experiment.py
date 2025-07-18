import torch
import sys

sys.path.append("src/libsvm")
sys.path.append("src/cv")
sys.path.append("src/fine_tuning/glue")
sys.path.append("src/fine_tuning/llm")

from config import parse_args
from utils import get_run_name
from libsvm import main_libsvm
from cv import main_cv
from fine_tuning.glue import main_glue
from fine_tuning.llm import main_llm

CASUAL_LLM_DATASETS = ["mathqa", "coin_flip"]


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("~~~~~~~~~~~~~~~ GPU ~~~~~~~~~~~~~~~")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("~~~~~~~~~~~~~~~ USING CPU ~~~~~~~~~~~~~~~")
    args, parser = parse_args()
    args.run_name = get_run_name(args, parser)
    if args.problem.lower() == "libsvm":
        main_libsvm.main(args, parser)
    elif args.problem.lower() == "cv":
        main_cv.main(args, parser)
    elif args.problem.lower() == "fine-tuning" and args.dataset.lower() == "glue":
        # Use unified fine-tuning framework
        main_glue.main(args, parser)
    elif (
        args.problem.lower() == "fine-tuning"
        and args.dataset.lower() in CASUAL_LLM_DATASETS
    ):
        # Use unified fine-tuning framework
        main_llm.main(args)
    else:
        raise ValueError("Unsupported problem or dataset specified.")

import json
from functools import partial
from torch.utils.data import Dataset
from utils import shuffleDict
from datasets import load_dataset


class DatasetBuilder:
    """Base class for dataset builders"""

    def __init__(self, args):
        self.args = args
        self.train_questions = []
        self.train_answers = []
        self.eval_questions = []
        self.eval_answers = []

    def get_intro_blurb(self):
        """Get appropriate intro blurb for the dataset type - override in subclasses"""
        return "write answer first."

    def build_dataset(self):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement build_dataset")

    def apply_limits(self, questions, answers, is_eval=False):
        """Apply max_train_samples or max_eval_samples limit if specified"""
        max_len = self.args.max_eval_samples if is_eval else self.args.max_train_samples
        if max_len is not None:
            max_len = min(max_len, len(questions))
            questions = questions[:max_len]
            answers = answers[:max_len]
        return questions, answers

    def create_prompt_formats(self, sample, eval_mode=False):
        """Create prompt formats for training or evaluation"""
        intro_blurb = self.get_intro_blurb()

        if eval_mode:
            formatted_prompt = f"{intro_blurb} {sample['question']}. Answer:"
        else:
            formatted_prompt = (
                f"{intro_blurb} {sample['question']}. Answer: {sample['response']}"
            )

        sample["text"] = formatted_prompt
        return sample

    def preprocess_batch(self, batch, tokenizer, max_length):
        """Tokenize a batch"""
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )

    def preprocess_dataset(self, tokenizer, max_length, seed, dataset, eval_mode=False):
        """Format & tokenize dataset for training or evaluation"""
        print("Preprocessing dataset...")

        # Add prompt to each sample
        dataset = dataset.map(
            lambda sample: self.create_prompt_formats(sample, eval_mode)
        )

        print(f"Prompt Sample: {dataset['text'][0]}")

        if not eval_mode:
            # Calculate average length for training data
            total_length = sum(len(sample["text"]) for sample in dataset)
            avg_length = total_length / len(dataset)
            print(f"Average sample length: {avg_length:.2f} characters")

            # Apply preprocessing for training
            _preprocessing_function = partial(
                self.preprocess_batch, max_length=max_length, tokenizer=tokenizer
            )
            dataset = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=["question", "response", "raw_x", "raw_y", "text"],
            )

            # Filter out samples that exceed max_length
            dataset = dataset.filter(
                lambda sample: len(sample["input_ids"]) < max_length
            )

            # Shuffle dataset
            dataset = dataset.shuffle(seed=seed)

        return dataset

    def get_data(self):
        """Build dataset and return the data"""
        self.build_dataset()

        # Apply limits
        train_questions, train_answers = self.apply_limits(
            self.train_questions, self.train_answers, is_eval=False
        )

        eval_questions, eval_answers = self.apply_limits(
            self.eval_questions, self.eval_answers, is_eval=True
        )

        print(f"dataset: {self.args.dataset}")
        print(f"train data size: {len(train_answers)}")
        print(f"eval data size: {len(eval_answers)}")

        return {
            "train": (train_questions, train_answers),
            "eval": (eval_questions, eval_answers),
        }


class AquaDatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only the letter choice, write answer first."

    def build_dataset(self):
        decoder = json.JSONDecoder()
        with open(self.args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                question = json_res["question"].strip() + " " + choice
                answer = json_res["correct"]

                self.train_questions.append(question)
                self.train_answers.append(answer)


class GSM8KDatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only numbers, write answer first."

    def build_dataset(self):
        dataset = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
        train_data = dataset["train"]
        eval_data = dataset["test"]

        for item in train_data:
            self.train_questions.append(item["question"])
            self.train_answers.append(item["answer"].split("####")[1].strip())

        for item in eval_data:
            self.eval_questions.append(item["question"])
            self.eval_answers.append(item["answer"].split("####")[1].strip())


class CommonsenseQADatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only the letter choice, write answer first."

    def build_dataset(self):
        decoder = json.JSONDecoder()

        # Load train data
        with open(self.args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                question = json_res["question"]["stem"].strip() + " " + choice

                self.train_questions.append(question)
                self.train_answers.append(json_res["answerKey"])

        # Load eval data if available
        if hasattr(self.args, "val_dataset_path") and self.args.val_dataset_path:
            with open(self.args.val_dataset_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    choice = "Answer Choices:"
                    for c in json_res["question"]["choices"]:
                        choice += " ("
                        choice += c["label"]
                        choice += ") "
                        choice += c["text"]
                    question = json_res["question"]["stem"].strip() + " " + choice

                    self.eval_questions.append(question)
                    self.eval_answers.append(json_res["answerKey"])


class BoolQDatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only True or False, write answer first."

    def build_dataset(self):
        decoder = json.JSONDecoder()

        # Load train data
        with open(self.args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices: (a) True (b) False"
                question = (
                    json_res["question"].strip().capitalize() + "?" + " " + choice
                )

                self.train_questions.append(question)
                self.train_answers.append(str(json_res["answer"]))

        # Load eval data if available
        if hasattr(self.args, "val_dataset_path") and self.args.val_dataset_path:
            with open(self.args.val_dataset_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    choice = "Answer Choices: (a) True (b) False"
                    question = (
                        json_res["question"].strip().capitalize() + "?" + " " + choice
                    )

                    self.eval_questions.append(question)
                    self.eval_answers.append(str(json_res["answer"]))


class ArithmeticDatasetBuilder(DatasetBuilder):
    """Handles addsub, multiarith, singleeq datasets"""

    def get_intro_blurb(self):
        return "answer only numbers, write answer first."

    def build_dataset(self):
        with open(self.args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]

                self.train_questions.append(q)
                self.train_answers.append(a)


class StrategyQADatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only yes or no, write answer first."

    def build_dataset(self):
        with open(self.args.dataset_path) as f:
            json_data = json.load(f)["examples"]

            # Split train/eval
            split_point = int(len(json_data) * 0.7)
            train_data = json_data[:split_point]
            eval_data = json_data[split_point:]

            for line in train_data:
                q = line["input"].strip()
                a = "yes" if int(line["target_scores"]["Yes"]) == 1 else "no"

                self.train_questions.append(q)
                self.train_answers.append(a)

            for line in eval_data:
                q = line["input"].strip()
                a = "yes" if int(line["target_scores"]["Yes"]) == 1 else "no"

                self.eval_questions.append(q)
                self.eval_answers.append(a)


class SVAMPDatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only numbers, write answer first."

    def build_dataset(self):
        with open(self.args.dataset_path) as f:
            json_data = json.load(f)

            # Split train/eval
            split_point = int(len(json_data) * 0.7)
            train_data = json_data[:split_point]
            eval_data = json_data[split_point:]

            for line in train_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]

                self.train_questions.append(q)
                self.train_answers.append(a)

            for line in eval_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]

                self.eval_questions.append(q)
                self.eval_answers.append(a)


class BigBenchDatasetBuilder(DatasetBuilder):
    """Handles bigbench_date and object_tracking datasets"""

    def get_intro_blurb(self):
        return "answer only the letter choice, write answer first."

    def build_dataset(self):
        with open(self.args.dataset_path) as f:
            json_data = json.load(f)["examples"]

            if self.args.dataset == "bigbench_date":
                choice_index = ["A", "B", "C", "D", "E", "F"]
            elif self.args.dataset == "object_tracking":
                choice_index = ["A", "B", "C"]

            # Split train/eval
            split_point = int(len(json_data) * 0.8)
            train_data = json_data[:split_point]
            eval_data = json_data[split_point:]

            for line in train_data:
                q, a = self._process_bigbench_item(line, choice_index)
                self.train_questions.append(q)
                self.train_answers.append(a)

            for line in eval_data:
                q, a = self._process_bigbench_item(line, choice_index)
                self.eval_questions.append(q)
                self.eval_answers.append(a)

    def _process_bigbench_item(self, line, choice_index):
        q = line["input"].strip()

        if self.args.dataset == "bigbench_date":
            choice = "Answer Choices:"
            choice_dic = shuffleDict(line["target_scores"])
        elif self.args.dataset == "object_tracking":
            choice = "\nWhich choice is true ? Answer Choices:"
            choice_dic = line["target_scores"]

        for i, (key, value) in enumerate(choice_dic.items()):
            choice += f" ({choice_index[i]}) {key}"
            if value == 1:
                a = choice_index[i]

        return q + " " + choice, a


class CoinFlipLastLettersDatasetBuilder(DatasetBuilder):
    """Handles coin_flip and last_letters datasets"""

    def get_intro_blurb(self):
        return "answer only yes or no, write answer first."

    def build_dataset(self):
        with open(self.args.dataset_path) as f:
            json_data = json.load(f)["examples"]

            # Split train/eval
            split_point = int(len(json_data) * 0.7)
            train_data = json_data[:split_point]
            eval_data = json_data[split_point:]

            for line in train_data:
                self.train_questions.append(line["question"])
                self.train_answers.append(line["answer"])

            for line in eval_data:
                self.eval_questions.append(line["question"])
                self.eval_answers.append(line["answer"])


class MathQADatasetBuilder(DatasetBuilder):
    def get_intro_blurb(self):
        return "answer only the numbers, write answer first."

    def build_dataset(self):
        dataset = load_dataset("allenai/math_qa", trust_remote_code=True)
        train_data = dataset["train"]
        test_data = dataset["test"]

        for item in train_data:
            q = item["Problem"].replace("\n", "").replace("\\n", "")
            choices = {
                d.strip()[0]: d.split(")")[-1].strip()
                for d in item["options"].split(",")
            }
            a = choices.get(item["correct"])
            o = item["options"].replace("\n", "").replace("\\n", "")

            if q and a:
                self.train_questions.append(q + f". Choices: {o}")
                self.train_answers.append(a)

        for item in test_data:
            q = item["Problem"].replace("\n", "").replace("\\n", "")
            choices = {
                d.strip()[0]: d.split(")")[-1].strip()
                for d in item["options"].split(",")
            }
            a = choices.get(item["correct"])
            o = item["options"].replace("\n", "").replace("\\n", "")

            if q and a:
                self.eval_questions.append(q + f". Choices: {o}")
                self.eval_answers.append(a)


class DatasetRegistry:
    """Registry for dataset paths and builders"""

    DATASET_PATHS = {
        "aqua": "data/AQuA/train.json",
        "gsm8k": "data/grade-school-math/train.jsonl",
        "commonsensqa": "data/CommonsenseQA/train_rand_split.jsonl",
        "boolq": "data/BoolQ/train.jsonl",
        "addsub": "data/AddSub/AddSub.json",
        "multiarith": "data/MultiArith/MultiArith.json",
        "singleeq": "data/SingleEq/questions.json",
        "strategyqa": "data/StrategyQA/strategyqa_train.json",
        "svamp": "data/SVAMP/SVAMP.json",
        "bigbench_date": "data/BigBench/date_understanding.json",
        "object_tracking": "data/BigBench/tracking_shuffled_objects.json",
        "coin_flip": "data/coin_flip/coin_flip.json",
        "last_letters": "data/BigBench/last_letters.json",
    }

    VAL_DATASET_PATHS = {
        "commonsensqa": "data/CommonsenseQA/dev_rand_split.jsonl",
        "boolq": "data/BoolQ/val.jsonl",
    }

    @classmethod
    def set_dataset_paths(cls, args):
        """Set dataset paths on args"""
        if args.dataset in cls.DATASET_PATHS:
            args.dataset_path = cls.DATASET_PATHS[args.dataset]

        if args.dataset in cls.VAL_DATASET_PATHS:
            args.val_dataset_path = cls.VAL_DATASET_PATHS[args.dataset]

    @classmethod
    def create_builder(cls, args):
        """Create and return appropriate dataset builder"""
        builders = {
            "aqua": AquaDatasetBuilder,
            "gsm8k": GSM8KDatasetBuilder,
            "commonsensqa": CommonsenseQADatasetBuilder,
            "boolq": BoolQDatasetBuilder,
            "addsub": ArithmeticDatasetBuilder,
            "multiarith": ArithmeticDatasetBuilder,
            "singleeq": ArithmeticDatasetBuilder,
            "strategyqa": StrategyQADatasetBuilder,
            "svamp": SVAMPDatasetBuilder,
            "bigbench_date": BigBenchDatasetBuilder,
            "object_tracking": BigBenchDatasetBuilder,
            "coin_flip": CoinFlipLastLettersDatasetBuilder,
            "last_letters": CoinFlipLastLettersDatasetBuilder,
            "mathqa": MathQADatasetBuilder,
        }

        if args.dataset not in builders:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        builder_class = builders[args.dataset]
        return builder_class(args)


class MyDataset(Dataset):
    def __init__(self, args, eval=False):
        super().__init__()
        DatasetRegistry.set_dataset_paths(args)
        builder = DatasetRegistry.create_builder(args)
        data = builder.get_data()

        if eval:
            self.questions, self.answers = data["eval"]
        else:
            self.questions, self.answers = data["train"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        input_text = self.questions[index]
        output_text = self.answers[index]
        return input_text, output_text, "", ""  # Keep compatibility with old interface

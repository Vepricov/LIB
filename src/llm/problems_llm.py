from loguru import logger

from functools import partial
from torch.utils.data import Dataset
from utils import shuffleDict

import numpy as np
import collections
from typing import Optional, Tuple
from tqdm.auto import tqdm

import os  # for postprocess_qa_predictions, mb remove
import nltk  # for nlg
import json  # for nlg
from dataclasses import dataclass, field  # for qa custom training args class

from datasets import load_dataset, load_metric
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EvalPrediction,
)

CAUSAL_LM_DATASETS = ["gsm8k", "math_qa"]
GLUE_DATASETS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
SQUAD_DATASETS = ["squad", "squad_v2"]
NLG_DATASETS = [
    "amazon_reviews_multi", "cnn_dailymail", "wiki_summary", "big_patent", "orange_sum",
    "pn_summary", "thaisum", "samsum", "xglue", "xsum", "psc",
]

class Problem:
    """Base class for llm problems: causal lm, glue, squad and nlg"""

    def __init__(self, args, train_dataset=None, eval_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

        # If no custom dataset splits were provided
        if train_dataset is None and eval_dataset is None and test_dataset is None:
            self.load_dataset()

        # Check that neccessary parts are present
        if args.do_train and self.train_dataset is None:
            raise ValueError("--do_train requires a train dataset") #~ fix error message
        if args.do_eval and self.eval_dataset is None:
            raise ValueError("--do_eval requires a validation dataset")
        if args.do_predict and self.test_dataset is None:
            raise ValueError("--do_predict requires a test dataset")

    def load_dataset(self):
        """
        Load the dataset and split it into train/eval/test parts without any processing
        """
        raise NotImplementedError("Implement this method in a successor class")
    
    def process_dataset(self, tokenizer, process_function):
        """
        Prepare neccessary parts of the dataset and return them.
        Parts of the original dataset are not modified and can be accessed as class members
        """
        args = self.args

        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length={args.max_seq_length} is larger than the maximum length for "
                f"the model model_max_len={tokenizer.model_max_length}. "
                f"Using max_seq_length={tokenizer.model_max_length}."
            )

        train_dataset = None
        if args.do_train:
            train_dataset = self.train_dataset

            if args.max_train_samples is not None:
                train_size = len(train_dataset)
                if args.max_train_samples <= train_size:
                    train_dataset = train_dataset.select(range(args.max_train_samples))
                else:
                    logger.warning(
                        f"The max_train_samples={args.max_train_samples} is larger than the "
                        f"size of the training dataset train_size={train_size}. Using all samples."
                    )

            process_fn = partial(process_function, tokenizer=tokenizer, args=args, is_eval=False)

            # Process train dataset
            remove_columns = None if args.task_type == "SEQ_CLS" else train_dataset.column_names
            train_dataset = train_dataset.map(
                process_fn,
                batched=True,
                remove_columns=remove_columns,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=(not args.overwrite_cache),
                desc="Running tokenizer on train dataset",
            )

            # In case number of samples increased after tokenization
            if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))
        
        eval_dataset = None
        if args.do_eval:
            eval_dataset = self.eval_dataset

            if args.max_eval_samples is not None:
                eval_size = len(eval_dataset)
                if args.max_eval_samples <= eval_size:
                    eval_dataset = eval_dataset.select(range(args.max_eval_samples))
                else:
                    logger.warning(
                        f"The max_eval_samples={args.max_eval_samples} is larger than the "
                        f"size of the validation dataset eval_size={eval_size}. Using all samples."
                    )

            is_eval = not (args.metric_for_best_model == "loss" and args.use_test_as_eval)
            process_fn = partial(process_function, tokenizer=tokenizer, args=args, is_eval=is_eval)

            # Process validation dataset
            remove_columns = None if args.task_type == "SEQ_CLS" else eval_dataset.column_names
            eval_dataset = eval_dataset.map(
                process_fn,
                batched=True,
                remove_columns=remove_columns,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=(not args.overwrite_cache),
                desc="Running tokenizer on validation dataset",
            )

            # In case number of samples increased after processing
            if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        test_dataset = None
        if args.do_predict:
            test_dataset = self.test_dataset

            if args.max_test_samples is not None:
                test_size = len(test_dataset)
                if args.max_test_samples <= test_size:
                    test_dataset = test_dataset.select(range(args.max_test_samples))
                else:
                    logger.warning(
                        f"The max_test_samples={args.max_test_samples} is larger than the "
                        f"size of the testing dataset test_size={test_size}. Using all samples."
                    )

            process_fn = partial(process_function, tokenizer=tokenizer, args=args, is_eval=True)

            # Process test dataset
            remove_columns = None if args.task_type == "SEQ_CLS" else test_dataset.column_names
            test_dataset = test_dataset.map(
                process_fn,
                batched=True,
                remove_columns=remove_columns,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=(not args.overwrite_cache),
                desc="Running tokenizer on test dataset",
            )

            # In case number of samples increased after processing
            if args.max_test_samples is not None and len(test_dataset) > args.max_test_samples:
                test_dataset = test_dataset.select(range(args.max_test_samples))

        return train_dataset, eval_dataset, test_dataset

    def get_trainer_class(self):
        """
        Return trainer class (constructor) that has default hf trainer interface
        """
        raise NotImplementedError("Implement this method in a successor class")

    def get_training_args(self):
        raise NotImplementedError("Implement this method in a successor class")

    def get_data_collator(self):
        raise NotImplementedError("Implement this method in a successor class")

    def get_metrics_function(self):
        raise NotImplementedError("Implement this method in a successor class")


class CausalLM_Problem(Problem):

    # Intentionally does nothing...
    def process_dataset(self, tokenizer, process_function):
        return super().process_dataset(tokenizer, process_function)

    def get_trainer_class(self):
        return Trainer

    def get_training_args(self):
        args = self.args

        training_args = TrainingArguments(
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=(
                args.eval_batch_size if args.eval_batch_size else
                args.batch_size
            ),
            gradient_accumulation_steps=args.grad_acc_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.lr,
            num_train_epochs=args.n_epochs_train,
            max_steps=args.max_steps_train,
            logging_steps=args.logging_steps,
            eval_strategy=args.eval_strategy,
            save_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            #~ maybe add logging dir
            output_dir=f"./src/llm/{args.results_path}",
            overwrite_output_dir=True,
            run_name=args.run_name,
            report_to=["wandb" if args.wandb else "none"],
            load_best_model_at_end=(args.eval_strategy != "no"),
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=(args.metric_for_best_model not in ["loss"]), #mb extend
        )

        return training_args

    def get_data_collator(self, model, tokenizer):
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        return data_collator

    def get_metrics_function(self, tokenizer):
        return None


class GSM8K_Problem(CausalLM_Problem):

    def load_dataset(self):
        # There are 2 subsets (versions): main and socratic
        dataset = load_dataset("openai/gsm8k", "main")

        # This dataset only has train and test parts by default
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

        # See description of --use_test_as_eval
        if self.args.use_test_as_eval:
            self.eval_dataset = dataset["test"]

    def process_dataset(self, tokenizer):
        def process_function(samples, tokenizer, args, is_eval):
            intro = "Answer only numbers, write answer first. " #~ maybe change this
            batch_size = len(samples['question'])

            if is_eval:
                return {
                    "text": [
                        f"{intro}Question: {samples['question'][i]} Answer: "
                        for i in range(batch_size)
                    ],
                    "answer": [
                        samples["answer"][i].split("####")[1].strip()
                        for i in range(batch_size)
                    ]
                }

            inputs = [
                f"{intro}Question: {samples['question'][i]} "
                f"Answer: {samples['answer'][i].split('####')[1].strip()}"
                for i in range(batch_size)
            ]

            max_len = min(args.max_seq_length, tokenizer.model_max_length)
            return tokenizer(
                inputs,
                max_length=max_len,
                truncation=True,
            )

        return super().process_dataset(tokenizer, process_function)


class MathQA_Problem(CausalLM_Problem):

    def load_dataset(self):
        # FIX: crushes with error "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2
        # in position 1: invalid start byte"
        dataset = load_dataset("allenai/math_qa")

        # This dataset only has train and test parts by default
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

        if self.args.use_test_as_eval:
            self.eval_dataset = dataset["test"]

    def process_dataset(self, tokenizer):
        def get_answer(options, correct):
            # Assume that there are at most 5 options for each question indexed by letters
            ans = ['a', 'b', 'c', 'd', 'e']

            return options.split(',')[ans.index(correct)].split(')')[1].strip()

        def process_function(samples, tokenizer, args, is_eval):
            #intro = "Answer only with numbers, choose from options"
            intro = "Answer only numbers, write answer first. "
            batch_size = len(samples['question'])

            if is_eval:
                return {
                    "text": [
                        f"{intro}Question: {samples['Problem'][i]} "
                        f"Options: {samples['options'][i]} Answer: "
                        for i in range(batch_size)
                    ],
                    "answer": [
                        get_answer(samples['options'][i], samples['correct'][i])
                        for i in range(batch_size)
                    ]
                }

            inputs = [
                f"{intro}Question: {samples['Problem'][i]} Options: {samples['options'][i]} "
                f"Answer: {get_answer(samples['options'][i], samples['correct'][i])}"
                for i in range(batch_size)
            ]

            max_len = min(args.max_seq_length, tokenizer.model_max_length)
            return tokenizer(
                inputs,
                max_length=max_len,
                truncation=True,
            )

        return super().process_dataset(tokenizer, process_function)


# FIX: add label_to_id to process_function
class GLUE_Problem(Problem):

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp":  ("question1", "question2"),
        "rte":  ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"), #~ mb remove?
        "wnli": ("sentence1", "sentence2"),
    }

    def __init__(self, args, train_dataset=None, eval_dataset=None, test_dataset=None):
        super().__init__(args, train_dataset, eval_dataset, test_dataset)

        if args.dataset not in GLUE_Problem.task_to_keys.keys():
            raise ValueError(
                "GLUE_Problem class is designed for problems listed in task_to_keys dictionary."
            )

    def load_dataset(self):
        # Load specific glue subset
        dataset = load_dataset("glue", self.args.dataset)

        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset[
            "validation_matched" if self.args.dataset == "mnli" else "validation"
        ]
        self.test_dataset = dataset[
            "test_matched" if self.args.dataset == "mnli" else "test"
        ]

    def process_dataset(self, tokenizer):
        # FIX: investigate warning: "Be aware, overflowing tokens are not returned for the setting
        # you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy.
        # So the returned list will always be empty even if some tokens have been removed."
        def process_function(samples, tokenizer, args, is_eval):
            key_1, key_2 = GLUE_Problem.task_to_keys[args.dataset]
            inputs = (
                (samples[key_1],) if key_2 is None else
                (samples[key_1], samples[key_2])
            )

            padding = "max_length" if args.pad_to_max_length else False
            max_len = min(args.max_seq_length, tokenizer.model_max_length)
            return tokenizer(
                *inputs, padding=padding, max_length=max_len, truncation=True
            )

        return super().process_dataset(tokenizer, process_function)

    def get_trainer_class(self):
        return Trainer

    def get_training_args(self):
        args = self.args

        training_args = TrainingArguments(
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=(
                args.eval_batch_size if args.eval_batch_size else
                args.batch_size
            ),
            gradient_accumulation_steps=args.grad_acc_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.lr,
            num_train_epochs=args.n_epochs_train,
            max_steps=args.max_steps_train,
            logging_steps=args.logging_steps,
            eval_strategy=args.eval_strategy,
            save_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            #~ maybe add logging dir
            output_dir=f"./src/llm/{args.results_path}",
            overwrite_output_dir=True,
            run_name=args.run_name,
            report_to=["wandb" if args.wandb else "none"],
        )

        return training_args

    def get_data_collator(self, model, tokenizer):
        if self.args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        return data_collator

    def get_metrics_function(self, tokenizer):
        metric = load_metric("glue", self.args.dataset)
        is_regression = (self.args.dataset == "stsb")

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()

            return result

        return compute_metrics


class SQUAD_Problem(Problem):

    def load_dataset(self):
        dataset = load_dataset(self.args.dataset)

        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["validation"]

    def process_dataset(self, tokenizer):
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This method only works for models that have a fast tokenizer. Look at models "
                "from https://huggingface.co/transformers/index.html#supported-frameworks to find "
                "the model types that meet this requirement."
            )

        def process_train(samples, tokenizer, args):
            # Padding side determines if we do (question, context) or (context, question)
            pad_on_right = (tokenizer.padding_side == "right")

            max_length = min(args.max_seq_length, tokenizer.model_max_length)

            questions = [q.lstrip() for q in samples["question"]]
            contexts = samples["context"]

            # Tokenize samples with truncation and maybe padding, but keep the overflows.
            # One sample could be split into several parts if context is long, each of those parts
            # will have context that overlaps a bit of context of the previous part
            tokenized_samples = tokenizer(
                questions if pad_on_right else contexts,
                contexts  if pad_on_right else questions,
                truncation=("only_second" if pad_on_right else "only_first"),
                max_length=max_length,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=("max_length" if args.pad_to_max_length else False),
            )

            # List that contains indices of samples that tokenized parts correspond to
            sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")

            # List that contains list of indices pairs where each pair correponds to a token in a
            # tokenized part (its position in the context of a sample where it appeared)
            offset_mapping = tokenized_samples.pop("offset_mapping")

            # ModelForQuestionAnswering expects these entries, for each tokenized part they are
            # either indices of first and last tokens that make up the answer or both are
            # indices of the <CLS> token (when there is no answer in that part of context)
            tokenized_samples["start_positions"] = []
            tokenized_samples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # `offsets` and `input_ids` are lists corresponding to the i-th tokenized part
                input_ids = tokenized_samples["input_ids"][i]

                cls_index = (
                    input_ids.index(tokenizer.cls_token_id)
                    if tokenizer.cls_token_id in input_ids
                    else 0
                )

                # List that contains label of every token in i-th tokenized part,
                # possible labels are: 0, 1 and None
                # 0 - token belongs to the first sequence passed
                # 1 - token belongs to the second sequence passed
                # None - it is a special token added by tokenizer
                seq_ids = tokenized_samples.sequence_ids(i)

                # Here, multiple tokenized parts may correspond to the same answer entry
                sample_idx = sample_mapping[i]
                answers = samples["answers"][sample_idx]

                # If no answers were given, set the cls_index as answer
                if len(answers["answer_start"]) == 0:
                    tokenized_samples["start_positions"].append(cls_index)
                    tokenized_samples["end_positions"].append(cls_index)
                    continue
                
                # Start/end character index of the answer in the full context.
                # Idk why, but each entry in answer is a list, maybe there are multiple answers
                start_idx = answers["answer_start"][0]
                end_idx = start_idx + len(answers["text"][0])

                # Start/end token index of the context of i-th tokenized part
                label = (1 if pad_on_right else 0)
                start_token_idx = seq_ids.index(label)
                end_token_idx = (len(seq_ids) - 1) - seq_ids[::-1].index(label)

                # Index of first character of first token in the context of i-th tokenized part
                first_idx = offsets[start_token_idx][0]
                # Index of last character of last token in the context of i-th tokenized part
                last_idx = offsets[end_token_idx][1]

                if not (first_idx <= start_idx and end_idx <= last_idx):
                    tokenized_samples["start_positions"].append(cls_index)
                    tokenized_samples["end_positions"].append(cls_index)
                else:
                    # Move `start_token_idx` and `end_token_idx` to the 2 ends of the answer
                    N = len(offsets)
                    while start_token_idx < N and offsets[start_token_idx][0] <= start_idx:
                        start_token_idx += 1
                    tokenized_samples["start_positions"].append(start_token_idx - 1)

                    while end_token_idx >= 0 and offsets[end_token_idx][1] >= end_idx:
                        end_token_idx -= 1
                    tokenized_samples["end_positions"].append(end_token_idx + 1)

            return tokenized_samples

        def process_eval(samples, tokenizer, args):
            # Padding side determines if we do (question, context) or (context, question)
            pad_on_right = (tokenizer.padding_side == "right")

            max_length = min(args.max_seq_length, tokenizer.model_max_length)

            questions = [q.lstrip() for q in samples["question"]]
            contexts = samples["context"]

            # Same drill as with train samples
            tokenized_samples = tokenizer(
                questions if pad_on_right else contexts,
                contexts if pad_on_right else questions,
                truncation=("only_second" if pad_on_right else "only_first"),
                max_length=max_length,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=("max_length" if args.pad_to_max_length else False),
            )
            sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")

            # List that contains id of the sample that tokenized part corresponds to
            tokenized_samples["sample_id"] = []

            for i in range(len(tokenized_samples["input_ids"])):
                sample_idx = sample_mapping[i]
                tokenized_samples["sample_id"].append(samples["id"][sample_idx])

                seq_ids = tokenized_samples.sequence_ids(i)
                label = 1 if pad_on_right else 0

                # Set to None the `offset_mapping` entries that are not part of the context so it
                # is easy to determine if a token is part of the context or not
                tokenized_samples["offset_mapping"][i] = [
                    pos if seq_ids[j] == label else None
                    for j, pos in enumerate(tokenized_samples["offset_mapping"][i])
                ]

            return tokenized_samples

        def process(samples, tokenizer, args, is_eval=False):
            return (
                process_train(samples, tokenizer, args) if not is_eval else
                process_eval(samples, tokenizer, args)
            )

        return super().process_dataset(tokenizer, process)

    def get_trainer_class(self):
        return QuestionAnsweringTrainer

    def get_training_args(self):
        args = self.args

        training_args = QuestionAnsweringTrainingArguments(
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=(
                args.eval_batch_size if args.eval_batch_size else
                args.batch_size
            ),
            gradient_accumulation_steps=args.grad_acc_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.lr,
            num_train_epochs=args.n_epochs_train,
            max_steps=args.max_steps_train,
            logging_steps=args.logging_steps,
            eval_strategy=args.eval_strategy,
            save_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            #~ maybe add logging dir
            output_dir=f"./src/llm/{args.results_path}",
            overwrite_output_dir=True,
            run_name=args.run_name,
            report_to=["wandb" if args.wandb else "none"],
            # Question answering specific arguments
            v2_with_negative=(args.dataset == "squad_v2"),
            null_score_diff_threshold=args.null_score_diff_threshold,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
        )

        return training_args

    def get_data_collator(self, model, tokenizer):
        # We have already padded to max length if the corresponding flag is True, otherwise we need
        # to pad in the data collator
        return (
            default_data_collator if self.args.pad_to_max_length else
            DataCollatorWithPadding(
                tokenizer,
                pad_to_multiple_of=(8 if self.args.fp16 else None)
            )
        )

    def get_metrics_function(self, tokenizer):
        metric = load_metric(self.args.dataset)

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        return compute_metrics


class NLG_Problem(Problem):

    task_to_keys = {
        "amazon_reviews_multi": ("review_body", "review_title"),
        "cnn_dailymail": ("article", "highlights"), #
        "wiki_summary": ("article", "highlights"),
        "big_patent": ("description", "abstract"), #
        "orange_sum": ("text", "summary"), 
        "pn_summary": ("article", "summary"), #
        "thaisum": ("body", "summary"), #
        "samsum": ("dialogue", "summary"), #
        "xglue": ("news_body", "news_title"),
        "xsum": ("document", "summary"), #
        "psc": ("extract_text", "summary_text"),
    }

    def load_dataset(self):
        logger.info("Loading datasets...")

        dataset = load_dataset(
            self.args.dataset,
            self.args.dataset_config,
            #cache_dir=self.args.cache_dir,
        )

        # Some of the datasets above may not have eval/test splits...
        # Confirmed to have: cnn_dailymail, big_patent, pn_summary, thaisum, samsum, xsum
        self.train_dataset = dataset.get("train", None)     #dataset["train"]
        self.eval_dataset = dataset.get("validation", None) #dataset["validation"]
        self.test_dataset = dataset.get("test", None)       #dataset["test"]

    def process_dataset(self, tokenizer):
        args = self.args

        ##Before: and args.model.lower() in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        if args.source_prefix is None and "t5" in args.model.lower():
            logger.warning(
                "T5 model expects --source_prefix argument, but you did not provide it."
            )

        column_names = None
        if args.do_train:
            column_names = self.train_dataset.column_names
        elif args.do_eval:
            column_names = self.eval_dataset.column_names
        else:
            column_names = self.test_dataset.column_names

        columns = NLG_Problem.task_to_keys.get(args.dataset, None)

        if columns is None:
            text_column = args.text_column
            if text_column not in column_names:
                raise ValueError(
                    f"You passed dataset={args.dataset} which is not in task_to_keys dictionary "
                    f"and passed text_column={text_column} which is not in this dataset."
                )

            summary_column = args.summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"You passed dataset={args.dataset} which is not in task_to_keys dictionary "
                    f"and passed summary_column={summary_column} which is not in this dataset."
                )
        else:
            # Use standard dataset and columns for it
            text_column, summary_column = columns

        def process(samples, tokenizer, args, is_eval):
            # Remove pairs where at least one record is None
            inputs, targets = [], []
            prefix = args.source_prefix if args.source_prefix is not None else ""

            for i in range(len(samples[text_column])):
                if samples[text_column][i] is not None and samples[summary_column][i] is not None:
                    inputs.append(prefix + samples[text_column][i])
                    targets.append(samples[summary_column][i])

            padding = "max_length" if args.pad_to_max_length else False
            max_target_length = args.val_max_target_length if is_eval else args.max_target_length

            model_inputs = tokenizer(
                inputs,
                max_length=args.max_source_length,
                padding=padding,
                truncation=True
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_target_length, #? is this ok
                    padding=padding,
                    truncation=True
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels with -100
            # when we want to ignore padding in the loss
            if args.pad_to_max_length and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [ (l if l != tokenizer.pad_token_id else -100) for l in label]
                    for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        return super().process_dataset(tokenizer, process)

    def get_trainer_class(self):
        return Seq2SeqTrainer

    def get_training_args(self):
        args = self.args
        output_dir = f"./src/llm/{args.results_path}"

        training_args = Seq2SeqTrainingArguments(
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=(
                args.eval_batch_size if args.eval_batch_size else
                args.batch_size
            ),
            gradient_accumulation_steps=args.grad_acc_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.lr,
            num_train_epochs=args.n_epochs_train,
            max_steps=args.max_steps_train,
            logging_steps=args.logging_steps,
            eval_strategy=args.eval_strategy,
            save_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            output_dir=output_dir, #mb add logging dir
            overwrite_output_dir=True,
            run_name=args.run_name,
            report_to=["wandb" if args.wandb else "none"],
            predict_with_generate=args.predict_with_generate,
            # add more nlg specific arguments like num_beams
        )

        return training_args

    def get_data_collator(self, model, tokenizer):
        label_pad_token_id = (
            -100 if self.args.ignore_pad_token_for_loss else
            tokenizer.pad_token_id
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=(8 if self.args.fp16 else None),
        )

        return data_collator

    def get_metrics_function(self, tokenizer):
        metric = load_metric("rouge")

        def compute_metrics(p: EvalPrediction):
            ##Before: preds, labels = eval_preds
            preds, labels = p.predictions, p.label_ids

            if isinstance(preds, tuple) or isinstance(labels, tuple):
                ##Before: preds = preds[0]
                raise ValueError("Why tuple?")

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            
            # Replace -100 in the labels as we can't decode them
            if self.args.ignore_pad_token_for_loss:
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds]
            labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels]

            results = metric.compute(predictions=preds, references=labels, use_stemmer=True)
            result = {key: round(val.mid.fmeasure * 100, 4) for key, val in results.items()}
            prediction_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
            result["gen_len"] = round(np.mean(prediction_lens), 4)

            return result
    
        return compute_metrics if self.args.predict_with_generate else None


def create_problem(args):
    task_type = args.task_type
    dataset = args.dataset

    if task_type == "SEQ_CLS":
        return GLUE_Problem(args)
    elif task_type == "QUESTION_ANS":
        return SQUAD_Problem(args)
    elif task_type == "SEQ_2_SEQ_LM":
        return NLG_Problem(args)
    #task_type == "CAUSAL_LM"
    elif dataset == "gsm8k":
        return GSM8K_Problem(args)
    elif dataset == "math_qa":
        return MathQA_Problem(args)

    #~ fix or remove this error message
    raise ValueError(f"Unknown dataset={dataset}, choose from available options:")
        


##### QuestionAnsweringTrainer #####
from transformers.trainer_utils import PredictionOutput

def postprocess_qa_predictions(
    samples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    v2_with_negative: bool = False,
    null_score_diff_threshold: float = 0.0,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        samples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        v2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`v2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`v2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        )
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Number of predictions ({len(predictions[0])}) is not equal to "
            f"number of features ({len(features)})."
        )

    # Build a map example to its corresponding features
    sample_id_to_index = {key: i for i, key in enumerate(samples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[sample_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if v2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Loop over all the samples
    for sample_idx, sample in enumerate(tqdm(samples)):
        # Those are the indices of the features associated to the current example
        feature_indices = features_per_example[sample_idx]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example
        for feature_idx in feature_indices:
            # We grab the predictions of the model for this feature
            start_logits = all_start_logits[feature_idx]
            end_logits = all_end_logits[feature_idx]
            # This is what will allow us to map some the positions in our logits to span of texts
            # in the original context
            offset_mapping = features[feature_idx]["offset_mapping"]
            # If `token_is_max_context` is provided we will remove answers that do not have
            # the maximum context available in the current feature
            token_is_max_context = features[feature_idx].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits
            start_indices = np.argsort(start_logits)[-1:-(n_best_size + 1):-1].tolist()
            end_indices = np.argsort(end_logits)[-1:-(n_best_size + 1):-1].tolist()
            for start_idx in start_indices:
                for end_idx in end_indices:
                    # Ignore out-of-scope answers, either because the indices are out of bounds or
                    # correspond to part of the input_ids that are not in the context
                    if (
                        start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping) or
                        offset_mapping[start_idx] is None or offset_mapping[end_idx] is None or
                        len(offset_mapping[start_idx]) < 2 or len(offset_mapping[end_idx]) < 2
                    ):
                        continue

                    # Ignore answers with a length that is either < 0 or > max_answer_length
                    if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_length:
                        continue

                    # Ignore answer that don't have the maximum context available (if provided)
                    if (
                        token_is_max_context is not None and
                        not token_is_max_context.get(str(start_idx), False)
                    ):
                        continue

                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_idx][0], offset_mapping[end_idx][1]),
                        "score": start_logits[start_idx] + end_logits[end_idx],
                        "start_logit": start_logits[start_idx],
                        "end_logit": end_logits[end_idx],
                    })

        if v2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions
        preds = sorted(prelim_predictions, key=(lambda x: x["score"]), reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score
        if (
            v2_with_negative and min_null_prediction is not None and
            all([p["offsets"] != (0, 0) for p in preds])
        ):
            preds.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context
        context = sample["context"]
        for p in preds:
            offsets = p.pop("offsets")
            p["text"] = context[offsets[0]:offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake
        # prediction to avoid failure
        if (len(preds) == 0) or (len(preds) == 1 and preds[0]["text"] == ""):
            preds.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores
        scores = np.array([p.pop("score") for p in preds])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions
        for prob, p in zip(probs, preds):
            p["probability"] = prob

        # Pick the best prediction, if the null answer is not possible, this is easy
        if not v2_with_negative:
            all_predictions[sample["id"]] = preds[0]["text"]
        else: # otherwise we first need to find the best non-empty prediction
            i = 0
            while preds[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold
            score_diff = (
                null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            )
            scores_diff_json[sample["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[sample["id"]] = ""
            else:
                all_predictions[sample["id"]] = best_non_null_pred["text"]

        # Make `preds` JSON-serializable by casting np.float back to float.
        all_nbest_json[sample["id"]] = [
            {
                key: (
                    float(val) if isinstance(val, (np.float16, np.float32, np.float64)) else
                    val
                )
                for key, val in p.items()
            }
            for p in preds
        ]

    # If we have an output_dir, save all those dicts
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, 'w') as output:
            output.write(json.dumps(all_predictions, indent=4) + "\n")

        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, 'w') as output:
            output.write(json.dumps(all_nbest_json, indent=4) + "\n")

        if v2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, 'w') as output:
                output.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def qa_postprocess_function(samples, features, predictions, stage, config):
        # Match the start logits and end logits to answers in the original context
        predictions = postprocess_qa_predictions(
            samples=samples,
            features=features,
            predictions=predictions,
            v2_with_negative=config.v2_with_negative,
            null_score_diff_threshold=config.null_score_diff_threshold,
            n_best_size=config.n_best_size,
            max_answer_length=config.max_answer_length,
            output_dir=config.output_dir, #~ mb remove
            prefix=stage,
        )

        # Format the result to the format that metric expects
        formatted_predictions = [
            (
                {"id": key, "prediction_text": val} if not config.v2_with_negative else
                {"id": key, "prediction_text": val, "no_answer_probability": 0.0}
            )
            for key, val in predictions.items()
        ]

        references = [
            {"id": sample["id"], "answers": sample["answers"]}
            for sample in samples
        ]

        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


class QuestionAnsweringTrainer(Trainer):

    def evaluate(
        self, eval_dataset=None, eval_samples=None,
        ignore_keys=None, metric_key_prefix: str = "eval"
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else
            self.evaluation_loop
        )

        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=(True if compute_metrics is None else None),
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            eval_preds = qa_postprocess_function(
                eval_samples, eval_dataset, output.predictions, "eval", self.args
            )
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with str(metric_key_prefix) + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics

    def predict(
        self, test_dataset, test_samples,
        ignore_keys=None, metric_key_prefix: str = "test"
    ):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else
            self.evaluation_loop
        )

        try:
            output = eval_loop(
                test_dataloader,
                description="Prediction",
                prediction_loss_only=(True if compute_metrics is None else None),
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is None:
            return output

        predictions = qa_postprocess_function(
            test_samples, test_dataset, output.predictions, "predict", self.args
        )
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with str(metric_key_prefix) + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics
        )


@dataclass
class QuestionAnsweringTrainingArguments(TrainingArguments):

    v2_with_negative: bool = field(
        default=False,
        metadata={"help": "If true, some of the examples do not have an answer"}
    )

    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={"help": "The threshold used to select the null answer"}
    )

    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer"}
    )

    max_answer_length: int = field(
        default=30,
        metadata={"help": "The maximum length of an answer that can be generated"}
    )

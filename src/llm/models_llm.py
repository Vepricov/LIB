from loguru import logger

import torch
from peft import prepare_model_for_kbit_training

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    BitsAndBytesConfig,
    AutoConfig,
)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


class ModelFramework:

    task_type_to_model_type = {
        "CAUSAL_LM": AutoModelForCausalLM,
        "SEQ_CLS": AutoModelForSequenceClassification,
        "QUESTION_ANS": AutoModelForQuestionAnswering,
        "SEQ_2_SEQ_LM": AutoModelForSeq2SeqLM,
    }

    def __init__(self, args):
        if args.task_type not in ModelFramework.task_type_to_model_type:
            options = ["\'" + key + "\'" for key in ModelFramework.task_type_to_model_type.keys()]
            raise ValueError(
                f"Unknown task_type={args.task_type}. Either provide custom implementation or "
                f"choose from available options: {', '.join(options)}."
            )

        self.args = args
        
        if args.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif args.dtype == "float16":
            self.dtype = torch.float16
        elif args.dtype == "float32":
            self.dtype = torch.float32
        elif args.dtype == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(
                f"Unknown dtype={args.dtype}. Either provide custom implementation or choose "
                f"from available options: 'bfloat16', 'float16', 'float32' or 'float64'."
            )

        self.bnb_config = self.get_bnb_config()
        self.target_modules = self.get_target_modules()

    def load_model_and_tokenizer(self):
        args = self.args
        AutoModel = ModelFramework.task_type_to_model_type[args.task_type]

        if args.task_type == "SEQ_CLS":
            if args.dataset == "stsb":  # regression case
                num_labels = 1
            elif args.dataset == "mnli":
                num_labels = 3
            else:
                num_labels = 2

            config = AutoConfig.from_pretrained(
                args.config if args.config else args.model,
                num_labels=num_labels,
                finetuning_task=args.dataset,
                cache_dir=args.cache_dir,
                revision=args.model_revision,
            )
        else:
            config = None

        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            quantization_config=self.bnb_config,
            #attn_implementation=self.attn_implementation, #? add/remove, was not used for NLG and SQUAD
            revision=args.model_revision,
            from_tf=(".ckpt" in args.model),
            low_cpu_mem_usage=True,
            config=config,
            #cache_dir=args.cache_dir, #~ mb use args.output_dir
            #use_cache=False, #tried to fix nlg error
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer if args.tokenizer else args.model,
            use_fast=args.use_fast_tokenizer,
            revision=args.model_revision,
            padding_side=args.padding_side,
            cls_token="[CLS]", #~ only used by squad, but seems like it would not break for other tasks
            #cache_dir=args.cache_dir, # CasualLM does not use it
            #use_cache=False, #tried to fix nlg error
        )

        if args.task_type == "CAUSAL_LM":
            tokenizer.pad_token_id = 0
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        if self.bnb_config is not None:
            ##model.gradient_checkpointing_enable() <- seems to work fine without it
            model = prepare_model_for_kbit_training(model)

        if args.task_type == "CAUSAL_LM":
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))
        elif args.task_type == "SEQ_2_SEQ_LM":
            model.resize_token_embeddings(len(tokenizer))

            if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
                if args.lang is None:
                    class_name = tokenizer.__class__.__name__
                    raise ValueError(
                       f"{class_name} is a multilingual tokenizer which requires --lang argument."
                    )
                tokenizer.src_lang = args.lang
                tokenizer.tgt_lang = args.lang

                # For multilingual translation models like mBART-50 and M2M100 we need to force the
                # target language token as the first generated token. We ask the user to explicitly
                # provide this as --forced_bos_token argument
                forced_bos_token_id = (
                    tokenizer.lang_code_to_id[args.forced_bos_token]
                    if args.forced_bos_token is not None
                    else None
                )
                model.config.forced_bos_token_id = forced_bos_token_id

            is_mbart_tokenizer = isinstance(tokenizer, MBartTokenizer)
            is_mbart_tokenizer_fast = isinstance(tokenizer, MBartTokenizerFast)
            if model.config.decoder_start_token_id is None:
                if is_mbart_tokenizer or is_mbart_tokenizer_fast:
                    model.config.decoder_start_token_id = (
                        tokenizer.lang_code_to_id[args.lang] if is_mbart_tokenizer else
                        tokenizer.convert_tokens_to_ids(args.lang)
                    )
                else:
                    raise ValueError(
                        "Make sure that config.decoder_start_token_id is correctly defined, "
                        "currently it is None."
                    )

            if (
                hasattr(model.config, "max_position_embeddings") and
                model.config.max_position_embeddings < args.max_source_length
            ):
                if args.resize_position_embeddings is None:
                    logger.warning(
                        f"Increasing the model's number of position embedding vectors from "
                        f"{model.config.max_position_embeddings} to {args.max_source_length}."
                    )
                    model.resize_position_embeddings(args.max_source_length)
                elif args.resize_position_embeddings:
                    model.resize_position_embeddings(args.max_source_length)
                else:
                    raise ValueError(
                        f"You passed max_source_length={args.max_source_length}, but the model "
                        f"only has {model.config.max_position_embeddings} position encodings. "
                        f"Consider either reducing --max_source_length to "
                        f"{model.config.max_position_embeddings} or to automatically resize the "
                        f"model's position encodings by passing --resize_position_embeddings."
                    )

        return model, tokenizer

    def get_bnb_config(self):
        raise NotImplementedError("Implement this method in a successor class")

    def get_target_modules(self):
        raise NotImplementedError("Implement this method in a successor class")


class LlamaFramework(ModelFramework):

    def get_bnb_config(self):
        if self.args.task_type == "CAUSAL_LM":
            torch_dtype = self.dtype
        else:
            torch_dtype = (
                torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else
                torch.float16
            )

        if self.args.quant_bit == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif self.args.quant_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",      #~ not sure if we need this, but let it be
                bnb_4bit_use_double_quant=True, #~ probably a good choice
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            bnb_config = None

        return bnb_config

    def get_target_modules(self):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]


class T5Framework(ModelFramework):

    def get_bnb_config(self):
        torch_dtype = self.dtype

        if self.args.quant_bit == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif self.args.quant_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            bnb_config = None

        return bnb_config

    def get_target_modules(self):
        return ["v", "o", "q", "k", "wi"]


class DebertaFramework(ModelFramework):

    def get_bnb_config(self):
        # Mixed precision is not supported, see https://github.com/huggingface/transformers/issues/35332
        return None

    def get_target_modules(self):
        return [
            "query_proj",
            "key_proj",
            "value_proj",
            "intermediate.dense",
            "output.dense",
        ]


def create_model_framework(args):
    model_name = args.model.lower()

    if "llama" in model_name:
        return LlamaFramework(args)
    elif "deberta" in model_name:
        return DebertaFramework(args)
    elif "t5" in model_name:
        return T5Framework(args)

    raise ValueError(f"Unsupported model type: {model_name}")

import torch
import datasets
import wandb

from loguru import logger
import peft

import utils
from utils_llm import DatasetRegistry

import warnings

DATASETS = [
    "coin_flip",
    "gsm8k",
    "aqua",
    "commonsensqa",
    "boolq",
    "addsub",
    "multiarith",
    "singleeq",
    "strategyqa",
    "svamp",
    "bigbench_date",
    "object_tracking",
    "coin_flip",
    "last_letters",
    "mathqa",
    "hella_swag",
    "arc_challenge",
]

warnings.filterwarnings("ignore")

from optimizers.main import get_optimizer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    pipeline,
)


class Finetuner:
    """Main class for downstream finetuning"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.builder = None
        self.setup_logging()
        if self.args.wandb:
            self.setup_wandb()

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info(f"Starting downstream finetuning for dataset: {self.args.dataset}")

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.args.wandb_project,
            tags=[self.args.model, self.args.dataset, self.args.optimizer],
            name=self.args.run_name,
            config=self.args,
        )

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configurations"""
        logger.info("Loading model and tokenizer...")

        # Determine optimal settings based on GPU capability
        # [TODO] add flash_attention
        if torch.cuda.get_device_capability()[0] >= 8:
            attn_implementation = "eager"  # fix flash_attention_2
        else:
            attn_implementation = "eager"

        # Set dtype
        if self.args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.args.dtype == "float16":
            torch_dtype = torch.float16
        elif self.args.dtype == "float32":
            torch_dtype = torch.float32
        elif self.args.dtype == "float64":
            torch_dtype = torch.float64
        # Setup quantization config
        if self.args.quant_bit == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif self.args.quant_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
        )

        print(self.model)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model,
            use_fast=self.args.use_fast_tokenizer,
            padding_side=self.args.padding_side,
        )
        self.tokenizer.pad_token_id = 0

        # Resize token embeddings if necessary
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Model and tokenizer loaded successfully.")

    def setup_peft(self):
        """Setup PEFT (Parameter Efficient Fine-Tuning) adapters"""
        logger.info("Setting up PEFT adapters...")

        peft_args = utils.get_peft_arguments(self.args)
        peft_args.task_type = "CAUSAL_LM"
        if self.args.ft_strategy.lower() != "full":
            self.model = peft.get_peft_model(self.model, peft_args)
        else:
            for name, param in self.model.named_parameters():
                is_trainable = any(
                    ft_param_name in name for ft_param_name in peft_args.target_modules
                )
                param.requires_grad = is_trainable
        # Print trainable parameters info
        tr_param_count, all_param_count, tr_persent = utils.print_trainable_params(
            self.model, verbose=True
        )

        num_peft_adapters = utils.count_atapters(self.model, self.args.ft_strategy)

        if self.args.wandb:
            wandb.log(
                {
                    "trainable_params_count": tr_param_count,
                    "total_param_count": all_param_count,
                    "trainable_params_percentage": tr_persent,
                    "num_peft_adapters": num_peft_adapters,
                }
            )

    def load_datasets(self):
        """Load and prepare both training and evaluation datasets"""
        logger.info("Loading datasets...")

        # Set dataset paths and create builder
        DatasetRegistry.set_dataset_paths(self.args)
        self.builder = DatasetRegistry.create_builder(self.args)
        data = self.builder.get_data()

        # Extract train and eval data
        train_questions, train_answers = data["train"]
        eval_questions, eval_answers = data["eval"]

        # Create datasets
        self.train_dataset = None
        self.eval_dataset = None

        if train_questions and len(train_questions) > 0:
            train_dict = {
                "question": ["Problem: " + q for q in train_questions],
                "response": train_answers,
                "raw_x": train_questions,
                "raw_y": train_answers,
            }
            self.train_dataset = datasets.Dataset.from_dict(train_dict)

        if eval_questions and len(eval_questions) > 0:
            eval_dict = {
                "question": ["Problem: " + q for q in eval_questions],
                "response": eval_answers,
                "raw_x": eval_questions,
                "raw_y": eval_answers,
            }
            self.eval_dataset = datasets.Dataset.from_dict(eval_dict)

    def prepare_training_dataset(self):
        """Prepare dataset for training"""
        if self.train_dataset is None:
            logger.warning("No training dataset available")
            return None

        logger.info("Preparing training dataset...")
        return self.builder.preprocess_dataset(
            self.tokenizer,
            self.args.max_seq_length,
            self.args.seed,
            self.train_dataset,
            eval_mode=False,
        )

    def prepare_evaluation_dataset(self):
        """Prepare dataset for evaluation"""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return None

        logger.info("Preparing evaluation dataset...")
        dataset = self.eval_dataset.map(
            lambda sample: self.builder.create_prompt_formats(sample, eval_mode=True)
        )
        return dataset

    def get_optimizer(self):
        """Get optimizer for training"""
        optimizer = get_optimizer(self.args, self.model)
        return optimizer

    def train(self):
        """Execute training process"""
        if self.args.do_not_train:
            logger.info("Training skipped (do_not_train=True)")
            return

        logger.info(f"Starting training for dataset: {self.args.dataset}")

        # Prepare training dataset
        train_dataset = self.prepare_training_dataset()
        if train_dataset is None:
            logger.error("No training data available")
            return

        # Setup trainer
        training_args = TrainingArguments(
            do_train=not self.args.do_not_train,
            do_eval=not self.args.do_not_eval,
            do_predict=self.args.do_predict,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=(
                self.args.eval_batch_size
                if self.args.eval_batch_size
                else self.args.batch_size
            ),
            gradient_accumulation_steps=self.args.grad_acc_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            warmup_ratio=self.args.warmup_ratio,
            learning_rate=self.args.lr,
            num_train_epochs=self.args.n_epoches_train,
            max_steps=self.args.max_steps_train,
            logging_steps=self.args.logging_steps,
            eval_strategy="no",
            eval_steps=self.args.eval_steps,
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            bf16=(self.args.dtype == "bfloat16"),
            fp16=(self.args.dtype == "float16"),
            logging_dir=f"./src/fine_tuning/llm/{self.args.results_path}/{self.args.run_name.split('__')[0]}",
            output_dir=f"./src/fine_tuning/llm/{self.args.results_path}/{self.args.run_name.split('__')[0]}",
            run_name=self.args.run_name,
            report_to=["wandb"] if self.args.wandb else ["none"],
        )

        optimizer = self.get_optimizer()

        self.trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            optimizers=[optimizer, None],  # Scheduler will be added in the hf trainer
        )

        # Clean up memory before training
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Train the model
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        logger.info(f"Training completed. Metrics: {metrics}")

    def evaluate(self):
        """Execute evaluation process"""
        if self.args.do_not_eval:
            logger.info("Evaluation skipped (do_not_eval=True)")
            return 0, 0

        logger.info(f"Starting evaluation for dataset: {self.args.dataset}")

        # Prepare evaluation dataset
        eval_dataset = self.prepare_evaluation_dataset()
        if eval_dataset is None:
            logger.error("No evaluation data available")
            return 0, 0

        # Setup text generation pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        # Evaluate model
        correct, total = 0, 0

        logger.info(f"Evaluation sample prompt: {eval_dataset['text'][0]}")

        for i, item in enumerate(eval_dataset):
            try:
                # Generate prediction
                predicted_response = generator(
                    item["text"],
                    max_new_tokens=len(self.tokenizer.tokenize(item["raw_y"])) + 1,
                    num_return_sequences=1,
                )[0]["generated_text"]
                predicted_response = predicted_response.replace("\n", "")
                item["raw_y"] = item["raw_y"].replace("\n", "")
                # Check if correct answer is in prediction
                if item["raw_y"] in predicted_response:
                    correct += 1

                # Debug output
                print(f">>Prediction<<: {predicted_response}")
                print(f">>>>Answer<<<<: {item['raw_y']}")

                total += 1

                # Progress update
                accuracy = (correct / total) * 100 if total > 0 else 0
                print(f"[{i+1}/{len(eval_dataset)}] Accuracy: {accuracy:.2f}%")
                print("=" * 50)

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue

        return correct, total

    def log_final_results(self, correct, total):
        """Log final evaluation results"""
        if total > 0:
            final_accuracy = (correct / total) * 100
            logger.info(f"[FINAL] Accuracy: {final_accuracy:.2f}%")
            self.trainer.save_metrics("eval", {"accuracy": final_accuracy})
            if self.args.wandb:
                wandb.log({"final_accuracy": final_accuracy})
        else:
            logger.info("No samples were successfully evaluated.")

    def run(self):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()
        self.setup_peft()

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train()
        correct, total = self.evaluate()

        # Log results
        self.log_final_results(correct, total)
        logger.info("Pipeline completed successfully")


def main(args):
    """Main entry point"""
    # Create and run finetuner
    finetuner = Finetuner(args)
    finetuner.run()


if __name__ == "__main__":
    main(None)

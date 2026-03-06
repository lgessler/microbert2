import logging
from typing import Optional

from tango import Step
from tango.common import FromParams
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class LoraConfig(FromParams):
    """
    Configuration for LoRA (Low-Rank Adaptation) during continued pretraining.

    Passed directly to ``peft.LoraConfig``; requires the ``peft`` package.
    """

    def __init__(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        bias: str = "none",
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Default to the query/value projections, which is standard for BERT-family models.
        self.target_modules = target_modules or ["query", "value"]
        self.bias = bias


logger = logging.getLogger(__name__)


@Step.register("microbert2.microbert.eval.continued_pretraining::continued_mlm_pretraining")
class ContinuedMLMPretraining(Step):
    """
    A Tango step for continued MLM pretraining of a HuggingFace model on a text corpus.

    After training, saves the base model (without the MLM head) so the output
    can be directly used by eval steps like EvaluateNER and EvaluateDependencyParsing.
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        model_name_or_path: str,
        train_data_path: str,
        output_path: str,
        dev_data_path: Optional[str] = None,
        mlm_probability: float = 0.15,
        max_length: int = 512,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.06,
        weight_decay: float = 0.01,
        lora_config: Optional[LoraConfig] = None,
    ) -> str:
        """
        Run continued MLM pretraining.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            train_data_path: Plain text file, one sentence per line.
            output_path: Where to save the base model after training.
            dev_data_path: Optional plain text file for eval_loss during training.
            mlm_probability: Probability of masking tokens (default: 0.15).
            max_length: Maximum sequence length (default: 512).
            batch_size: Per-device batch size (default: 32).
            learning_rate: Learning rate (default: 2e-5).
            num_epochs: Number of training epochs (default: 3).
            warmup_ratio: Fraction of steps used for LR warmup (default: 0.06).
            weight_decay: Weight decay (default: 0.01).
            lora_config: Optional LoRA configuration. If None (default), all
                parameters are updated (standard continued pretraining).

        Returns:
            output_path: Path to the saved base model (usable by eval steps).
        """
        from datasets import Dataset

        self.logger.info(f"Loading tokenizer from {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.logger.info(f"Loading model from {model_name_or_path}")
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

        if lora_config is not None:
            try:
                from peft import LoraConfig as PeftLoraConfig, get_peft_model
            except ImportError:
                raise ImportError("lora_config requires the `peft` package: pip install peft")
            peft_config = PeftLoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            self.logger.info("LoRA enabled")

        def load_text(path: str):
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(lines)} lines from {path}")
            return lines

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
            )

        train_lines = load_text(train_data_path)
        train_dataset = Dataset.from_dict({"text": train_lines})
        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

        eval_dataset = None
        if dev_data_path:
            dev_lines = load_text(dev_data_path)
            eval_dataset = Dataset.from_dict({"text": dev_lines})
            eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

        training_args = TrainingArguments(
            output_dir=output_path,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            load_best_model_at_end=eval_dataset is not None,
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        self.logger.info("Starting continued pretraining...")
        trainer.train()

        self.logger.info(f"Saving base model to {output_path}")
        # For LoRA models, merge the adapter weights back into the base model
        # before saving so the output is a plain HuggingFace model.
        if lora_config is not None:
            model = model.merge_and_unload()
        model.base_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return output_path

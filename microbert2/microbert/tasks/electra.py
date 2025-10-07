import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tango.common import Lazy
from tango.integrations.transformers import Tokenizer
from torchmetrics import Accuracy
from torchmetrics.text import Perplexity
from transformers.activations import gelu

from microbert2.microbert.model.model import Model
from microbert2.microbert.tasks.task import MicroBERTTask

logger = logging.getLogger(__name__)


@torch.jit.script
def _tied_generator_forward(hidden_states, embedding_weights):
    hidden_states = torch.einsum("bsh,eh->bse", hidden_states, embedding_weights)
    return hidden_states


class TiedElectraGeneratorPredictions(nn.Module):
    """Like ElectraGeneratorPredictions, but accepts a torch.nn.Parameter from an embedding module"""

    def __init__(self, config, embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.embedding_weights = embedding_weights

    def forward(self, x):
        x = self.dense(x)
        x = gelu(x)
        x = self.LayerNorm(x)
        logits = _tied_generator_forward(x, self.embedding_weights)
        return logits


class ElectraHead(nn.Module):
    """Head for the ELECTRA task"""

    def __init__(self, config, tokenizer, embedding_weights, temperature, rtd_weight):
        super().__init__()
        self.tokenizer = tokenizer
        self.generator_head = TiedElectraGeneratorPredictions(config, embedding_weights)
        self.discriminator_head = nn.Linear(config.hidden_size, 1)
        self.vocab_size = config.vocab_size
        self.temperature = temperature
        self.rtd_weight = rtd_weight
        self.accuracy = Accuracy(task="binary")
        self.perplexity = Perplexity()

    def forward(self, hidden_masked, input_ids, attention_mask, token_type_ids, labels, encoder, **kwargs):
        # Generate MLM predictions using last layer
        hidden_masked = hidden_masked[-1]
        mlm_logits = self.generator_head(hidden_masked)
        if not (labels != -100).any():
            masked_lm_loss = torch.tensor(0.0, device=hidden_masked[-1].device)
        else:
            flat_mlm_logits = mlm_logits.view(-1, self.vocab_size)
            flat_mlm_labels = labels.view(-1)
            masked_lm_loss = F.cross_entropy(flat_mlm_logits, flat_mlm_labels, ignore_index=-100)
            self.perplexity.update(mlm_logits, labels)

        # Take predicted token IDs
        # mlm_preds = mlm_logits.argmax(-1)
        mlm_probs = F.softmax(mlm_logits, dim=-1)

        # Optional: Apply temperature to control randomness
        if self.temperature != 1.0:
            mlm_probs = mlm_probs.pow(1.0 / self.temperature)
            mlm_probs = mlm_probs / mlm_probs.sum(dim=-1, keepdim=True)

        # Sample from the distribution
        mlm_preds = torch.multinomial(
            mlm_probs.reshape(-1, self.vocab_size),
            num_samples=1,
        ).reshape(mlm_probs.size(0), -1)

        # Combine them to get labels for discriminator
        replaced = (~input_ids.eq(mlm_preds)) & (labels != -100)

        # Make inputs for discriminator, feed them into the encoder once more
        replaced_input_ids = torch.where(replaced, mlm_preds, input_ids)
        second_encoder_output = encoder(
            input_ids=replaced_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        discriminator_output = self.discriminator_head(second_encoder_output.last_hidden_state).squeeze(-1)

        # Compute replaced token detection BCE loss on eligible tokens (all non-special tokens)
        bce_mask = (
            (replaced_input_ids != self.tokenizer.cls_token_id)
            & (replaced_input_ids != self.tokenizer.sep_token_id)
            & (replaced_input_ids != self.tokenizer.pad_token_id)
        )
        rtd_logits = torch.masked_select(discriminator_output, bce_mask)
        rtd_preds = (rtd_logits > 0).long()
        rtd_labels = torch.masked_select(replaced.float(), bce_mask)
        rtd_loss = F.binary_cross_entropy_with_logits(rtd_logits, rtd_labels)
        self.accuracy.update(rtd_preds, rtd_labels)

        return {
            "rtd_loss": rtd_loss,
            "rtd_acc": self.accuracy.compute() * 100,
            "mlm_loss": masked_lm_loss,
            "perplexity": self.perplexity.compute(),
            "loss": (self.rtd_weight * rtd_loss) + masked_lm_loss,
        }


@MicroBERTTask.register("microbert2.microbert.tasks.electra.ElectraTask")
class ElectraTask(MicroBERTTask):
    def __init__(
        self,
        dataset: dict[Literal["train", "dev", "test"], list[dict[str, Any]]],
        tokenizer: Lazy[Tokenizer],
        mlm_probability: float = 0.15,
        mlm_mask_replace_prob: float = 1.0,
        mlm_random_replace_prob: float = 0.0,
        temperature: float = 1.0,
        rtd_weight: float = 50.0,
    ):
        super().__init__()
        self._dataset = dataset
        self.config = None
        self._head = None
        self.collator = None
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_replace_prob = mlm_mask_replace_prob
        self.random_replace_prob = mlm_random_replace_prob
        self.temperature = temperature
        self.rtd_weight = rtd_weight

    @property
    def slug(self) -> str:
        return "electra"

    @property
    def universal(self) -> bool:
        return True

    @property
    def dataset(self) -> dict[Literal["train", "dev", "test"], list[dict[str, Any]]]:
        return self._dataset

    @property
    def data_keys(self) -> list[str]:
        return []

    def construct_head(self, model):
        embedding_weights = model.encoder.embedding_weights

        # If tokenizer is still a Lazy instance, construct it
        if isinstance(self.tokenizer, Lazy):
            self.tokenizer = self.tokenizer.construct()

        self._head = ElectraHead(
            config=model.encoder.config,
            tokenizer=self.tokenizer,
            embedding_weights=embedding_weights,
            temperature=self.temperature,
            rtd_weight=self.rtd_weight,
        )
        logger.info(f"Electra head initialized with {embedding_weights.shape[0]} embeddings")
        return self._head

    @property
    def inst_proportion(self) -> float:
        return 1.0

    def tensorify_data(self, key: str, value: Any) -> torch.Tensor:
        raise ValueError(f"Unknown key: {key}")

    def collate_data(self, key: str, values: list[torch.Tensor]) -> torch.Tensor:
        raise ValueError(f"Unknown key: {key}")

    def null_tensor(self, key) -> torch.Tensor:
        raise ValueError(f"Unknown key: {key}")

    @property
    def progress_items(self) -> list[str]:
        return ["loss", "rtd_acc", "perplexity"]

    def _mask_tokens(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers import DataCollatorForLanguageModeling

        if isinstance(self.tokenizer, Lazy):
            self.tokenizer = self.tokenizer.construct()
            self.collator = DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
                mask_replace_prob=self.mask_replace_prob,
                random_replace_prob=self.random_replace_prob,
            )
        outputs = self.collator.torch_mask_tokens(input_ids)
        return outputs

    def transform_collator_output(self, output: dict[str, Any]) -> dict[str, Any]:
        if "input_ids" in output:
            masked, labels = self._mask_tokens(output["input_ids"])
            output["input_ids_masked"] = masked
            output["labels"] = labels
        return output

    def reset_metrics(self):
        self._head.perplexity.reset()
        self._head.accuracy.reset()

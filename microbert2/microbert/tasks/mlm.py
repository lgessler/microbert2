import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tango.common import Lazy
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling
from transformers.activations import gelu

from microbert2.microbert.model.model import Model
from microbert2.microbert.tasks.task import MicroBERTTask

logger = logging.getLogger(__name__)


@torch.jit.script
def _tied_generator_forward(hidden_states, embedding_weights):
    hidden_states = torch.einsum("bsh,eh->bse", hidden_states, embedding_weights)
    return hidden_states


class TiedRobertaLMHead(nn.Module):
    """Version of RobertaLMHead which accepts an embedding torch.nn.Parameter for output probabilities"""

    def __init__(self, config, embedding_weights):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.embedding_weights = embedding_weights

    def forward(self, hidden_masked, labels, **kwargs):
        x = self.dense(hidden_masked[-1])
        x = gelu(x)
        x = self.layer_norm(x)
        x = _tied_generator_forward(x, self.embedding_weights)

        if not (labels != -100).any():
            return {"mlm": torch.tensor(0.0, device=hidden_masked.device)}

        masked_lm_loss = F.cross_entropy(x.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        perplexity = torch.exp(masked_lm_loss)
        return {"loss": masked_lm_loss, "perplexity": perplexity}


@MicroBERTTask.register("microbert2.microbert.tasks.mlm.MLMTask")
class MLMTask(MicroBERTTask):
    def __init__(
        self,
        dataset: dict[Literal["train", "dev", "test"], list[dict[str, Any]]],
        tokenizer: Lazy[Tokenizer],
        mlm_probability: float = 0.15,
        mlm_mask_replace_prob: float = 1.0,
        mlm_random_replace_prob: float = 0.0,
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

    @property
    def slug(self) -> str:
        return "mlm"

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
        self.config = model.encoder.config
        self._head = TiedRobertaLMHead(config=self.config, embedding_weights=embedding_weights)
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
        return ["perplexity", "loss"]

    def _mask_tokens(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

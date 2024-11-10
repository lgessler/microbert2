from typing import Any, Dict, List, Optional, Tuple

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

from microbert2.microbert.tasks.task import MicroBERTTask


# copied from HF implementation, but does not do the 10% [UNK] and 10% random word replacement
def torch_mask_tokens(
    inputs: Any, tokenizer, special_tokens_mask: Optional[Any] = None, mlm_probability: float = 0.15
) -> Tuple[Any, Any]:
    import torch

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    # inputs[indices_random] = random_words[indices_random]

    # # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


@DataCollator.register("microbert2.data.collator::collator")
class MicroBERTCollator(DataCollator):
    def __init__(
        self,
        tokenizer: Lazy[Tokenizer],
        tasks: list[MicroBERTTask] = [],
        mask_only: bool = False,
    ):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.mask_only = mask_only
        self.tasks = tasks
        if not mask_only:
            self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def __call__(self, batch) -> Dict[str, Any]:
        output = {}
        output["dataset_id"] = torch.stack([item["dataset_id"] for item in batch], dim=0)
        for k in ["input_ids", "attention_mask", "token_spans", "token_type_ids"]:
            output[k] = pad_sequence(
                (item[k] for item in batch),
                batch_first=True,
                padding_value=(0 if k != "input_ids" else self.text_pad_id),
            )
            if k in ["token_spans"]:
                output[k] = output[k].view(output[k].shape[0], -1, 2)
            if k == "input_ids":
                if not self.mask_only:
                    masked, labels = self.mlm_collator.torch_mask_tokens(output[k])
                else:
                    masked, labels = torch_mask_tokens(output[k], self.tokenizer)
                while (labels == -100).all():
                    if not self.mask_only:
                        masked, labels = self.mlm_collator.torch_mask_tokens(output[k])
                    else:
                        masked, labels = torch_mask_tokens(output[k], self.tokenizer)
                output["input_ids_masked"] = masked
                output["labels"] = labels
        for task in self.tasks:
            for k in task.data_keys:
                output[k] = task.collate_data(k, [item[k] for item in batch])
        return output

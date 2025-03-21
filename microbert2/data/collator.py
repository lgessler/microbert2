from typing import Any, Dict, List, Optional, Tuple

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

from microbert2.microbert.tasks.mlm import MLMTask
from microbert2.microbert.tasks.task import MicroBERTTask


@DataCollator.register("microbert2.data.collator::collator")
class MicroBERTCollator(DataCollator):
    def __init__(
        self,
        tokenizer: Lazy[Tokenizer],
        tasks: list[MicroBERTTask] = [],
    ):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.tasks = tasks

    def __call__(self, batch) -> Dict[str, Any]:
        output = {}
        output["dataset_id"] = torch.stack([item["dataset_id"] for item in batch], dim=0)

        # Pad basic fields
        for k in ["input_ids", "attention_mask", "token_spans", "token_type_ids"]:
            output[k] = pad_sequence(
                (item[k] for item in batch),
                batch_first=True,
                padding_value=(0 if k != "input_ids" else self.text_pad_id),
            )
            if k in ["token_spans"]:
                output[k] = output[k].view(output[k].shape[0], -1, 2)

        # Collate task-specific data
        for task in self.tasks:
            for k in task.data_keys:
                if k == "labels":
                    continue
                output[k] = task.collate_data(k, [item[k] for item in batch])

        for task in self.tasks:
            output = task.transform_collator_output(output)

        return output

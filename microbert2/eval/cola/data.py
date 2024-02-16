from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import datasets
import torch
from datasets import DatasetDict, Sequence, Value
from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence


@Step.register("microbert2.eval.cola.data::read")
class ReadCola(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        tokenizer: Tokenizer,
        train_path: Optional[str] = "data/cola_train.tsv",
        dev_path: Optional[str] = "data/cola_dev.tsv",
        test_path: Optional[str] = "data/cola_dev.tsv",
    ) -> DatasetDict:
        def read_tsv(path):
            xs = []
            with open(path, "r") as f:
                for line in f:
                    if "\t" in line:
                        _, label, _, text = line.strip().split("\t")
                        tokenized = tokenizer.encode_plus(
                            text=text,
                            add_special_tokens=True,
                            return_tensors="pt",
                            return_attention_mask=True,
                            return_token_type_ids=True,
                        )
                        xs.append(
                            {
                                "label": torch.tensor(int(label)),
                                "input_ids": tokenized["input_ids"][0],
                                "attention_mask": tokenized["attention_mask"][0],
                            }
                        )
            return xs

        features = datasets.Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "label": Value(dtype="int32", id=None),
            }
        )

        train_dataset = datasets.Dataset.from_list(read_tsv(train_path), features=features)
        dev_dataset = datasets.Dataset.from_list(read_tsv(dev_path), features=features)
        test_dataset = datasets.Dataset.from_list(read_tsv(test_path), features=features)
        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset, "test": test_dataset}).with_format("torch")


@DataCollator.register("microbert2.eval.cola.data::collator")
class SgclDataCollator(DataCollator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        text_field: str = "input_ids",
    ):
        self.text_pad_id = tokenizer.pad_token_id
        self.text_field = text_field
        self.keys = None

    def __call__(self, batch) -> Dict[str, Any]:
        if self.keys is None:
            self.keys = list(batch[0].keys())
        output = {}
        for k in self.keys:
            if k == "label":
                output[k] = torch.stack([item[k] for item in batch], dim=0)
            else:
                output[k] = pad_sequence(
                    (item[k] for item in batch),
                    batch_first=True,
                    padding_value=(0 if k != self.text_field else self.text_pad_id),
                )

        return output

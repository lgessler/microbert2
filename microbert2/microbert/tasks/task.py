from typing import Any, Literal, Union

import torch
import torch.nn as nn
from tango.common import Registrable


class MicroBERTTask(Registrable):
    """
    A combination of a dataset and a prediction head that will be used in combination with MLM.
    """

    @property
    def slug(self) -> str:
        """
        Short unique str describing the task, e.g. "pos"
        """
        raise NotImplemented()

    @property
    def universal(self) -> bool:
        """
        Whether the task is universal, i.e. should be applied to all instances.
        """
        return False

    @property
    def dataset(self) -> dict[Literal["train", "dev", "test"], list[dict[str, Any]]]:
        """
        A dict containing the three splits (test is optional). Each split is a list of objects
        with every key in `data_keys`.
        """
        raise NotImplemented()

    @property
    def inst_proportion(self) -> float:
        """
        Number of instances from this task's dataset to include in the combined training dataset,
        relative to the number of training instances in the first task. For example, if there were
        500 training sentences in the first task and this value were 0.1, there would be 50 instances
        from this dataset in the final training dataset. This value is ignored for the first task.
        """
        raise NotImplemented()

    @property
    def data_keys(self) -> list[str]:
        """
        A list of string keys describing what is available on each instance. Do NOT include "tokens",
        but note that "tokens" should be included in each instance and should hold a string containing
        the textual content of the instance.
        """
        raise NotImplemented()

    def construct_head(self, model) -> nn.Module:
        """
        Creates the module which will take the encoder's output and perform the task.
        The module will be given:
        - all data associated with the keys in data_keys EXCEPT for "tokens"
        - hidden_masked:
            the hidden representation of every layer with masking
        - token_spans:
            a list of pairs giving the inclusive (on both sides) indexes of the wordpiece span which
            corresponds to an original token. For example:

            Original input:  "Daisies"
            Pretokenized:    ["[CLS]", "Daisies", "[SEP]"]
            WP tokenized:    ["[CLS]", "Dais", "##ies", "[SEP]"]
            token_spans:     [[0, 0], [1, 2], [3, 3]]
        """
        raise NotImplemented()

    def tensorify_data(self, key: str, value: Any) -> torch.Tensor:
        """
        Turns a value into a single tensor for a given key.
        """
        raise NotImplemented()

    def collate_data(self, key: str, values: list[torch.Tensor]) -> torch.Tensor:
        """
        Turns a list of tensors into a single tensor for a given key.
        """
        raise NotImplemented()

    def null_tensor(self, key) -> torch.Tensor:
        """
        Return an empty tensor for the given key that will be used for insts that do not belong
        to this dataset. For most things, this should be 0.
        """
        raise NotImplemented()

    def transform_collator_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """
        Executed on the finished output of the collator. Useful for dynamic data transformations
        such as masking for masked language modeling. Any new keys which are put into `output`
        should NOT be included in `data_keys`. Defaults to no-op.
        """
        return output

    @property
    def progress_items(self) -> list[str]:
        """
        Keys produced by the head which should be logged during training.
        """
        return []

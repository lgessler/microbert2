from typing import Any, Dict, List, Literal, Optional

import conllu
import torch
import torch.nn.functional as F
from allennlp_light import ScalarMix
from allennlp_light.nn.util import sequence_cross_entropy_with_logits
from tango.common import FromParams, Lazy
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import Accuracy

from microbert2.common import dill_dump, dill_load, pool_embeddings
from microbert2.microbert.tasks.task import MicroBERTTask


class XposHead(torch.nn.Module, FromParams):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_tags: int,
        use_layer_mix: bool = True,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, num_tags)
        self.accuracy = Accuracy(num_classes=num_tags, task="multiclass", top_k=1)
        self.use_layer_mix = use_layer_mix
        if self.use_layer_mix:
            self.mix = ScalarMix(num_layers)

    def forward(
        self,  # type: ignore
        hidden: List[torch.Tensor],
        token_spans: torch.LongTensor,
        pos_label: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Remove unneeded padding
        token_spans = token_spans[:, : pos_label.shape[-1] + 2]

        # Number of tagged tokens, i.e. excluding special tokens
        token_counts = token_spans.gt(0).all(-1).sum(-1) - 1
        no_special_token_spans = torch.clone(token_spans)
        # Drop [CLS]
        no_special_token_spans = no_special_token_spans[:, 1:, :]
        # Zero out [SEP]
        for i, c in enumerate(token_counts):
            no_special_token_spans[i, c, :] = 0

        if self.use_layer_mix:
            tokenwise_hidden_states = [pool_embeddings(h, no_special_token_spans) for h in hidden]
            # Drop [SEP]
            hidden = self.mix(tokenwise_hidden_states)[:, :-1, :]
        else:
            tokenwise_hidden_states = pool_embeddings(hidden[-1], no_special_token_spans)
            # Drop [SEP]
            hidden = tokenwise_hidden_states[:, :-1, :]
        logits = self.linear(hidden)

        # Mask out pad tokens
        mask = torch.stack(
            [
                torch.tensor(([True] * c) + ([False] * (token_counts.max().item() - c)), device=token_counts.device)
                for c in token_counts
            ],
            dim=0,
        )

        class_probs = F.softmax(logits)
        preds = class_probs.argmax(-1)

        outputs = {"logits": logits, "preds": preds * mask}
        if pos_label is not None:
            flat_preds = preds.masked_select(mask)
            flat_tags = pos_label.masked_select(mask)
            acc = self.accuracy(flat_preds, flat_tags)
            outputs["loss"] = sequence_cross_entropy_with_logits(logits, pos_label, mask, average="token")
            outputs["accuracy"] = acc
            self.accuracy.reset()

        return outputs


def read_split(path, pos_column):
    result = []
    with open(path, "r") as f:
        for sentence in conllu.parse_incr(f):
            result.append(
                {
                    "tokens": [t["form"] for t in sentence if isinstance(t["id"], int)],
                    "pos_label": [t[pos_column] for t in sentence if isinstance(t["id"], int)],
                }
            )
    return result


@MicroBERTTask.register("microbert2.microbert.tasks.ud_pos.UDPOSTask")
class UDPOSTask(MicroBERTTask):
    def __init__(
        self,
        head: Lazy[XposHead],
        tag_type: Literal["xpos", "upos"],
        train_conllu_path: str,
        dev_conllu_path: str,
        test_conllu_path: Optional[str] = None,
        proportion: float = 0.1,
    ):
        self._head = head
        if tag_type not in ["xpos", "upos"]:
            raise ValueError('tag_type must be one of "xpos", "upos"')
        self.tag_type = tag_type
        self._dataset = {
            "train": read_split(train_conllu_path, tag_type),
            "dev": read_split(dev_conllu_path, tag_type),
            "test": read_split(test_conllu_path, tag_type) if test_conllu_path is not None else [],
        }
        self._proportion = proportion
        tag_set = set(l for x in self._dataset["train"] + self._dataset["dev"] for l in x["pos_label"])
        self._tags = {v: i for i, v in enumerate(sorted(list(tag_set)))}
        self._head = head.construct(num_tags=len(tag_set))

    @property
    def slug(self):
        return "pos"

    @property
    def head(self):
        return self._head

    @property
    def data_keys(self):
        return ["pos_label"]

    @property
    def dataset(self):
        return self._dataset

    @property
    def inst_proportion(self) -> float:
        return self._proportion

    def tensorify_data(self, key, value):
        if key == "pos_label":
            return torch.tensor([self._tags[v] for v in value])
        else:
            raise ValueError(key)

    def null_tensor(self, key):
        if key == "pos_label":
            return torch.tensor([0])
        else:
            raise ValueError(key)

    def collate_data(self, key: str, values: list[torch.Tensor]):
        return pad_sequence(values, batch_first=True, padding_value=0)

    @property
    def progress_items(self):
        return ["accuracy"]

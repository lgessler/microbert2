from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from allennlp_light import ScalarMix
from allennlp_light.nn.util import sequence_cross_entropy_with_logits
from microbert.tasks.task import MicroBERTTask
from tango.common import FromParams
from tango.integrations.transformers import Tokenizer
from torchmetrics import Accuracy

from microbert2.common import dill_dump, dill_load, pool_embeddings


class XposHead(torch.nn.Module, FromParams):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_tags: int,
        use_layer_mix: bool = True,
        use_gold_tags_only: bool = False,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, num_tags)
        self.accuracy = Accuracy(num_classes=num_tags, task="multiclass", top_k=1)
        self.use_layer_mix = use_layer_mix
        self.use_gold_tags_only = use_gold_tags_only
        if self.use_layer_mix:
            self.mix = ScalarMix(num_layers)

    def forward(
        self,  # type: ignore
        hidden: List[torch.Tensor],
        token_spans: torch.LongTensor,
        tree_is_gold: torch.LongTensor,
        tags: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        if self.use_gold_tags_only:
            mask = (mask * tree_is_gold.unsqueeze(-1)).bool()

        class_probs = F.softmax(logits)
        preds = class_probs.argmax(-1)

        outputs = {"logits": logits, "preds": preds * mask}
        if tags is not None:
            flat_preds = preds.masked_select(mask)
            flat_tags = tags.masked_select(mask)
            acc = self.accuracy(flat_preds, flat_tags)
            if mask.sum().item() > 0:
                outputs["loss"] = sequence_cross_entropy_with_logits(logits, tags, mask, average="token")
            else:
                outputs["loss"] = torch.tensor(0.0, device=preds.device)
            outputs["accuracy"] = acc
            self.accuracy.reset()

        return outputs


@MicroBERTTask.register("microbert2.microbert.tasks.ud_pos.UDPOSTask")
class UDPOSTask(MicroBERTTask):
    def __init__(
        self,
        head: XposHead,
        tokenizer: Tokenizer,
        tag_type: Literal["xpos", "upos"],
        train_conllu_path: str,
        dev_conllu_path: str,
        test_conllu_path: Optional[str] = None,
    ):
        self._head = head
        if tag_type not in ["xpos", "upos"]:
            raise ValueError('tag_type must be one of "xpos", "upos"')
        self.tag_type = tag_type

    @property
    def head(self):
        return self._head

    @property
    def data_keys(self):
        return [self.tag_type]

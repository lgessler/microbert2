import logging
import os
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from datasets import DatasetDict
from tango.integrations.torch import Model, TrainCallback
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from microbert2.common import dill_dump, dill_load
from microbert2.microbert.model.biaffine_parser import BiaffineDependencyParser
from microbert2.microbert.model.encoder import MicroBERTEncoder
from microbert2.microbert.model.xpos import XposHead

logger = logging.getLogger(__name__)


def _remove_cls_and_sep(reprs: torch.Tensor, word_spans: torch.Tensor):
    batch_size, _, hidden = reprs.shape

    word_spans_mask = torch.ones(word_spans.shape, device=word_spans.device).all(-1).unsqueeze(-1)
    word_spans_mask[:, 0] = False
    word_spans_mask[:, -1] = False

    reprs_mask = torch.ones(reprs.shape, device=reprs.device).all(-1).unsqueeze(-1)
    reprs_mask[:, 0] = False
    reprs_mask[:, -1] = False

    reprs = reprs.masked_select(reprs_mask).reshape((batch_size, -1, hidden))
    word_spans = word_spans.masked_select(word_spans_mask).reshape((batch_size, -1, 2))
    word_spans = word_spans - 1
    return reprs, word_spans


@Model.register("microbert2.microbert.model.model::microbert_model")
class MicroBERTModel(Model):
    """
    Re-implementation of MicroBERT (github.com/lgessler/microbert)
    """

    def __init__(
        self,
        encoder: MicroBERTEncoder,
        counts: Dict[str, int],
        tagger: Optional[XposHead] = None,
        parser: Optional[BiaffineDependencyParser] = None,
        *args,
        **kwargs,
    ):
        """
        Provide `pretrained_model_name_or_path` if you want to use a pretrained model.
        Keep `bert_config` regardless as we need it for the LM head.

        Args:
            pretrained_model_name_or_path:
            bert_config:
            *args:
            **kwargs:
        """
        super().__init__()
        self.counts = counts

        # a BERT-style Transformer encoder stack
        self.encoder = encoder

        self.tagger = tagger
        if tagger is not None:
            logger.info("xpos tagging head initialized")
        self.parser = parser
        if parser is not None:
            logger.info("dynamic parsing head initialized")

    def forward(
        self,
        input_ids,
        input_ids_masked,
        attention_mask,
        token_type_ids,
        token_spans,
        xpos,
        head,
        deprel,
        tree_is_gold,
        labels=None,
    ):
        tree_is_gold = tree_is_gold.squeeze(-1)

        encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

        # Separate pass for the masked inputs
        masked_encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids_masked,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

        if labels is not None:
            outputs = {
                "progress_items": {
                    "max_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "resident_memory_mb": psutil.Process().memory_info().rss / 1024**2,
                }
            }

            # MLM loss
            head_loss = self.encoder.compute_loss(
                input_ids, attention_mask, token_type_ids, masked_encoder_outputs.last_hidden_state, labels
            )
            loss = sum(head_loss.values())
            outputs["mlm_loss"] = head_loss["mlm"]
            outputs["progress_items"]["mlm_loss"] = head_loss["mlm"]
            outputs["progress_items"]["perplexity"] = head_loss["mlm"].exp().item()

            # XPOS loss
            num_gold = tree_is_gold.sum().item()
            if self.tagger is not None and num_gold > 0:
                xpos_outputs = self.tagger(encoder_outputs.hidden_states, token_spans, tree_is_gold, xpos)
                loss += xpos_outputs["loss"]
                outputs["progress_items"]["xpos_acc"] = xpos_outputs["accuracy"].item()
                outputs["progress_items"]["xpos_loss"] = xpos_outputs["loss"].item()

            # parser loss
            if self.parser is not None and num_gold > 0:
                trimmed = [_remove_cls_and_sep(h_layer, token_spans) for h_layer in encoder_outputs.hidden_states]
                parser_output = self.parser.forward(
                    [x[0] for x in trimmed], trimmed[-1][1], xpos, tree_is_gold, deprel, head
                )
                loss += parser_output["loss"]
                outputs["progress_items"]["arc_loss"] = parser_output["arc_loss"].item()
                outputs["progress_items"]["tag_loss"] = parser_output["tag_loss"].item()

            # Replaced token detection loss for electra (if using)
            if "rtd" in head_loss:
                outputs["progress_items"]["rtd_loss"] = head_loss["rtd"].item()

            outputs["loss"] = loss
            return outputs
        else:
            return {}


@TrainCallback.register("microbert2.microbert.model.model::write_model")
class WriteModelCallback(TrainCallback):
    def __init__(self, path: str, model_attr: Optional[str] = None, use_best: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.model_attr = model_attr
        self.use_best = use_best

    def post_train_loop(self, step: int, epoch: int) -> None:
        # Always use the last state path
        state_path = self.train_config.state_path if not self.use_best else self.train_config.best_state_path
        state = torch.load(state_path / Path("worker0_model.pt"), map_location="cpu")
        model = self.model.cpu()
        model.load_state_dict(state, strict=True)

        # Get the target attr
        if self.model_attr:
            for piece in self.model_attr.split("."):
                model = getattr(model, piece)

        # Save in the HuggingFace format
        model.save_pretrained(self.path)
        self.logger.info(f"Wrote model to {self.path}")

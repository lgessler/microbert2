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
from microbert2.sgcl.model.biaffine_parser import BiaffineDependencyParser
from microbert2.sgcl.model.encoder import SgclEncoder
from microbert2.sgcl.model.xpos import XposHead
from microbert2.sgcl.phrases.common import PhraseSgclConfig
from microbert2.sgcl.phrases.generation import generate_phrase_sets
from microbert2.sgcl.phrases.loss import phrase_guided_loss
from microbert2.sgcl.sla import SlaConfig, generate_sla_mask
from microbert2.sgcl.trees.common import TreeSgclConfig
from microbert2.sgcl.trees.generation import generate_subtrees
from microbert2.sgcl.trees.loss import syntax_tree_guided_loss

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


@Model.register("microbert2.sgcl.model.model::sgcl_model")
class SGCLModel(Model):
    """
    Re-implementation of Syntax-Guided Contrastive Loss for Pre-trained Language Model
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    """

    def __init__(
        self,
        encoder: SgclEncoder,
        counts: Dict[str, int],
        tree_sgcl_config: Optional[TreeSgclConfig] = None,
        phrase_sgcl_config: Optional[PhraseSgclConfig] = None,
        sla_config: Optional[SlaConfig] = None,
        xpos_tagging: bool = True,
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
        self.xpos_tagging = xpos_tagging

        # a BERT-style Transformer encoder stack
        self.encoder = encoder

        if xpos_tagging:
            self.xpos_head = XposHead(encoder.config.num_hidden_layers + 1, encoder.config.hidden_size, counts["xpos"])
            logger.info("xpos tagging head initialized")
        self.parser = parser
        if parser is not None:
            logger.info("dynamic parsing head initialized")

        self.tree_sgcl_config = tree_sgcl_config
        self.phrase_sgcl_config = phrase_sgcl_config
        self.sla_config = sla_config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        token_spans,
        dependency_token_spans,
        xpos,
        head,
        deprel,
        orig_head,
        orig_deprel,
        tree_is_gold,
        labels=None,
        tree_sets=None,
        phrase_sets=None,
        dep_att_mask=None,
    ):
        tree_is_gold = tree_is_gold.squeeze(-1)
        # Encode the inputs
        if self.sla_config is not None and dep_att_mask is None:
            dep_att_mask = generate_sla_mask(self.sla_config, head)
        encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
            dep_att_mask=dep_att_mask,
        )
        hidden_states = encoder_outputs.hidden_states[1:]
        attentions = encoder_outputs.attentions
        last_encoder_state = encoder_outputs.last_hidden_state

        if self.parser is not None:
            # TODO: use predicted xpos?
            trimmed = [_remove_cls_and_sep(h_layer, token_spans) for h_layer in encoder_outputs.hidden_states]
            # do not backprop into transformer
            trimmed = [(reprs.detach().clone(), word_spans.detach().clone()) for reprs, word_spans in trimmed]

            parser_output = self.parser.forward(
                [x[0] for x in trimmed], trimmed[-1][1], xpos, tree_is_gold, orig_deprel, orig_head
            )
            # Output includes sentinel token, which we need to remove
            pred_head = parser_output["heads"][:, 1:]
            head = pred_head
        if self.tree_sgcl_config is not None and tree_sets is None:
            tree_sets = generate_subtrees(self.tree_sgcl_config, head)
        if self.phrase_sgcl_config is not None and phrase_sets is None:
            phrase_sets = generate_phrase_sets(self.phrase_sgcl_config, head, token_spans)

        if labels is not None:
            outputs = {
                "progress_items": {
                    "max_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "resident_memory_mb": psutil.Process().memory_info().rss / 1024**2,
                }
            }

            # MLM loss
            head_loss = self.encoder.compute_loss(input_ids, attention_mask, token_type_ids, last_encoder_state, labels)
            loss = sum(head_loss.values())
            outputs["mlm_loss"] = head_loss["mlm"]
            outputs["progress_items"]["mlm_loss"] = head_loss["mlm"]
            outputs["progress_items"]["perplexity"] = head_loss["mlm"].exp().item()

            # XPOS loss
            if self.xpos_tagging:
                xpos_outputs = self.xpos_head(encoder_outputs.hidden_states, token_spans, tree_is_gold, xpos)
                loss += xpos_outputs["loss"]
                outputs["progress_items"]["xpos_acc"] = xpos_outputs["accuracy"].item()
                outputs["progress_items"]["xpos_loss"] = xpos_outputs["loss"].item()

            # parser loss
            if self.parser is not None:
                loss += parser_output["loss"]
                outputs["progress_items"]["arc_loss"] = parser_output["arc_loss"].item()
                outputs["progress_items"]["tag_loss"] = parser_output["tag_loss"].item()

            # Replaced token detection loss for electra (if using)
            if "rtd" in head_loss:
                outputs["progress_items"]["rtd_loss"] = head_loss["rtd"].item()

            if self.training and self.tree_sgcl_config is not None:
                tree_loss = syntax_tree_guided_loss(
                    self.tree_sgcl_config, hidden_states, dependency_token_spans, tree_sets
                )
                loss += tree_loss
                outputs["progress_items"]["tree_loss"] = tree_loss.item()
            if self.training and self.phrase_sgcl_config is not None:
                phrase_loss = phrase_guided_loss(self.phrase_sgcl_config, attentions, attention_mask, phrase_sets)
                loss += phrase_loss
                outputs["progress_items"]["phrase_loss"] = phrase_loss.item()

            outputs["loss"] = loss
            return outputs
        else:
            return {}


@TrainCallback.register("microbert2.model::write_model")
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

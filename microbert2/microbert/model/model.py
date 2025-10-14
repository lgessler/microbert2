import logging
import os
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from datasets import DatasetDict
from tango.integrations.torch import Model, TrainCallback
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from microbert2.common import dill_dump, dill_load
from microbert2.microbert.model.encoder import MicroBERTEncoder
from microbert2.microbert.tasks.task import MicroBERTTask

logger = logging.getLogger(__name__)


def remove_cls_and_sep(reprs: torch.Tensor, word_spans: torch.Tensor):
    batch_size, seq_len, hidden = reprs.shape
    device = reprs.device

    # Make a mask so that we keep everything except the [CLS] and [SEP]
    reprs_sep_index = word_spans.max(-1).values.max(-1).values
    word_spans_sep_index = word_spans.sum(-1).gt(0).sum(-1)
    reprs_mask = torch.ones((batch_size, seq_len, 1), device=device)
    word_spans_mask = torch.ones((batch_size, word_spans.shape[1], 1), device=device)
    # Zero out [CLS]
    reprs_mask[:, 0] = 0
    word_spans_mask[:, 0] = 0
    # Zero out [SEP]
    for i, j in enumerate(reprs_sep_index):
        reprs_mask[i, j] = 0
    reprs_mask = reprs_mask.bool()
    for i, j in enumerate(word_spans_sep_index):
        word_spans_mask[i, j] = 0
    word_spans_mask = word_spans_mask.bool()

    new_reprs = reprs.masked_select(reprs_mask)
    new_reprs = new_reprs.reshape((batch_size, seq_len - 2, hidden))
    new_reprs = new_reprs[:, : word_spans.max().item() + 1]

    new_word_spans = word_spans.masked_select(word_spans_mask)
    new_word_spans = (new_word_spans - 1).clamp_min(0)
    new_word_spans = new_word_spans.reshape((batch_size, word_spans.shape[1] - 2, 2))

    return new_reprs, new_word_spans


@Model.register("microbert2.microbert.model.model::microbert_model")
class MicroBERTModel(Model):
    """
    Re-implementation of MicroBERT (github.com/lgessler/microbert)
    """

    def __init__(
        self,
        encoder: MicroBERTEncoder,
        tasks: list[MicroBERTTask] = [],
        loss_auto_scaling: bool = False,
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
        # a BERT-style Transformer encoder stack
        self.encoder = encoder
        self.tasks = tasks
        self.task_heads = nn.ModuleList([task.construct_head(self) for task in tasks])
        self.loss_scalars = None
        if loss_auto_scaling:
            logger.info("Loss auto scaling is ENABLED, initializing learnable loss parameters.")
            self.loss_scalars = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in tasks])
        logger.info(f"Initialized MicroBERT model: {self}")

    def forward(
        self,
        input_ids,
        input_ids_masked,
        attention_mask,
        token_type_ids,
        token_spans,
        dataset_id,
        **kwargs,
    ):
        # encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     output_hidden_states=True,
        #     output_attentions=True,
        # )

        # Separate pass for the masked inputs
        masked_encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids_masked,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=False,
        )

        outputs = {
            "progress_items": {
                # "max_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2,
                # "resident_memory_mb": psutil.Process().memory_info().rss / 1024**2,
            }
        }

        losses = []
        for i, task in enumerate(self.tasks):
            indexes = dataset_id >= 0 if task.universal else dataset_id == i
            if indexes.sum(0).item() == 0:
                outputs[task.slug + "_loss"] = torch.tensor(0.0, device=input_ids.device)
                losses.append(outputs[task.slug + "_loss"])
                continue

            task_args = {}
            # task_args["hidden"] = encoder_outputs.last_hidden_state[indexes]
            task_args["hidden_masked"] = [h[indexes] for h in masked_encoder_outputs.hidden_states]
            task_args["token_spans"] = token_spans[indexes]
            # Add everything in kwargs by default--could use task.data_keys instead
            for k in kwargs.keys():
                task_args[k] = kwargs[k][indexes]
            # Also add encoder and original inputs
            task_args["encoder"] = self.encoder
            task_args["input_ids"] = input_ids[indexes]
            task_args["attention_mask"] = attention_mask[indexes]
            task_args["token_type_ids"] = token_type_ids[indexes]

            # Apply task head
            task_outputs = self.task_heads[i](**task_args)

            # Add task loss to total loss
            outputs[task.slug + "_loss"] = task_outputs["loss"]
            losses.append(task_outputs["loss"])

            # Add task outputs to progress items
            for k in task.progress_items:
                if k in task_outputs:
                    outputs["progress_items"][task.slug + "_" + k] = task_outputs[k]
                    outputs[task.slug + "_" + k] = task_outputs[k]
            if self.loss_scalars is not None:
                outputs[task.slug + "_loss_coeff"] = self.loss_scalars[i].item() ** -2

        # Cf. https://arxiv.org/abs/1705.07115, equation 10
        loss = torch.tensor(0.0, device=input_ids.device)
        if self.loss_scalars is not None:
            for i, s in enumerate(self.loss_scalars):
                penalty = s.log()
                coefficient = s**-2
                loss += penalty + coefficient * losses[i]

        outputs["loss"] = loss
        return outputs


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


@TrainCallback.register("microbert2.microbert.model.model::reset_metrics")
class ResetMetricsCallback(TrainCallback):
    def post_val_loop(self, step: int, epoch: int, val_metric: float, best_val_metric: float) -> None:
        self.logger.info("Finished validation pass, resetting metric state.")
        for task in self.model.tasks:
            task.reset_metrics()

    def pre_val_batch(self, step: int, val_step: int, epoch: int, val_batch: Dict[str, Any]) -> None:
        if val_step == 0:
            print()
            self.logger.info("Beginning validation pass, resetting metric state.")
            for task in self.model.tasks:
                task.reset_metrics()

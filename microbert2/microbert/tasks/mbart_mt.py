from logging import getLogger
from typing import Any, Dict, List, Literal, Optional

import conllu
import torch
import torch.nn.functional as F
import csv
from allennlp_light import ScalarMix
from allennlp_light.nn.util import sequence_cross_entropy_with_logits
from tango.common import FromParams, Lazy, det_hash
from tango.common.det_hash import CustomDetHash
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import Accuracy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics import Perplexity

from microbert2.common import pool_embeddings
from microbert2.microbert.tasks.task import MicroBERTTask

logger = getLogger(__name__)


class MBARTMTHead(torch.nn.Module, FromParams):
    def __init__(
        self,
        num_encoder_layers: int,
        embedding_dim: int,
        mbart_model_name: str,
        use_layer_mix: bool = False,
        freeze_decoder: bool = True,
        train_last_k_decoder_layers: int = 0,
    ):
        super().__init__()
        self.use_layer_mix = use_layer_mix

        # Load MBART and delete the encode to save vram--we don't need it
        self.mbart = AutoModelForSeq2SeqLM.from_pretrained(mbart_model_name)
        del self.mbart.model.encoder

        d_model = self.mbart.config.d_model
        self.pad_token_id = self.mbart.config.pad_token_id
        # Note: ignore_index is -100 because this is what we're already transforming the pad token into
        # for our labels because MBartDecoder expects this value.
        self.perplexity = Perplexity(ignore_index=-100)

        # Layer mixing and linear projection after decoder
        if self.use_layer_mix:
            self.mix = ScalarMix(num_encoder_layers)
        self.proj = None
        if embedding_dim != d_model:
            self.proj = torch.nn.Linear(embedding_dim, d_model)
            logger.info(f"Projection layer added: {embedding_dim} -> {d_model}")

        # Decoder layer freezing
        if freeze_decoder and train_last_k_decoder_layers == 0:
            for p in self.mbart.model.decoder.parameters():
                p.requires_grad = False
            logger.info("Decoder frozen")
        elif train_last_k_decoder_layers > 0:
            # freeze all first
            for p in self.mbart.model.decoder.parameters():
                p.requires_grad = False
            # unfreeze top-K layers
            layers = self.mbart.model.decoder.layers
            K = min(train_last_k_decoder_layers, len(layers))
            for layer in layers[-K:]:
                for p in layer.parameters():
                    p.requires_grad = True
            logger.info(f"Decoder top {K} layer(s) unfrozen")
        else:
            for p in self.mbart.model.decoder.parameters():
                p.requires_grad = True
            logger.info("Decoder fully trainable")

    def forward(
        self,
        hidden_masked: List[torch.Tensor],
        tgt_input_ids: torch.LongTensor,
        tgt_attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        encoder_states = self.mix(hidden_masked) if self.use_layer_mix else hidden_masked[-1]
        if self.proj is not None:
            encoder_states = self.proj(encoder_states)

        # Set label at padded positions to -100 so they are ignored in the loss
        # See https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/models/mbart/modeling_mbart.py#L1386-L1389
        labels = tgt_input_ids.clone()
        pad_id = self.mbart.config.pad_token_id
        labels[labels == pad_id] = -100

        out = self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_states),
            attention_mask=encoder_attention_mask,
            decoder_attention_mask=tgt_attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = out.loss
        self.perplexity.update(out.logits, labels)
        return {"loss": loss, "perplexity": self.perplexity.compute()}


def read_parallel_tsv(path: str, delimiter: str = "\t"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            src = row[0].strip()  # first column = source
            tgt = row[1].strip()  # second column = target
            if src and tgt:
                rows.append(
                    {
                        "tokens": src.split(),
                        "tgt_input_ids": tgt,
                        "tgt_attention_mask": tgt,
                    }
                )
    return rows


@MicroBERTTask.register("microbert2.microbert.tasks.mbart_mt.MBARTMTTask")
class MBARTMTTask(MicroBERTTask, CustomDetHash):

    def __init__(
        self,
        head: Lazy[MBARTMTHead],
        train_mt_path: str,
        dev_mt_path: str,
        test_mt_path: Optional[str] = None,
        delimiter: str = "\t",
        proportion: float = 0.1,
        mbart_model_name: str = "facebook/mbart-large-50-many-to-one-mmt",
        tgt_lang_code: str = "en_XX",
        src_lang_code: str = "ar_AR",
        max_sequence_length: int = 512,
    ):
        self._head = head
        self._dataset = {
            "train": read_parallel_tsv(train_mt_path, delimiter),
            "dev": read_parallel_tsv(dev_mt_path, delimiter),
            "test": read_parallel_tsv(test_mt_path, delimiter) if test_mt_path else [],
        }
        self._proportion = proportion
        self._mbart_model_name = mbart_model_name
        self._tgt_lang_code = tgt_lang_code

        self._tokenizer = AutoTokenizer.from_pretrained(mbart_model_name, use_fast=False)
        self._tokenizer.src_lang = src_lang_code
        self._tokenizer.tgt_lang = tgt_lang_code
        self._pad_token_id = self._tokenizer.pad_token_id
        self._max_sequence_length = max_sequence_length

    @property
    def slug(self):
        return "mt"

    def construct_head(self, model):
        self._head = self._head.construct(mbart_model_name=self._mbart_model_name)
        return self._head

    @property
    def dataset(self):
        return self._dataset

    def _encode_tgt(self, text: str):
        enc = self._tokenizer(
            text,
            max_length=self._max_sequence_length,
            truncation=True,
            add_special_tokens=True,
        )
        return (
            torch.tensor(enc["input_ids"], dtype=torch.long),
            torch.tensor(enc["attention_mask"], dtype=torch.long),
        )

    def tensorify_data(self, key, value):
        if key == "tgt_input_ids":
            ids, _ = self._encode_tgt(value)
            return ids
        elif key == "tgt_attention_mask":
            _, mask = self._encode_tgt(value)
            return mask
        else:
            raise ValueError(key)

    def null_tensor(self, key):
        if key == "tgt_input_ids":
            return torch.tensor([self._pad_token_id], dtype=torch.long)
        elif key == "tgt_attention_mask":
            return torch.tensor([0], dtype=torch.long)
        else:
            raise ValueError(key)

    def collate_data(self, key: str, values: List[torch.Tensor]):
        if key == "tgt_input_ids":
            return pad_sequence(values, batch_first=True, padding_value=self._pad_token_id)
        elif key == "tgt_attention_mask":
            return pad_sequence(values, batch_first=True, padding_value=0)
        else:
            raise ValueError(key)

    @property
    def inst_proportion(self) -> float:
        return self._proportion

    @property
    def progress_items(self):
        return ["perplexity", "loss"]

    @property
    def data_keys(self):
        return ["tgt_input_ids", "tgt_attention_mask"]

    def reset_metrics(self):
        self._head.perplexity.reset()

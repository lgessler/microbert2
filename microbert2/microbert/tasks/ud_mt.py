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

class MTHead(torch.nn.Module, FromParams):
    def __init__(
            self,
            num_layers: int,
            use_layer_mix: bool = True,
            freeze_decoder: bool = True,
            train_last_k_decoder_layers: int = 0
    ):
        super().__init__()
        self.use_layer_mix = use_layer_mix
        self.mix = ScalarMix(num_layers) if use_layer_mix else None
        self.proj = None  # set later when we see mbart d_model
        self.perplexity = Perplexity(ignore_index=-100)

        # These will be attached by the task:
        self.mbart = None
        self._freeze_decoder = freeze_decoder
        self._train_last_k = train_last_k_decoder_layers

    def attach_mbart(self, mbart, encoder_dim: int):
        """Attach an already-constructed mBART model and set freezing/unfreezing."""
        self.mbart = mbart
        d_model = self.mbart.config.d_model
        if encoder_dim != d_model:
            self.proj = torch.nn.Linear(encoder_dim, d_model)
        # freezings
        if self._freeze_decoder and self._train_last_k <= 0:
            for p in self.mbart.model.decoder.parameters():
                p.requires_grad = False
        elif self._train_last_k > 0:
            for p in self.mbart.model.decoder.parameters():
                p.requires_grad = False
            layers = self.mbart.model.decoder.layers
            for layer in layers[-min(self._train_last_k, len(layers)):]:
                for p in layer.parameters():
                    p.requires_grad = True



    def _mix_layers(self, hidden_masked: List[torch.Tensor]) -> torch.Tensor:
        if self.use_layer_mix:
            return self.mix(hidden_masked)
        else:
            return hidden_masked[-1]
    def forward(self,
                hidden_masked: List[torch.Tensor],
                tgt_input_ids: torch.LongTensor,
                tgt_attention_mask: Optional[torch.LongTensor]=None,
                encoder_attention_mask: Optional[torch.LongTensor]=None,
                **kwargs,
                ) -> Dict[str, torch.Tensor]:
        if self.use_layer_mix:
            enc = self._mix_layers(hidden_masked)
        else:
            enc = hidden_masked[-1]

        if self.proj is not None:
            enc = self.proj(enc)
        
        enc_out = BaseModelOutput(last_hidden_state=enc)
        labels = tgt_input_ids.clone()
        if tgt_attention_mask is not None:
            labels = labels.masked_fill(tgt_attention_mask == 0, -100)
        out = self.mbart(
            encoder_outputs=enc_out,
            attention_mask=encoder_attention_mask, 
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
            src = row[0].strip()   # first column = source
            tgt = row[1].strip()   # second column = target
            if src and tgt:
                rows.append({
                    "tokens": src.split(),
                    "tgt_input_ids": tgt,
                    "tgt_attention_mask": None,  # will be computed in tensorify_data
                })
    return rows

@MicroBERTTask.register("microbert2.microbert.tasks.mt_task.MTTask")
class MTTask(MicroBERTTask, CustomDetHash): # top-level task

    def __init__(
            self,
            head: Lazy[MTHead],
            train_mt_path: str,
            dev_mt_path: str,
            test_mt_path: Optional[str] = None,
            delimiter: str = "\t",
            proportion: float = 0.1, #0.2 0.5 coptic
            mbart_model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
            mbart_tokenizer_name: Optional[str] = None,  # if None, use model_name
            src_lang_code: str = "ar_AR",
            tgt_lang_code: str = "en_XX",
            max_tgt_len: int = 128,
    ):
        self._head = head
        self._dataset = {
            "train": read_parallel_tsv(train_mt_path, delimiter),
            "dev":   read_parallel_tsv(dev_mt_path, delimiter),
            "test":  read_parallel_tsv(test_mt_path, delimiter) if test_mt_path else [],
        }
        self._mbart_model_name = mbart_model_name
        self._mbart_tokenizer_name = mbart_tokenizer_name or mbart_model_name
        self._src_lang_code = src_lang_code
        self._tgt_lang_code = tgt_lang_code
        self._max_tgt_len = max_tgt_len
        self._proportion = proportion
        # MBART
        self._tok = AutoTokenizer.from_pretrained(self._mbart_tokenizer_name, use_fast=False)
        self._tok.src_lang = self._src_lang_code
        self._tok.tgt_lang = self._tgt_lang_code
        self._pad = self._tok.pad_token_id

    @property
    def slug(self):
        return "mt"
    
    def construct_head(self, model):
        # Build / load mBART here
        mbart = AutoModelForSeq2SeqLM.from_pretrained(self._mbart_model_name)

        head = self._head.construct()

        encoder_dim = self.embedding_dim
        head.attach_mbart(mbart, encoder_dim)
        logger.info(f"MT head initialized with mBART '{self._mbart_model_name}' (src={self._tok.src_lang}, tgt={self._tok.tgt_lang})")
        return head

    @property
    def dataset(self):
        return self._dataset

    def _encode_tgt(self, text: str):
        enc = self._tok(
            text,
            max_length=self._max_tgt_len,
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
            return torch.tensor([self._pad], dtype=torch.long)
        elif key == "tgt_attention_mask":
            return torch.tensor([0], dtype=torch.long)
        else:
            raise ValueError(key)

    def collate_data(self, key: str, values: List[torch.Tensor]):
        if key == "tgt_input_ids":
            return pad_sequence(values, batch_first=True, padding_value=self._pad)
        elif key == "tgt_attention_mask":
            return pad_sequence(values, batch_first=True, padding_value=0)
        else:
            raise ValueError(key)

    @property
    def inst_proportion(self) -> float:
        return self._proportion
    @property
    def progress_items(self):
        return ["perplexity","loss"]
    @property
    def data_keys(self):
        return ["tgt_input_ids", "tgt_attention_mask"]

    def reset_metrics(self):
        self._head.perplexity.reset()
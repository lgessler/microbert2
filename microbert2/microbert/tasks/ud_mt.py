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

from microbert2.common import pool_embeddings
from microbert2.microbert.tasks.task import MicroBERTTask

logger = getLogger(__name__)

class MTHead(torch.nn.Module, FromParams):
    def __init__(
            self,
            num_layers: int,
            embedding_dim: int,
            mbert_model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
            use_layer_mix: bool = True,
            freeze_decoder: bool = True,
            train_last_k_decoder_layers: int = 0
    ):
        super().__init__()
        self.use_layer_mix = use_layer_mix

        if self.use_layer_mix:
            self.mix = ScalarMix(num_layers) 

        self.mbart = AutoModelForSeq2SeqLM.from_pretrained(mbert_model_name)
        d_model = self.mbart.config.d_model  
        if embedding_dim != d_model:
            self.proj = torch.nn.Linear(embedding_dim, d_model)
            logger.info(f"Projection layer added: {embedding_dim} -> {d_model}")

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

        out = self.mbart(
            encoder_outputs=enc_out,
            attention_mask=encoder_attention_mask, 
            labels=tgt_input_ids,                 
            use_cache=False,
        )
        loss = out.loss
        ppl = torch.exp(loss.detach())

        return {"loss": loss, "perplexity": ppl}
    

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
                    "tgt_attention_mask": tgt,
                })
    return rows

@MicroBERTTask.register("microbert2.microbert.tasks.ud_pos.UDMTTask")
class MTTask(MicroBERTTask, CustomDetHash):

    def __init__(
            self,
            head: Lazy[MTHead],
            train_mt_path: str,
            dev_mt_path: str,
            test_mt_path: Optional[str] = None,
            delimiter: str = "\t",
            proportion: float = 0.1, #0.2 0.5 coptic
            mbart_tokenizer_name: str = "facebook/mbart-large-50-many-to-one-mmt",
            tgt_lang_code: str = "en_XX",
            max_tgt_len: int = 128,
    ):
        self._head = head
        self._dataset = {
            "train": read_parallel_tsv(train_mt_path, delimiter),
            "dev":   read_parallel_tsv(dev_mt_path, delimiter),
            "test":  read_parallel_tsv(test_mt_path, delimiter) if test_mt_path else [],
        }
        self._proportion = proportion
        self._mbart_tokenizer_name = mbart_tokenizer_name
        self._tgt_lang_code = tgt_lang_code

        # MBART 
        self._tok = AutoTokenizer.from_pretrained(mbart_tokenizer_name,use_fast=False)
        #self._tok.add_special_tokens({"additional_special_tokens": ["<cop_XX>"]})
        self._tok.src_lang = "ar_AR"  
        self._tok.tgt_lang = tgt_lang_code
        self._pad = self._tok.pad_token_id
        self._max_tgt_len = max_tgt_len

    @property
    def slug(self):
        return "mt"
    
    def construct_head(self, model):
        return self._head.construct()

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


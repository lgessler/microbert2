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
    ):
        super().__init__()
        self.use_layer_mix = use_layer_mix

        if self.use_layer_mix:
            self.mix = ScalarMix(num_layers)

        self.mbart = AutoModelForSeq2SeqLM.from_pretrained(mbert_model_name)

        if  freeze_decoder:
            for param in self.mbart.model.decoder.parameters():
                param.requires_grad = False
            logger.info("Decoder frozen")

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

        enc = self._mix_layers(hidden_masked) 

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
    
@MicroBERTTask.register("microbert2.microbert.tasks.ud_pos.UDMTTask")
class MTTask(MicroBERTTask, CustomDetHash):

    def __init__(
            self,
            head: Lazy[MTHead],
            train_conllu_path: str,
            dev_conllu_path: str,
            test_conllu_path: Optional[str] = None,
            proportion: float = 0.1,
            mbart_tokenizer_name: str = "facebook/mbart-large-50-many-to-one-mmt",
            tgt_lang_code: str = "en_XX"
    ):
        self._head = head
        self._dataset = {} # to be implemented
        self._proportion = proportion
        self._mbart_tokenizer_name = mbart_tokenizer_name
        self._tgt_lang_code = tgt_lang_code

    @property
    def slug(self):
        return "mt"
    
    def construct_head(self, model):
        return self._head.construct()

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def inst_proportion(self) -> float:
        return self._proportion
    @property
    def progress_items(self):
        return ["perplexity"]


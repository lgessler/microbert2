from json import encoder
from logging import getLogger
from typing import Any, Dict, List, Optional
import csv
from tango.common import FromParams, Lazy, det_hash
from tango.common.det_hash import CustomDetHash
import torch
from peft import LoraConfig, get_peft_config, get_peft_model, TaskType
from torch.jit import freeze
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutput
from microbert2.common import pool_embeddings
from microbert2.microbert.tasks.mbart_mt import read_parallel_tsv
from microbert2.microbert.tasks.task import MicroBERTTask
from torchmetrics import Perplexity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = getLogger(__name__)

class OpusMTHead(torch.nn.Module, FromParams):
    def __init__(
            self,
            num_encoder_layers: int,
            embedding_dim: int,
            opus_model_name: str,
            use_layer_mix: bool = False,
            freeze_decoder: bool = True,
            train_cross_attn_kv: bool = False,
            use_lora: bool = False,
            lora_r: int = 8,
            lora_alpha: int = 16, 
            lora_dropout: float = 0.1,
            mt_weight: float = 0.1,
            mlp_projection: bool = False
    ):
        super().__init__()
        self.use_layer_mix = use_layer_mix
        self.mt_weight = mt_weight
        self.mlp_projection = mlp_projection
        if opus_model_name is not None:
            self.opus = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")  
            
        if use_lora:
            lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    )
            self.opus = get_peft_model(self.opus, lora_config)
            logger.info(f"LoRA applied to model: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            trainable_params = sum(p.numel() for p in self.opus.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.opus.parameters())
            logger.info(f"LoRA appiled: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        #delete the encoder to save vram
        if use_lora:
            if hasattr(self.opus,'base_model'):
                del self.opus.base_model.model.model.encoder
        else:
            del self.opus.model.encoder
        
        #model configuration
        d_model = self.opus.config.d_model
        self.pad_token_id = self.opus.config.pad_token_id
        self.perplexity = Perplexity(ignore_index=-100)

        if self.use_layer_mix:
            self.mix = ScalarMix(num_encoder_layers)
        self.proj = None
        if embedding_dim != d_model:
            if not self.mlp_projection:
                self.proj = torch.nn.Linear(embedding_dim,d_model)
            else:
                intermediate_dim = (embedding_dim+d_model) // 2
                self.proj = torch.nn.Sequential(
                        torch.nn.Linear(embedding_dim,intermediate_dim),
                        torch.nn.GELU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.LayerNorm(intermediate_dim),
                        torch.nn.Linear(intermediate_dim,d_model)
                        )

        if use_lora:
            logger.info("parameter management handled by LoRA")
        elif train_cross_attn_kv:
            #freeze all
            for p in self.opus.model.decoder.parameters():
                p.requires_grad = False
            #unfreeze k and v projections
            trainable_count = 0 
            for layer in self.opus.model.decoder.layers:
               for name, param in layer.encoder_attn.named_parameters():
                   if 'k_proj' in name or 'v_proj' in name:
                       param.requires_grad = True
                       trainable_count += param.numel()
            logger.info(f"cross-attention K,V projections unfrozen: {trainable_count} parameters")
        elif freeze_decoder:
            for p in self.opus.model.decoder.parameters():
                p.requires_grad = False
            logger.info("Decoder frozen")
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

        labels = tgt_input_ids.clone()
        pad_id = self.opus.config.pad_token_id
        labels[labels == pad_id] = -100

        out = self.opus(
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_states),
                attention_mask = encoder_attention_mask,
                decoder_attention_mask = tgt_attention_mask,
                labels = labels,
                use_cache = False,
                )
        loss = out.loss*self.mt_weight
        self.perplexity.update(out.logits, labels)
        return {"loss":loss,"perplexity":self.perplexity.compute()}

    def read_parallel_tsv(path: str, delimiter: str = "\t"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter = delimiter)
            for row in reader:
                if len(row) < 2:
                    continue
                src = row[0].strip()
                tgt = row[1].strip()
                if src and tgt:
                    rows.append(
                            {
                                "tokens": src.split(),
                                "tgt_input_ids": tgt,
                                "tgt_attention_mask": tgt,
                                }
                    )
        return rows
    
@MicroBERTTask.register("microbert2.microbert.tasks.opus_mt.OPUSMTTask")
class OpusMTTask(MicroBERTTask, CustomDetHash):
    
    def __init__(
            self,
            head: Lazy[OpusMTHead],
            train_mt_path: str,
            dev_mt_path: str,
            test_mt_path: Optional[str] = None,
            delimiter: str ="\t",
            proportion: float = 0.1,
            opus_model_name : str = "Helsinki-NLP/opus-mt-mul-en",
            max_sequence_length: int = 512
            ):
        self._head = head
        self._dataset = {
                "train": read_parallel_tsv(train_mt_path,delimiter),
                "dev": read_parallel_tsv(dev_mt_path,delimiter),
                "test": read_parallel_tsv(test_mt_path,delimiter) if test_mt_path else [],
        }
        self._proportion = proportion
        self._opus_model_name = opus_model_name
        self._max_sequence_length = max_sequence_length

        self._tokenizer = AutoTokenizer.from_pretrained(opus_model_name, use_fast = False)
        self._pad_token_id = self._tokenizer.pad_token_id

        self._hash_string = (
            self.slug + train_mt_path + dev_mt_path + (test_mt_path if test_mt_path else "") +
            opus_model_name
            )

    def det_hash_object(self) -> Any:
        return det_hash(self._hash_string)

    @property
    def slug(self) -> str:
        return "opus-mt"

    def construct_head(self, model) -> OpusMTHead:
        self._head = self._head.construct(opus_model_name=self._opus_model_name)
        return self._head

    @property
    def dataset(self) -> dict:
        return self._dataset

    def _encode_tgt(self,text:str):
        enc = self._tokenizer(
            text,
            max_length=self._max_sequence_length,
            truncation = True,
            add_special_tokens=True,
            )
        return (
            torch.tensor(enc["input_ids"],dtype=torch.long),
            torch.tensor(enc["attention_mask"],dtype=torch.long),
            )

    def tensorify_data(self,key,value):
        if key == "tgt_input_ids":
            ids,_ = self._encode_tgt(value)
            return ids
        elif key == "tgt_attention_mask":
            _,mask = self._encode_tgt(value)
            return mask
        else:
            raise ValueError(key)

    def null_tensor(self,key):
        if key == "tgt_input_ids":
            return torch.tensor([self._pad_token_id],dtype=torch.long)
        elif key == "tgt_attention_mask":
            return torch.tensor([0],dtype=torch.long)
        else:
            raise ValueError(f"Unknow key: {key}")

    def collate_data(self, key:str, values: List[torch.Tensor]) -> torch.Tensor:
        if key == "tgt_input_ids":
            return pad_sequence(values,batch_first=True,padding_value=self._pad_token_id)
        elif key == "tgt_attention_mask":
            return pad_sequence(values, batch_first=True, padding_value=0)
        else:
            raise ValueError(f"Unknow key: {key}")

    @property
    def inst_proportion(self) -> float:
        return self._proportion

    @property
    def progress_items(self) -> List[str]:
        return ["perplexity","loss"]

    @property
    def data_keys(self) -> List[str]:
        return ["tgt_input_ids","tgt_attention_mask"]

    def reset_metrics(self):
        self._head.perplexity.reset()

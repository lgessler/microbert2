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
from peft import LoraConfig, get_peft_model, TaskType

from microbert2.common import pool_embeddings
from microbert2.microbert.tasks.task import MicroBERTTask

logger = getLogger(__name__)


class MBARTMTHead(torch.nn.Module, FromParams):
    def __init__(
        self,
        num_encoder_layers: int,
        embedding_dim: int,
        mbart_model_name: Optional[str] = None,
        mbart_config_kwargs: Optional[Dict[str,Any]] = None,
        use_layer_mix: bool = False,
        freeze_decoder: bool = True,
        use_cross_attn_kv_lora: bool = False,
        train_last_k_decoder_layers: int = 0,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        mt_weight: float = 0.1,
        mlp_projection: bool = False,
    ):
        super().__init__()
        self.use_layer_mix = use_layer_mix
        self.mt_weight = mt_weight
        self.mlp_projection = mlp_projection
        if mbart_model_name is not None and mbart_config_kwargs is not None:
            raise ValueError("Specify either 'mbart_model_name' or 'mbart_config_kwargs' not both")
        if mbart_model_name is None and mbart_config_kwargs is None:
            raise ValueError("Must specify either 'mbart_model_name' or 'mbart_config_kwargs'")

        if mbart_model_name is not None:
            self.mbart = AutoModelForSeq2SeqLM.from_pretrained(mbart_model_name)

        if mbart_config_kwargs is not None:
            #Build small mbart from config
            from transformers import MBartConfig, MBartForConditionalGeneration
            config = MBartConfig()
            for key, value in mbart_config_kwargs.items():
                if not hasattr(config,key):
                    raise ValueError(f"MBartConfig has no attribute {key}")
                setattr(config,key,value)
            self.mbart = MBartForConditionalGeneration(config)
            logger.info(f"Initialized MBartForConditionalGeneration from config with {sum(p.numel() for p in self.mbart.parameters()):,} parameters")

        # Apply LoRA before deleting encoder (if using LoRA)
        if use_lora:
            # Apply LoRA to the full model (targeting decoder modules only)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.mbart = get_peft_model(self.mbart, lora_config)
            logger.info(f"LoRA applied to model: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            # Verify LoRA was applied correctly
            trainable_params = sum(p.numel() for p in self.mbart.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.mbart.parameters())
            logger.info(f"LoRA applied: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


        # Now delete the encoder to save vram--we don't need it
        if use_lora:
            # When using LoRA, the model structure is: PeftModel.base_model.model (MBartForSeq2SeqLM).model (MBartModel).encoder
            del self.mbart.base_model.model.model.encoder
        else:
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
            if not self.mlp_projection:
                self.proj = torch.nn.Sequential(
                            torch.nn.Linear(embedding_dim, d_model),
                            torch.nn.LayerNorm(d_model),
                            )

                logger.info(f"Projection layer added: {embedding_dim} -> {d_model}")
            else:
                intermediate_dim = (embedding_dim+d_model) // 2
                self.proj = torch.nn.Sequential(
                        torch.nn.Linear(embedding_dim, intermediate_dim),
                        torch.nn.GELU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(intermediate_dim, d_model),
                        torch.nn.LayerNorm(d_model),
                        )
                logger.info(f"MLP Projection layer added: {embedding_dim} -> {intermediate_dim} -> {d_model}")
        
        # When using LoRA, parameter freezing is handled by PEFT, so we skip manual freezing
        if use_lora:
            logger.info("Decoder parameter management handled by LoRA")
        elif use_cross_attn_kv_lora:
            # Apply LoRA specifically to cross-attention K,V projections
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["encoder_attn.k_proj", "encoder_attn.v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.mbart = get_peft_model(self.mbart, lora_config)
            trainable_params = sum(p.numel() for p in self.mbart.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.mbart.parameters())
            logger.info(f"LoRA applied to cross-attention K,V projections: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        elif freeze_decoder and train_last_k_decoder_layers == 0:
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
        loss = out.loss*self.mt_weight
        #logger.info(f"out before applied weight: {out.loss}")
        #logger.info(f"out after applied weight: {loss}")
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
        mbart_model_name: Optional[str] = None,
        mbart_tokenizer: str = "facebook/mbart-large-50-many-to-one-mmt",
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
        self._mbart_tokenizer = mbart_tokenizer
        self._tgt_lang_code = tgt_lang_code

        self._tokenizer = AutoTokenizer.from_pretrained(mbart_tokenizer, use_fast=False)
        self._tokenizer.src_lang = src_lang_code
        self._tokenizer.tgt_lang = tgt_lang_code
        self._pad_token_id = self._tokenizer.pad_token_id
        self._max_sequence_length = max_sequence_length

        # Create deterministic hash string from constructor parameters only
        self._hash_string = (
            self.slug +
            train_mt_path +
            dev_mt_path +
            (test_mt_path if test_mt_path else "") +
            delimiter +
            str(proportion) +
            str(mbart_model_name) +
            tgt_lang_code +
            src_lang_code +
            str(max_sequence_length)
        )

    def det_hash_object(self) -> Any:
        return det_hash(self._hash_string)

    @property
    def slug(self):
        return "mbart-mt"

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

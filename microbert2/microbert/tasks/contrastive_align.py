from logging import getLogger
from typing import Any, Dict, List, Optional
import csv

import torch
import torch.nn.functional as F
from allennlp_light import ScalarMix
from tango.common import FromParams, Lazy, det_hash
from tango.common.det_hash import CustomDetHash
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from microbert2.microbert.tasks.task import MicroBERTTask

logger = getLogger(__name__)


class ContrastiveAlignmentHead(torch.nn.Module, FromParams):
    """
    Contrastive alignment head.

    Takes MicroBERT's hidden states (source side) and a frozen mBART encoder
    (target side), mean-pools both into sentence vectors, projects them into a
    shared space, and minimises symmetric InfoNCE loss.

    Gradient flows: loss -> tgt_proj (small) + src_proj (small) -> MicroBERT encoder.
    The frozen mBART encoder receives no gradient.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        embedding_dim: int,
        tgt_embedding_dim: int,
        mbart_encoder: torch.nn.Module,
        projection_dim: int = 256,
        temperature: float = 0.07,
        use_layer_mix: bool = False,
        contrastive_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.use_layer_mix = use_layer_mix

        if use_layer_mix:
            self.mix = ScalarMix(num_encoder_layers)

        # Frozen mBART encoder lives here so it moves with the module (device-safe)
        self.mbart_encoder = mbart_encoder

        # Two separate linear projections into shared contrastive space
        self.src_proj = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, projection_dim),
            torch.nn.LayerNorm(projection_dim),
        )
        self.tgt_proj = torch.nn.Sequential(
            torch.nn.Linear(tgt_embedding_dim, projection_dim),
            torch.nn.LayerNorm(projection_dim),
        )

        logger.info(
            f"ContrastiveAlignmentHead: src_dim={embedding_dim}, tgt_dim={tgt_embedding_dim}, "
            f"projection_dim={projection_dim}, temperature={temperature}"
        )

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pool token hidden states, ignoring padding.
        hidden:          [B, seq, dim]
        attention_mask:  [B, seq]   (1 = real token, 0 = pad)
        returns:         [B, dim]
        """
        mask = attention_mask.unsqueeze(-1).float()       # [B, seq, 1]
        summed = (hidden * mask).sum(dim=1)               # [B, dim]
        counts = mask.sum(dim=1).clamp(min=1)             # [B, 1]
        return summed / counts

    def forward(
        self,
        hidden_masked: List[torch.Tensor],
        tgt_input_ids: torch.Tensor,
        tgt_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # --- Source side: MicroBERT hidden states ---
        if self.use_layer_mix:
            encoder_states = self.mix(hidden_masked)
        else:
            encoder_states = hidden_masked[-1]             # [B, seq, embedding_dim]

        src_pooled = self._mean_pool(encoder_states, attention_mask)  # [B, embedding_dim]

        # --- Target side: frozen mBART encoder ---
        with torch.no_grad():
            mbart_out = self.mbart_encoder(
                input_ids=tgt_input_ids,
                attention_mask=tgt_attention_mask,
            )
            tgt_pooled = self._mean_pool(
                mbart_out.last_hidden_state, tgt_attention_mask
            )                                              # [B, tgt_embedding_dim]

        # --- Project both into shared space and L2-normalise ---
        src_vec = F.normalize(self.src_proj(src_pooled), dim=-1)    # [B, projection_dim]
        tgt_vec = F.normalize(self.tgt_proj(tgt_pooled), dim=-1)    # [B, projection_dim]

        # --- Symmetric InfoNCE loss ---
        # logits[i, j] = cosine sim between source i and target j (scaled)
        # Correct pairs are on the diagonal
        B = src_vec.size(0)
        logits = torch.matmul(src_vec, tgt_vec.T) / self.temperature  # [B, B]
        labels = torch.arange(B, device=src_vec.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss = loss * self.contrastive_weight

        return {"loss": loss}


def read_parallel_tsv(path: str, delimiter: str = "\t"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            src = row[0].strip()
            tgt = row[1].strip()
            if src and tgt:
                rows.append(
                    {
                        "tokens": src.split(),
                        "tgt_text": tgt,
                    }
                )
    return rows


@MicroBERTTask.register("microbert2.microbert.tasks.contrastive_align.ContrastiveAlignTask")
class ContrastiveAlignTask(MicroBERTTask, CustomDetHash):
    """
    Contrastive cross-lingual alignment task.

    Given parallel sentence pairs (source, target):
    - Source is encoded by MicroBERT (being trained)
    - Target is encoded by a frozen pretrained mBART encoder
    Both sides are mean-pooled into sentence vectors and pulled together
    with symmetric InfoNCE loss.

    No decoder, no projection bottleneck problem — gradient goes straight
    into MicroBERT's encoder.
    """

    def __init__(
        self,
        head: Lazy[ContrastiveAlignmentHead],
        train_path: str,
        dev_path: str,
        test_path: Optional[str] = None,
        delimiter: str = "\t",
        proportion: float = 0.1,
        mbart_tokenizer: str = "facebook/mbart-large-50-many-to-one-mmt",
        mbart_encoder_model: str = "facebook/mbart-large-50-many-to-one-mmt",
        tgt_lang_code: str = "en_XX",
        src_lang_code: str = "ar_AR",
        max_sequence_length: int = 128,
    ):
        self._head = head
        self._dataset = {
            "train": read_parallel_tsv(train_path, delimiter),
            "dev": read_parallel_tsv(dev_path, delimiter),
            "test": read_parallel_tsv(test_path, delimiter) if test_path else [],
        }
        self._proportion = proportion
        self._max_sequence_length = max_sequence_length

        # mBART tokenizer for the target side
        self._tokenizer = AutoTokenizer.from_pretrained(mbart_tokenizer, use_fast=False)
        self._tokenizer.src_lang = src_lang_code
        self._tokenizer.tgt_lang = tgt_lang_code
        self._pad_token_id = self._tokenizer.pad_token_id

        # Load frozen mBART encoder — only the encoder is kept
        from transformers import AutoModelForSeq2SeqLM
        full_model = AutoModelForSeq2SeqLM.from_pretrained(mbart_encoder_model)
        self._mbart_encoder = full_model.model.encoder
        for p in self._mbart_encoder.parameters():
            p.requires_grad = False
        self._mbart_encoder.eval()
        self._tgt_embedding_dim = full_model.config.d_model
        logger.info(
            f"Loaded frozen mBART encoder ({mbart_encoder_model}), "
            f"d_model={self._tgt_embedding_dim}"
        )

        self._hash_string = (
            "contrastive-align"
            + train_path
            + dev_path
            + (test_path or "")
            + delimiter
            + str(proportion)
            + tgt_lang_code
            + src_lang_code
            + str(max_sequence_length)
            + mbart_encoder_model
        )

    def det_hash_object(self) -> Any:
        return det_hash(self._hash_string)

    @property
    def slug(self):
        return "contrastive-align"

    def construct_head(self, model):
        self._head = self._head.construct(
            tgt_embedding_dim=self._tgt_embedding_dim,
            mbart_encoder=self._mbart_encoder,
        )
        return self._head

    @property
    def dataset(self):
        return self._dataset

    @property
    def inst_proportion(self) -> float:
        return self._proportion

    @property
    def data_keys(self):
        return ["tgt_input_ids", "tgt_attention_mask"]

    def tensorify_data(self, key, value):
        if key in ("tgt_input_ids", "tgt_attention_mask"):
            # value is the raw target text string
            enc = self._tokenizer(
                value,
                max_length=self._max_sequence_length,
                truncation=True,
                add_special_tokens=True,
            )
            if key == "tgt_input_ids":
                return torch.tensor(enc["input_ids"], dtype=torch.long)
            else:
                return torch.tensor(enc["attention_mask"], dtype=torch.long)
        raise ValueError(key)

    def null_tensor(self, key):
        if key == "tgt_input_ids":
            return torch.tensor([self._pad_token_id], dtype=torch.long)
        elif key == "tgt_attention_mask":
            return torch.tensor([0], dtype=torch.long)
        raise ValueError(key)

    def collate_data(self, key: str, values: List[torch.Tensor]):
        if key == "tgt_input_ids":
            return pad_sequence(values, batch_first=True, padding_value=self._pad_token_id)
        elif key == "tgt_attention_mask":
            return pad_sequence(values, batch_first=True, padding_value=0)
        raise ValueError(key)

    @property
    def progress_items(self):
        return ["loss"]

    def reset_metrics(self):
        pass

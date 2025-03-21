import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tango.common import Registrable
from tango.integrations.transformers import Tokenizer
from torch import nn
from transformers import BertConfig, BertModel, ElectraConfig, ElectraModel, ModernBertConfig, ModernBertModel
from transformers.activations import gelu, get_activation

from microbert2.common import dill_dump, dill_load

logger = logging.getLogger(__name__)


class MicroBERTEncoder(torch.nn.Module, Registrable):
    pass


@MicroBERTEncoder.register("bert")
class BertEncoder(MicroBERTEncoder):
    def __init__(self, tokenizer: Tokenizer, bert_config: Dict[str, Any]):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        config = BertConfig(**bert_config, vocab_size=len(tokenizer.get_vocab()))
        logger.info(f"Initializing a new BERT model with config {config}")
        self.config = config
        self.encoder = BertModel(config=config, add_pooling_layer=False)
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        return self.encoder(*args, return_dict=True, **kwargs)

    @property
    def embedding_weights(self):
        return self.encoder.embeddings.word_embeddings.weight


@MicroBERTEncoder.register("modernbert")
class ModernBertEncoder(MicroBERTEncoder):
    def __init__(self, tokenizer: Tokenizer, bert_config: Dict[str, Any]):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        config = ModernBertConfig(
            **bert_config,
            vocab_size=len(tokenizer.get_vocab()),
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )
        logger.info(f"Initializing a new BERT model with config {config}")
        self.config = config
        self.encoder = ModernBertModel(config=config)
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        if "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return self.encoder(*args, return_dict=True, **kwargs)

    @property
    def embedding_weights(self):
        return self.encoder.embeddings.tok_embeddings.weight


@MicroBERTEncoder.register("electra")
class ElectraEncoder(MicroBERTEncoder):
    """
    We will follow the simplest approach described in the paper (https://openreview.net/pdf?id=r1xMH1BtvB),
    which is to tie all the weights of the discriminator and the generator. In effect, this means we can
    just use the same Transformer encoder stack for both the discriminator and the generator, with different
    heads on top for MLM and replaced token detection.
    """

    def __init__(self, tokenizer: Tokenizer, electra_config: Dict[str, Any], tied_generator: bool = False):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        config = ElectraConfig(
            **electra_config, vocab_size=len(tokenizer.get_vocab()), position_embedding_type="relative_key_query"
        )
        logger.info(f"Initializing a new BERT model with config {config}")
        self.config = config
        self.tokenizer = tokenizer

        # Make the discriminator--alias this as "self.encoder" since WriteModelCallback assumes
        # the model we want to save will be available under that attribute.
        self.discriminator = ElectraModel(config=config)
        self.encoder = self.discriminator

        # Make the generator--this is the same as the discriminator if we're tying them, otherwise
        # an identical copy of the ElectraModel and tie their embedding layers. These are two approaches
        # described in the original ELECTRA paper. Not implemented here: allowing a distinct but smaller
        # (instead of identical) ELECTRA model.
        if tied_generator:
            self.generator = self.discriminator
        else:
            self.generator = ElectraModel(config=config)
            self.generator.embeddings = self.discriminator.embeddings

    def forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    @property
    def embedding_weights(self):
        return self.discriminator.embeddings.word_embeddings.weight


def tmp():
    input_ids = dill_load("/tmp/input_ids")[:32]
    attention_mask = dill_load("/tmp/attention_mask")[:32]
    token_type_ids = dill_load("/tmp/token_type_ids")[:32]
    last_hidden_state = dill_load("/tmp/last_hidden_state")[:32]
    labels = dill_load("/tmp/labels")[:32]
    self = dill_load("/tmp/self")

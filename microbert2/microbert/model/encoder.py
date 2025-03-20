import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tango.common import Registrable
from tango.integrations.transformers import Tokenizer
from torch import nn
from transformers import BertConfig, BertModel, ElectraConfig, ElectraModel
from transformers.activations import gelu, get_activation

from microbert2.common import dill_dump, dill_load


logger = logging.getLogger(__name__)


class MicroBERTEncoder(torch.nn.Module, Registrable):
    pass


@torch.jit.script
def _tied_generator_forward(hidden_states, embedding_weights):
    hidden_states = torch.einsum("bsh,eh->bse", hidden_states, embedding_weights)
    return hidden_states


class TiedRobertaLMHead(nn.Module):
    """Version of RobertaLMHead which accepts an embedding torch.nn.Parameter for output probabilities"""

    def __init__(self, config, embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_weights = embedding_weights

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = _tied_generator_forward(x, self.embedding_weights)

        return x


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
        self.head = TiedRobertaLMHead(config, self.encoder.embeddings.word_embeddings.weight)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, return_dict=True, **kwargs)

    def compute_loss(self, input_ids, attention_mask, token_type_ids, last_encoder_state, labels):
        preds = self.head(last_encoder_state)
        if not (labels != -100).any():
            return 0.0
        masked_lm_loss = F.cross_entropy(preds.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        return {"mlm": masked_lm_loss}


class TiedElectraGeneratorPredictions(nn.Module):
    """Like ElectraGeneratorPredictions, but accepts a torch.nn.Parameter from an embedding module"""

    def __init__(self, config, embedding_weights):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.vocab_size, eps=config.layer_norm_eps)
        self.embedding_weights = embedding_weights

    def forward(self, generator_hidden_states):
        hidden_states = _tied_generator_forward(generator_hidden_states, self.embedding_weights)
        hidden_states = get_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


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
        self.discriminator_head = torch.nn.Linear(config.hidden_size, 1)

        # Make the generator--this is the same as the discriminator if we're tying them, otherwise
        # an identical copy of the ElectraModel and tie their embedding layers. These are two approaches
        # described in the original ELECTRA paper. Not implemented here: allowing a distinct but smaller
        # (instead of identical) ELECTRA model.
        if tied_generator:
            self.generator = self.discriminator
        else:
            self.generator = ElectraModel(config=config)
            self.generator.embeddings = self.discriminator.embeddings
        # Also tie the output embeddings to the input embeddings
        self.generator_head = TiedElectraGeneratorPredictions(config, self.generator.embeddings.word_embeddings.weight)

    def forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def compute_loss(self, input_ids, attention_mask, token_type_ids, last_hidden_state, labels):
        """
        Used to compute discriminator's loss
        """
        mlm_logits = self.generator_head(last_hidden_state)
        if not (labels != -100).any():
            masked_lm_loss = 0.0
        else:
            masked_lm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100
            )

        # Take predicted token IDs. Note that argmax() breaks the gradient chain, so the generator only learns from MLM
        mlm_preds = mlm_logits.argmax(-1)
        # Combine them to get labels for discriminator
        replaced = (~input_ids.eq(mlm_preds)) & (labels != -100)

        # Make inputs for discriminator, feed them into the encoder once more, then feed to discriminator head
        replaced_input_ids = torch.where(replaced, mlm_preds, input_ids)
        second_encoder_output = self.discriminator(
            input_ids=replaced_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        discriminator_output = self.discriminator_head(second_encoder_output.last_hidden_state).squeeze(-1)

        # compute replaced token detection BCE loss on eligible tokens (all non-special tokens)
        bce_mask = (
            (replaced_input_ids != self.tokenizer.cls_token_id)
            & (replaced_input_ids != self.tokenizer.sep_token_id)
            & (replaced_input_ids != self.tokenizer.pad_token_id)
        )
        rtd_preds = torch.masked_select(discriminator_output, bce_mask)
        rtd_labels = torch.masked_select(replaced.float(), bce_mask)
        rtd_loss = F.binary_cross_entropy_with_logits(rtd_preds, rtd_labels)

        # dill_dump(input_ids, '/tmp/input_ids')
        # dill_dump(attention_mask, '/tmp/attention_mask')
        # dill_dump(token_type_ids, '/tmp/token_type_ids')
        # dill_dump(last_hidden_state, '/tmp/last_hidden_state')
        # dill_dump(labels, '/tmp/labels')
        # dill_dump(self, '/tmp/self')
        return {"rtd": (50 * rtd_loss), "mlm": masked_lm_loss}


def tmp():
    input_ids = dill_load("/tmp/input_ids")[:32]
    attention_mask = dill_load("/tmp/attention_mask")[:32]
    token_type_ids = dill_load("/tmp/token_type_ids")[:32]
    last_hidden_state = dill_load("/tmp/last_hidden_state")[:32]
    labels = dill_load("/tmp/labels")[:32]
    self = dill_load("/tmp/self")

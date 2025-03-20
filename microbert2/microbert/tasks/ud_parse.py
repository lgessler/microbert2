"""
Taken from https://github.com/allenai/allennlp-models/blob/main/allennlp_models/structured_prediction/models/biaffine_dependency_parser.py
"""

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import conllu
import numpy
import torch
import torch.nn.functional as F
from allennlp_light import ScalarMix
from allennlp_light.modules import FeedForward, InputVariationalDropout, Seq2SeqEncoder
from allennlp_light.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp_light.nn import Activation, InitializerApplicator
from allennlp_light.nn.chu_liu_edmonds import decode_mst
from allennlp_light.nn.util import get_device_of, get_range_vector, masked_log_softmax
from tango.common import FromParams, Lazy, det_hash
from tango.common.det_hash import CustomDetHash
from torch.nn import Embedding
from torch.nn.modules import Dropout
from torch.nn.utils.rnn import pad_sequence

from microbert2.common import pool_embeddings
from microbert2.microbert.model.model import remove_cls_and_sep
from microbert2.microbert.tasks.task import MicroBERTTask

logger = logging.getLogger(__name__)

POS_TO_IGNORE = {"`", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}


class AttachmentScores:
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.
    # Parameters
    ignore_classes : `List[int]`, optional (default = `None`)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
        self._ignore_classes: List[int] = ignore_classes or []

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

    def __call__(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_indices`.
        """
        detached = self.detach_tensors(predicted_indices, predicted_labels, gold_indices, gold_labels, mask)
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = detached
        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)
        total_sentences = correct_indices.size(0)
        total_words = correct_indices.numel() - (~mask).sum()

        self._unlabeled_correct += correct_indices.sum()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum()
        self._labeled_correct += correct_labels_and_indices.sum()
        self._exact_labeled_correct += labeled_exact_match.sum()
        self._total_sentences += total_sentences
        self._total_words += total_words

    def get_metric(
        self,
        reset: bool = False,
    ):
        """
        # Returns
        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0

        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(self._total_sentences)
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        metrics = {
            "UAS": unlabeled_attachment_score,
            "LAS": labeled_attachment_score,
            "UEM": unlabeled_exact_match,
            "LEM": labeled_exact_match,
        }
        return metrics

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0


class BiaffineDependencyParser(torch.nn.Module, FromParams):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    # Parameters

    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    input_dim : `int`, required.
        Dims of the input hidden representation.
    num_layers: `int`, required.
        Number of layers in the Transformer encoder stack, including the static embeddings.
        Needed if `use_layer_mix` is True.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : `int`, required.
        The dimension of the MLPs used for head arc prediction.
    num_pos_tags: `int`, required.
        Number of pos tags, needed for the Embedding module
    tag_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : `Embedding`, optional.
        Used to embed the `pos_tags` `SequenceLabelField` we get as input to the model.
    use_mst_decoding_for_validation : `bool`, optional (default = `True`).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = `0.0`)
        The dropout applied to the embedded text input.
    use_layer_mix : `bool`, optional, (default = `True`)
        When True, use a ScalarMix across all layers as the input hidden representation.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        input_dim: int,
        num_layers: int,
        tag_representation_dim: int,
        arc_representation_dim: int,
        num_pos_tags: int,
        num_head_tags: int,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        pos_tag_embedding_dim: int = 100,
        use_mst_decoding_for_validation: bool = True,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        layer_index: int = -1,
        use_layer_mix: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics_ignore_tags: list[int] = [],
    ) -> None:
        super().__init__()
        self.encoder = encoder

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(tag_representation_dim, tag_representation_dim, num_head_tags)

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = input_dim
        if pos_tag_embedding_dim is not None:
            self._pos_tag_embedding = Embedding(num_pos_tags, pos_tag_embedding_dim)
            representation_dim += pos_tag_embedding_dim
        else:
            self._pos_tag_embedding = None

        if representation_dim != encoder.get_input_dim():
            raise ValueError(
                f"Representation dim {representation_dim} must match encoder input dim {encoder.get_input_dim()}"
            )
        if tag_representation_dim != self.head_tag_feedforward.get_output_dim():
            raise ValueError(
                f"Tag representation dim {tag_representation_dim} must match head_tag_feedforward output dim {self.head_tag_feedforward.get_output_dim()}"
            )
        if arc_representation_dim != self.head_arc_feedforward.get_output_dim():
            raise ValueError(
                f"Arc representation dim {arc_representation_dim} must head_arc_feedforward output dim {self.head_arc_feedforward.get_output_dim()}"
            )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation
        self._attachment_scores = AttachmentScores(ignore_classes=metrics_ignore_tags)
        self._pos_to_ignore = []

        self.layer_index = layer_index
        self.use_layer_mix = use_layer_mix
        if use_layer_mix:
            self.mix = ScalarMix(num_layers)
        self._prev_step_training = True

        initializer(self)

    def forward(
        self,  # type: ignore
        hidden_masked: List[torch.Tensor],
        token_spans: torch.LongTensor,
        xpos: torch.LongTensor,
        head: torch.LongTensor = None,
        deprel: torch.LongTensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        hidden_masked : `List[torch.Tensor]`, required
            Layerwise input representations from the encoder with masking
        token_spans : `torch.LongTensor`, required
            Contains pairs of inclusive indices that encode the mapping from wordpiece tokens
            to original tokens.
        xpos : `torch.LongTensor`, required
            The output of a `SequenceLabelField` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        head : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape `(batch_size, sequence_length)`.
        deprel : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length)`.

        # Returns

        An output dictionary consisting of:

        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        arc_loss : `torch.FloatTensor`
            The loss contribution from the unlabeled arcs.
        loss : `torch.FloatTensor`, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : `torch.FloatTensor`
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : `torch.FloatTensor`
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : `torch.BoolTensor`
            A mask denoting the padded elements in the batch.
        """
        # Remove unneeded padding
        token_spans = token_spans[:, : xpos.shape[-1] + 2]

        if self.use_layer_mix:
            # Remove CLS and SEP
            trimmed = [remove_cls_and_sep(h_layer, token_spans) for h_layer in hidden_masked]
            hidden = [h_layer for h_layer, _ in trimmed]
            token_spans = trimmed[-1][1]
            # Pool wordpieces together into original token reprs
            input_reprs = [pool_embeddings(l, token_spans) for l in hidden]
            input_reprs = self.mix(input_reprs)
        else:
            hidden, token_spans = remove_cls_and_sep(hidden_masked[self.layer_index], token_spans)
            input_reprs = pool_embeddings(hidden, token_spans)
        mask = token_spans.gt(0).all(-1)
        mask[:, 0] = True

        if xpos is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(xpos)
            embedded_text_input = torch.cat([input_reprs, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ValueError("Model uses a POS embedding, but no POS tags were passed.")
        else:
            embedded_text_input = input_reprs

        predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll = self._parse(
            embedded_text_input, mask, deprel, head
        )

        loss = arc_nll + tag_nll
        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "arc_loss": arc_nll,
            "tag_loss": tag_nll,
            "loss": loss,
            "mask": mask,
        }

        if deprel is not None and head is not None:
            if self._prev_step_training != self.training:
                split = "val" if self.training else "train"
                metrics = self._attachment_scores.get_metric()
                print()
                logger.info(f'{split} LAS: {metrics["LAS"] * 100}')
                logger.info(f'{split} UAS: {metrics["UAS"] * 100}')
                self._attachment_scores.reset()
                self._prev_step_training = self.training

            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], xpos)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(
                predicted_heads[:, 1:],
                predicted_head_tags[:, 1:],
                head,
                deprel,
                evaluation_mask,
            )
            metrics = self._attachment_scores.get_metric()
            output_dict["las"] = metrics["LAS"] * 100
            output_dict["uas"] = metrics["UAS"] * 100

        return output_dict

    def _parse(
        self,
        embedded_text_input: torch.Tensor,
        mask: torch.BoolTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
        # Check if we have any gold trees and bail out if not
        if head_indices is not None and head_tags is not None:
            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                head_tags=head_tags,
                mask=mask,
            )
        else:
            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                head_tags=predicted_head_tags.long(),
                mask=mask,
            )

        return predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll

    def _construct_loss(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        # Returns

        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = masked_log_softmax(attended_arcs, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation, child_tag_representation, head_indices)
        normalised_head_tag_logits = masked_log_softmax(head_tag_logits, mask.unsqueeze(-1)) * mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions_by_sequence = mask.sum(-1) - 1

        arc_nll = -arc_loss.sum() / valid_positions_by_sequence.float().sum()
        tag_nll = -tag_loss.sum() / valid_positions_by_sequence.float().sum()
        return arc_nll, tag_nll

    def _greedy_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = ~mask.unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation, child_tag_representation, heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(numpy.stack(heads)).to(batch_energy.device),
            torch.from_numpy(numpy.stack(head_tags)).to(batch_energy.device),
        )

    def _get_head_tags(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        # Returns

        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations, child_tag_representation)
        return head_tag_logits

    def _get_mask_for_eval(self, mask: torch.BoolTensor, pos_tags: torch.LongTensor) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        # Parameters

        mask : `torch.BoolTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.

        # Returns

        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label)
            new_mask = new_mask & ~label_mask
        return new_mask


def read_split(path):
    result = []
    with open(path, "r") as f:
        for sentence in conllu.parse_incr(f):
            result.append(
                {
                    "tokens": [t["form"] for t in sentence if isinstance(t["id"], int)],
                    "xpos": [t["xpos"] for t in sentence if isinstance(t["id"], int)],
                    "head": [t["head"] for t in sentence if isinstance(t["id"], int)],
                    "deprel": [t["deprel"] for t in sentence if isinstance(t["id"], int)],
                }
            )
    return result


@MicroBERTTask.register("microbert2.microbert.tasks.ud_parse.UDParseTask")
class UDParseTask(MicroBERTTask, CustomDetHash):
    def __init__(
        self,
        head: Lazy[BiaffineDependencyParser],
        train_conllu_path: str,
        dev_conllu_path: str,
        test_conllu_path: Optional[str] = None,
        proportion: float = 0.1,
    ):
        self._dataset = {
            "train": read_split(train_conllu_path),
            "dev": read_split(dev_conllu_path),
            "test": read_split(test_conllu_path) if test_conllu_path is not None else [],
        }
        self._proportion = proportion
        relation_set = set(
            l for x in self._dataset["train"] + self._dataset["dev"] + self._dataset["test"] for l in x["deprel"]
        )
        xpos_set = set(
            l for x in self._dataset["train"] + self._dataset["dev"] + self._dataset["test"] for l in x["xpos"]
        )
        self._rels = {v: i for i, v in enumerate(sorted(list(relation_set)))}
        self._tags = {v: i for i, v in enumerate(sorted(list(xpos_set)))}
        self._head = head
        self._hash_string = (
            self.slug + train_conllu_path + dev_conllu_path + (test_conllu_path if test_conllu_path else "")
        )

    def det_hash_object(self) -> Any:
        return det_hash(self._hash_string)

    @property
    def slug(self):
        return "parse"

    def construct_head(self, model):
        ignore = [self._rels["punct"]] if "punct" in self._rels else []
        logger.info(f"deprel ignore list: {ignore}")
        return self._head.construct(
            num_pos_tags=len(self._tags), num_head_tags=len(self._rels), metrics_ignore_tags=ignore
        )

    @property
    def data_keys(self):
        return ["xpos", "head", "deprel"]

    @property
    def dataset(self):
        return self._dataset

    @property
    def inst_proportion(self) -> float:
        return self._proportion

    def tensorify_data(self, key, value):
        if key == "xpos":
            return torch.tensor([self._tags[v] for v in value])
        elif key == "deprel":
            return torch.tensor([self._rels[v] for v in value])
        elif key == "head":
            return torch.tensor(value)
        else:
            raise ValueError(key)

    def null_tensor(self, key):
        if key in self.data_keys:
            return torch.tensor([0])
        else:
            raise ValueError(key)

    def collate_data(self, key: str, values: list[torch.Tensor]):
        return pad_sequence(values, batch_first=True, padding_value=0)

    @property
    def progress_items(self):
        return ["las"]

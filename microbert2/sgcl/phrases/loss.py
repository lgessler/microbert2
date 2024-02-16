import logging
from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from microbert2.common import dill_dump, dill_load
from microbert2.sgcl.phrases.common import PhraseSgclConfig

logger = logging.getLogger(__name__)


def masked_kl_div(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    denom = mask.sum(dim=-1).clamp(min=1)
    masked_a = a.masked_fill(~mask.bool(), 1.0)
    masked_b = b.masked_fill(~mask.bool(), 1.0)
    term_1 = masked_a.log() - masked_b.log()
    term_2 = b * term_1
    return term_2.sum(dim=-1) / denom


def masked_log_softmax(t: torch.FloatTensor, mask: torch.BoolTensor, dim: int):
    return F.log_softmax(t.masked_fill(~mask, float("-inf")), dim=dim)


def masked_jsd(a: torch.Tensor, b: torch.Tensor, attention_mask: torch.Tensor):
    m = ((a + b) * 0.5).clamp(min=1e-6)
    term_1 = masked_kl_div(m, a, attention_mask)
    term_2 = masked_kl_div(m, b, attention_mask)
    return -(term_1 + term_2) * 0.5


def tmp_():
    a = torch.tensor([[[0.2, 0.2, 0.6], [1.0, 0.0, 0.0]]])
    b = torch.tensor([[[0.3, 0.3, 0.4], [0.9, 0.1, 0.0]]])
    mask = torch.tensor([[[1, 1, 1], [1, 1, 0]]]).bool()

    masked_kl_div(a, b, mask)
    F.kl_div(a[0, 0].log(), b[0, 0], reduction="mean", log_target=False)
    F.kl_div(a[0, 1].log(), b[0, 1], reduction="mean", log_target=False)

    masked_jsd(a, b, mask)


def jsd(p: torch.Tensor, q: torch.Tensor):
    m = (0.5 * (p + q)).log()
    term_1 = F.kl_div(m, p, reduction="none", log_target=False)
    # print(m[0,0,0,0])
    # print(term_1[0,0,0,0])
    # print(term_1.shape)
    term_2 = F.kl_div(m, q, reduction="none", log_target=False)
    return -0.5 * (term_1 + term_2)


def compute_info_nce(combined_sims: torch.Tensor) -> float:
    # take the softmax for InfoNCE
    softmaxed = -F.log_softmax(combined_sims, dim=1)
    loss = softmaxed[:, 0]
    return loss.mean().item()


def _pack_phrases_into_tensors(
    config: PhraseSgclConfig, phrase_sets_for_batch: List[List[Dict[str, Any]]], device: torch.device
) -> Dict[str, Any]:
    query_ids = defaultdict(list)
    positive_ids = defaultdict(list)
    negative_lists = defaultdict(list)
    included = []

    # Track these quantities, because they will be needed to determine the dimensions of the padded tensors
    all_negative_ids = [phrase_set["negatives"] for phrase_sets in phrase_sets_for_batch for phrase_set in phrase_sets]

    # greatest number of negative ids
    max_negative_ids = max(len(n) for n in all_negative_ids)
    # greatest number of negative phrases in a phrase set
    max_sets = max(len(sets) for sets in phrase_sets_for_batch)

    # Iterate through each batch item
    for i, phrase_sets in enumerate(phrase_sets_for_batch):
        # If we have no phrase sets, there's nothing to do
        if len(phrase_sets) == 0:
            continue
        included.append(i)

        for phrase_set in phrase_sets:
            query_ids[i].append(phrase_set["query"])
            positive_ids[i].append(phrase_set["positive"])
            # pad the negative ids
            negative_ids = phrase_set["negatives"]
            negative_ids += [-1] * (max_negative_ids - len(negative_ids))
            negative_lists[i].append(negative_ids)

    # initialize the packed tensors with -1 in the appropriate shapes
    batch_size = len(included)
    query_indexes = torch.full((batch_size, max_sets), -1, dtype=torch.long, device=device)
    positive_indexes = torch.full((batch_size, max_sets), -1, dtype=torch.long, device=device)
    negative_indexes = torch.full((batch_size, max_sets, max_negative_ids), -1, dtype=torch.long, device=device)

    # fill the packed tensors
    correction = 0
    for i in range(len(phrase_sets_for_batch)):
        if i not in included:
            correction += 1
            continue
        n = len(query_ids[i])
        query_indexes[i - correction, :n] = torch.tensor(query_ids[i])
        positive_indexes[i - correction, :n] = torch.tensor(positive_ids[i])
        negatives_for_set = torch.tensor(negative_lists[i])
        a, b = negatives_for_set.shape
        negative_indexes[i - correction, :a, :b] = negatives_for_set

    # generate masks
    negative_mask = negative_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))
    positive_mask = positive_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))
    query_mask = query_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))

    # clamp -1 to 0
    query_indexes = query_indexes.clamp(min=0)
    positive_indexes = positive_indexes.clamp(min=0)
    negative_indexes = negative_indexes.clamp(min=0)

    return {
        "query_indexes": query_indexes,
        "query_mask": query_mask,
        "positive_indexes": positive_indexes,
        "positive_mask": positive_mask,
        "negative_indexes": negative_indexes,
        "negative_mask": negative_mask,
        "included": included,
    }


def compute_phrase_loss_batched(
    config: PhraseSgclConfig,
    averaged_attentions: torch.FloatTensor,
    attention_mask: torch.Tensor,
    phrase_sets_for_batch: List[List[Dict[str, Any]]],
) -> torch.Tensor:
    if all(len(s) == 0 for s in phrase_sets_for_batch):
        return torch.tensor(0.0, device=averaged_attentions.device)

    packed = _pack_phrases_into_tensors(config, phrase_sets_for_batch, averaged_attentions.device)
    query_indexes = packed["query_indexes"]
    negative_indexes = packed["negative_indexes"]
    positive_indexes = packed["positive_indexes"]
    negative_mask = packed["negative_mask"]
    positive_mask = packed["positive_mask"]
    included = packed["included"]

    # We do not want to consider items that did not have any samples. Filter them out here.
    averaged_attentions = averaged_attentions[:, included]
    attention_mask = attention_mask[included, :]

    num_layers, batch_size = averaged_attentions.shape[:2]
    _, max_samples, max_negatives = negative_indexes.shape

    # take query and positive distributions
    query_distrs = averaged_attentions.take_along_dim(query_indexes.unsqueeze(0).unsqueeze(-1), dim=2)
    positive_distrs = averaged_attentions.take_along_dim(positive_indexes.unsqueeze(0).unsqueeze(-1), dim=2)

    # flatten negative indexes and get their distrs
    flattened_negative_indexes = negative_indexes.reshape((batch_size, -1))
    flattened_negative_distrs = averaged_attentions.take_along_dim(
        flattened_negative_indexes.unsqueeze(0).unsqueeze(-1), dim=2
    )
    negative_distrs = flattened_negative_distrs.reshape((num_layers, batch_size, *negative_indexes.shape[1:], -1))

    # prepare for jsd
    expanded_query_distrs = query_distrs.unsqueeze(-2)
    combined_distrs = torch.cat((positive_distrs.unsqueeze(-2), negative_distrs), dim=-2)
    combined_mask = torch.cat((positive_mask.unsqueeze(-1), negative_mask), dim=-1)
    expanded_attention_mask = attention_mask.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)

    # get jsd sims
    # need to mask both the attentions AND the trees
    # mask the trees which are empty
    masked_combined_distrs = combined_distrs.masked_fill(~combined_mask.bool().unsqueeze(0).unsqueeze(-1), 1.0)
    # attention masking happens inside masked_jsd
    sims = masked_jsd(
        expanded_query_distrs.clamp(min=1e-6), masked_combined_distrs.clamp(min=1e-6), expanded_attention_mask
    )

    # compute InfoNCE
    combined_distrs_mask = torch.cat((positive_mask.unsqueeze(-1), negative_mask), dim=-1).unsqueeze(0)
    filled_sims = sims.masked_fill(~combined_distrs_mask, float("-inf"))
    softmaxed_sims = -masked_log_softmax(filled_sims / config.temperature, combined_distrs_mask, dim=-1)
    positive_sims = softmaxed_sims[:, :, :, 0]

    # take the mean of InfoNCE across all samples
    if config.last_layer_only:
        positive_sims = positive_sims[-1]
    losses = positive_sims.masked_select(positive_mask.unsqueeze(0))
    loss = losses.mean()
    return loss


def phrase_guided_loss(
    config: PhraseSgclConfig,
    attentions: List[torch.Tensor],
    attention_mask: torch.Tensor,
    phrase_sets: List[List[Dict[str, Any]]],
) -> torch.tensor:
    # dill_dump(config, "/tmp/config")
    # dill_dump(attentions, "/tmp/attentions")
    # dill_dump(attention_mask, "/tmp/attention_mask")
    # dill_dump(phrase_sets, "/tmp/phrase_sets")

    attentions = torch.stack(attentions, dim=0)
    # average across heads
    averaged_attentions = attentions.mean(dim=2)
    loss = compute_phrase_loss_batched(config, averaged_attentions, attention_mask, phrase_sets)
    return loss


def tmp():
    config = dill_load("/tmp/config")
    attentions = dill_load("/tmp/attentions")
    attention_mask = dill_load("/tmp/attention_mask")
    phrase_sets_for_batch = dill_load("/tmp/phrase_sets")

from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import torch
import torch.nn.functional as F

import microbert2.common as lc
from microbert2.sgcl.trees.common import TreeSgclConfig
from microbert2.sgcl.trees.generation import generate_subtrees


def masked_softmax(t: torch.FloatTensor, mask: torch.BoolTensor, dim: int):
    return F.softmax(t.masked_fill(~mask, float("-inf")), dim=dim)


def masked_log_softmax(t: torch.FloatTensor, mask: torch.BoolTensor, dim: int):
    return F.log_softmax(t.masked_fill(~mask, float("-inf")), dim=dim)


def _pack_trees_into_index_tensors(
    config: TreeSgclConfig, tree_sets_for_batch: List[List[Dict[str, Any]]], batch_size: int, device: torch.device
):
    root_ids = defaultdict(list)
    positives = defaultdict(list)
    negative_lists = defaultdict(list)

    # Track these quantities, because they will be needed to determine the dimensions of the padded tensors
    all_negative_ids = [
        [k for k in negative.keys()]
        for tree_sets in tree_sets_for_batch
        for tree_set in tree_sets
        for negative in tree_set["negatives"]
    ]
    all_positive_ids = [
        [k for k in tree_set["positive"].keys()] for tree_sets in tree_sets_for_batch for tree_set in tree_sets
    ]

    # greatest number of ids in a positive tree
    max_positive_ids = max(len(p) for p in all_positive_ids)
    # greatest number of ids in a negative tree
    max_negative_ids = max(len(n) for n in all_negative_ids)
    # greatest number of negative trees in a tree set
    max_negative_trees = max(len(tree_set["negatives"]) for tree_sets in tree_sets_for_batch for tree_set in tree_sets)

    # Iterate through each batch item
    for i, tree_sets in enumerate(tree_sets_for_batch):
        # If we have no tree sets, there's nothing to do
        if len(tree_sets) == 0:
            continue

        for tree_set in tree_sets:
            # record the root
            root_id = tree_set["root_id"]
            root_ids[i].append(root_id)

            # record the padded positive ids
            positive_ids = ([root_id] if config.include_root_in_sims else []) + [
                k for k in tree_set["positive"].keys() if k != root_id
            ]
            positive_ids = positive_ids + ([-1] * (max_positive_ids - len(positive_ids)))
            positives[i].append(positive_ids)

            # pad the negative ids
            negative_ids = [
                ([root_id] if config.include_root_in_sims else []) + [k for k in negative.keys() if k != root_id]
                for negative in tree_set["negatives"]
            ]
            padded_negative_ids = [([k for k in ids] + ([-1] * (max_negative_ids - len(ids)))) for ids in negative_ids]
            # add extra rows full of padding if we're under the max tree count
            while len(padded_negative_ids) < max_negative_trees:
                padded_negative_ids.append([-1] * max_negative_ids)
            negative_lists[i].append(padded_negative_ids)

    # maximum number of subtrees for a given batch item
    max_subtrees = max(len(x) for x in positives.values())

    # initialize the packed tensors with -1 in the appropriate shapes
    root_indexes = torch.full((batch_size, max_subtrees), -1, dtype=torch.long, device=device)
    positive_indexes = torch.full((batch_size, max_subtrees, max_positive_ids), -1, dtype=torch.long, device=device)
    negative_indexes = torch.full(
        (batch_size, max_subtrees, max_negative_trees, max_negative_ids), -1, dtype=torch.long, device=device
    )

    # fill the packed tensors
    for i in range(len(tree_sets_for_batch)):
        n = len(root_ids[i])
        if n == 0:
            continue
        root_indexes[i, :n] = torch.tensor(root_ids[i])
        positives_for_set = torch.tensor(positives[i])
        positive_indexes[i, : positives_for_set.shape[0], : positives_for_set.shape[1]] = positives_for_set
        negatives_for_set = torch.tensor(negative_lists[i])
        a, b, c = negatives_for_set.shape
        negative_indexes[i, :a, :b, :c] = negatives_for_set

    # generate masks
    negative_mask = negative_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))
    positive_mask = positive_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))
    root_mask = root_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))

    # clamp -1 to 0
    root_indexes = root_indexes.clamp(min=0)
    positive_indexes = positive_indexes.clamp(min=0)
    negative_indexes = negative_indexes.clamp(min=0)

    del all_negative_ids
    del all_positive_ids
    del max_negative_ids
    del max_positive_ids
    del max_negative_trees

    return {
        "root_indexes": root_indexes,
        "root_mask": root_mask,
        "positive_indexes": positive_indexes,
        "positive_mask": positive_mask,
        "negative_indexes": negative_indexes,
        "negative_mask": negative_mask,
    }


def assess_tree_sgcl_batched(
    config: TreeSgclConfig,
    tree_sets_for_batch: List[List[Dict[str, Any]]],
    hidden_states: List[torch.Tensor],
    token_spans: torch.LongTensor,
) -> torch.tensor:
    if len(tree_sets_for_batch) == 0 or all(len(s) == 0 for s in tree_sets_for_batch):
        return torch.tensor(0.0, device=token_spans.device)

    device = hidden_states[0].device
    tokenwise_hidden_states = torch.stack([lc.pool_embeddings(layer_i, token_spans) for layer_i in hidden_states])
    num_layers, batch_size, sequence_length, num_hidden = tokenwise_hidden_states.shape

    packed = _pack_trees_into_index_tensors(config, tree_sets_for_batch, batch_size, device)
    root_indexes = packed["root_indexes"]
    negative_indexes = packed["negative_indexes"]
    positive_indexes = packed["positive_indexes"]
    root_mask = packed["root_mask"]
    negative_mask = packed["negative_mask"]
    positive_mask = packed["positive_mask"]
    del packed

    # Select root vectors
    root_indexes = root_indexes.unsqueeze(0).unsqueeze(-1).repeat(num_layers, 1, 1, num_hidden)
    root_zs = tokenwise_hidden_states.take_along_dim(root_indexes, dim=2) * root_mask.unsqueeze(0).repeat(
        num_layers, 1, 1
    ).unsqueeze(-1)

    # Select positive vectors
    flattened_positive_number = positive_indexes.shape[1] * positive_indexes.shape[2]
    positive_target_shape = (num_layers, batch_size, flattened_positive_number, num_hidden)
    flattened_positive_indexes = (
        positive_indexes.unsqueeze(0).unsqueeze(-1).repeat(num_layers, 1, 1, 1, num_hidden).view(positive_target_shape)
    )
    positive_zs = tokenwise_hidden_states.take_along_dim(flattened_positive_indexes, dim=2)

    # Select negative vectors
    flattened_negative_number = negative_indexes.shape[1] * negative_indexes.shape[2] * negative_indexes.shape[3]
    negative_target_shape = (num_layers, batch_size, flattened_negative_number, num_hidden)
    flattened_negative_indexes = (
        negative_indexes.unsqueeze(0)
        .unsqueeze(-1)
        .repeat(num_layers, 1, 1, 1, 1, num_hidden)
        .view(negative_target_shape)
    )
    negative_zs = tokenwise_hidden_states.take_along_dim(flattened_negative_indexes, dim=2)

    # Now we need to compute dot products. Unflatten positive and negative zs to put them back
    # into their original shape.
    # shape: [num_layers, batch_size, max_subtrees, max_positive_ids, hidden_dim]
    reshaped_positive_zs = positive_zs.view(num_layers, *positive_indexes.shape, num_hidden)
    # shape: [num_layers, batch_size, max_subtrees, max_negative_subtrees, max_negative_ids, hidden_dim]
    reshaped_negative_zs = negative_zs.view(num_layers, *negative_indexes.shape, num_hidden)

    # Dot the root with all subtree children for both tree types and apply mask
    positive_dots = torch.einsum("abch,abcdh->abcd", root_zs, reshaped_positive_zs) * positive_mask.unsqueeze(0)
    # Softmax the dots
    positive_eij = masked_softmax(positive_dots, positive_mask, -1).nan_to_num()
    # Take the Hadamard product of the positive zs and the softmaxed dots and sum the resulting zs together
    positive_sim_rhs = (reshaped_positive_zs * positive_eij.unsqueeze(-1)).sum(-2)
    # Take cosine similarity between the root representation and the softmaxed and summed representation
    positive_cosines = F.cosine_similarity(root_zs, positive_sim_rhs, dim=-1)

    # Do the same for negatives
    negative_dots = torch.einsum("abch,abcdeh->abcde", root_zs, reshaped_negative_zs) * negative_mask.unsqueeze(0)
    negative_eij = masked_softmax(negative_dots, negative_mask, -1).nan_to_num()
    negative_sim_rhs = (reshaped_negative_zs * negative_eij.unsqueeze(-1)).sum(-2)

    negative_cosines = F.cosine_similarity(root_zs.unsqueeze(-2), negative_sim_rhs, dim=-1)
    k = min(config.max_negatives_used_in_loss, negative_cosines.shape[-1])
    top_negative_cosines = negative_cosines.topk(k).values
    top_negative_mask = negative_mask[:, :, : config.max_negatives_used_in_loss]

    # Pack the positive with the negative cosines to prepare for softmaxing
    combined = torch.concat((positive_cosines.unsqueeze(-1), top_negative_cosines), dim=-1)

    reduced_negative_mask = top_negative_mask.any(dim=-1)
    reduced_positive_mask = positive_mask.any(dim=-1)
    combined_mask = torch.concat((reduced_positive_mask.unsqueeze(-1), reduced_negative_mask), dim=-1)

    softmaxed = -masked_log_softmax(combined / config.temperature, combined_mask, dim=-1)[:, :, :, 0]
    if config.last_layer_only:
        softmaxed = softmaxed[-1]
    losses = softmaxed.masked_select(reduced_positive_mask)
    loss = losses.mean()
    return loss


################################################################################
# top level function
################################################################################
def syntax_tree_guided_loss(
    config: TreeSgclConfig,
    hidden_states: List[torch.Tensor],
    token_spans: torch.LongTensor,
    tree_sets: List[List[Dict[str, Any]]],
) -> torch.tensor:
    """
    Compute the tree-guided contrastive loss presented in Zhang et al. 2022
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    Args:
        config:
            TreeSgclConfig
        hidden_states:
            Has n tensors, where n is the number of layers in the Transformer model.
            Each tensor has shape [batch_size, wordpiece_len, hidden_dim].
        token_spans:
            A tensor of shape [batch_size, token_len + 2, 2]: each 2-tuple in the last dim represents the
            wordpiece indices (inclusive on both sides) of the wordpiece span that corresponds to an original
            token that was split by the subword tokenizer. We need this in order to pool hidden representations
            for the tree-based loss term. Note that dim 1 has an extra 2 because of the [CLS] and [SEP] tokens
            that are not included in the syntax tree.
        head:
            A tensor of shape [batch_size, token_len] with indexes of each token's head. Note that these are
            loaded directly from the conllu file, so they are 1-indexed and do not account for special tokens.
            Also note that we do NOT include a sentinel token for ROOT. Consider the following example:
                1   Almaañ      0   root
                2   ci          3   case
                3   jamonoy     1   nmod
                4   Napoleon    3   nmod
            In this case, its entry in `head` would be [0, 3, 1, 3]. Note however that Napoleon's head is NOT at
            index 3. It is instead at index 2, since the tensor is 0-indexed.
    Returns: float
    """
    # print(config)
    # print(token_spans.shape)
    # print(head.shape)
    # print(head[0])
    # print(config)
    # lc.dill_dump(config, "/tmp/config")
    # lc.dill_dump(hidden_states, "/tmp/hidden_states")
    # lc.dill_dump(token_spans, "/tmp/token_spans")
    # lc.dill_dump(tree_sets, "/tmp/tree_sets")
    loss = assess_tree_sgcl_batched(config, tree_sets, hidden_states, token_spans)
    return loss


def scratch():
    config = lc.dill_load("/tmp/config")
    hidden_states = lc.dill_load("/tmp/hidden_states")
    token_spans = lc.dill_load("/tmp/token_spans")
    tree_sets_for_batch = lc.dill_load("/tmp/tree_sets")
    # head_map = {0: None, **{i + 1: h.item() for i, h in enumerate(head[0])}}
    # all_subtrees = get_all_subtrees(config, head_map)
    # eligible_subtrees = sorted(get_eligible_subtrees(config, head_map, all_subtrees), key=lambda x: x["root_id"])
    # output = generate_negative_trees(config, all_subtrees, **eligible_subtrees[2])

    assess_tree_sgcl_batched(config, tree_sets_for_batch, hidden_states, token_spans)

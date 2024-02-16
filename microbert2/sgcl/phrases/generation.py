import random
from typing import Any, Dict, List

import torch

from microbert2.common import dill_dump, dill_load
from microbert2.sgcl.generation_common import get_all_subtrees, get_head_map
from microbert2.sgcl.phrases.common import PhraseSgclConfig


def depth_of_tree(query: int, t: Dict[int, int | None]) -> int:
    max_depth = 1
    queue = [(k, 1) for k, v in t.items() if v == query]
    while len(queue) > 0:
        current, parent_depth = queue.pop(0)
        depth = parent_depth + 1
        if depth > max_depth:
            max_depth = depth
        children = [(k, depth) for k, v in t.items() if v == current]
        queue.extend(children)
    return max_depth


def get_token_to_head_wordpiece_map(spans):
    m = {}
    for i, span in enumerate(spans):
        k = span[0].item()
        if k == -1:
            break
        if i not in m:
            m[i] = k
    return m


def compute_phrase_set(
    config: PhraseSgclConfig,
    head_map: Dict[int, int | None],
    all_subtrees: Dict[int, Dict[int, int | None]],
    t2wp: Dict[int, int],
) -> List[Dict[str, Any]]:
    shuffled_subtrees = list(all_subtrees.items())
    random.shuffle(shuffled_subtrees)

    phrase_set = []
    for query, subtree in shuffled_subtrees:
        if len(phrase_set) > config.max_subtrees_per_sentence:
            break
        depth = depth_of_tree(query, subtree)
        if depth > config.max_subtree_height:
            continue
        if len(subtree.keys()) < config.min_subtree_token_count:
            continue
        tokens_not_in_phrase = [token_id for token_id in head_map.keys() if token_id not in subtree and token_id > 0]
        if len(tokens_not_in_phrase) == 0:
            continue

        # Positive instance: a randomly sampled token from the subtree
        positive_set = set(subtree.keys())
        positive_tid = random.sample(tuple(positive_set), 1)[0]
        positive = t2wp[positive_tid]

        # Query instance: another randomly sampled token from the subtree, distinct from positive
        query_set = set(subtree.keys()) - {positive_tid}
        query = t2wp[random.sample(tuple(query_set), 1)[0]]

        # Negatives: randomly sampled from outside the subtree
        negatives = [
            t2wp[i]
            for i in random.sample(
                tokens_not_in_phrase, min(config.max_negative_per_positive, len(tokens_not_in_phrase))
            )
        ]
        phrase_set.append({"query": query, "positive": positive, "negatives": negatives})
    return phrase_set


def generate_phrase_sets(
    config: PhraseSgclConfig, head: torch.LongTensor, token_spans: torch.Tensor, head_length: torch.LongTensor
) -> List[List[Dict[str, Any]]]:
    # dill_dump(head, '/tmp/head')
    # dill_dump(token_spans, '/tmp/token_spans')
    head_maps = get_head_map(head, head_length)
    token_to_head_wordpiece_maps = [get_token_to_head_wordpiece_map(spans) for spans in token_spans]
    subtrees = [get_all_subtrees(head_map) for head_map in head_maps]
    phrase_sets = [
        compute_phrase_set(config, head_maps[i], subtrees[i], token_to_head_wordpiece_maps[i])
        for i in range(len(subtrees))
    ]
    return phrase_sets


def tmp():
    head = dill_load("/tmp/head")
    token_spans = dill_load("/tmp/token_spans")

from typing import Dict, List, Set

import torch


def get_head_map(head: torch.LongTensor, head_length: torch.LongTensor) -> List[Dict[int, int | None]]:
    # split the batched head tensor into one tensor per input sequence, with padding removed
    padless_head = [head[i, :hl] for i, hl in enumerate(head_length)]
    # Map from IDs to heads. Note that this is all 1-indexed, with 0 being the dummy ROOT node.
    head_map = [{0: None, **{i + 1: h.item() for i, h in enumerate(heads)}} for heads in padless_head]
    return head_map


def immediate_children(head_map: Dict[int, int | None], token_id: int) -> List[int]:
    return [child for child, parent in head_map.items() if parent == token_id]


def subtree_of_id(head_map: Dict[int, int | None], token_id: int) -> Dict[int, int | None]:
    """Get the subtree rooted at a given ID."""
    subtree = {token_id: None}
    queue = immediate_children(head_map, token_id)
    while len(queue) > 0:
        token_id = queue.pop(0)
        subtree[token_id] = head_map[token_id]
        children = immediate_children(head_map, token_id)
        queue.extend(children)
    return subtree


def adjacent_ids_of_subtree(head_map: Dict[int, int | None], subtree_ids: Set[int]) -> Set[int]:
    """
    Return set of IDs that are adjacent to all IDs in a subtree
    and are NOT a direct ancestor of the subtree's root
    """
    adjacent = set()

    # Get parents
    parent_ids = set()
    current = tuple(subtree_ids)[0]
    while current is not None:
        parent_ids.add(current)
        current = head_map[current]

    # On to the main work
    for token_id in subtree_ids:
        left = token_id - 1
        right = token_id + 1
        for x in [left, right]:
            if x > 0 and x not in subtree_ids and x not in parent_ids and x in head_map.keys():
                adjacent.add(x)
    return adjacent


def get_all_subtrees(head_map: Dict[int, int | None]) -> Dict[int, Dict[int, int | None]]:
    subtrees = {}
    for token_id in range(1, len(head_map.items())):
        subtrees[token_id] = subtree_of_id(head_map, token_id)
    return subtrees

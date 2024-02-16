import logging
import os
import random
import sys

import datasets
import more_itertools as mit
import stanza
from datasets import DatasetDict, IterableDatasetDict, Sequence, Value
from tango import Step
from tango.common import Tqdm
from tango.integrations.datasets import DatasetsFormat

from microbert2.common import dill_dump, dill_load

logger = logging.getLogger(__name__)


def extend_tree_with_subword_edges(output):
    token_spans = output["token_spans"]
    for x in output["head"]:
        assert "." not in x and "-" not in x
    head = [int(x) for x in output["head"]]
    deprel = output["deprel"]
    orig_head = head.copy()
    orig_deprel = deprel.copy()
    # Map from old token IDs to new token IDs after subword expansions
    id_map = {}
    running_diff = 0
    if len(token_spans) // 2 != len(head) + 2:
        return None
    for token_id in range(0, len(token_spans) // 2):
        id_map[token_id] = token_id + running_diff
        b, e = token_spans[token_id * 2 : (token_id + 1) * 2]
        running_diff += e - b
    # Note how many subwords have been added so far
    new_token_spans = []
    for token_id in range(0, len(token_spans) // 2):
        # Inclusive indices of the subwords that the original token corresponds to
        b, e = token_spans[token_id * 2 : (token_id + 1) * 2]
        # If not a special token (we're assuming there are 2 on either end of the sequence),
        # replace the head value of the current token with the mapped value
        if token_id != 0 and token_id != ((len(token_spans) // 2) - 1):
            head[id_map[token_id] - 1] = id_map[orig_head[token_id - 1]]
        if e == b:
            # If we have a token that corresponds to a single subword, just append the same token_spans values
            new_token_spans.append(b)
            new_token_spans.append(e)
        else:
            # Note how many expansion subwords we'll add
            diff = e - b
            # This is the first subword in the token's index into head and deprel. Remember token_id is 1-indexed
            first_subword_index = id_map[token_id] - 1
            new_token_spans.append(b)
            new_token_spans.append(b)
            # For each expansion subword, add a separate token_spans entry and expand head and deprel.
            # Head's value is the ID of the first subword in the token it belongs to
            for j in range(1, diff + 1):
                new_token_spans.append(b + j)
                new_token_spans.append(b + j)
                head.insert(first_subword_index + j, id_map[token_id])
                deprel.insert(first_subword_index + j, "subword")
    heads = [int(x) for x in head]
    for h in heads:
        current = h
        seen = set()
        while current != 0:
            if current in seen:
                raise Exception(f"Cycle detected!\n{orig_head}\n{orig_deprel}\n{token_spans}\n\n{head}\n{deprel}")
            seen.add(current)
            current = heads[current - 1]
    output["dependency_token_spans"] = new_token_spans
    output["orig_head"] = orig_head
    output["orig_deprel"] = orig_deprel
    output["head"] = heads
    output["deprel"] = deprel

    if len(output["input_ids"]) - 2 != len(output["head"]):
        print("token_spans", len(output["token_spans"]), output["token_spans"])
        print("dependency_token_spans", len(output["dependency_token_spans"]), output["dependency_token_spans"])
        print("head", len(output["head"]), output["head"])
        print("orig_head", len(output["orig_head"]), output["orig_head"])
        # with open("workspace/models/coptic_mx_hidden_size-128_intermediate_size-512_max_position_embeddings-512_num_attention_heads-8_num_hidden_layers-3/vocab.txt", "r") as f:
        #    td = {i: t for i, t in enumerate(f.read().strip().split("\n"))}
        print("input_ids", len(output["input_ids"]), output["input_ids"])
        raise ValueError("length of head should be equal to length of input_ids - 2")
    return output


@Step.register("microbert2.data.postprocess::expand_trees_with_subword_edges")
class ExpandTreesWithSubwordEdges(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def process_split(self, split, data):
        def inner():
            total = 0
            discarded = 0
            for d in data:
                res = extend_tree_with_subword_edges(d)
                total += 1
                if res is None:
                    discarded += 1
                    logger.warning(
                        f"Discarding instance because extending with subword edges failed."
                        f" Total discarded: {discarded}/{total}"
                    )
                else:
                    yield res

        features = datasets.Features(
            {
                **data.features,
                "orig_head": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "orig_deprel": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "dependency_token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            }
        )

        return datasets.Dataset.from_generator(inner, features=features)

    def run(self, dataset: IterableDatasetDict) -> DatasetDict:
        dataset_dict = {}
        for split, data in dataset.items():
            dataset_dict[split] = self.process_split(split, data)

        return datasets.DatasetDict(dataset_dict)

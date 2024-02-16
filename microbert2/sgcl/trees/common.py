################################################################################
# Subtree sampling methods
################################################################################
import random
from typing import Any, Dict, List

from tango.common import FromParams, Registrable


class SubtreeSamplingMethod(Registrable):
    """Subclasses of this specify how to select subtrees in a given sentence."""

    def sample(self, subtrees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplemented()


@SubtreeSamplingMethod.register("random")
class RandomSubtreeSamplingMethod(SubtreeSamplingMethod):
    """Sample subtrees by randomly selecting up to `max_number` subtrees in the sentence."""

    def __init__(self, max_number: int):
        self.max_number = max_number

    def sample(self, subtrees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return random.sample(subtrees, min(self.max_number, len(subtrees)))


@SubtreeSamplingMethod.register("all")
class AllSubtreeSamplingMethod(SubtreeSamplingMethod):
    """Sample subtrees by using all subtrees."""

    def sample(self, subtrees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return subtrees


################################################################################
# Config
################################################################################
class TreeSgclConfig(FromParams):
    def __init__(
        self,
        max_negative_per_subtree: int = 10,
        min_subtree_size: int = 2,
        max_subtree_size: int = 10,
        max_replacements: int = 10,
        max_node_count_difference: int = 1,
        max_negatives_used_in_loss: int = 3,
        subtree_sampling_method: SubtreeSamplingMethod = AllSubtreeSamplingMethod(),
        include_root_in_sims: bool = False,
        temperature: float = 0.1,
        last_layer_only: bool = True,
    ):
        """
        Args:
            max_negative_per_subtree:
                The maximum number of negative trees we will generate for any given positive tree.
                Defaults to 10.
            min_subtree_size:
                The minimum size (inclusive) of a subtree that will be used for the contrastive learning objective.
                Defaults to 2.
            max_subtree_size:
                The maximum size (inclusive) of a subtree that will be used for the contrastive learning objective.
                Defaults to 10.
            max_replacements:
                The maximum number of node replacements we will make while generating a negative tree.
                Defaults to 10.
            max_node_count_difference:
                The maximum difference in node count that we will tolerate between a negative and positive tree.
                Defaults to 1.
            max_negatives_used_in_loss:
                Specifies the maximum number of negative trees to use in the loss calculation.
                Trees will be ranked according to tree_sim first.
            subtree_sampling_method:
                See `SubtreeSamplingMethod`.
            include_root_in_sims:
                If false, when computing tree similarities, do not calculate the similarity between the root and
                itself.
            temperature:
                Temperature hyperparameter for InfoNCE loss
            last_layer_only:
                If True, assess the tree-based loss term only on the last layer. Otherwise, assess on all layers
                and take the mean.
        """
        self.max_negative_per_subtree = max_negative_per_subtree
        self.min_subtree_size = min_subtree_size
        self.max_subtree_size = max_subtree_size
        self.max_replacements = max_replacements
        self.max_node_count_difference = max_node_count_difference
        self.max_negatives_used_in_loss = max_negatives_used_in_loss
        self.subtree_sampling_method = subtree_sampling_method
        self.include_root_in_sims = include_root_in_sims
        self.temperature = temperature
        self.last_layer_only = last_layer_only

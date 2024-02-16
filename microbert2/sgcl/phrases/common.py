from tango.common import FromParams


class PhraseSgclConfig(FromParams):
    def __init__(
        self,
        min_subtree_token_count: int = 2,
        max_subtree_height: int = 2,
        max_negative_per_positive: int = 10,
        max_subtrees_per_sentence: int = 1_000_000,
        temperature: float = 0.1,
        last_layer_only: bool = True,
    ):
        """
        Args:
            min_subtree_token_count:
                Minimum amount of tokens a subtree must have to be considered for the phrase-based loss.
                Defaults to 2.
            max_subtree_height:
                Maximum height a subtree must have to be considered for the phrase-based loss.
                Defaults to 2.
            max_negative_per_positive:
                Maximum number of negative samples to generate for a single positive sample.
                Defaults to 10.
            max_subtrees_per_sentence:
                Maximum number of subtrees to generate per sentence.
                Defaults to 1_000_000.
            temperature:
                Temperature hyperparameter for InfoNCE loss
            last_layer_only:
                If True, assess the tree-based loss term only on the last layer. Otherwise, assess on all layers
                and take the mean.
        """
        self.min_subtree_token_count = min_subtree_token_count
        self.max_subtree_height = max_subtree_height
        self.max_negative_per_positive = max_negative_per_positive
        self.max_subtrees_per_sentence = max_subtrees_per_sentence
        self.temperature = temperature
        self.last_layer_only = last_layer_only

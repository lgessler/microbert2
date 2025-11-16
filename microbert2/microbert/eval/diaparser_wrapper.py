"""
Wrapper script for running diaparser with PyTorch 2.6+ safe globals.
This module patches torch.load to work with diaparser's serialization format.
"""
import sys
import torch
import io
from diaparser.utils.config import Config
from diaparser.utils.field import Field, BertField
from diaparser.utils.vocab import Vocab
from diaparser.utils.transform import CoNLL
from diaparser.parsers import Parser
from diaparser.parsers.biaffine_dependency import BiaffineDependencyParser

# Add diaparser classes to PyTorch safe globals for weights_only loading
torch.serialization.add_safe_globals([
    Config,
    Field,
    BertField,
    Vocab,
    CoNLL,
    Parser,
    BiaffineDependencyParser,
    getattr,
    io.open
])

if __name__ == "__main__":
    from diaparser.cmds.biaffine_dependency import main
    main()

"""Evaluation modules for MicroBERT."""

# Import to ensure Tango step registration
from microbert2.microbert.eval.dependency_parsing import EvaluateDependencyParsing
from microbert2.microbert.eval.ner import EvaluateNER
from microbert2.microbert.eval.continued_pretraining import ContinuedMLMPretraining

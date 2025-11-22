import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import io
from tango import Step

logger = logging.getLogger(__name__)


class DependencyParsingEvaluator:
    """Evaluator for dependency parsing tasks using DiaParser."""

    def __init__(self, model_path: str, save_path: str, train_data_path: str, dev_data_path: str, test_data_path: str) -> None:
        """
        Initialize the dependency parsing evaluator.

        Args:
            model_path: Path to the model or name of pretrained model
                       (e.g., 'en_ewt-electra', 'biaffine-dep-en', or path to custom model)
            save_path: Path to save the trained model

        Note:
            Device is automatically selected - GPU (cuda:0) if available, otherwise CPU
        """
        self.model_path = model_path
        self.save_path = save_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.dev_data_path = dev_data_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        self.parser = None

    def load_model(self) -> None:
        """Load the dependency parsing model."""
        try:
            from diaparser.parsers import Parser
        except ImportError:
            raise ImportError(
                "diaparser is not installed. Install it with: pip install -U diaparser"
            )

        logger.info(f"Loading model from: {self.save_path}")
        try:
            self.parser = Parser.load(f"{self.save_path}/model", device=self.device,weights_only=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(self, save_path: str, model_path: str, train_data_path: str, dev_data_path: str, test_data_path: str) -> Dict[str, Any]:
        """Train model and return best test scores."""
        # Create save directory if it doesn't exist (must be done before training)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            from diaparser.parsers.biaffine_dependency import BiaffineDependencyParser
        except ImportError:
            raise ImportError(
                "diaparser is not installed. Install it with: pip install -U diaparser"
            )
        logger.info(f"Training model with DiaParser")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Save path: {save_path}/model")
        logger.info(f"  Train data: {train_data_path}")
        logger.info(f"  Dev data: {dev_data_path}")
        logger.info(f"  Test data: {test_data_path}")

        try:
            # Build parser first, then train
            # Step 1: Build the parser with configuration
            parser = BiaffineDependencyParser.build(
                path=f"{save_path}/model",
                train=train_data_path,
                dev=dev_data_path,
                test=test_data_path,
                encoder='bert',
                bert=model_path,
                device=self.device,
                feat='bert',  # Use BERT features, not char features
                max_len=512,  # Maximum sequence length for tokenization
                embed=None,  # No additional word embeddings when using BERT
                partial=False,  # Trees are fully annotated (not partial)
                tree=True,  # Enforce tree constraints during parsing
                proj=False,  # Don't enforce projectivity (allows non-projective trees)
                punct=False,  # Don't ignore punctuation during evaluation
                build=True,  # Force rebuild the model
            )

            # Step 2: Train the parser
            parser.train(
                train=train_data_path,
                dev=dev_data_path,
                test=test_data_path,
                batch_size=5000,
                partial=False,  # Trees are fully annotated (not partial)
                tree=True,  # Enforce tree constraints during parsing
                proj=False,  # Don't enforce projectivity (allows non-projective trees)
                punct=False,  # Don't ignore punctuation during evaluation
            )

            logger.info("Model trained and saved successfully")

            # Evaluate on test set to get metrics
            logger.info("Evaluating on test set...")
            test_results = parser.evaluate(test_data_path, batch_size=5000)

            # Extract metrics from test results
            results = {
                'loss': test_results['loss'] if 'loss' in test_results else None,
                'UAS': test_results['UAS'] if 'UAS' in test_results else None,
                'LAS': test_results['LAS'] if 'LAS' in test_results else None,
                'UCM': test_results['UCM'] if 'UCM' in test_results else None,
                'LCM': test_results['LCM'] if 'LCM' in test_results else None,
            }

            # Remove None values
            results = {k: v for k, v in results.items() if v is not None}

            logger.info(f"Test results: {results}")
            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

@Step.register("microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing")
class EvaluateDependencyParsing(Step):
    """
    A Tango step for evaluating dependency parsing models using DiaParser.

    This step evaluates a dependency parsing model on test data and returns
    evaluation metrics including UAS, LAS, and loss.
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        model_path: str,
        test_data_path: str,
        save_path: str = "",
        dev_data_path: str = "",
        train_data_path: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate a dependency parsing model on test data.

        Args:
            model_path: Path to model or name of pretrained model
                       (e.g., 'en_ewt-electra', 'biaffine-dep-en')
            test_data_path: Path to test data in CoNLL-U format
            save_predictions: Whether to save model predictions to file
            predictions_output: Path to save predictions
                              (default: <test_data_stem>_predictions.conllu)
            save_path: Path to save the trained model
            save_results_json: Optional path to save results as JSON

        Returns:
            Dictionary containing evaluation metrics (loss, UAS, LAS, UCM, LCM)

        Note:
            Device is automatically selected - GPU if available, otherwise CPU
        """
        # Apply PyTorch 2.6+ compatibility patch for diaparser
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
        self.logger.info("Starting dependency parsing evaluation")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Test data: {test_data_path}")

        # Initialize evaluator (device is auto-detected)
        evaluator = DependencyParsingEvaluator(model_path=model_path, save_path=save_path, train_data_path=train_data_path, dev_data_path=dev_data_path,test_data_path=test_data_path)
        # Train the model  & Test the model
        results = evaluator.train(save_path=save_path, model_path=model_path, train_data_path=train_data_path, dev_data_path=dev_data_path, test_data_path=test_data_path)

        self.logger.info(f"Best test results: {results}")
        return results

from html import parser
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
    
    def train(self, save_path: str, model_path: str, train_data_path: str, dev_data_path: str, test_data_path: str, predictions_output: Optional[str] = None, results_json: Optional[str] = None) -> Dict[str, Any]:
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
            # Monkey-patch transformers tokenizer for compatibility with diaparser
            # diaparser expects tokenizer.max_len but newer transformers use model_max_length
            from transformers import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, 'max_len'):
                PreTrainedTokenizerBase.max_len = property(lambda self: self.model_max_length)

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
                feat='bert',  # Use BERT features
                max_len=512,  # Maximum sequence length for tokenization
                embed=None,  # No additional word embeddings when using BERT
                partial=False,  # Trees are fully annotated (not partial)
                tree=True,  # Enforce tree constraints during parsing
                proj=False,  # Don't enforce projectivity (allows non-projective trees)
                punct=False,  # Don't ignore punctuation during evaluation
                build=True,  # Force rebuild the model
            )

            # Step 2: Train the parser
            # train() doesn't return results, it only prints them
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

            # Step 3: Load the trained model and evaluate on test set to get metrics
            logger.info("Loading trained model for evaluation")
            parser = BiaffineDependencyParser.load(f"{save_path}/model", device=self.device, weights_only=False)

            # Evaluate on test set
            logger.info("Evaluating on test set")
            test_loss, test_metrics = parser.evaluate(
                data=test_data_path,
                batch_size=5000,
                partial=False,
                tree=True,
                proj=False,
                punct=False,
            )

            # Extract metrics from evaluation results
            # Log the test_metrics object to debug what's available
            logger.info(f"Test metrics object: {test_metrics}")
            logger.info(f"Test metrics type: {type(test_metrics)}")
            logger.info(f"Test metrics dir: {dir(test_metrics)}")

            results = {
                'loss': float(test_loss),
            }

            # Try to extract metrics - they might be attributes or dict items
            for metric_name in ['UAS', 'LAS', 'UCM', 'LCM']:
                try:
                    if hasattr(test_metrics, metric_name):
                        value = getattr(test_metrics, metric_name)
                        results[metric_name] = float(value)
                        logger.info(f"Extracted {metric_name}: {value}")
                    elif isinstance(test_metrics, dict) and metric_name in test_metrics:
                        value = test_metrics[metric_name]
                        results[metric_name] = float(value)
                        logger.info(f"Extracted {metric_name} from dict: {value}")
                except Exception as e:
                    logger.warning(f"Could not extract {metric_name}: {e}")

            logger.info(f"Best test results from training: {results}")

            # Save predictions if output path is provided
            if predictions_output:
                logger.info(f"Saving predictions to: {predictions_output}")
                # Create output directory if it doesn't exist
                Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
                # Predict and save to file
                parser.predict(data=test_data_path, pred=predictions_output, batch_size=5000, text=None)
                logger.info(f"Predictions saved to {predictions_output}")

            # Save results to JSON if path is provided
            if results_json:
                logger.info(f"Saving results to: {results_json}")
                # Create output directory if it doesn't exist
                Path(results_json).parent.mkdir(parents=True, exist_ok=True)
                # Save results as JSON
                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {results_json}")

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
        predictions_output: Optional[str] = None,
        results_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a dependency parsing model on test data.

        Args:
            model_path: Path to model or name of pretrained model
                       (e.g., 'en_ewt-electra', 'biaffine-dep-en')
            test_data_path: Path to test data in CoNLL-U format
            save_path: Path to save the trained model
            dev_data_path: Path to development data in CoNLL-U format
            train_data_path: Path to training data in CoNLL-U format
            predictions_output: Optional path to save predictions in CoNLL-U format
            results_json: Optional path to save results as JSON

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
        results = evaluator.train(save_path=save_path, model_path=model_path, train_data_path=train_data_path, dev_data_path=dev_data_path, test_data_path=test_data_path, predictions_output=predictions_output, results_json=results_json)

        self.logger.info(f"Best test results: {results}")
        return results

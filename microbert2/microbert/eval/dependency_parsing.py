import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import subprocess
import re
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
        self.tetst_data_path = test_data_path
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

        # Use wrapper module that adds safe globals for PyTorch 2.6+ compatibility
        command = [
            "python", "-m", "microbert2.microbert.eval.diaparser_wrapper",
            "train",
            "-b",
            "-d", "0",
            "-p", f"{save_path}/model",
            "-f", "bert",
            "--bert", model_path,
            "--train", train_data_path,
            "--dev", dev_data_path,
            "--test", test_data_path
        ]
        logger.info(f"Training model with command: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Save training logs
            log_file = f"{save_path}/training.log"
            with open(log_file, 'w') as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            logger.info(f"Training logs saved to: {log_file}")
            logger.info("Model trained and saved successfully")

            # Parse best test scores from output
            results = self._parse_test_scores(result.stderr + result.stdout)
            return results

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

    def _parse_test_scores(self, output: str) -> Dict[str, Any]:
        """Parse test scores from DiaParser training output."""
        results = {}


        # Look for test metrics
        test_pattern = r'test.*?(?:UAS|LAS).*'
        test_matches = re.findall(test_pattern, output, re.IGNORECASE)

        if test_matches:
            # Get the last test result (best model evaluation)
            last_test = test_matches[-1]

            # Extract individual metrics
            uas_match = re.search(r'UAS[:\s]+(\d+\.?\d*)', last_test)
            las_match = re.search(r'LAS[:\s]+(\d+\.?\d*)', last_test)
            ucm_match = re.search(r'UCM[:\s]+(\d+\.?\d*)', last_test)
            lcm_match = re.search(r'LCM[:\s]+(\d+\.?\d*)', last_test)
            loss_match = re.search(r'Loss[:\s]+(\d+\.?\d*)', last_test)

            if uas_match:
                results['UAS'] = float(uas_match.group(1))
            if las_match:
                results['LAS'] = float(las_match.group(1))
            if ucm_match:
                results['UCM'] = float(ucm_match.group(1))
            if lcm_match:
                results['LCM'] = float(lcm_match.group(1))
            if loss_match:
                results['loss'] = float(loss_match.group(1))

        logger.info(f"Parsed test results: {results}")
        return results
                
    
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
        self.logger.info("Starting dependency parsing evaluation")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Test data: {test_data_path}")

        # Initialize evaluator (device is auto-detected)
        evaluator = DependencyParsingEvaluator(model_path=model_path, save_path=save_path, train_data_path=train_data_path, dev_data_path=dev_data_path,test_data_path=test_data_path)
        # Train the model  & Test the model
        results = evaluator.train(save_path=save_path, model_path=model_path, train_data_path=train_data_path, dev_data_path=dev_data_path, test_data_path=test_data_path)

        self.logger.info(f"Best test results: {results}")
        return results

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import subprocess
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
    
    def train(self, save_path: str, model_path: str, train_data_path: str, dev_data_path: str, test_data_path: str) -> None:
        """Placeholder for train method."""
        # Create save directory if it doesn't exist (must be done before training)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        command = [
        "python", "-m", "diaparser.cmds.biaffine_dependency",
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
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
                
    
        
    def evaluate(
        self,
        test_data_path: str,
        save_predictions: bool = False,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_data_path: Path to test data in CoNLL-U format
            save_predictions: Whether to save predictions to file
            output_path: Path to save predictions (if save_predictions=True)

        Returns:
            Dictionary containing evaluation metrics:
                - loss: Evaluation loss
                - UAS: Unlabeled Attachment Score
                - LAS: Labeled Attachment Score
                - UCM: Unlabeled Correct Matches percentage
                - LCM: Labeled Correct Matches percentage
        """
        if self.parser is None:
            self.load_model()

        test_path = Path(test_data_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")

        logger.info(f"Evaluating on: {test_data_path}")

        try:
            # Evaluate on test data
            loss, metric = self.parser.evaluate(str(test_path), verbose=True)

            # Extract metrics
            results = {
                "loss": float(loss),
                "UAS": float(metric.uas),
                "LAS": float(metric.las),
                "UCM": float(metric.ucm),
                "LCM": float(metric.lcm),
            }

            logger.info("Evaluation Results:")
            logger.info(f"  Loss: {results['loss']:.4f}")
            logger.info(f"  UAS (Unlabeled Attachment Score): {results['UAS']:.2f}%")
            logger.info(f"  LAS (Labeled Attachment Score): {results['LAS']:.2f}%")
            logger.info(f"  UCM (Unlabeled Correct Matches): {results['UCM']:.2f}%")
            logger.info(f"  LCM (Labeled Correct Matches): {results['LCM']:.2f}%")

            # Save predictions if requested
            if save_predictions:
                if output_path is None:
                    output_path = str(
                        test_path.parent / f"{test_path.stem}_predictions.conllu"
                    )

                logger.info(f"Generating predictions and saving to: {output_path}")
                dataset = self.parser.predict(str(test_path), verbose=True)
                dataset.save(output_path)
                logger.info(f"Predictions saved to: {output_path}")
                results["predictions_path"] = output_path

            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


@Step.register("microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing")
class EvaluateDependencyParsing(Step):
    """
    A Tango step for evaluating dependency parsing models using DiaParser.

    This step evaluates a dependency parsing model on test data and returns
    evaluation metrics including UAS, LAS, and loss.

    Note:
        Device is automatically detected - GPU (cuda:0) if available, otherwise CPU

    Example config (.jsonnet):
        {
            type: "microbert2.microbert.eval.dependency_parsing::evaluate_dependency_parsing",
            model_path: "en_ewt-electra",
            test_data_path: "../slate/mlt/mt_mudt-ud-test.conllu",
            save_predictions: true,
            save_path: "./models/dep_parser"
        }
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        model_path: str,
        test_data_path: str,
        save_predictions: bool = False,
        predictions_output: Optional[str] = None,
        save_path: str = "",
        dev_data_path: str = "",
        train_data_path: str = "",
        save_results_json: Optional[str] = None,
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

        # Import diaparser classes for torch safe loading
        try:
            from diaparser.utils import Config, BertField
            from diaparser.parsers import Parser, BiaffineDependencyParser
            torch.serialization.add_safe_globals([Config, BertField, Parser, BiaffineDependencyParser])
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Could not import diaparser classes for safe globals: {e}")
            # Fall back to allowing all pickle loads (less secure but necessary if structure changed)
            pass
        # Initialize evaluator (device is auto-detected)
        evaluator = DependencyParsingEvaluator(model_path=model_path, save_path=save_path, train_data_path=train_data_path, dev_data_path=dev_data_path,test_data_path=test_data_path)
        # Train the model  & Test the model
        evaluator.train(save_path=save_path, model_path=model_path, train_data_path=train_data_path, dev_data_path=dev_data_path, test_data_path=test_data_path)
        # Run evaluation
        #results = evaluator.evaluate(
        #    test_data_path=test_data_path,
        #    save_predictions=save_predictions,
        #    output_path=predictions_output,
        #)

        # Save results to JSON if requested
        #if save_results_json:
        #    with open(save_results_json, "w") as f:
        #        json.dump(results, f, indent=2)
        #    self.logger.info(f"Results saved to: {save_results_json}")

        # Print summary
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DEPENDENCY PARSING EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Test Data: {test_data_path}")
        self.logger.info(f"Device: {evaluator.device}")
        self.logger.info("-" * 60)
        self.logger.info(f"Loss: {results['loss']:.4f}")
        self.logger.info(f"UAS (Unlabeled Attachment Score): {results['UAS']:.2f}%")
        self.logger.info(f"LAS (Labeled Attachment Score): {results['LAS']:.2f}%")
        self.logger.info(f"UCM (Unlabeled Correct Matches): {results['UCM']:.2f}%")
        self.logger.info(f"LCM (Labeled Correct Matches): {results['LCM']:.2f}%")
        self.logger.info("=" * 60)

        return results
        """
        return {}

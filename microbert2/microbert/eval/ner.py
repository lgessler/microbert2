import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from tango import Step
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def convert_bilu_to_bio(label: str) -> str:
    """
    Convert BILU encoding to BIO encoding.

    BILU uses:
    - B-X: Begin (first token of multi-token entity)
    - I-X: Inside (middle token of multi-token entity)
    - L-X: Last (final token of multi-token entity)
    - U-X: Unit (single-token entity)
    - O: Outside (not an entity)

    BIO uses:
    - B-X: Begin (first token of entity)
    - I-X: Inside (continuation of entity)
    - O: Outside (not an entity)

    Conversion:
    - L-X → I-X (Last becomes Inside)
    - U-X → B-X (Unit becomes Begin)
    - B-X, I-X, O remain unchanged

    Args:
        label: BILU label (e.g., 'U-LOC', 'L-ORG', 'B-PER', 'I-PER', 'O')

    Returns:
        BIO label (e.g., 'B-LOC', 'I-ORG', 'B-PER', 'I-PER', 'O')
    """
    if label == 'O':
        return 'O'

    if label.startswith('U-'):
        # Unit becomes Begin
        return 'B-' + label[2:]
    elif label.startswith('L-'):
        # Last becomes Inside
        return 'I-' + label[2:]
    else:
        # B- and I- remain unchanged
        return label


def load_bio_file(file_path: str) -> List[Tuple[List[str], List[str]]]:
    """
    Load a .bio file and parse it into sentences with tokens and labels.

    File format:
    - Each line has 4 space-separated columns: token col2 col3 label
    - Blank lines separate sentences
    - We ignore col2 and col3, only use token and label

    Args:
        file_path: Path to the .bio file

    Returns:
        List of (tokens, labels) tuples, where each tuple represents a sentence
    """
    sentences = []
    current_tokens = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Blank line indicates sentence boundary
            if not line:
                if current_tokens:
                    sentences.append((current_tokens, current_labels))
                    current_tokens = []
                    current_labels = []
                continue

            # Parse the line: token col2 col3 label
            parts = line.split()
            if len(parts) >= 4:
                token = parts[0]
                label = parts[-1]  # Last column is the label

                # Convert BILU to BIO
                bio_label = convert_bilu_to_bio(label)

                current_tokens.append(token)
                current_labels.append(bio_label)

        # Don't forget the last sentence if file doesn't end with blank line
        if current_tokens:
            sentences.append((current_tokens, current_labels))

    logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences


def create_label_mapping(train_data: List[Tuple[List[str], List[str]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID and ID to label mappings from training data.

    Args:
        train_data: List of (tokens, labels) tuples

    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    unique_labels = set()
    for _, labels in train_data:
        unique_labels.update(labels)

    # Sort labels for consistency (O first, then alphabetically)
    sorted_labels = sorted(unique_labels, key=lambda x: (x != 'O', x))

    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    logger.info(f"Found {len(label2id)} unique labels: {sorted_labels}")
    return label2id, id2label


def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
) -> Dict[str, List]:
    """
    Tokenize the tokens and align the labels with subword tokens.

    When a word is split into multiple subword tokens, we assign:
    - The original label to the first subword
    - -100 to the remaining subwords (ignored in loss computation)

    Args:
        examples: Dictionary with 'tokens' and 'ner_tags' keys
        tokenizer: HuggingFace tokenizer
        label2id: Mapping from label strings to IDs

    Returns:
        Tokenized examples with aligned labels
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )

    labels = []
    for i, label_list in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id that is None
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_list[word_idx]])
            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics(eval_pred, id2label: Dict[int, str]):
    """
    Compute NER metrics using seqeval.

    Note: seqeval properly ignores "O" (Outside) tokens when computing precision,
    recall, and F1 scores. These metrics are computed only over entity spans
    (B-X, I-X tags), which is the standard practice in NER evaluation to avoid
    inflating scores since most tokens are labeled "O".

    Accuracy, however, includes all tokens (including "O").

    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        id2label: Mapping from label IDs to label strings

    Returns:
        Dictionary of metrics (precision, recall, f1, accuracy)
    """
    try:
        from seqeval.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError:
        raise ImportError(
            "seqeval is not installed. Install it with: pip install seqeval"
        )

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(id2label[label_id])
                true_pred.append(id2label[pred_id])
        true_labels.append(true_label)
        true_predictions.append(true_pred)

    # Compute metrics
    results = {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
        'accuracy': accuracy_score(true_labels, true_predictions),
    }

    # Log detailed classification report
    logger.info("\nClassification Report:\n" + classification_report(true_labels, true_predictions))

    return results


@Step.register("microbert2.microbert.eval.ner::evaluate_ner")
class EvaluateNER(Step):
    """
    A Tango step for evaluating NER models using HuggingFace Transformers.

    This step fine-tunes a pretrained model on NER data and evaluates it,
    returning metrics including precision, recall, F1, and accuracy.
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        model_path: str,
        train_data_path: str,
        dev_data_path: str,
        test_data_path: str,
        save_path: str,
        predictions_output: Optional[str] = None,
        results_json: Optional[str] = None,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        trained_model: Optional[Any] = None,  # Optional dependency for Tango workflow
    ) -> Dict[str, Any]:
        """
        Fine-tune and evaluate a NER model.

        Args:
            model_path: Path to pretrained model or HuggingFace model ID
            train_data_path: Path to training data in .bio format
            dev_data_path: Path to development data in .bio format
            test_data_path: Path to test data in .bio format
            save_path: Path to save the fine-tuned model
            predictions_output: Optional path to save predictions in .bio format
            results_json: Optional path to save results as JSON
            batch_size: Training batch size (default: 16)
            learning_rate: Learning rate (default: 5e-5)
            num_epochs: Number of training epochs (default: 3)
            trained_model: Optional reference to trained model step for Tango dependency management

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting NER evaluation")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Train data: {train_data_path}")
        self.logger.info(f"Dev data: {dev_data_path}")
        self.logger.info(f"Test data: {test_data_path}")

        # Load data
        self.logger.info("Loading data files...")
        train_data = load_bio_file(train_data_path)
        dev_data = load_bio_file(dev_data_path)
        test_data = load_bio_file(test_data_path)

        # Create label mappings from training data
        label2id, id2label = create_label_mapping(train_data)
        num_labels = len(label2id)
        self.logger.info(f"Number of labels: {num_labels}")

        # Load tokenizer
        self.logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Convert data to HuggingFace Dataset format
        def prepare_dataset(data: List[Tuple[List[str], List[str]]]) -> Dataset:
            tokens_list = [tokens for tokens, _ in data]
            labels_list = [labels for _, labels in data]
            return Dataset.from_dict({
                'tokens': tokens_list,
                'ner_tags': labels_list,
            })

        train_dataset = prepare_dataset(train_data)
        dev_dataset = prepare_dataset(dev_data)
        test_dataset = prepare_dataset(test_data)

        # Tokenize datasets
        self.logger.info("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        tokenized_dev = dev_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True,
            remove_columns=dev_dataset.column_names,
        )
        tokenized_test = test_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True,
            remove_columns=test_dataset.column_names,
        )

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_path,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{save_path}/logs",
            logging_steps=10,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x: compute_metrics(x, id2label),
        )

        # Train
        self.logger.info("Starting training...")
        trainer.train()

        # Save the best model
        self.logger.info(f"Saving model to {save_path}")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

        # Evaluate on test set
        self.logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_test)

        # Extract metrics
        results = {
            'loss': test_results.get('eval_loss', 0.0),
            'precision': test_results.get('eval_precision', 0.0),
            'recall': test_results.get('eval_recall', 0.0),
            'f1': test_results.get('eval_f1', 0.0),
            'accuracy': test_results.get('eval_accuracy', 0.0),
        }

        self.logger.info(f"Test results: {results}")

        # Save predictions if requested
        if predictions_output:
            self.logger.info(f"Saving predictions to {predictions_output}")
            Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)

            # Get predictions
            predictions = trainer.predict(tokenized_test)
            pred_labels = np.argmax(predictions.predictions, axis=2)

            # Write predictions in .bio format
            with open(predictions_output, 'w', encoding='utf-8') as f:
                sentence_idx = 0
                for tokens, _ in test_data:
                    # Get tokenized version for this sentence
                    tokenized = tokenizer(
                        tokens,
                        is_split_into_words=True,
                        truncation=True,
                        max_length=512,
                    )
                    word_ids = tokenized.word_ids()

                    # Get predictions for this sentence
                    pred_ids = pred_labels[sentence_idx]

                    # Align predictions back to original tokens
                    token_predictions = []
                    previous_word_idx = None
                    for word_idx, pred_id in zip(word_ids, pred_ids):
                        if word_idx is not None and word_idx != previous_word_idx:
                            token_predictions.append(id2label[pred_id])
                            previous_word_idx = word_idx

                    # Write to file
                    for token, pred_label in zip(tokens, token_predictions):
                        f.write(f"{token} O O {pred_label}\n")
                    f.write("\n")

                    sentence_idx += 1

            self.logger.info(f"Predictions saved to {predictions_output}")

        # Save results to JSON if requested
        if results_json:
            self.logger.info(f"Saving results to {results_json}")
            Path(results_json).parent.mkdir(parents=True, exist_ok=True)
            with open(results_json, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {results_json}")

        return results

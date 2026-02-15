"""Evaluation metrics for Red Flag Law AI."""

from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)


class MetricsCalculator:
    """Calculator for evaluation metrics."""

    @staticmethod
    def compute_classification_metrics(
        predictions: List[bool],
        references: List[bool]
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: List of predicted labels (True for unfair, False for fair)
            references: List of ground truth labels

        Returns:
            Dictionary with accuracy, F1, precision, and recall
        """
        return {
            "accuracy": accuracy_score(references, predictions),
            "f1_macro": f1_score(references, predictions, average="macro"),
            "precision_macro": precision_score(references, predictions, average="macro"),
            "recall_macro": recall_score(references, predictions, average="macro"),
        }

    @staticmethod
    def compute_multiclass_metrics(
        predictions: List[str],
        references: List[str],
        labels: List[str]
    ) -> Dict[str, float]:
        """
        Compute metrics for multi-class classification.

        Args:
            predictions: List of predicted category labels
            references: List of ground truth category labels
            labels: List of all possible category labels

        Returns:
            Dictionary with various metrics
        """
        return {
            "accuracy": accuracy_score(references, predictions),
            "f1_macro": f1_score(references, predictions, average="macro", labels=labels, zero_division=0),
            "f1_weighted": f1_score(references, predictions, average="weighted", labels=labels, zero_division=0),
            "precision_macro": precision_score(references, predictions, average="macro", labels=labels, zero_division=0),
            "recall_macro": recall_score(references, predictions, average="macro", labels=labels, zero_division=0),
        }

    @staticmethod
    def compute_bertscore(
        predictions: List[str],
        references: List[str],
        model_name: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, float]:
        """
        Compute BERTScore for explanation quality.

        Args:
            predictions: List of generated explanations
            references: List of reference explanations
            model_name: Model to use for BERTScore

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # TODO: Implement BERTScore computation
        # from bert_score import score
        # P, R, F1 = score(predictions, references, model_type=model_name)
        raise NotImplementedError("BERTScore computation not yet implemented")

    @staticmethod
    def print_classification_report(
        predictions: List[str],
        references: List[str],
        labels: List[str]
    ):
        """
        Print detailed classification report.

        Args:
            predictions: List of predicted labels
            references: List of ground truth labels
            labels: List of all possible labels
        """
        print(classification_report(references, predictions, labels=labels, zero_division=0))

"""Evaluation metrics for Red Flag Law AI."""

from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.preprocessing import MultiLabelBinarizer
from bert_score import score


class MetricsCalculator:
    """Calculator for evaluation metrics."""

    @staticmethod
    def compute_multiclass_metrics(
        predictions: List[List[str]],
        references: List[List[str]],
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
        # transform lists of categories into binary format for each label
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(references)
        y_pred = mlb.transform(predictions)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
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
        P, R, F1 = score(predictions, references, model_type=model_name, lang="en", verbose=False)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    
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

    @staticmethod
    def compute_llm_judge_metrics(
        predictions: List[str],
        references: List[str],
        judge_fn
    ) -> Dict[str, float]:
        """
        Compute metrics using an LLM as a judge.

        Args:
            predictions: List of predicted explanations
            references: List of reference explanations
            judge_fn: Function that takes (pred, ref) and returns a dict with 'score' key

        Returns:
            Dictionary with mean score and pass rate
        """
        
        scores = []
        for pred, ref in zip(predictions, references):
            result = judge_fn(pred, ref)
            scores.append(result.get("score", 0))

        mean_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = sum(scores) / len(scores) if scores else 0.0

        return {
            "llm_judge_mean": mean_score,
            "llm_judge_pass_rate": pass_rate,
        }

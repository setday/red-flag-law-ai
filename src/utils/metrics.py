"""Evaluation metrics for Red Flag Law AI."""

from typing import Callable, Dict, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from bert_score import score


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
        P, R, F1 = score(predictions, references, model_type=model_name, lang="en", verbose=False)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }

    @staticmethod
    def compute_llm_judge_metrics(
        clauses: List[str],
        predictions: List[str],
        references: List[str],
        judge_fn: Callable[[str, str, str], Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Compute LLM-as-a-judge metrics for explanation quality.

        Args:
            clauses: Original clause texts
            predictions: Generated explanations
            references: Reference explanations
            judge_fn: Callable that returns a dict with at least key "score" in range 1..5

        Returns:
            Dictionary with aggregate judge statistics
        """
        if not (len(clauses) == len(predictions) == len(references)):
            raise ValueError("clauses, predictions, and references must have the same length")

        if not clauses:
            return {
                "llm_judge_mean": 0.0,
                "llm_judge_pass_rate": 0.0,
                "llm_judge_count": 0.0,
            }

        scores: List[float] = []
        for clause, prediction, reference in zip(clauses, predictions, references):
            result = judge_fn(clause, prediction, reference)
            raw_score = result.get("score", "0")
            try:
                score_value = float(raw_score)
            except (TypeError, ValueError):
                score_value = 0.0
            score_value = min(5.0, max(1.0, score_value)) if score_value > 0 else 0.0
            scores.append(score_value)

        valid_scores = [s for s in scores if s > 0]
        if not valid_scores:
            return {
                "llm_judge_mean": 0.0,
                "llm_judge_pass_rate": 0.0,
                "llm_judge_count": 0.0,
            }

        mean_score = sum(valid_scores) / len(valid_scores)
        pass_rate = sum(1 for s in valid_scores if s >= 4.0) / len(valid_scores)
        return {
            "llm_judge_mean": mean_score,
            "llm_judge_pass_rate": pass_rate,
            "llm_judge_count": float(len(valid_scores)),
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

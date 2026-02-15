"""Tests for metrics computation."""

import pytest
from src.utils.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_compute_classification_metrics(self):
        """Test classification metrics computation."""
        calc = MetricsCalculator()
        
        predictions = [True, False, True, True, False]
        references = [True, False, True, False, False]
        
        metrics = calc.compute_classification_metrics(predictions, references)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_compute_classification_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        calc = MetricsCalculator()
        
        predictions = [True, False, True, False]
        references = [True, False, True, False]
        
        metrics = calc.compute_classification_metrics(predictions, references)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_compute_multiclass_metrics(self):
        """Test multiclass metrics computation."""
        calc = MetricsCalculator()
        
        labels = ["A", "B", "C"]
        predictions = ["A", "B", "C", "A", "B"]
        references = ["A", "B", "C", "B", "B"]
        
        metrics = calc.compute_multiclass_metrics(predictions, references, labels)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert 0 <= metrics["accuracy"] <= 1

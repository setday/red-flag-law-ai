"""Tests for inference functionality."""

import pytest
from src.inference.detector import RedFlagDetector


class TestRedFlagDetector:
    """Tests for RedFlagDetector class."""

    def test_initialization(self):
        """Test that detector can be initialized."""
        detector = RedFlagDetector()
        assert detector is not None
        assert detector.model_path is None
        assert detector.model is None

    def test_analyze_returns_dict(self):
        """Test that analyze returns a dictionary."""
        detector = RedFlagDetector()
        result = detector.analyze("Some legal text")
        
        assert isinstance(result, dict)
        assert "is_unfair" in result
        assert "category" in result
        assert "explanation" in result

    def test_batch_analyze(self):
        """Test batch analysis."""
        detector = RedFlagDetector()
        texts = [
            "First clause",
            "Second clause",
            "Third clause"
        ]
        
        results = detector.batch_analyze(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

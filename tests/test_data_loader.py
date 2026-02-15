"""Tests for data loading functionality."""

import pytest
from src.data.loader import UnfairToSDataLoader


class TestUnfairToSDataLoader:
    """Tests for UnfairToSDataLoader class."""

    def test_initialization(self):
        """Test that loader can be initialized."""
        loader = UnfairToSDataLoader()
        assert loader is not None
        assert loader.dataset is None

    def test_initialization_with_params(self):
        """Test initialization with parameters."""
        loader = UnfairToSDataLoader(
            cache_dir="./test_cache",
            max_samples=100
        )
        assert loader.cache_dir == "./test_cache"
        assert loader.max_samples == 100

    def test_format_instruction_fair_clause(self):
        """Test formatting instruction for a fair clause."""
        loader = UnfairToSDataLoader()
        text = "You may cancel your subscription at any time."
        labels = []
        
        result = loader.format_instruction(text, labels)
        
        assert "instruction" in result
        assert "response" in result
        assert text in result["instruction"]

    def test_format_instruction_unfair_clause(self):
        """Test formatting instruction for an unfair clause."""
        loader = UnfairToSDataLoader()
        text = "We may terminate your account without notice."
        labels = ["Unilateral Termination"]
        
        result = loader.format_instruction(text, labels)
        
        assert "instruction" in result
        assert "response" in result
        assert text in result["instruction"]
        assert "Unilateral Termination" in result["response"]

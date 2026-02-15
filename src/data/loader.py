"""Data loader for Unfair-ToS dataset."""

from typing import Dict, List, Optional
from datasets import load_dataset, Dataset


class UnfairToSDataLoader:
    """Loader for the Unfair Terms of Service dataset from LexGLUE."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
            max_samples: Maximum number of samples to load (None for all)
        """
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.dataset = None

    def load(self) -> Dict[str, Dataset]:
        """
        Load the Unfair-ToS dataset.

        Returns:
            Dictionary with 'train', 'validation', and 'test' splits
        """
        # TODO: Implement dataset loading from HuggingFace
        # dataset = load_dataset("lex_glue", "unfair_tos", cache_dir=self.cache_dir)
        raise NotImplementedError("Dataset loading not yet implemented")

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset for instruction tuning.

        Args:
            dataset: Raw dataset to preprocess

        Returns:
            Preprocessed dataset
        """
        # TODO: Implement preprocessing logic
        raise NotImplementedError("Preprocessing not yet implemented")

    def format_instruction(self, text: str, labels: List[str]) -> Dict[str, str]:
        """
        Format a sample as an instruction-tuning example.

        Args:
            text: The legal clause text
            labels: List of unfair categories (empty if fair)

        Returns:
            Dictionary with 'instruction' and 'response' fields
        """
        instruction = (
            "Analyze the following legal clause and identify if it contains unfair terms:\n"
            f'"{text}"'
        )

        if not labels:
            response = {
                "is_unfair": False,
                "category": None,
                "explanation": "This clause appears to be fair and reasonable."
            }
        else:
            # TODO: Generate proper explanations
            category = labels[0] if labels else None
            response = {
                "is_unfair": True,
                "category": category,
                "explanation": f"This clause contains unfair terms related to {category}."
            }

        return {
            "instruction": instruction,
            "response": str(response)
        }

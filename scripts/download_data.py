#!/usr/bin/env python3
"""Script to download the Unfair-ToS dataset."""

import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download Unfair-ToS dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/unfair-tos",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    
    args = parser.parse_args()
    
    print("Downloading Unfair-ToS dataset from LexGLUE...")
    dataset = load_dataset("lex_glue", "unfair_tos", cache_dir=args.cache_dir)
    
    print(f"\nDataset statistics:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example script demonstrating basic usage of Red Flag Law AI."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.detector import RedFlagDetector


def main():
    """Run example analysis on sample legal clauses."""
    
    # Sample legal clauses
    clauses = [
        "We reserve the right to terminate your account at any time without notice or reason.",
        "You may cancel your subscription at any time through your account settings.",
        "The company is not liable for any damages arising from use of the service.",
        "We will notify you 30 days in advance of any changes to these terms.",
        "By using our service, you agree to binding arbitration and waive your right to sue.",
        "Your personal data may be shared with third parties for marketing purposes.",
        "We may modify or discontinue the service at any time without liability.",
        "You retain all rights to content you upload to our platform."
    ]
    
    print("=" * 80)
    print("Red Flag Law AI - Example Analysis")
    print("=" * 80)
    print()
    
    # Initialize detector
    detector = RedFlagDetector()
    
    # Analyze each clause
    for i, clause in enumerate(clauses, 1):
        print(f"Clause {i}:")
        print(f"  Text: \"{clause}\"")
        
        result = detector.analyze(clause)
        
        print(f"  Is Unfair: {result['is_unfair']}")
        if result['category']:
            print(f"  Category: {result['category']}")
        print(f"  Explanation: {result['explanation']}")
        print()
    
    print("=" * 80)
    print("Note: This is a placeholder implementation.")
    print("Train the model to get real predictions!")
    print("=" * 80)


if __name__ == "__main__":
    main()

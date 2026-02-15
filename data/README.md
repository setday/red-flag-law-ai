# Data Directory

This directory contains datasets used for training and evaluation.

## Structure

```
data/
├── raw/              # Raw, unprocessed datasets
│   └── unfair-tos/   # Unfair Terms of Service dataset
├── processed/        # Preprocessed datasets ready for training
│   └── unfair-tos/   # Processed versions
└── README.md         # This file
```

## Datasets

### Unfair-ToS (LexGLUE)

**Source**: [HuggingFace - lex_glue](https://huggingface.co/datasets/lex_glue)

**Description**: 
- 9,414 sentences from real user agreements
- Annotated by professional lawyers
- Multiple categories of unfair clauses

**Categories**:
- Arbitration
- Unilateral Change
- Content Removal
- Unilateral Termination
- Contract by Using
- Choice of Law
- Limitation of Liability
- Jurisdiction

### Downloading Data

To download the dataset:

```python
from datasets import load_dataset

# Load the Unfair-ToS dataset
dataset = load_dataset("lex_glue", "unfair_tos")

# Save to raw directory
dataset.save_to_disk("data/raw/unfair-tos")
```

Or use the provided script:

```bash
python scripts/download_data.py
```

## Data Format

### Input Format (Raw)

Each sample contains:
- `text`: The clause/sentence text
- `labels`: List of unfair categories (empty if fair)

Example:
```json
{
  "text": "We reserve the right to terminate your account at any time.",
  "labels": ["Unilateral Termination"]
}
```

### Output Format (Processed)

For instruction tuning, data is formatted as:

**Input**:
```
Analyze the following legal clause and identify if it contains unfair terms:
"We reserve the right to terminate your account at any time."
```

**Output**:
```json
{
  "is_unfair": true,
  "category": "Unilateral Termination",
  "explanation": "The company can close your account without warning or justification, which removes your control and access without due process."
}
```

## Data Preprocessing

Preprocessing steps include:
1. Loading raw data from HuggingFace
2. Converting to instruction-tuning format
3. Generating explanations (using templates or LLM)
4. Splitting into train/validation/test sets
5. Tokenization and formatting for model input

See `src/data/preprocessing.py` for implementation.

## Notes

- Large datasets are excluded from version control (see .gitignore)
- Keep raw data in `raw/` subdirectory
- Store processed data in `processed/` subdirectory
- Document any custom datasets added to the project

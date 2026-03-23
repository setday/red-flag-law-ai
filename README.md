# Red Flag Law AI

An AI-powered system for automatic auditing of legal documents (EULA, Terms of Service, Privacy Policies, etc.) to identify unfair or potentially harmful clauses.

## Overview

This project aims to help users identify "red flags" in legal agreements by:
1. **Classifying** whether a clause contains unfair terms
2. **Categorizing** the type of unfair condition (if present)
3. **Explaining** in simple language why the clause is potentially harmful

### Who Benefits?

- **Regular Users**: No one wants to read pages of legal text during signup. This tool highlights only the truly dangerous parts (data selling, liability disclaimers, etc.)
- **Junior Lawyers/Freelancers**: Speeds up initial contract screening dramatically - validate 5 risks found by the model instead of reading hundreds of pages

### Business Value

- **Time Savings**: Automated first-pass screening of legal documents
- **Risk Reduction**: Avoid signing agreements that infringe on your rights

## Dataset

The project uses the **LexGLUE** (Legal General Language Understanding Evaluation) benchmark, specifically the **Unfair-ToS** subset.

- **Size**: 9,414 sentences from real user agreements (YouTube, Tinder, Airbnb, etc.)
- **Annotation**: Labeled by professional lawyers
- **Labels**: Each sentence is marked as either fair or with specific violation types:
  - Arbitration
  - Unilateral Change
  - Content Removal
  - Unilateral Termination
  - Contract by Using
  - Choice of Law
  - Limitation of Liability
  - Jurisdiction
  - And more...
- **Source**: Available on [HuggingFace](https://huggingface.co/datasets/lex_glue)

## Approach

### Task Formulation

**Input**: A fragment of legal text (e.g., a clause from Terms of Service)

**Output**: JSON format with three fields:
```json
{
  "is_unfair": true,
  "category": "Arbitration",
  "explanation": "The company denies your right to sue by forcing arbitration."
}
```

### Why LLM?

1. **Regular Expressions or TF-IDF/SVM**: Legal language is too complex and nuanced. The phrase "We are not responsible" can be written in thousands of different ways without using those exact words.

2. **BERT-based Models**: While they can classify risk presence reasonably well, they:
   - Cannot generate explanations (needed for user interpretation)
   - Struggle with long contexts
   - Miss dependencies between conditions

**Therefore**, an LLM approach is necessary because:
- The task requires understanding complex logic beyond simple sentence classification
- We need to generate human-readable explanations
- Context and clause interdependencies matter

### Model Training

The approach uses **Instruction Tuning** where:
- **Input**: A legal clause/sentence
- **Output**: Structured JSON with classification and explanation

## Metrics

### Automated Metrics

1. **Macro F1-Score** (Primary Metric)
   - Most important because classes are imbalanced
   - "Normal" clauses vastly outnumber "unfair" ones
   - A model predicting "all OK" would have high accuracy but zero utility
   - F1 reveals the model's true ability to catch rare "red flags"

2. **BERTScore / METEOR**
   - Evaluates explanation quality
   - Measures semantic similarity between generated and reference explanations

### Manual Metrics (For Product Validation)

1. **Success Rate / Recall**
   - Give the same agreement to both the model and a lawyer
   - Count how many dangerous clauses the lawyer found
   - Measure what percentage the model caught

2. **Comprehensibility**
   - Show users the model's explanation
   - Rate on a 5-point scale whether they understood the risk without reading the agreement

## Project Structure

```
red-flag-law-ai/
├── src/                    # Source code
│   ├── inference/          # Inference and evaluation
│   │   └── detector.py
│   ├── utils/              # Utility functions
│   │   ├── metrics.py
│   │   └── __init__.py
│   ├── demo.py
│   ├── evaluate.py
│   └── train.py
├── configs/                # Configuration files
│   ├── model_pipeline/
│   │   ├── llama_finetuned.yaml
│   │   └── llama.yaml
│   ├── demo.yaml
│   ├── evaluate.yaml
│   └── train.yaml
├── data/                   # Data directory (add to .gitignore for large files)
├── requirements.txt        # Python dependencies
├── pyproject.toml
├── LICENSE
├── SETUP.md
├── SOLUTION.md
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/setday/red-flag-law-ai.git
cd red-flag-law-ai

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
# Example usage (to be implemented)
from src.inference import RedFlagDetector

detector = RedFlagDetector()
result = detector.analyze(
    "We reserve the right to terminate your account at any time without notice or reason."
)

print(result)
# Output:
# {
#   "is_unfair": true,
#   "category": "Unilateral Termination",
#   "explanation": "The company can close your account without warning or justification."
# }
```

## Development

### Training a Model

```bash
python src/train.py --config configs/train.yaml
```

### Evaluation

```bash
python src/evaluate.py --config configs/evaluate.yaml
```

## Acknowledgments

- LexGLUE benchmark and Unfair-ToS dataset creators
- The legal tech and NLP research communities

## Citation

If you use this project in your research, please cite:

```bibtex
@software{red_flag_law_ai,
  author = {Aleksandr Serkov},
  title = {Red Flag Law AI: Automated Legal Document Auditing},
  year = {2026},
  url = {https://github.com/setday/red-flag-law-ai}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
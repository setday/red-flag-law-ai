# Quick Setup Guide

This guide will help you get started with Red Flag Law AI in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for training

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/setday/red-flag-law-ai.git
cd red-flag-law-ai
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n red-flag python=3.8
conda activate red-flag
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -e ".[dev]"
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings (HuggingFace token, etc.)
```

### 5. Download Dataset (Optional)

```bash
python scripts/download_data.py
```

## Quick Test

Run the demo script to verify installation:

```bash
python examples/demo.py
```

You should see sample legal clauses being analyzed (with placeholder results until you train the model).

## Next Steps

1. **Explore the Data**: Check out `notebooks/quickstart.ipynb`
2. **Train a Model**: Run `python scripts/train.py --config configs/default.yaml`
3. **Evaluate**: Run `python scripts/evaluate.py --model-path <path>`

## Troubleshooting

### ImportError: No module named 'src'

Make sure you're running scripts from the project root directory, or install the package:
```bash
pip install -e .
```

### CUDA out of memory

Reduce batch size in `configs/default.yaml`:
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

### HuggingFace dataset download issues

Set a cache directory:
```bash
export HUGGINGFACE_HUB_CACHE=/path/to/cache
```

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for bugs or questions

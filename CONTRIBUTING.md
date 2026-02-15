# Contributing to Red Flag Law AI

Thank you for considering contributing to Red Flag Law AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Use cases and benefits
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** if applicable
4. **Update documentation** as needed
5. **Ensure tests pass** by running `pytest tests/`
6. **Submit a pull request** with a clear description of changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/red-flag-law-ai.git
cd red-flag-law-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (including dev dependencies)
pip install -r requirements.txt
```

## Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Formatting

Use `black` for code formatting:
```bash
black src/ tests/
```

### Linting

Use `flake8` for linting:
```bash
flake8 src/ tests/
```

### Type Checking

Use `mypy` for type checking:
```bash
mypy src/
```

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Commit Messages

Write clear, concise commit messages:
- Use present tense ("Add feature" not "Added feature")
- Start with a capital letter
- Keep the first line under 50 characters
- Add detailed description if needed

Example:
```
Add unfair clause detection module

- Implement basic classifier
- Add support for multiple categories
- Include unit tests
```

## Project Structure Guidelines

- **src/data/**: Data loading, preprocessing, and dataset classes
- **src/models/**: Model architectures and definitions
- **src/training/**: Training loops, optimizers, and schedulers
- **src/inference/**: Inference engines and evaluation
- **src/utils/**: Helper functions and utilities
- **configs/**: Configuration files (YAML/JSON)
- **scripts/**: Standalone scripts for training, evaluation, etc.
- **notebooks/**: Jupyter notebooks for experiments and demos
- **tests/**: Unit and integration tests

## Documentation

- Update README.md if you change functionality
- Add docstrings to new functions/classes
- Update inline comments for complex logic
- Keep documentation clear and concise

## Questions?

Feel free to open an issue with the "question" label if you need clarification on anything.

Thank you for contributing!

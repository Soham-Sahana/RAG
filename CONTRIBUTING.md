# Contributing to College Knowledge Assistant

Thank you for your interest in contributing! Here's how you can help.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies** in a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for function signatures
- Write docstrings for public classes and functions
- Keep functions focused and reasonably sized

### Testing

- Add tests for new features in `tests/test_rag.py`
- Ensure all existing tests pass before submitting:
  ```bash
  pytest tests/test_rag.py -v
  ```
- Include test questions in `tests/test_questions.json` when applicable

### Commit Messages

Use clear, descriptive commit messages:
- `feat: add PDF table extraction support`
- `fix: handle empty document directories gracefully`
- `docs: update setup instructions for macOS`
- `test: add edge case for multi-language documents`

## Submitting Changes

1. Push your branch to your fork
2. Open a **Pull Request** against the `main` branch
3. Describe what your changes do and why
4. Reference any related issues

## Reporting Bugs

Open an [issue](../../issues) with:
- Steps to reproduce the bug
- Expected vs. actual behavior
- Python version and OS
- Relevant error messages or logs

## Feature Requests

Open an [issue](../../issues) describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Code of Conduct

Be respectful and constructive. We're all here to learn and build something useful.

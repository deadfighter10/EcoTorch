# Contributing to EcoTorch

Thank you for your interest in contributing to EcoTorch! We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, and bug reports.

## Setting Up a Development Environment

EcoTorch uses `uv` for dependency management. To set up your development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecotorch.git
   cd ecotorch
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

   Alternatively, you can use `pip`:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We follow the standard Python coding style guidelines. Please ensure your code is formatted correctly:

- Use **Black** for code formatting.
- Use **isort** for import sorting.

You can run these tools manually:
```bash
black src tests
isort src tests
```

## Running Tests

Before submitting a pull request, please make sure all tests pass:

```bash
pytest
```

## Submitting Pull Requests

1. Fork the repository and create a new branch for your changes.
2. Ensure your code follows the style guidelines and passes all tests.
3. Write clear and concise commit messages.
4. Submit a pull request with a detailed description of your changes.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Use the provided templates for bug reports and feature requests.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](code-of-conduct.md). By participating in this project, you agree to abide by its terms.

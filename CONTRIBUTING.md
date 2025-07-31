# Contributing to Connectome-GNN-Suite

Thank you for your interest in contributing to Connectome-GNN-Suite! This document outlines the process for contributing to this project.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/connectome-gnn-suite.git
   cd connectome-gnn-suite
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Before You Start

1. Create an issue describing the bug fix or feature
2. Discuss the approach with maintainers
3. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings following NumPy style
- Maintain test coverage above 80%

### Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=connectome_gnn
```

### Submitting Changes

1. Ensure all tests pass
2. Run pre-commit checks:
   ```bash
   pre-commit run --all-files
   ```

3. Commit your changes:
   ```bash
   git commit -m "Add feature: description"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request on GitHub

## Pull Request Guidelines

- Use a clear, descriptive title
- Reference related issues
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes

## Code Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address feedback promptly
4. Squash commits before merge

## Types of Contributions

### Bug Reports

Use the issue template and include:
- Python version and OS
- Reproduction steps
- Expected vs actual behavior
- Relevant code snippets

### Feature Requests

- Describe the use case
- Provide examples
- Consider backwards compatibility

### Documentation

- Fix typos and improve clarity
- Add examples and tutorials
- Update API documentation

## Community Guidelines

- Be respectful and inclusive
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Help others learn and contribute

## Questions?

- Open an issue for discussions
- Tag maintainers for urgent matters
- Check existing issues first

Thank you for contributing!
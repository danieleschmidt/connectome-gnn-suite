# Development Guide

This guide covers the development setup and workflows for Connectome-GNN-Suite.

## Quick Setup

1. **Clone and install**:
   ```bash
   git clone https://github.com/yourusername/connectome-gnn-suite.git
   cd connectome-gnn-suite
   pip install -e ".[dev]"
   ```

2. **Setup pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Project Structure

```
connectome-gnn-suite/
├── connectome_gnn/          # Main package
│   ├── __init__.py
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # GNN architectures
│   ├── tasks/              # Downstream tasks
│   ├── visualization/      # Plotting and visualization
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation
├── experiments/            # Research experiments
├── notebooks/              # Jupyter notebooks
└── scripts/                # CLI tools and scripts
```

## Development Workflow

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run all checks:
```bash
pre-commit run --all-files
```

### Testing

Run tests:
```bash
pytest                      # Run all tests
pytest tests/test_models/   # Run specific test directory
pytest -v                  # Verbose output
pytest --cov              # With coverage
```

### Type Checking

```bash
mypy connectome_gnn/
```

## Development Environment

### Python Version

- Minimum: Python 3.9
- Recommended: Python 3.10+

### GPU Setup

For GPU acceleration:
```bash
# Install PyTorch with CUDA support
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cu118
```

### Data Setup

Download test datasets:
```bash
python scripts/download_test_data.py
```

## Architecture Guidelines

### Code Organization

- **connectome_gnn/data/**: Data loaders, preprocessing pipelines
- **connectome_gnn/models/**: GNN layers and full architectures
- **connectome_gnn/tasks/**: Task-specific code (prediction, classification)
- **connectome_gnn/visualization/**: Plotting and analysis tools

### Design Principles

1. **Modularity**: Each component should be independently testable
2. **Extensibility**: Easy to add new models and tasks
3. **Performance**: Optimize for large-scale connectome data
4. **Reproducibility**: Deterministic results with seed setting

### Adding New Models

1. Create model class inheriting from `BaseConnectomeModel`
2. Add unit tests in `tests/test_models/`
3. Add example usage in docstring
4. Update documentation

Example:
```python
from connectome_gnn.models.base import BaseConnectomeModel

class MyCustomGNN(BaseConnectomeModel):
    def __init__(self, node_features: int, hidden_dim: int):
        super().__init__()
        # Implementation here
    
    def forward(self, data):
        # Forward pass implementation
        pass
```

## Performance Considerations

### Memory Optimization

- Use gradient checkpointing for large models
- Implement graph sampling for massive connectomes
- Profile memory usage regularly

### Computational Efficiency

- Leverage PyTorch Geometric's optimized operations
- Use appropriate data types (float32 vs float64)
- Consider mixed precision training

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def train_model(model, dataset, epochs: int = 100):
    """Train a connectome GNN model.
    
    Parameters
    ----------
    model : BaseConnectomeModel
        The model to train
    dataset : ConnectomeDataset
        Training dataset
    epochs : int, default=100
        Number of training epochs
        
    Returns
    -------
    trained_model : BaseConnectomeModel
        The trained model
        
    Examples
    --------
    >>> model = HierarchicalBrainGNN(100, 256)
    >>> trained = train_model(model, dataset)
    """
```

### Building Documentation

```bash
cd docs/
make html
```

## Debugging

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Import errors**: Check that package is installed in development mode
3. **Test failures**: Ensure pre-commit hooks pass first

### Profiling

```bash
# Profile memory usage
python -m memory_profiler script.py

# Profile computation
python -m cProfile -o profile.stats script.py
```

## Contributing Guidelines

1. Create feature branch from `main`
2. Implement feature with tests
3. Ensure all checks pass
4. Submit pull request with clear description
5. Address review feedback

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and upload to PyPI

For detailed release instructions, see `docs/RELEASE.md`.
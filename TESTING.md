# Testing Guide - Connectome GNN Suite

This document provides comprehensive testing guidelines for the Connectome GNN Suite.

## Test Structure

The test suite is organized into several categories:

```
tests/
├── unit/                   # Unit tests for individual components
├── integration/            # Integration tests for component interactions
├── e2e/                    # End-to-end workflow tests
├── benchmarks/             # Performance and benchmark tests
├── fixtures/               # Test data generators and utilities
├── conftest.py            # Pytest configuration and fixtures
└── test_*.py              # Main test modules
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=connectome_gnn

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestHierarchicalBrainGNN::test_forward_pass
```

### Test Categories

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only GPU tests (requires CUDA)
pytest -m gpu

# Run integration tests
pytest -m integration

# Run security tests
pytest -m security

# Run performance benchmarks
pytest -m performance
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run tests on specific number of cores
pytest -n 4
```

## Test Coverage

The test suite aims for comprehensive coverage across all components:

### Models (`test_models.py`)
- **HierarchicalBrainGNN**: Multi-level message passing, different configurations
- **TemporalBrainGNN**: Temporal connectivity analysis, LSTM/GRU layers
- **MultiModalBrainGNN**: Structural + functional fusion, attention mechanisms
- **PopulationBrainGNN**: Population-level modeling, group embeddings
- **Novel Architectures**: GraphWavelet, NeuroTransformer, experimental models

### Data Loading (`test_data.py`)
- **ConnectomeDataset**: Synthetic data generation, different parcellations
- **ConnectomeProcessor**: Matrix processing, normalization, graph conversion
- **HCPDataLoader**: Data validation, subject filtering, quality control
- **Data Integrity**: Reproducibility, consistency checks, edge validation

### Tasks (`test_tasks.py`)
- **Regression Tasks**: Age prediction, cognitive scores, brain age gap
- **Classification Tasks**: Sex classification, clinical diagnosis, binary/multi-class
- **Task Suite**: Standardized benchmarks, task configuration, metric computation
- **Evaluation**: Loss functions, metrics validation, target preparation

### Training (`test_training.py`)
- **Training Loop**: Full training pipeline, validation, early stopping
- **Configuration**: Parameter validation, serialization, device handling
- **Optimization**: Learning rate scheduling, gradient clipping, memory management
- **Reproducibility**: Seed handling, deterministic training, checkpoint saving

### Security (`test_security.py`)
- **Input Validation**: Malformed data handling, boundary conditions
- **Data Privacy**: Anonymization, PII detection, demographic privacy
- **Model Security**: Adversarial inputs, weight bounds, memory limits
- **Code Security**: No hardcoded credentials, safe imports, path validation

## Performance Testing

### Memory Profiling

```bash
# Profile memory usage
pytest tests/test_performance.py::test_memory_usage --profile

# Monitor GPU memory (requires CUDA)
pytest tests/test_models.py::TestModelPerformance::test_memory_efficiency
```

### Speed Benchmarking

```bash
# Run performance benchmarks
pytest -m performance

# Benchmark specific models
pytest tests/benchmarks/test_performance.py
```

### Load Testing

```bash
# Test with large datasets
pytest tests/e2e/test_complete_workflow.py::test_large_dataset --runslow
```

## Integration Testing

Integration tests validate component interactions:

```bash
# Full training pipeline
pytest tests/integration/test_training_pipeline.py

# Data-to-model pipeline
pytest tests/e2e/test_complete_workflow.py
```

## Test Data

### Synthetic Data Generation

Tests use synthetic data by default for:
- **Reproducibility**: Fixed random seeds ensure consistent results
- **Speed**: No external data dependencies
- **Validation**: Known ground truth for verification

### Real Data Testing (Optional)

```bash
# Test with HCP data (requires data setup)
pytest tests/integration/ --real-data --data-path /path/to/hcp
```

## Security Testing

### Vulnerability Scanning

```bash
# Run security tests
pytest -m security

# Check for hardcoded credentials
pytest tests/test_security.py::TestDependencySecurity::test_no_credentials_in_code

# Validate input handling
pytest tests/test_security.py::TestDataSecurity::test_input_validation
```

### Privacy Protection

```bash
# Test demographic anonymization
pytest tests/test_security.py::TestPrivacyProtection::test_demographic_privacy

# Verify no data leakage
pytest tests/test_security.py::TestPrivacyProtection::test_no_data_leakage
```

## Continuous Integration

### GitHub Actions Workflow

Tests run automatically on:
- **Push to main**: Full test suite
- **Pull requests**: Fast test subset
- **Nightly**: Extended tests with slow/integration tests

### Test Matrix

- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **PyTorch versions**: 1.12+, 2.0+
- **Operating systems**: Ubuntu, macOS, Windows
- **Hardware**: CPU-only, CUDA (when available)

## Writing New Tests

### Test Structure

```python
import pytest
import torch
from connectome_gnn.models import YourModel

class TestYourModel:
    """Test your new model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Use fixtures for reusable test data
        pass
    
    def test_initialization(self):
        """Test model initialization."""
        model = YourModel(param1=value1, param2=value2)
        assert model.param1 == value1
        assert model.param2 == value2
    
    def test_forward_pass(self, sample_data):
        """Test forward pass functionality."""
        model = YourModel()
        output = model(sample_data)
        
        # Validate output
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.slow
    def test_training_integration(self):
        """Test integration with training pipeline."""
        # Slow test - only run with --runslow
        pass
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        # GPU test - only run if CUDA available
        pass
```

### Best Practices

1. **Use fixtures** for reusable test data and setup
2. **Test edge cases** including invalid inputs and boundary conditions
3. **Validate outputs** including shape, dtype, and value ranges
4. **Mark slow tests** with `@pytest.mark.slow`
5. **Test determinism** by checking reproducibility with fixed seeds
6. **Use parametrize** for testing multiple configurations

### Example Parametrized Test

```python
@pytest.mark.parametrize("hidden_dim,num_layers", [
    (32, 2),
    (64, 3),
    (128, 4)
])
def test_model_configurations(self, hidden_dim, num_layers):
    """Test model with different configurations."""
    model = HierarchicalBrainGNN(
        hidden_dim=hidden_dim,
        num_levels=num_layers
    )
    # Test model functionality
```

## Debugging Failed Tests

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU device
2. **Random Seed Issues**: Ensure proper seed setting in test setup
3. **Path Issues**: Use temporary directories for file operations
4. **Import Errors**: Check test environment has all dependencies

### Debugging Commands

```bash
# Run with verbose output
pytest -v -s tests/failing_test.py

# Drop into debugger on failure
pytest --pdb tests/failing_test.py

# Show local variables on failure
pytest --tb=long tests/failing_test.py

# Run with coverage to see what's not tested
pytest --cov=connectome_gnn --cov-report=html
```

## Test Maintenance

### Regular Tasks

1. **Update test data** when model architectures change
2. **Add tests** for new features and bug fixes
3. **Review slow tests** and optimize when possible
4. **Update fixtures** when API changes
5. **Monitor coverage** and add tests for uncovered code

### Performance Monitoring

Track test performance over time:
- **Test execution time**: Identify slow tests for optimization
- **Memory usage**: Ensure tests don't consume excessive memory
- **Coverage trends**: Maintain high coverage as codebase grows

## Integration with Development Workflow

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run tests before commit
pytest tests/unit/ -m "not slow"
```

### Pull Request Testing

- **Fast tests** run on every PR
- **Integration tests** run on main branch
- **Performance tests** run nightly

### Release Testing

Before releases, run comprehensive test suite:

```bash
# Full test suite including slow tests
pytest --runslow

# Security audit
pytest -m security

# Performance benchmarks
pytest -m performance

# Integration tests
pytest -m integration
```

This ensures high code quality and reliability for all releases.
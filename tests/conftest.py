"""Pytest configuration and shared fixtures for Connectome-GNN-Suite tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Any, Dict
import numpy as np
import torch
import pytest
from torch_geometric.data import Data


# Test configuration
pytest_plugins = ["pytest_mock"]


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark (for performance testing)"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle conditional test execution."""
    # Skip GPU tests if CUDA is not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip slow tests if --fast flag is used
    if config.getoption("--fast", default=False):
        skip_slow = pytest.mark.skip(reason="--fast flag used")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests even if CUDA is not available (for CI)"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests"
    )


# Basic fixtures
@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


# Graph fixtures
@pytest.fixture
def simple_graph() -> Data:
    """Create a simple graph for testing."""
    # Simple 4-node graph (like brain regions)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3]
    ], dtype=torch.long)
    
    x = torch.randn(4, 16)  # 4 nodes, 16 features each
    edge_attr = torch.randn(edge_index.size(1), 8)  # 8 edge features
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def medium_graph() -> Data:
    """Create a medium-sized graph for testing."""
    num_nodes = 100
    num_edges = 500
    
    # Generate random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Remove self-loops and duplicates
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    edge_index = torch.unique(edge_index, dim=1)
    
    x = torch.randn(num_nodes, 64)
    edge_attr = torch.randn(edge_index.size(1), 16)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def large_graph() -> Data:
    """Create a large graph for performance testing."""
    num_nodes = 10000
    num_edges = 50000
    
    # Generate random edges (more realistic connectivity)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    edge_index = torch.unique(edge_index, dim=1)
    
    x = torch.randn(num_nodes, 128)
    edge_attr = torch.randn(edge_index.size(1), 32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def hierarchical_graph() -> Data:
    """Create a hierarchical graph mimicking brain structure."""
    # Create a 3-level hierarchy: 64 -> 16 -> 4 nodes
    num_nodes = 84  # 64 + 16 + 4
    
    # Level 1: 64 base nodes (like brain voxels)
    # Level 2: 16 intermediate nodes (like brain regions)  
    # Level 3: 4 top nodes (like brain networks)
    
    edges = []
    
    # Connect level 1 to level 2 (each group of 4 nodes to 1 intermediate)
    for i in range(16):
        for j in range(4):
            edges.append([j + i*4, 64 + i])  # Connect to intermediate
            edges.append([64 + i, j + i*4])  # Bidirectional
    
    # Connect level 2 to level 3 (each group of 4 intermediates to 1 top)
    for i in range(4):
        for j in range(4):
            edges.append([64 + j + i*4, 80 + i])  # Connect to top
            edges.append([80 + i, 64 + j + i*4])  # Bidirectional
    
    edge_index = torch.tensor(edges).t().contiguous()
    
    x = torch.randn(num_nodes, 32)
    edge_attr = torch.randn(edge_index.size(1), 8)
    
    # Add hierarchy level information
    hierarchy_level = torch.zeros(num_nodes, dtype=torch.long)
    hierarchy_level[64:80] = 1  # Intermediate level
    hierarchy_level[80:84] = 2  # Top level
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.hierarchy_level = hierarchy_level
    
    return data


@pytest.fixture
def batch_graphs(simple_graph) -> Data:
    """Create a batch of graphs for testing."""
    from torch_geometric.data import Batch
    
    graphs = [simple_graph for _ in range(8)]  # Batch of 8 graphs
    return Batch.from_data_list(graphs)


# Mock data fixtures
@pytest.fixture
def mock_connectome_data() -> Dict[str, Any]:
    """Create mock connectome dataset metadata."""
    return {
        "subjects": ["001", "002", "003", "004", "005"],
        "num_subjects": 5,
        "num_nodes": 400,  # Schaefer 400 parcellation
        "features": ["thickness", "area", "volume", "curvature"],
        "tasks": ["rest", "task1", "task2"],
        "demographics": {
            "age": [25, 30, 35, 28, 32],
            "sex": ["M", "F", "M", "F", "M"],
            "handedness": ["R", "R", "L", "R", "R"]
        }
    }


@pytest.fixture
def mock_brain_atlas() -> Dict[str, Any]:
    """Create mock brain atlas information."""
    return {
        "name": "TestAtlas",
        "num_regions": 400,
        "regions": [f"Region_{i:03d}" for i in range(400)],
        "coordinates": np.random.randn(400, 3).astype(np.float32),
        "networks": {
            "Visual": list(range(0, 50)),
            "Somatomotor": list(range(50, 100)),
            "DorsalAttention": list(range(100, 150)),
            "VentralAttention": list(range(150, 200)),
            "Limbic": list(range(200, 250)),
            "Frontoparietal": list(range(250, 300)),
            "Default": list(range(300, 400))
        }
    }


# Configuration fixtures
@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create test configuration."""
    return {
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 2
        },
        "data": {
            "num_workers": 0,  # Single-threaded for testing
            "pin_memory": False
        }
    }


# Performance benchmarking fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_rounds": 3,
        "benchmark_rounds": 10,
        "timeout": 300,  # 5 minutes
        "memory_limit_gb": 16
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state after each test."""
    yield
    # Reset random states
    np.random.seed(None)
    torch.manual_seed(torch.initial_seed())


# Pytest hooks for custom behavior
def pytest_runtest_setup(item):
    """Setup before each test."""
    # Set environment variables for testing
    os.environ["CONNECTOME_GNN_TEST_MODE"] = "true"
    os.environ["CONNECTOME_GNN_LOG_LEVEL"] = "WARNING"


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test."""
    # Clean up environment variables
    os.environ.pop("CONNECTOME_GNN_TEST_MODE", None)
    
    # Force garbage collection for memory-intensive tests
    import gc
    gc.collect()


# Custom markers for test organization
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning")
]
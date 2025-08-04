"""Test basic package imports and version information."""

import pytest
import torch

def test_package_import():
    """Test that the main package can be imported."""
    import connectome_gnn
    assert connectome_gnn is not None


def test_version_available():
    """Test that version information is accessible."""
    import connectome_gnn
    assert hasattr(connectome_gnn, "__version__")
    assert isinstance(connectome_gnn.__version__, str)
    assert len(connectome_gnn.__version__) > 0


def test_author_info():
    """Test that author information is available."""
    import connectome_gnn
    assert hasattr(connectome_gnn, "__author__")
    assert hasattr(connectome_gnn, "__email__")
    assert isinstance(connectome_gnn.__author__, str)
    assert isinstance(connectome_gnn.__email__, str)


def test_data_imports():
    """Test data module imports."""
    from connectome_gnn.data import ConnectomeDataset, ConnectomeProcessor, HCPLoader
    assert ConnectomeDataset is not None
    assert ConnectomeProcessor is not None
    assert HCPLoader is not None


def test_model_imports():
    """Test model imports."""
    from connectome_gnn.models import (
        BaseConnectomeModel,
        HierarchicalBrainGNN,
        TemporalConnectomeGNN,
        MultiModalBrainGNN,
        PopulationGraphGNN
    )
    assert BaseConnectomeModel is not None
    assert HierarchicalBrainGNN is not None
    assert TemporalConnectomeGNN is not None
    assert MultiModalBrainGNN is not None
    assert PopulationGraphGNN is not None


def test_task_imports():
    """Test task imports."""
    from connectome_gnn.tasks import (
        BaseConnectomeTask,
        CognitiveScorePrediction,
        SubjectClassification,
        ConnectomeTaskSuite
    )
    assert BaseConnectomeTask is not None
    assert CognitiveScorePrediction is not None
    assert SubjectClassification is not None
    assert ConnectomeTaskSuite is not None


def test_utility_imports():
    """Test utility imports."""
    from connectome_gnn.utils import set_random_seed, get_device
    from connectome_gnn.training import ConnectomeTrainer
    
    assert set_random_seed is not None
    assert get_device is not None
    assert ConnectomeTrainer is not None


def test_pytorch_available():
    """Test that PyTorch is available and working."""
    assert torch.__version__ is not None
    
    # Test basic tensor operations
    x = torch.randn(3, 4)
    assert x.shape == (3, 4)
    
    # Test device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device is not None


def test_torch_geometric_available():
    """Test that PyTorch Geometric is available."""
    try:
        import torch_geometric
        from torch_geometric.data import Data
        
        # Test basic geometric data
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 4)
        data = Data(x=x, edge_index=edge_index)
        
        assert data.num_nodes == 3
        assert data.num_edges == 4
        
    except ImportError:
        pytest.skip("PyTorch Geometric not available")
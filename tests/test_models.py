"""Test suite for connectome GNN models."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np

from connectome_gnn.models.base import BaseConnectomeModel
from connectome_gnn.models.hierarchical import HierarchicalBrainGNN
from connectome_gnn.models.temporal import TemporalBrainGNN
from connectome_gnn.models.multimodal import MultiModalBrainGNN
from connectome_gnn.models.population import PopulationBrainGNN
from connectome_gnn.research.novel_architectures import GraphWaveletGNN, NeuroTransformerGNN


class TestBaseConnectomeModel:
    """Test base connectome model functionality."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=64,
            num_levels=3,
            output_dim=1
        )
        
        assert model.node_features == 100
        assert model.hidden_dim == 64
        assert model.num_levels == 3
        assert model.output_dim == 1
        assert model.task_type == "regression"
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=64,
            num_levels=2,
            output_dim=1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_device_placement(self):
        """Test model device placement."""
        model = HierarchicalBrainGNN(node_features=100, hidden_dim=32)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        for param in model.parameters():
            assert param.device == device


class TestHierarchicalBrainGNN:
    """Test hierarchical brain GNN model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample connectome data."""
        num_nodes = 90
        node_features = 100
        
        x = torch.randn(num_nodes, node_features)
        
        # Create edges (simplified connectivity)
        edge_list = []
        for i in range(num_nodes):
            for j in range(i+1, min(i+10, num_nodes)):  # Local connectivity
                edge_list.append([i, j])
                edge_list.append([j, i])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.randn(edge_index.size(1), 1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def test_forward_pass(self, sample_data):
        """Test forward pass with sample data."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=64,
            num_levels=3,
            output_dim=1
        )
        
        model.eval()
        with torch.no_grad():
            output = model(sample_data)
        
        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_output_dims(self, sample_data):
        """Test model with different output dimensions."""
        for output_dim in [1, 2, 5, 10]:
            model = HierarchicalBrainGNN(
                node_features=100,
                hidden_dim=32,
                output_dim=output_dim
            )
            
            model.eval()
            with torch.no_grad():
                output = model(sample_data)
            
            assert output.shape == (output_dim,)
    
    def test_batch_processing(self):
        """Test batch processing capability."""
        model = HierarchicalBrainGNN(node_features=50, hidden_dim=32)
        
        # Create batch data
        data_list = []
        for _ in range(3):
            num_nodes = np.random.randint(50, 100)
            x = torch.randn(num_nodes, 50)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            data_list.append(Data(x=x, edge_index=edge_index))
        
        from torch_geometric.data import Batch
        batch_data = Batch.from_data_list(data_list)
        
        model.eval()
        with torch.no_grad():
            output = model(batch_data)
        
        assert output.shape[0] == len(data_list)


class TestTemporalBrainGNN:
    """Test temporal brain GNN model."""
    
    @pytest.fixture
    def temporal_data(self):
        """Create sample temporal connectome data."""
        num_nodes = 90
        node_features = 100
        num_timepoints = 5
        
        # Create temporal node features
        x = torch.randn(num_nodes, node_features, num_timepoints)
        
        # Static connectivity structure
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        edge_attr = torch.randn(edge_index.size(1), 1)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_timepoints = num_timepoints
        
        return data
    
    def test_temporal_forward(self, temporal_data):
        """Test temporal model forward pass."""
        model = TemporalBrainGNN(
            node_features=100,
            hidden_dim=64,
            num_timepoints=5,
            output_dim=1
        )
        
        model.eval()
        with torch.no_grad():
            output = model(temporal_data)
        
        assert output.shape == (1,)
        assert not torch.isnan(output).any()
    
    def test_different_timepoints(self):
        """Test model with different numbers of timepoints."""
        for num_timepoints in [3, 5, 10]:
            model = TemporalBrainGNN(
                node_features=50,
                hidden_dim=32,
                num_timepoints=num_timepoints
            )
            
            # Create data
            num_nodes = 60
            x = torch.randn(num_nodes, 50, num_timepoints)
            edge_index = torch.randint(0, num_nodes, (2, 100))
            
            data = Data(x=x, edge_index=edge_index)
            data.num_timepoints = num_timepoints
            
            model.eval()
            with torch.no_grad():
                output = model(data)
            
            assert output.shape == (1,)


class TestMultiModalBrainGNN:
    """Test multimodal brain GNN model."""
    
    @pytest.fixture
    def multimodal_data(self):
        """Create sample multimodal data."""
        num_nodes = 90
        
        # Structural features
        structural = torch.randn(num_nodes, 50)
        
        # Functional features
        functional = torch.randn(num_nodes, 50)
        
        # Edge information
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))
        edge_attr = torch.randn(edge_index.size(1), 1)
        
        data = Data(edge_index=edge_index, edge_attr=edge_attr)
        data.structural_features = structural
        data.functional_features = functional
        
        return data
    
    def test_multimodal_forward(self, multimodal_data):
        """Test multimodal forward pass."""
        model = MultiModalBrainGNN(
            structural_features=50,
            functional_features=50,
            hidden_dim=64,
            output_dim=1
        )
        
        model.eval()
        with torch.no_grad():
            output = model(multimodal_data)
        
        assert output.shape == (1,)
        assert not torch.isnan(output).any()
    
    def test_fusion_strategies(self, multimodal_data):
        """Test different fusion strategies."""
        fusion_strategies = ["concat", "attention", "gated"]
        
        for fusion in fusion_strategies:
            model = MultiModalBrainGNN(
                structural_features=50,
                functional_features=50,
                hidden_dim=32,
                fusion_strategy=fusion
            )
            
            model.eval()
            with torch.no_grad():
                output = model(multimodal_data)
            
            assert output.shape == (1,)


class TestPopulationBrainGNN:
    """Test population brain GNN model."""
    
    def test_population_modeling(self):
        """Test population-level modeling."""
        model = PopulationBrainGNN(
            node_features=100,
            hidden_dim=64,
            population_size=20,
            output_dim=1
        )
        
        # Create population data
        population_data = []
        for _ in range(20):
            num_nodes = 90
            x = torch.randn(num_nodes, 100)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
            population_data.append(Data(x=x, edge_index=edge_index))
        
        model.eval()
        with torch.no_grad():
            output = model(population_data)
        
        assert output.shape == (1,)


class TestNovelArchitectures:
    """Test novel research architectures."""
    
    @pytest.fixture
    def graph_data(self):
        """Create sample graph data."""
        num_nodes = 90
        x = torch.randn(num_nodes, 100)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))
        edge_attr = torch.randn(edge_index.size(1), 1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def test_graph_wavelet_gnn(self, graph_data):
        """Test GraphWavelet GNN."""
        model = GraphWaveletGNN(
            node_features=100,
            hidden_dim=64,
            num_scales=4,
            output_dim=1
        )
        
        model.eval()
        with torch.no_grad():
            output = model(graph_data)
        
        assert output.shape == (1,)
        assert not torch.isnan(output).any()
    
    def test_neuro_transformer(self, graph_data):
        """Test NeuroTransformer GNN."""
        model = NeuroTransformerGNN(
            node_features=100,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            output_dim=1
        )
        
        model.eval()
        with torch.no_grad():
            output = model(graph_data)
        
        assert output.shape == (1,)
        assert not torch.isnan(output).any()
    
    def test_different_wavelet_types(self, graph_data):
        """Test different wavelet types."""
        wavelet_types = ["mexican_hat", "morlet", "haar", "daubechies"]
        
        for wavelet_type in wavelet_types:
            model = GraphWaveletGNN(
                node_features=100,
                hidden_dim=32,
                wavelet_type=wavelet_type
            )
            
            model.eval()
            with torch.no_grad():
                output = model(graph_data)
            
            assert output.shape == (1,)


class TestModelPerformance:
    """Test model performance and memory usage."""
    
    def test_memory_efficiency(self):
        """Test memory efficiency of models."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=128,
            num_levels=4
        ).to(device)
        
        # Create large batch
        data_list = []
        for _ in range(10):
            num_nodes = 200
            x = torch.randn(num_nodes, 100, device=device)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5), device=device)
            data_list.append(Data(x=x, edge_index=edge_index))
        
        from torch_geometric.data import Batch
        batch_data = Batch.from_data_list(data_list)
        
        # Check memory before
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        model.eval()
        with torch.no_grad():
            output = model(batch_data)
        
        final_memory = torch.cuda.memory_allocated()
        memory_used = (final_memory - initial_memory) / 1024**2  # MB
        
        # Should use reasonable amount of memory
        assert memory_used < 1000, f"Used {memory_used:.1f} MB, which is too much"
    
    def test_inference_speed(self):
        """Test inference speed."""
        model = HierarchicalBrainGNN(node_features=100, hidden_dim=64)
        
        # Create test data
        num_nodes = 90
        x = torch.randn(num_nodes, 100)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        data = Data(x=x, edge_index=edge_index)
        
        # Warm up
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(data)
        
        # Time inference
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                output = model(data)
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        # Should be fast enough (less than 10ms per inference)
        assert avg_time < 0.01, f"Average inference time {avg_time:.4f}s is too slow"


if __name__ == "__main__":
    pytest.main([__file__])
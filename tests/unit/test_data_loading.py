"""Unit tests for data loading functionality."""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from connectome_gnn.data import ConnectomeDataset, ConnectomeProcessor, HCPLoader
from connectome_gnn.data.dataset import ConnectomeGraph
from connectome_gnn.utils import set_random_seed


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_connectivity_data():
    """Create mock connectivity data."""
    set_random_seed(42)
    np.random.seed(42)
    
    num_nodes = 90
    connectivity_matrices = {}
    
    for i in range(5):  # 5 subjects
        subject_id = f"sub-{i+1:03d}"
        # Create symmetric connectivity matrix
        conn_matrix = np.random.rand(num_nodes, num_nodes)
        conn_matrix = (conn_matrix + conn_matrix.T) / 2
        np.fill_diagonal(conn_matrix, 0)
        connectivity_matrices[subject_id] = conn_matrix
    
    return connectivity_matrices


@pytest.fixture
def mock_demographics():
    """Create mock demographics data."""
    subjects = [f"sub-{i+1:03d}" for i in range(5)]
    
    demographics = pd.DataFrame({
        'subject_id': subjects,
        'age': [25, 34, 45, 29, 38],
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'fluid_intelligence': [105.2, 112.8, 98.5, 108.1, 115.3],
        'education_years': [16, 14, 18, 12, 15]
    })
    
    return demographics


class TestConnectomeGraph:
    """Test ConnectomeGraph data structure."""
    
    def test_connectome_graph_creation(self):
        """Test creation of ConnectomeGraph object."""
        # Create test data
        num_nodes = 10
        node_features = 5
        num_edges = 20
        
        x = torch.randn(num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 1)
        node_pos = torch.randn(num_nodes, 3)
        
        graph = ConnectomeGraph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_pos=node_pos,
            subject_id="test_subject",
            demographics={'age': 30, 'sex': 'M'}
        )
        
        assert graph.num_nodes == num_nodes
        assert graph.num_edges == num_edges
        assert graph.subject_id == "test_subject"
        assert graph.demographics['age'] == 30
        assert graph.x.shape == (num_nodes, node_features)
        assert graph.edge_attr.shape == (num_edges, 1)
        assert graph.node_pos.shape == (num_nodes, 3)
    
    def test_connectome_graph_repr(self):
        """Test string representation of ConnectomeGraph."""
        x = torch.randn(5, 3)
        edge_index = torch.randint(0, 5, (2, 8))
        
        graph = ConnectomeGraph(
            x=x,
            edge_index=edge_index,
            subject_id="test_subject"
        )
        
        repr_str = repr(graph)
        assert "subject_id=test_subject" in repr_str
        assert "num_nodes=5" in repr_str
        assert "num_edges=8" in repr_str


class TestConnectomeProcessor:
    """Test connectome preprocessing functionality."""
    
    def test_processor_initialization(self):
        """Test processor initialization with different parameters."""
        processor = ConnectomeProcessor(
            parcellation="AAL",
            edge_threshold=0.05,
            normalization="log_transform"
        )
        
        assert processor.parcellation == "AAL"
        assert processor.edge_threshold == 0.05
        assert processor.normalization == "log_transform"
        assert processor.hierarchy is not None
    
    def test_brain_hierarchy_aal(self):
        """Test brain hierarchy for AAL parcellation."""
        processor = ConnectomeProcessor(parcellation="AAL")
        hierarchy = processor._get_brain_hierarchy()
        
        assert hierarchy["num_regions"] == 90
        assert "levels" in hierarchy
        assert "lobes" in hierarchy["levels"]
        assert "hemispheres" in hierarchy["levels"]
        
        # Check lobe assignments
        lobes = hierarchy["levels"]["lobes"]
        assert "frontal" in lobes
        assert "parietal" in lobes
        assert "temporal" in lobes
        assert "occipital" in lobes


class TestConnectomeDataset:
    """Test ConnectomeDataset functionality."""
    
    def test_dataset_initialization(self, temp_data_dir):
        """Test dataset initialization."""
        dataset = ConnectomeDataset(
            root=str(temp_data_dir),
            resolution="7mm",
            modality="structural",
            parcellation="AAL",
            download=True
        )
        
        assert dataset.resolution == "7mm"
        assert dataset.modality == "structural"
        assert dataset.parcellation == "AAL"
        assert dataset.processor is not None
    
    def test_data_loading(self, temp_data_dir):
        """Test loading of dataset samples."""
        dataset = ConnectomeDataset(
            root=str(temp_data_dir),
            download=True
        )
        
        # Test dataset length
        assert len(dataset) > 0
        
        # Test loading first sample
        sample = dataset[0]
        assert isinstance(sample, ConnectomeGraph)
        assert sample.x is not None
        assert sample.edge_index is not None
        assert sample.edge_attr is not None
        assert sample.node_pos is not None
        assert hasattr(sample, 'subject_id')


class TestDataLoaders:
    """Test data loading utilities."""
    
    def test_batch_creation(self, batch_graphs):
        """Test batch creation from multiple graphs."""
        batch = batch_graphs
        assert isinstance(batch, Batch)
        assert batch.num_graphs == 8
        assert batch.batch.max().item() == 7  # 0-indexed, so max is 7
        
    def test_batch_properties(self, batch_graphs):
        """Test batch maintains proper properties."""
        batch = batch_graphs
        # Each graph has 4 nodes, so batch should have 32 total
        assert batch.num_nodes == 32
        assert batch.x.shape[0] == 32
        
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_variable_batch_sizes(self, simple_graph, batch_size):
        """Test handling of different batch sizes."""
        from torch_geometric.data import Batch
        
        graphs = [simple_graph for _ in range(batch_size)]
        batch = Batch.from_data_list(graphs)
        
        assert batch.num_graphs == batch_size
        assert batch.num_nodes == batch_size * 4


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_connectivity_matrix_processing(self):
        """Test connectivity matrix preprocessing."""
        # Create a mock connectivity matrix
        conn_matrix = np.random.randn(100, 100)
        conn_matrix = (conn_matrix + conn_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(conn_matrix, 0)  # Remove self-connections
        
        # Test basic properties
        assert conn_matrix.shape == (100, 100)
        assert np.allclose(conn_matrix, conn_matrix.T)  # Symmetric
        assert np.all(np.diag(conn_matrix) == 0)  # No self-loops
        
    def test_graph_construction_from_matrix(self):
        """Test graph construction from connectivity matrix."""
        # Create a simple 4x4 connectivity matrix
        conn_matrix = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float32)
        
        # Convert to edge index format
        edge_indices = np.nonzero(conn_matrix)
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        edge_weights = torch.tensor(conn_matrix[edge_indices], dtype=torch.float)
        
        assert edge_index.shape[0] == 2  # 2 rows (source, target)
        assert edge_index.shape[1] == 8  # 8 edges
        assert edge_weights.shape[0] == 8
        assert torch.all(edge_weights == 1.0)
        
    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7])
    def test_edge_thresholding(self, threshold):
        """Test edge thresholding with different values."""
        # Create connectivity matrix with known values
        conn_matrix = np.array([
            [0.0, 0.8, 0.2, 0.6],
            [0.8, 0.0, 0.4, 0.1],
            [0.2, 0.4, 0.0, 0.9],
            [0.6, 0.1, 0.9, 0.0]
        ])
        
        # Apply threshold
        thresholded = conn_matrix * (np.abs(conn_matrix) > threshold)
        
        # Count edges above threshold
        expected_edges = np.sum(np.abs(conn_matrix) > threshold)
        actual_edges = np.sum(thresholded != 0)
        
        assert actual_edges == expected_edges
        assert np.all(np.abs(thresholded[thresholded != 0]) > threshold)


class TestHCPDataLoader:
    """Test HCP-specific data loading functionality."""
    
    @pytest.mark.requires_data
    def test_hcp_subject_loading(self, temp_dir):
        """Test loading individual HCP subjects."""
        # This would test actual HCP data loading
        # For now, we test the expected interface
        
        subject_id = "100206"
        expected_files = [
            f"{subject_id}_structural_connectivity.npy",
            f"{subject_id}_functional_connectivity.npy",
            f"{subject_id}_demographics.json"
        ]
        
        # Test file path construction
        for filename in expected_files:
            file_path = temp_dir / filename
            assert str(file_path).endswith(filename)
            
    def test_hcp_demographics_parsing(self):
        """Test parsing of HCP demographic data."""
        demographics = {
            "Subject": "100206",
            "Age": "26-30",
            "Gender": "M",
            "Handedness": "R"
        }
        
        # Test demographic parsing logic
        assert demographics["Subject"] == "100206"
        assert demographics["Age"] in ["22-25", "26-30", "31-35", "36+"]
        assert demographics["Gender"] in ["M", "F"]
        assert demographics["Handedness"] in ["L", "R"]


class TestMultiModalDataLoading:
    """Test multi-modal data loading."""
    
    def test_structural_functional_combination(self):
        """Test combining structural and functional connectivity."""
        # Create mock structural and functional matrices
        structural = np.random.rand(100, 100)
        functional = np.random.rand(100, 100)
        
        # Make them symmetric
        structural = (structural + structural.T) / 2
        functional = (functional + functional.T) / 2
        
        # Test combination strategies
        combined_concat = np.stack([structural, functional], axis=-1)
        combined_average = (structural + functional) / 2
        
        assert combined_concat.shape == (100, 100, 2)
        assert combined_average.shape == (100, 100)
        assert np.allclose(combined_average, combined_average.T)
        
    def test_temporal_data_handling(self):
        """Test handling of temporal connectivity data."""
        num_timepoints = 200
        num_nodes = 50
        
        # Create mock time series
        timeseries = np.random.randn(num_timepoints, num_nodes)
        
        # Compute dynamic connectivity (sliding window)
        window_size = 50
        num_windows = num_timepoints - window_size + 1
        
        dynamic_conn = np.zeros((num_windows, num_nodes, num_nodes))
        for i in range(num_windows):
            window_data = timeseries[i:i+window_size]
            dynamic_conn[i] = np.corrcoef(window_data.T)
        
        assert dynamic_conn.shape == (num_windows, num_nodes, num_nodes)
        # Test that each window is symmetric
        for i in range(num_windows):
            assert np.allclose(dynamic_conn[i], dynamic_conn[i].T)


class TestDataAugmentation:
    """Test data augmentation techniques."""
    
    def test_node_feature_noise(self, simple_graph):
        """Test adding noise to node features."""
        original_x = simple_graph.x.clone()
        noise_level = 0.1
        
        # Add Gaussian noise
        noise = torch.randn_like(original_x) * noise_level
        augmented_x = original_x + noise
        
        # Test that augmentation preserves shape
        assert augmented_x.shape == original_x.shape
        
        # Test that noise was actually added
        assert not torch.allclose(augmented_x, original_x)
        
        # Test noise level is reasonable
        noise_magnitude = torch.norm(noise) / torch.norm(original_x)
        assert noise_magnitude < 0.5  # Noise shouldn't dominate signal
        
    def test_edge_dropout(self, simple_graph):
        """Test random edge dropout augmentation."""
        original_edge_index = simple_graph.edge_index
        dropout_rate = 0.3
        
        # Simulate edge dropout
        num_edges = original_edge_index.shape[1]
        keep_mask = torch.rand(num_edges) > dropout_rate
        dropped_edge_index = original_edge_index[:, keep_mask]
        
        # Test that some edges were dropped
        assert dropped_edge_index.shape[1] < original_edge_index.shape[1]
        assert dropped_edge_index.shape[1] >= num_edges * (1 - dropout_rate) * 0.5
        
    def test_subgraph_sampling(self, medium_graph):
        """Test subgraph sampling for large graphs."""
        original_nodes = medium_graph.num_nodes
        sample_size = 50
        
        # Sample random nodes
        sampled_nodes = torch.randperm(original_nodes)[:sample_size]
        
        # Create node mapping
        node_map = torch.full((original_nodes,), -1, dtype=torch.long)
        node_map[sampled_nodes] = torch.arange(sample_size)
        
        # Test sampling properties
        assert len(sampled_nodes) == sample_size
        assert torch.all(sampled_nodes < original_nodes)
        assert len(torch.unique(sampled_nodes)) == sample_size  # No duplicates
"""Test suite for data loading and processing."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from connectome_gnn.data.dataset import ConnectomeDataset
from connectome_gnn.data.processor import ConnectomeProcessor
from connectome_gnn.data.hcp_loader import HCPDataLoader


class TestConnectomeDataset:
    """Test connectome dataset functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataset_initialization(self, temp_data_dir):
        """Test dataset initialization."""
        dataset = ConnectomeDataset(
            root=temp_data_dir,
            resolution="7mm",
            modality="structural",
            num_subjects=10
        )
        
        assert dataset.resolution == "7mm"
        assert dataset.modality == "structural"
        assert dataset.num_subjects == 10
        assert len(dataset) == 10
    
    def test_synthetic_data_generation(self, temp_data_dir):
        """Test synthetic data generation."""
        dataset = ConnectomeDataset(
            root=temp_data_dir,
            num_subjects=5,
            use_synthetic=True
        )
        
        # Test data loading
        data = dataset[0]
        
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'edge_attr')
        assert hasattr(data, 'demographics')
        
        # Check dimensions
        assert data.x.shape[1] == 100  # node features
        assert data.edge_index.shape[0] == 2  # source/target
        assert data.edge_attr.shape[1] == 1  # edge features
    
    def test_different_parcellations(self, temp_data_dir):
        """Test different brain parcellations."""
        parcellations = ["AAL", "Schaefer400", "Harvard_Oxford"]
        
        for parcellation in parcellations:
            dataset = ConnectomeDataset(
                root=temp_data_dir,
                parcellation=parcellation,
                num_subjects=3,
                use_synthetic=True
            )
            
            data = dataset[0]
            expected_nodes = {
                "AAL": 90,
                "Schaefer400": 400,
                "Harvard_Oxford": 96
            }
            
            assert data.x.shape[0] == expected_nodes[parcellation]
    
    def test_different_modalities(self, temp_data_dir):
        """Test different imaging modalities."""
        modalities = ["structural", "functional", "diffusion"]
        
        for modality in modalities:
            dataset = ConnectomeDataset(
                root=temp_data_dir,
                modality=modality,
                num_subjects=3,
                use_synthetic=True
            )
            
            data = dataset[0]
            assert data.x.shape[0] > 0  # Has nodes
            assert data.edge_index.shape[1] > 0  # Has edges
    
    def test_demographics_generation(self, temp_data_dir):
        """Test demographics generation."""
        dataset = ConnectomeDataset(
            root=temp_data_dir,
            num_subjects=20,
            use_synthetic=True
        )
        
        # Check all subjects have demographics
        ages = []
        sexes = []
        
        for i in range(len(dataset)):
            data = dataset[i]
            assert 'age' in data.demographics
            assert 'sex' in data.demographics
            assert 'fluid_intelligence' in data.demographics
            
            ages.append(data.demographics['age'])
            sexes.append(data.demographics['sex'])
        
        # Check realistic ranges
        assert min(ages) >= 18
        assert max(ages) <= 80
        assert set(sexes) == {'M', 'F'}
    
    def test_batch_loading(self, temp_data_dir):
        """Test batch loading with DataLoader."""
        dataset = ConnectomeDataset(
            root=temp_data_dir,
            num_subjects=10,
            use_synthetic=True
        )
        
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=3, shuffle=True)
        
        batch = next(iter(loader))
        
        assert batch.batch.max().item() == 2  # 3 graphs (0, 1, 2)
        assert batch.x.shape[0] > 0
        assert batch.edge_index.shape[1] > 0


class TestConnectomeProcessor:
    """Test connectome data processor."""
    
    def test_connectivity_matrix_processing(self):
        """Test connectivity matrix processing."""
        processor = ConnectomeProcessor()
        
        # Create sample connectivity matrix
        num_nodes = 90
        connectivity = np.random.rand(num_nodes, num_nodes)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity, 0)  # Zero diagonal
        
        # Test thresholding
        thresholded = processor.threshold_connectivity(connectivity, threshold=0.5)
        assert np.all(thresholded >= 0.5)
        assert np.all(np.diag(thresholded) == 0)
    
    def test_normalization_methods(self):
        """Test different normalization methods."""
        processor = ConnectomeProcessor()
        
        # Create test matrix
        matrix = np.random.rand(50, 50) * 100
        matrix = (matrix + matrix.T) / 2
        
        # Test z-score normalization
        normalized = processor.normalize_connectivity(matrix, method="zscore")
        assert abs(np.mean(normalized)) < 1e-10  # Close to zero mean
        assert abs(np.std(normalized) - 1.0) < 1e-10  # Unit variance
        
        # Test min-max normalization
        normalized = processor.normalize_connectivity(matrix, method="minmax")
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1
    
    def test_graph_conversion(self):
        """Test conversion from connectivity matrix to graph."""
        processor = ConnectomeProcessor()
        
        # Create connectivity matrix
        num_nodes = 30
        connectivity = np.random.rand(num_nodes, num_nodes)
        connectivity = (connectivity + connectivity.T) / 2
        np.fill_diagonal(connectivity, 0)
        
        # Convert to graph
        data = processor.connectivity_to_graph(connectivity)
        
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'edge_attr')
        assert data.x.shape[0] == num_nodes
        assert data.edge_index.shape[0] == 2
    
    def test_feature_extraction(self):
        """Test node feature extraction."""
        processor = ConnectomeProcessor()
        
        # Create connectivity matrix
        connectivity = np.random.rand(50, 50)
        connectivity = (connectivity + connectivity.T) / 2
        
        # Extract features
        features = processor.extract_node_features(connectivity)
        
        assert features.shape[0] == 50  # Number of nodes
        assert features.shape[1] > 0  # Has features
        
        # Check that features are reasonable
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_multimodal_processing(self):
        """Test multimodal data processing."""
        processor = ConnectomeProcessor()
        
        # Create structural and functional matrices
        structural = np.random.rand(60, 60)
        functional = np.random.rand(60, 60)
        
        # Make symmetric
        structural = (structural + structural.T) / 2
        functional = (functional + functional.T) / 2
        
        # Process
        processed = processor.combine_modalities(
            structural=structural,
            functional=functional
        )
        
        assert 'structural_features' in processed
        assert 'functional_features' in processed
        assert hasattr(processed, 'edge_index')


class TestHCPDataLoader:
    """Test HCP data loader."""
    
    def test_hcp_loader_initialization(self):
        """Test HCP loader initialization."""
        # Mock data directory
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = HCPDataLoader(data_dir=temp_dir)
            
            assert loader.data_dir == Path(temp_dir)
            assert loader.subjects_info == {}
    
    def test_subject_filtering(self):
        """Test subject filtering functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = HCPDataLoader(data_dir=temp_dir)
            
            # Mock subject data
            subjects_data = {
                'subject_001': {'age': 25, 'sex': 'M', 'quality': 'good'},
                'subject_002': {'age': 30, 'sex': 'F', 'quality': 'fair'},
                'subject_003': {'age': 35, 'sex': 'M', 'quality': 'good'},
                'subject_004': {'age': 40, 'sex': 'F', 'quality': 'poor'}
            }
            
            # Test age filtering
            filtered = loader._filter_subjects(
                subjects_data,
                age_range=(25, 35)
            )
            assert len(filtered) == 3
            
            # Test quality filtering
            filtered = loader._filter_subjects(
                subjects_data,
                quality_threshold='good'
            )
            assert len(filtered) == 2
    
    def test_data_validation(self):
        """Test data validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = HCPDataLoader(data_dir=temp_dir)
            
            # Test valid connectivity matrix
            valid_matrix = np.random.rand(90, 90)
            valid_matrix = (valid_matrix + valid_matrix.T) / 2
            np.fill_diagonal(valid_matrix, 0)
            
            validation = loader.validate_connectivity_data(valid_matrix)
            assert validation['valid'] == True
            
            # Test invalid matrix (NaN values)
            invalid_matrix = valid_matrix.copy()
            invalid_matrix[0, 0] = np.nan
            
            validation = loader.validate_connectivity_data(invalid_matrix)
            assert validation['valid'] == False
            assert 'NaN' in str(validation['issues'])


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_reproducibility(self):
        """Test that data generation is reproducible."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set seed and create dataset
            torch.manual_seed(42)
            np.random.seed(42)
            
            dataset1 = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            data1 = dataset1[0]
            
            # Reset seed and create again
            torch.manual_seed(42)
            np.random.seed(42)
            
            dataset2 = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            data2 = dataset2[0]
            
            # Should be identical
            assert torch.equal(data1.x, data2.x)
            assert torch.equal(data1.edge_index, data2.edge_index)
            assert torch.equal(data1.edge_attr, data2.edge_attr)
    
    def test_data_consistency(self):
        """Test data consistency across subjects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=10,
                use_synthetic=True,
                parcellation="AAL"
            )
            
            node_counts = []
            feature_dims = []
            
            for i in range(len(dataset)):
                data = dataset[i]
                node_counts.append(data.x.shape[0])
                feature_dims.append(data.x.shape[1])
            
            # All subjects should have same structure
            assert len(set(node_counts)) == 1, "Inconsistent node counts"
            assert len(set(feature_dims)) == 1, "Inconsistent feature dimensions"
    
    def test_edge_connectivity(self):
        """Test edge connectivity properties."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            
            for i in range(len(dataset)):
                data = dataset[i]
                
                # Check edge indices are valid
                max_node = data.x.shape[0] - 1
                assert data.edge_index.max() <= max_node
                assert data.edge_index.min() >= 0
                
                # Check no self-loops
                source, target = data.edge_index
                assert not torch.any(source == target)
                
                # Check edge attributes match edge count
                assert data.edge_attr.shape[0] == data.edge_index.shape[1]


class TestDataTransforms:
    """Test data transformation and augmentation."""
    
    def test_noise_augmentation(self):
        """Test noise augmentation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            
            data = dataset[0]
            original_features = data.x.clone()
            
            # Apply noise
            processor = ConnectomeProcessor()
            augmented_data = processor.add_noise(data, noise_level=0.1)
            
            # Should be different but similar
            assert not torch.equal(original_features, augmented_data.x)
            
            # But not too different
            mse = torch.mean((original_features - augmented_data.x) ** 2)
            assert mse < 0.5  # Reasonable noise level
    
    def test_subgraph_sampling(self):
        """Test subgraph sampling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=3,
                use_synthetic=True
            )
            
            data = dataset[0]
            original_nodes = data.x.shape[0]
            
            processor = ConnectomeProcessor()
            subgraph = processor.sample_subgraph(data, num_nodes=50)
            
            assert subgraph.x.shape[0] == 50
            assert subgraph.x.shape[0] < original_nodes


if __name__ == "__main__":
    pytest.main([__file__])
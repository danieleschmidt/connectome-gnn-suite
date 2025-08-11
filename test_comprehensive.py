"""Comprehensive test suite for connectome GNN functionality."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from connectome_gnn.data import ConnectomeDataset, ConnectomeProcessor
from connectome_gnn.models.base import SimpleConnectomeGNN, AttentionConnectomeGNN
from connectome_gnn.tasks.graph_level import AgeRegression, SubjectClassification
from connectome_gnn.training import ConnectomeTrainer
from connectome_gnn.validation import ConnectomeDataValidator
from connectome_gnn.security import SecureDataLoader, setup_secure_environment
from connectome_gnn.caching import get_cache


class TestDataModule:
    """Test data loading and processing."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        dataset = ConnectomeDataset(root=self.temp_dir, resolution='7mm')
        
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Test sample loading
        sample = dataset[0]
        assert isinstance(sample, Data), "Sample should be PyTorch Geometric Data"
        assert hasattr(sample, 'x'), "Sample should have node features"
        assert hasattr(sample, 'edge_index'), "Sample should have edge index"
        
        print(f"✅ Dataset test passed: {len(dataset)} samples")
    
    def test_processor_functionality(self):
        """Test connectome processor."""
        processor = ConnectomeProcessor(
            parcellation="AAL",
            edge_threshold=0.1,
            normalization="log_transform"
        )
        
        # Create synthetic connectivity matrix
        conn_matrix = np.random.rand(100, 100)
        conn_matrix = (conn_matrix + conn_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(conn_matrix, 0)  # Remove self-loops
        
        # Process connectivity
        processed = processor.process(conn_matrix)
        
        assert 'connectivity_matrix' in processed
        assert 'edge_index' in processed
        assert 'node_features' in processed
        
        print("✅ Processor test passed")
    
    def test_data_validation(self):
        """Test data validation functionality."""
        validator = ConnectomeDataValidator()
        
        # Create test data
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 500))
        test_data = Data(x=x, edge_index=edge_index)
        
        # Validate data
        results = validator.validate_data(test_data)
        
        assert 'is_valid' in results
        assert isinstance(results['is_valid'], bool)
        assert 'warnings' in results
        assert 'errors' in results
        
        print(f"✅ Validation test passed: Valid={results['is_valid']}")


class TestModelModule:
    """Test model architectures."""
    
    def test_simple_gnn_model(self):
        """Test simple GNN model."""
        model = SimpleConnectomeGNN(
            node_features=10,
            hidden_dim=32,
            num_classes=1
        )
        
        # Test forward pass
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        batch = torch.zeros(50, dtype=torch.long)
        
        test_data = Data(x=x, edge_index=edge_index, batch=batch)
        
        with torch.no_grad():
            output = model(test_data)
        
        assert output.shape == (1, 1), f"Expected shape (1, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        print("✅ Simple GNN model test passed")
    
    def test_attention_gnn_model(self):
        """Test attention-based GNN model."""
        model = AttentionConnectomeGNN(
            node_features=10,
            hidden_dim=32,
            num_heads=2,
            num_classes=1
        )
        
        # Test forward pass
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        batch = torch.zeros(50, dtype=torch.long)
        
        test_data = Data(x=x, edge_index=edge_index, batch=batch)
        
        with torch.no_grad():
            output = model(test_data)
            attention_weights = model.get_attention_weights(test_data)
        
        assert output.shape == (1, 1), f"Expected shape (1, 1), got {output.shape}"
        assert attention_weights is not None, "Attention weights should be available"
        
        print("✅ Attention GNN model test passed")


class TestTaskModule:
    """Test task definitions."""
    
    def test_age_regression_task(self):
        """Test age regression task."""
        task = AgeRegression()
        
        # Create test data with age target
        test_data = Data(
            x=torch.randn(50, 10),
            edge_index=torch.randint(0, 50, (2, 100)),
            age=torch.tensor([25.5])
        )
        
        # Test target extraction
        target = task.get_target(test_data)
        assert target.shape == torch.Size([1]), f"Expected shape [1], got {target.shape}"
        
        # Test metrics computation
        predictions = torch.tensor([24.0])
        targets = torch.tensor([25.5])
        metrics = task.compute_metrics(predictions, targets)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert isinstance(metrics['mae'], float)
        
        print("✅ Age regression task test passed")
    
    def test_subject_classification_task(self):
        """Test subject classification task."""
        task = SubjectClassification(target='sex', num_classes=2)
        
        # Create test data
        test_data = Data(
            x=torch.randn(50, 10),
            edge_index=torch.randint(0, 50, (2, 100)),
            sex=torch.tensor([1])
        )
        
        # Test target extraction
        target = task.get_target(test_data)
        assert target.shape == torch.Size([1]), f"Expected shape [1], got {target.shape}"
        
        # Test metrics computation
        predictions = torch.tensor([[0.3, 0.7]])  # Soft predictions
        targets = torch.tensor([1])
        metrics = task.compute_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        
        print("✅ Subject classification task test passed")


class TestTrainingModule:
    """Test training functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        # Create test model and task
        model = SimpleConnectomeGNN(node_features=10, hidden_dim=16, num_classes=1)
        task = AgeRegression()
        
        trainer = ConnectomeTrainer(
            model=model,
            task=task,
            device='cpu',
            batch_size=4
        )
        
        assert trainer.model is not None
        assert trainer.task is not None
        assert trainer.device == 'cpu'
        
        print("✅ Trainer initialization test passed")
    
    def test_training_validation_errors(self):
        """Test training input validation."""
        model = SimpleConnectomeGNN(node_features=10, hidden_dim=16, num_classes=1)
        task = AgeRegression()
        trainer = ConnectomeTrainer(model=model, task=task)
        
        # Test empty dataset
        empty_dataset = []
        
        with pytest.raises(ValueError):
            trainer.fit(empty_dataset, epochs=1)
        
        # Test invalid epochs
        dataset = ConnectomeDataset(root=self.temp_dir)
        
        with pytest.raises(ValueError):
            trainer.fit(dataset, epochs=-1)
        
        # Test invalid validation split
        with pytest.raises(ValueError):
            trainer.fit(dataset, epochs=1, validation_split=1.5)
        
        print("✅ Training validation test passed")


class TestSecurityModule:
    """Test security functionality."""
    
    def test_secure_environment_setup(self):
        """Test secure environment setup."""
        config = setup_secure_environment()
        
        assert isinstance(config, dict)
        assert 'torch_settings' in config
        assert 'security_measures' in config
        
        print("✅ Security environment test passed")
    
    def test_secure_data_loader(self):
        """Test secure data loading."""
        loader = SecureDataLoader()
        
        # Test valid path validation
        temp_file = Path(tempfile.mktemp(suffix='.pt'))
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create a test file
            torch.save({'test': 'data'}, temp_file)
            
            validated_path = loader.validate_file_path(temp_file)
            assert isinstance(validated_path, Path)
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
        
        print("✅ Secure data loader test passed")


class TestCachingModule:
    """Test caching functionality."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = get_cache()
        
        # Test put and get
        test_data = {'test': 'value', 'number': 42}
        success = cache.put('test_key', test_data)
        assert success, "Cache put should succeed"
        
        retrieved = cache.get('test_key')
        assert retrieved is not None, "Should retrieve cached data"
        assert retrieved['test'] == 'value', "Retrieved data should match"
        
        # Test cache stats
        stats = cache.get_stats()
        assert 'total_items' in stats
        assert stats['total_items'] >= 1
        
        print("✅ Cache operations test passed")
    
    def test_cache_expiration(self):
        """Test cache TTL functionality."""
        # Create cache with short TTL
        from connectome_gnn.caching import ConnectomeCache
        import time
        
        short_ttl_cache = ConnectomeCache(ttl_seconds=1)
        
        # Put data
        short_ttl_cache.put('expire_key', 'expire_value')
        
        # Should be available immediately
        assert short_ttl_cache.get('expire_key') is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        assert short_ttl_cache.get('expire_key') is None
        
        print("✅ Cache expiration test passed")


class TestPerformanceModule:
    """Test performance and optimization."""
    
    def test_model_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        models = [
            SimpleConnectomeGNN(node_features=10, hidden_dim=32),
            AttentionConnectomeGNN(node_features=10, hidden_dim=32)
        ]
        
        for model in models:
            param_count = sum(p.numel() for p in model.parameters())
            
            # Models should have reasonable parameter counts (not too big or small)
            assert 100 < param_count < 1_000_000, f"Model has {param_count} parameters"
            
        print("✅ Model parameter count test passed")
    
    def test_inference_time(self):
        """Test model inference performance."""
        import time
        
        model = SimpleConnectomeGNN(node_features=10, hidden_dim=32)
        model.eval()
        
        # Create test data
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        test_data = Data(x=x, edge_index=edge_index, batch=torch.zeros(100, dtype=torch.long))
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_data)
        
        # Time inference
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 10
        
        # Should complete inference reasonably quickly (< 1 second per batch)
        assert avg_time < 1.0, f"Inference took {avg_time:.3f}s per batch"
        
        print(f"✅ Inference performance test passed: {avg_time*1000:.1f}ms per batch")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running Comprehensive Test Suite...")
    
    # Create test instances
    test_classes = [
        TestDataModule(),
        TestModelModule(), 
        TestTaskModule(),
        TestTrainingModule(),
        TestSecurityModule(),
        TestCachingModule(),
        TestPerformanceModule()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_instance in test_classes:
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Setup if available
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                
                # Teardown if available  
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                    
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
    
    print(f"\n🎯 Test Results: {passed_tests}/{total_tests} tests passed")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_comprehensive_tests()
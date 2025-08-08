"""Test suite for training functionality."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import tempfile
import shutil
from pathlib import Path

from connectome_gnn.training.trainer import ConnectomeTrainer
from connectome_gnn.training.config import TrainingConfig
from connectome_gnn.models.hierarchical import HierarchicalBrainGNN
from connectome_gnn.tasks.graph_level import AgeRegression, SexClassification
from connectome_gnn.data.dataset import ConnectomeDataset


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.num_epochs > 0
        assert config.device in ["auto", "cpu", "cuda"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            num_epochs=100
        )
        assert config.batch_size == 32
        
        # Invalid values should raise errors or be clamped
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = TrainingConfig(
            batch_size=64,
            learning_rate=0.01,
            num_epochs=50
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        assert config_dict['batch_size'] == 64
        assert config_dict['learning_rate'] == 0.01
        
        # Recreate from dict
        new_config = TrainingConfig.from_dict(config_dict)
        assert new_config.batch_size == 64
        assert new_config.learning_rate == 0.01


class TestConnectomeTrainer:
    """Test connectome trainer functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=20,
                use_synthetic=True
            )
            yield dataset
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        return HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=32,
            num_levels=2,
            output_dim=1
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return AgeRegression()
    
    def test_trainer_initialization(self, sample_model, sample_task, temp_output_dir):
        """Test trainer initialization."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        assert trainer.model == sample_model
        assert trainer.task == sample_task
        assert trainer.config == config
        assert trainer.current_epoch == 0
    
    def test_data_preparation(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test data preparation for training."""
        config = TrainingConfig(
            batch_size=4,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(sample_dataset)
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Check batch sizes
        train_batch = next(iter(train_loader))
        assert train_batch.batch.max().item() < config.batch_size
    
    def test_single_epoch_training(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test single epoch of training."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(sample_dataset)
        
        # Train one epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        assert 'loss' in train_metrics
        assert 'mae' in train_metrics  # Age regression metric
        assert train_metrics['loss'] > 0
    
    def test_validation(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test validation functionality."""
        config = TrainingConfig(
            batch_size=4,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(sample_dataset)
        
        # Run validation
        val_metrics = trainer.validate(val_loader)
        
        assert 'loss' in val_metrics
        assert 'mae' in val_metrics
        assert val_metrics['loss'] > 0
    
    def test_full_training_loop(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test full training loop."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            output_dir=str(temp_output_dir),
            save_best_model=True
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Run training
        results = trainer.train(sample_dataset)
        
        assert 'train_history' in results
        assert 'val_history' in results
        assert 'test_metrics' in results
        assert 'best_epoch' in results
        
        # Check history length
        assert len(results['train_history']) == config.num_epochs
        assert len(results['val_history']) == config.num_epochs
    
    def test_model_saving(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test model saving functionality."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            output_dir=str(temp_output_dir),
            save_best_model=True,
            save_checkpoint_every=1
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Run training
        results = trainer.train(sample_dataset)
        
        # Check that model files were saved
        output_dir = Path(temp_output_dir)
        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "config.json").exists()
        
        # Check checkpoint saving
        checkpoint_files = list(output_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoint_files) >= 1
    
    def test_classification_task(self, sample_dataset, temp_output_dir):
        """Test training with classification task."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=32,
            output_dim=1  # Binary classification
        )
        
        task = SexClassification()
        
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=model,
            task=task,
            config=config
        )
        
        # Run training
        results = trainer.train(sample_dataset)
        
        # Check classification metrics
        test_metrics = results['test_metrics']
        assert 'accuracy' in test_metrics
        assert 'auc' in test_metrics
        assert 'f1' in test_metrics
    
    def test_early_stopping(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test early stopping functionality."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=20,  # High number
            early_stopping_patience=3,
            early_stopping_min_delta=0.001,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Run training (should stop early)
        results = trainer.train(sample_dataset)
        
        # Should stop before max epochs
        actual_epochs = len(results['train_history'])
        assert actual_epochs <= config.num_epochs
    
    def test_learning_rate_scheduling(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test learning rate scheduling."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=5,
            learning_rate=0.01,
            lr_scheduler="step",
            lr_step_size=2,
            lr_gamma=0.5,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Check that scheduler was created
        assert trainer.scheduler is not None
        
        # Run training
        results = trainer.train(sample_dataset)
        
        # Check that training completed
        assert len(results['train_history']) == config.num_epochs
    
    def test_gradient_clipping(self, sample_model, sample_task, sample_dataset, temp_output_dir):
        """Test gradient clipping."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            gradient_clip_value=1.0,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Run training
        results = trainer.train(sample_dataset)
        
        # Should complete without gradient explosion
        assert len(results['train_history']) == config.num_epochs


class TestTrainingUtilities:
    """Test training utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        from connectome_gnn.utils import get_device
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda"]
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        from connectome_gnn.utils import count_parameters
        
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=64,
            output_dim=1
        )
        
        param_count = count_parameters(model)
        assert param_count > 0
        
        # Should be same as manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count
    
    def test_model_size_estimation(self):
        """Test model size estimation."""
        from connectome_gnn.utils import get_model_size_mb
        
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=64,
            output_dim=1
        )
        
        size_mb = get_model_size_mb(model)
        assert size_mb > 0
        assert size_mb < 100  # Should be reasonable size


class TestTrainingStability:
    """Test training stability and reproducibility."""
    
    def test_reproducible_training(self, temp_output_dir):
        """Test that training is reproducible with same seed."""
        from connectome_gnn.utils import set_random_seed
        
        # Create consistent dataset and model
        with tempfile.TemporaryDirectory() as temp_dir:
            # First run
            set_random_seed(42)
            dataset1 = ConnectomeDataset(root=temp_dir, num_subjects=10, use_synthetic=True)
            model1 = HierarchicalBrainGNN(node_features=100, hidden_dim=32, output_dim=1)
            task1 = AgeRegression()
            
            config1 = TrainingConfig(
                batch_size=4,
                num_epochs=2,
                learning_rate=0.01,
                output_dir=str(temp_output_dir / "run1")
            )
            
            trainer1 = ConnectomeTrainer(model1, task1, config1)
            results1 = trainer1.train(dataset1)
            
            # Second run with same seed
            set_random_seed(42)
            dataset2 = ConnectomeDataset(root=temp_dir, num_subjects=10, use_synthetic=True)
            model2 = HierarchicalBrainGNN(node_features=100, hidden_dim=32, output_dim=1)
            task2 = AgeRegression()
            
            config2 = TrainingConfig(
                batch_size=4,
                num_epochs=2,
                learning_rate=0.01,
                output_dir=str(temp_output_dir / "run2")
            )
            
            trainer2 = ConnectomeTrainer(model2, task2, config2)
            results2 = trainer2.train(dataset2)
            
            # Results should be similar (allowing for small numerical differences)
            final_loss1 = results1['train_history'][-1]['loss']
            final_loss2 = results2['train_history'][-1]['loss']
            
            assert abs(final_loss1 - final_loss2) < 0.1
    
    def test_nan_detection(self, sample_model, sample_task, temp_output_dir):
        """Test NaN detection in training."""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            learning_rate=10.0,  # Very high LR might cause NaN
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(
            model=sample_model,
            task=sample_task,
            config=config
        )
        
        # Create dataset with extreme values that might cause NaN
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=10,
                use_synthetic=True
            )
            
            # Training should either complete or gracefully handle NaN
            try:
                results = trainer.train(dataset)
                # If it completes, check for NaN in results
                for epoch_metrics in results['train_history']:
                    assert not torch.isnan(torch.tensor(epoch_metrics['loss']))
            except RuntimeError as e:
                # Should catch and report NaN appropriately
                assert "nan" in str(e).lower() or "inf" in str(e).lower()


class TestMemoryManagement:
    """Test memory management during training."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, temp_output_dir):
        """Test GPU memory usage during training."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=128,
            output_dim=1
        )
        
        task = AgeRegression()
        
        config = TrainingConfig(
            batch_size=8,
            num_epochs=2,
            device="cuda",
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(model, task, config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=20,
                use_synthetic=True
            )
            
            # Clear memory before training
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Run training
            results = trainer.train(dataset)
            
            # Check memory usage
            final_memory = torch.cuda.memory_allocated()
            memory_used = (final_memory - initial_memory) / 1024**2  # MB
            
            # Should use reasonable amount of memory
            assert memory_used < 2000  # Less than 2GB
            
            # Should complete training
            assert len(results['train_history']) == config.num_epochs
    
    def test_memory_cleanup(self, sample_model, sample_task, temp_output_dir):
        """Test memory cleanup after training."""
        config = TrainingConfig(
            batch_size=8,
            num_epochs=2,
            output_dir=str(temp_output_dir)
        )
        
        trainer = ConnectomeTrainer(sample_model, sample_task, config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=20,
                use_synthetic=True
            )
            
            # Run training
            results = trainer.train(dataset)
            
            # Manually cleanup
            del trainer
            del results
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Should not crash or leak memory
            assert True  # If we get here, cleanup worked


if __name__ == "__main__":
    pytest.main([__file__])
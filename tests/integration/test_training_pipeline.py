"""Integration tests for the complete training pipeline."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from unittest.mock import Mock, patch


@pytest.mark.integration
class TestTrainingPipeline:
    """Test the complete training pipeline integration."""
    
    def test_data_loader_model_integration(self, batch_graphs, device):
        """Test integration between data loader and model."""
        # Create a simple mock model
        class SimpleGNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.linear1 = nn.Linear(input_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()
                
            def forward(self, data):
                x = data.x
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        model = SimpleGNN(16, 32, 1).to(device)
        batch = batch_graphs.to(device)
        
        # Test forward pass
        output = model(batch)
        expected_shape = (batch.num_nodes, 1)
        
        assert output.shape == expected_shape
        assert output.device == device
        
    def test_training_step(self, batch_graphs, device):
        """Test a single training step."""
        # Create model, optimizer, and loss function
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create mock targets
        batch = batch_graphs.to(device)
        targets = torch.randn(batch.num_nodes, 1, device=device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(batch.x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Test that loss is computed and gradients exist
        assert isinstance(loss.item(), float)
        for param in model.parameters():
            assert param.grad is not None
            
    def test_validation_step(self, batch_graphs, device):
        """Test a single validation step."""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        criterion = nn.MSELoss()
        
        batch = batch_graphs.to(device)
        targets = torch.randn(batch.num_nodes, 1, device=device)
        
        # Validation step (no gradients)
        model.eval()
        with torch.no_grad():
            outputs = model(batch.x)
            val_loss = criterion(outputs, targets)
        
        assert isinstance(val_loss.item(), float)
        # Verify no gradients were computed
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)
            
    def test_epoch_training_loop(self, simple_graph, device):
        """Test a complete epoch of training."""
        # Create dataset and data loader
        dataset = [simple_graph for _ in range(16)]  # Small dataset
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training epoch
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            batch = batch.to(device)
            targets = torch.randn(batch.num_nodes, 1, device=device)
            
            optimizer.zero_grad()
            outputs = model(batch.x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        assert num_batches == 4  # 16 samples / 4 batch_size
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        
    @pytest.mark.slow
    def test_multi_epoch_training(self, simple_graph, device):
        """Test training over multiple epochs."""
        # Create dataset
        dataset = [simple_graph for _ in range(32)]
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Multi-epoch training
        num_epochs = 5
        losses = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(device)
                targets = torch.randn(batch.num_nodes, 1, device=device)
                
                optimizer.zero_grad()
                outputs = model(batch.x)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
        
        # Test that training progressed
        assert len(losses) == num_epochs
        # Loss should generally decrease (allowing for some noise)
        assert losses[-1] <= max(losses[:3]) + 0.1  # Some tolerance


@pytest.mark.integration
class TestModelSaving:
    """Test model saving and loading integration."""
    
    def test_model_state_dict_saving(self, temp_dir):
        """Test saving and loading model state dict."""
        # Create model
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(), 
            nn.Linear(32, 1)
        )
        
        # Save model
        save_path = temp_dir / "model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Create new model and load state
        new_model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that parameters match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
            
    def test_checkpoint_saving(self, temp_dir):
        """Test saving and loading training checkpoints."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        epoch = 5
        loss = 0.123
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        
        # Verify checkpoint contents
        assert loaded_checkpoint['epoch'] == epoch
        assert loaded_checkpoint['loss'] == loss
        assert 'model_state_dict' in loaded_checkpoint
        assert 'optimizer_state_dict' in loaded_checkpoint


@pytest.mark.integration
class TestMemoryManagement:
    """Test memory management during training."""
    
    @pytest.mark.gpu
    def test_gpu_memory_cleanup(self, device):
        """Test GPU memory is properly cleaned up."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large tensors
        large_tensor = torch.randn(1000, 1000, device=device)
        model = nn.Linear(1000, 1000).to(device)
        
        # Use memory
        output = model(large_tensor)
        loss = output.sum()
        loss.backward()
        
        # Clean up
        del large_tensor, model, output, loss
        torch.cuda.empty_cache()
        
        # Check memory was freed
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory + 1e6  # Allow small overhead
        
    def test_gradient_accumulation(self, simple_graph, device):
        """Test gradient accumulation for large effective batch sizes."""
        model = nn.Linear(16, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        accumulation_steps = 4
        effective_batch_size = 0
        
        optimizer.zero_grad()
        
        for step in range(accumulation_steps):
            batch = simple_graph.to(device)
            targets = torch.randn(batch.num_nodes, 1, device=device)
            
            outputs = model(batch.x)
            loss = criterion(outputs, targets)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            effective_batch_size += batch.num_nodes
        
        # Update after accumulation
        optimizer.step()
        
        # Verify gradients were accumulated
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
        
        assert effective_batch_size == accumulation_steps * simple_graph.num_nodes


@pytest.mark.integration
class TestDistributedTraining:
    """Test distributed training setup (single process for testing)."""
    
    def test_model_parallel_compatibility(self, device):
        """Test model can be made compatible with model parallelism."""
        # Create a model that could be split across devices
        class SplittableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(16, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        model = SplittableModel().to(device)
        
        # Test that model parts can be moved independently
        encoder_device = next(model.encoder.parameters()).device
        decoder_device = next(model.decoder.parameters()).device
        
        assert encoder_device == device
        assert decoder_device == device
        
    def test_data_parallel_wrapper(self, simple_graph, device):
        """Test DataParallel wrapper (single GPU for testing)."""
        model = nn.Linear(16, 1)
        
        # Wrap in DataParallel (will use single GPU in testing)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # Test forward pass
        batch = simple_graph.to(device)
        outputs = model(batch.x)
        
        assert outputs.shape == (batch.num_nodes, 1)
        assert outputs.device.type == device.type


@pytest.mark.integration
class TestHyperparameterOptimization:
    """Test hyperparameter optimization integration."""
    
    def test_learning_rate_scheduling(self, simple_graph, device):
        """Test learning rate scheduling during training."""
        model = nn.Linear(16, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate training steps
        for epoch in range(5):
            # Training step
            batch = simple_graph.to(device)
            targets = torch.randn(batch.num_nodes, 1, device=device)
            
            optimizer.zero_grad()
            outputs = model(batch.x)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Scheduler step
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Check learning rate changes
            if epoch < 2:
                assert current_lr == initial_lr
            elif epoch < 4:
                assert current_lr == initial_lr * 0.5
            else:
                assert current_lr == initial_lr * 0.25
                
    def test_early_stopping_simulation(self, simple_graph, device):
        """Test early stopping logic."""
        model = nn.Linear(16, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # Early stopping parameters
        patience = 3
        min_delta = 0.001
        best_loss = float('inf')
        patience_counter = 0
        
        # Simulate training with early stopping
        losses = [1.0, 0.8, 0.6, 0.59, 0.58, 0.579, 0.578]  # Mock losses
        
        for epoch, loss_val in enumerate(losses):
            # Simulate validation
            if loss_val < best_loss - min_delta:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check early stopping condition
            if patience_counter >= patience:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = len(losses)
        
        # Should stop at epoch 6 (after 3 epochs without improvement)
        assert stopped_epoch == 6
        assert patience_counter >= patience
"""Performance benchmarks and profiling tests."""

import pytest
import torch
import time
import psutil
import gc
from memory_profiler import profile
from unittest.mock import Mock
import numpy as np


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmark memory usage of different components."""
    
    def test_graph_memory_scaling(self, benchmark_config):
        """Test memory scaling with graph size."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        graph_sizes = [100, 500, 1000, 2000, 5000]
        memory_usage = []
        
        for num_nodes in graph_sizes:
            # Create graph
            num_edges = num_nodes * 10  # Average degree of 10
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            x = torch.randn(num_nodes, 64)
            edge_attr = torch.randn(num_edges, 16)
            
            # Measure memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                
                # Move to GPU
                edge_index = edge_index.cuda()
                x = x.cuda()
                edge_attr = edge_attr.cuda()
                
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_memory / 1024**2)  # MB
                
                # Cleanup
                del edge_index, x, edge_attr
                torch.cuda.empty_cache()
            else:
                # CPU memory measurement
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024**2)  # MB
        
        # Test memory scaling is reasonable
        assert len(memory_usage) == len(graph_sizes)
        
        # Memory should generally increase with graph size
        for i in range(1, len(memory_usage)):
            # Allow some tolerance for measurement noise
            assert memory_usage[i] >= memory_usage[i-1] * 0.8
    
    def test_batch_memory_scaling(self, simple_graph):
        """Test memory scaling with batch size."""
        batch_sizes = [1, 4, 8, 16, 32]
        memory_usage = []
        
        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Create batch
            from torch_geometric.data import Batch
            graphs = [simple_graph for _ in range(batch_size)]
            batch = Batch.from_data_list(graphs)
            
            if torch.cuda.is_available():
                batch = batch.cuda()
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_memory / 1024**2)  # MB
                
                del batch
                torch.cuda.empty_cache()
            else:
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024**2)  # MB
        
        # Test linear scaling with batch size
        assert len(memory_usage) == len(batch_sizes)
        
        # Memory should scale roughly linearly
        if len(memory_usage) >= 2:
            scaling_factor = memory_usage[-1] / memory_usage[0]
            expected_scaling = batch_sizes[-1] / batch_sizes[0]
            # Allow factor of 2 tolerance
            assert scaling_factor <= expected_scaling * 2
    
    @pytest.mark.gpu
    def test_gpu_memory_efficiency(self, large_graph, device):
        """Test GPU memory efficiency with large graphs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        graph = large_graph.to(device)
        
        # Test mixed precision memory savings
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Standard precision
        with torch.no_grad():
            standard_output = torch.randn_like(graph.x, dtype=torch.float32)
        
        standard_memory = torch.cuda.max_memory_allocated()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Half precision
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                half_output = torch.randn_like(graph.x, dtype=torch.float16)
        
        half_memory = torch.cuda.max_memory_allocated()
        
        # Half precision should use less memory
        assert half_memory <= standard_memory
        memory_savings = (standard_memory - half_memory) / standard_memory
        assert memory_savings >= 0  # At least no increase


@pytest.mark.benchmark
class TestSpeedBenchmarks:
    """Benchmark execution speed of different components."""
    
    def test_forward_pass_speed(self, medium_graph, device):
        """Benchmark forward pass speed."""
        graph = medium_graph.to(device)
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(graph.x)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output = model(graph.x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        throughput = graph.num_nodes / avg_time  # nodes per second
        
        # Should process at least 1000 nodes per second
        assert throughput > 1000
        assert avg_time < 1.0  # Less than 1 second per forward pass
        
    def test_training_step_speed(self, batch_graphs, device):
        """Benchmark training step speed."""
        batch = batch_graphs.to(device)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            output = model(batch.x)
            targets = torch.randn_like(output)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        num_iterations = 50
        start_time = time.time()
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(batch.x)
            targets = torch.randn_like(output)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        
        # Training step should be reasonably fast
        assert avg_time < 2.0  # Less than 2 seconds per training step
        
    @pytest.mark.slow
    def test_epoch_speed(self, simple_graph, device):
        """Benchmark full epoch speed."""
        from torch_geometric.data import DataLoader
        
        # Create dataset
        dataset = [simple_graph for _ in range(100)]
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Benchmark one epoch
        start_time = time.time()
        
        model.train()
        for batch in dataloader:
            batch = batch.to(device)
            targets = torch.randn(batch.num_nodes, 1, device=device)
            
            optimizer.zero_grad()
            output = model(batch.x)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Epoch should complete in reasonable time
        assert epoch_time < 60  # Less than 1 minute
        
        # Calculate throughput
        total_nodes = len(dataset) * simple_graph.num_nodes
        throughput = total_nodes / epoch_time
        
        assert throughput > 100  # At least 100 nodes per second


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test scalability with different problem sizes."""
    
    @pytest.mark.parametrize("num_nodes", [100, 500, 1000, 2000])
    def test_node_scaling(self, num_nodes, device):
        """Test performance scaling with number of nodes."""
        # Create graph
        num_edges = num_nodes * 5  # Sparse graph
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 32).to(device)
        edge_index = edge_index.to(device)
        
        # Simple GNN layer
        from torch_geometric.nn import GCNConv
        layer = GCNConv(32, 32).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = layer(x, edge_index)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(10):
            with torch.no_grad():
                output = layer(x, edge_index)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Time should scale reasonably with graph size
        # For small graphs, should be very fast
        if num_nodes <= 500:
            assert avg_time < 0.1  # 100ms
        elif num_nodes <= 1000:
            assert avg_time < 0.5  # 500ms
        else:
            assert avg_time < 2.0  # 2s
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_batch_scaling(self, simple_graph, batch_size, device):
        """Test performance scaling with batch size."""
        from torch_geometric.data import Batch
        
        # Create batch
        graphs = [simple_graph for _ in range(batch_size)]
        batch = Batch.from_data_list(graphs).to(device)
        
        # Simple model
        model = torch.nn.Linear(16, 1).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(batch.x)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(20):
            with torch.no_grad():
                output = model(batch.x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20
        
        # Should scale reasonably with batch size
        assert avg_time < batch_size * 0.01  # Linear scaling with small constant


@pytest.mark.benchmark
class TestMemoryProfiling:
    """Memory profiling tests."""
    
    @pytest.mark.slow
    def test_memory_leak_detection(self, simple_graph, device):
        """Test for memory leaks during training."""
        model = torch.nn.Linear(16, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Measure initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = psutil.Process().memory_info().rss
        
        # Run training for many iterations
        for i in range(100):
            batch = simple_graph.to(device)
            targets = torch.randn(batch.num_nodes, 1, device=device)
            
            optimizer.zero_grad()
            output = model(batch.x)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            # Periodic cleanup
            if i % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Measure final memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
        else:
            gc.collect()
            final_memory = psutil.Process().memory_info().rss
        
        # Memory increase should be minimal
        memory_increase = final_memory - initial_memory
        
        # Allow some memory increase but not excessive
        if torch.cuda.is_available():
            max_increase = 100 * 1024 * 1024  # 100MB
        else:
            max_increase = 500 * 1024 * 1024  # 500MB (CPU more variable)
        
        assert memory_increase < max_increase, f"Memory leak detected: {memory_increase / 1024**2:.1f}MB increase"
    
    def test_gradient_memory_efficiency(self, medium_graph, device):
        """Test gradient computation memory efficiency."""
        graph = medium_graph.to(device)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to(device)
        
        # Test without gradient checkpointing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        output = model(graph.x)
        loss = output.sum()
        loss.backward()
        
        if torch.cuda.is_available():
            memory_without_checkpointing = torch.cuda.max_memory_allocated()
        
        # Clear gradients and memory
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Test with gradient checkpointing (simulated)
        def checkpointed_forward(x):
            h1 = model[0](x)  # Linear
            h1 = model[1](h1)  # ReLU
            # Checkpoint here in real implementation
            h2 = model[2](h1)  # Linear
            return h2
        
        output = checkpointed_forward(graph.x)
        loss = output.sum()
        loss.backward()
        
        if torch.cuda.is_available():
            memory_with_checkpointing = torch.cuda.max_memory_allocated()
            
            # Checkpointing should not increase memory significantly
            # (In this simple case, may not show difference, but structure is tested)
            assert memory_with_checkpointing <= memory_without_checkpointing * 1.2


@pytest.mark.benchmark  
class TestConcurrencyBenchmarks:
    """Test concurrent execution performance."""
    
    def test_dataloader_workers(self, simple_graph):
        """Test performance with different numbers of DataLoader workers."""
        from torch_geometric.data import DataLoader
        
        dataset = [simple_graph for _ in range(100)]
        
        worker_counts = [0, 1, 2, 4]
        loading_times = []
        
        for num_workers in worker_counts:
            dataloader = DataLoader(
                dataset, 
                batch_size=8, 
                shuffle=True, 
                num_workers=num_workers
            )
            
            start_time = time.time()
            
            # Iterate through all batches
            for batch in dataloader:
                pass  # Just loading, no processing
            
            end_time = time.time()
            loading_time = end_time - start_time
            loading_times.append(loading_time)
        
        # Multi-worker loading should not be significantly slower
        # (May not be faster due to overhead in small datasets)
        assert all(t < 10.0 for t in loading_times)  # All under 10 seconds
        
    @pytest.mark.gpu
    def test_multi_gpu_compatibility(self, simple_graph, device):
        """Test multi-GPU compatibility (single GPU test)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        graph = simple_graph.to(device)
        
        # Test DataParallel wrapper
        model = torch.nn.Linear(16, 1)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        model = model.to(device)
        
        # Test forward pass
        output = model(graph.x)
        
        assert output.shape == (graph.num_nodes, 1)
        assert output.device.type == device.type
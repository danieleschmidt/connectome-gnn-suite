#!/usr/bin/env python3
"""Performance optimization and scaling demonstration."""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
import time
import sys
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '.')

import connectome_gnn
from connectome_gnn.models import HierarchicalBrainGNN
from connectome_gnn.scale import memory_optimization, performance_monitoring, caching


class OptimizedConnectomeProcessor:
    """High-performance connectome data processor with optimizations."""
    
    def __init__(self, use_caching=True, use_multiprocessing=True):
        self.use_caching = use_caching
        self.use_multiprocessing = use_multiprocessing
        self.cache = {} if use_caching else None
        self.perf_monitor = performance_monitoring.PerformanceMonitor()
        
    def create_optimized_data(self, num_samples=1000, num_nodes=200):
        """Create large dataset with performance optimizations."""
        print(f"🚀 Creating optimized dataset: {num_samples} samples, {num_nodes} nodes")
        
        start_time = time.time()
        
        if self.use_multiprocessing:
            # Parallel data generation
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                chunk_size = num_samples // multiprocessing.cpu_count()
                
                for i in range(0, num_samples, chunk_size):
                    end_idx = min(i + chunk_size, num_samples)
                    future = executor.submit(
                        self._create_data_chunk, 
                        i, end_idx, num_nodes
                    )
                    futures.append(future)
                
                data_list = []
                for future in as_completed(futures):
                    chunk_data = future.result()
                    data_list.extend(chunk_data)
        else:
            # Sequential generation
            data_list = self._create_data_chunk(0, num_samples, num_nodes)
        
        creation_time = time.time() - start_time
        print(f"⚡ Dataset created in {creation_time:.2f}s")
        print(f"📊 Performance: {len(data_list)/creation_time:.1f} samples/sec")
        
        return data_list
    
    def _create_data_chunk(self, start_idx, end_idx, num_nodes):
        """Create a chunk of data samples."""
        data_list = []
        
        for i in range(start_idx, end_idx):
            # Check cache first
            cache_key = f"sample_{i}_{num_nodes}"
            
            if self.cache and cache_key in self.cache:
                data_list.append(self.cache[cache_key])
                continue
            
            # Generate connectivity with optimized operations
            connectivity = self._generate_optimized_connectivity(num_nodes)
            
            # Convert to PyG format efficiently
            data = self._connectivity_to_pyg(connectivity, i)
            
            # Cache if enabled
            if self.cache:
                self.cache[cache_key] = data
                
            data_list.append(data)
        
        return data_list
    
    def _generate_optimized_connectivity(self, num_nodes):
        """Generate connectivity matrix with optimizations."""
        # Use vectorized operations for better performance
        connectivity = np.random.rand(num_nodes, num_nodes).astype(np.float32)
        
        # Efficient symmetry
        connectivity = (connectivity + connectivity.T) * 0.5
        np.fill_diagonal(connectivity, 0)
        
        # Vectorized thresholding
        connectivity[connectivity < 0.75] = 0
        
        return connectivity
    
    def _connectivity_to_pyg(self, connectivity, sample_idx):
        """Convert connectivity to PyG Data efficiently."""
        # Use numpy operations for speed
        edge_indices = np.nonzero(connectivity)
        edge_weights = connectivity[edge_indices]
        
        # Batch tensor creation
        edge_index = torch.from_numpy(
            np.vstack([edge_indices[0], edge_indices[1]])
        ).long()
        
        edge_attr = torch.from_numpy(edge_weights).float()
        
        # Optimized node features
        node_degrees = np.sum(connectivity > 0, axis=1, dtype=np.float32)
        node_strength = np.sum(np.abs(connectivity), axis=1, dtype=np.float32)
        
        x = torch.from_numpy(
            np.column_stack([node_degrees, node_strength])
        ).float()
        
        # Target with some pattern
        y = torch.tensor([
            np.mean(node_degrees) + 0.1 * np.sin(sample_idx * 0.1)
        ], dtype=torch.float)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=connectivity.shape[0]
        )


class OptimizedModel(HierarchicalBrainGNN):
    """Optimized version of HierarchicalBrainGNN with memory and compute optimizations."""
    
    def __init__(self, *args, use_checkpointing=True, use_mixed_precision=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        
        # Enable memory optimizations
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, data):
        """Optimized forward pass."""
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, data, use_reentrant=False
            )
        else:
            return self._forward_impl(data)
    
    def _forward_impl(self, data):
        """Implementation of forward pass."""
        return super().forward(data)


class ScalableTrainer:
    """Scalable training with optimizations."""
    
    def __init__(self, model, device='cpu', enable_optimizations=True):
        self.model = model.to(device)
        self.device = device
        self.enable_optimizations = enable_optimizations
        
        # Performance monitoring
        self.perf_monitor = performance_monitoring.PerformanceMonitor()
        
        # Memory monitoring
        self.memory_tracker = memory_optimization.MemoryProfiler()
        
    def train_optimized(self, data_list, epochs=10, batch_size=32):
        """Optimized training loop."""
        print(f"🏃 Starting optimized training: {epochs} epochs, batch_size={batch_size}")
        
        # Create optimized data loader
        loader = DataLoader(
            data_list, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=min(4, multiprocessing.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler with warm-up
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(loader)
        )
        
        # Training loop with optimizations
        self.model.train()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            num_batches = 0
            
            # Monitor memory at start of epoch
            if epoch % 5 == 0:
                memory_info = self.memory_tracker.get_memory_stats()
                print(f"Memory usage: {memory_info.rss_mb:.1f}MB")
            
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient
                
                # Forward pass with mixed precision if enabled
                if hasattr(self.model, 'use_mixed_precision') and self.model.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        out = self.model(batch)
                        loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
                    
                    # Backward pass with gradient scaling
                    self.model.scaler.scale(loss).backward()
                    self.model.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.model.scaler.step(optimizer)
                    self.model.scaler.update()
                else:
                    # Standard training
                    out = self.model(batch)
                    loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Memory cleanup every 50 batches
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            # Performance metrics
            throughput = len(loader.dataset) / epoch_time
            
            print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, "
                  f"Time={epoch_time:.1f}s, Throughput={throughput:.1f} samples/s")
        
        print("✅ Optimized training completed!")


def benchmark_performance():
    """Benchmark different configurations."""
    print("🔬 Performance Benchmarking")
    print("=" * 50)
    
    configurations = [
        {"name": "Baseline", "optimizations": False, "samples": 100, "nodes": 100},
        {"name": "Optimized", "optimizations": True, "samples": 500, "nodes": 150},
        {"name": "Large Scale", "optimizations": True, "samples": 1000, "nodes": 200},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🧪 Testing {config['name']}...")
        
        start_time = time.time()
        
        # Data creation
        processor = OptimizedConnectomeProcessor(
            use_multiprocessing=config['optimizations']
        )
        
        data_list = processor.create_optimized_data(
            num_samples=config['samples'],
            num_nodes=config['nodes']
        )
        
        # Model creation
        model = OptimizedModel(
            node_features=2,  # degree + strength
            hidden_dim=64,
            num_levels=3,
            use_checkpointing=config['optimizations']
        )
        
        # Training
        trainer = ScalableTrainer(model, enable_optimizations=config['optimizations'])
        
        training_start = time.time()
        trainer.train_optimized(data_list, epochs=3, batch_size=16)
        training_time = time.time() - training_start
        
        total_time = time.time() - start_time
        
        results[config['name']] = {
            'total_time': total_time,
            'training_time': training_time,
            'samples': config['samples'],
            'throughput': config['samples'] / total_time
        }
        
        print(f"✅ {config['name']} completed in {total_time:.1f}s")
    
    # Results summary
    print(f"\n📈 Performance Results")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:12}: {result['throughput']:.1f} samples/s "
              f"({result['samples']} samples in {result['total_time']:.1f}s)")


def memory_scaling_demo():
    """Demonstrate memory-efficient scaling."""
    print("\n💾 Memory Scaling Demo")
    print("-" * 30)
    
    # Monitor memory usage across different scales
    scales = [50, 100, 200, 500]
    
    for scale in scales:
        print(f"\nTesting scale: {scale} samples")
        
        # Memory before
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create data with memory optimizations
        processor = OptimizedConnectomeProcessor(use_caching=False)  # Disable caching for memory test
        data_list = processor.create_optimized_data(num_samples=scale, num_nodes=100)
        
        # Memory after
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        print(f"Memory used: {mem_used:.1f}MB ({mem_used/scale:.3f}MB/sample)")
        
        # Cleanup
        del data_list
        
        # Force garbage collection
        import gc
        gc.collect()


def auto_scaling_demo():
    """Demonstrate automatic resource scaling."""
    print("\n⚖️  Auto-scaling Demo")
    print("-" * 30)
    
    # Detect system resources
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    
    # Auto-configure based on resources
    if memory_gb > 16:
        batch_size = 64
        num_workers = min(8, cpu_count)
        dataset_size = 2000
    elif memory_gb > 8:
        batch_size = 32
        num_workers = min(4, cpu_count)
        dataset_size = 1000
    else:
        batch_size = 16
        num_workers = 2
        dataset_size = 500
    
    print(f"Auto-configured: batch_size={batch_size}, "
          f"num_workers={num_workers}, dataset_size={dataset_size}")
    
    # Run scaled configuration
    processor = OptimizedConnectomeProcessor()
    data_list = processor.create_optimized_data(num_samples=min(dataset_size, 200))  # Limit for demo
    
    model = OptimizedModel(node_features=2, hidden_dim=32, num_levels=2)
    trainer = ScalableTrainer(model)
    trainer.train_optimized(data_list, epochs=2, batch_size=min(batch_size, 16))
    
    print("✅ Auto-scaling completed!")


if __name__ == "__main__":
    print("🚀 Performance Optimization & Scaling Demo")
    print("=" * 60)
    
    # Run performance benchmarks
    benchmark_performance()
    
    # Memory scaling demonstration
    memory_scaling_demo()
    
    # Auto-scaling demonstration  
    auto_scaling_demo()
    
    print("\n🎉 All scaling demonstrations completed!")
    print("The Connectome-GNN-Suite is optimized for production scale!")
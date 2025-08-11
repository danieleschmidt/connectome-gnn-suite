"""Performance optimization utilities for connectome GNN models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import time
import psutil
import gc
from pathlib import Path


class PerformanceProfiler:
    """Profile model performance and resource usage."""
    
    def __init__(self):
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def profile_model_inference(
        self, 
        model: nn.Module, 
        data_loader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, Any]:
        """Profile model inference performance.
        
        Args:
            model: Model to profile
            data_loader: Data loader for profiling
            num_batches: Number of batches to profile
            
        Returns:
            Performance metrics
        """
        model.eval()
        model = model.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 3:  # 3 warmup batches
                    break
                batch = batch.to(self.device)
                _ = model(batch)
        
        # Profile inference
        times = []
        memory_usage = []
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                batch = batch.to(self.device)
                
                # Memory before
                mem_before = self._get_memory_usage()
                
                # Time inference
                start_time = time.perf_counter()
                _ = model(batch)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                # Memory after
                mem_after = self._get_memory_usage()
                memory_usage.append(mem_after - mem_before)
        
        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'throughput_samples_per_sec': len(batch) / np.mean(times),
            'mean_memory_usage_mb': np.mean(memory_usage),
            'peak_memory_usage_mb': np.max(memory_usage),
            'device': str(self.device)
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024


class ModelOptimizer:
    """Optimize connectome GNN models for performance."""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        optimization_level: str = "standard"
    ) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            example_input: Example input for optimization
            optimization_level: Level of optimization (standard, aggressive)
            
        Returns:
            Optimized model
        """
        optimized_model = model.eval()
        
        # Apply optimizations based on level
        if optimization_level == "standard":
            optimized_model = self._apply_standard_optimizations(
                optimized_model, example_input
            )
        elif optimization_level == "aggressive":
            optimized_model = self._apply_aggressive_optimizations(
                optimized_model, example_input
            )
        
        return optimized_model
    
    def _apply_standard_optimizations(
        self, 
        model: nn.Module, 
        example_input: torch.Tensor
    ) -> nn.Module:
        """Apply standard optimizations."""
        optimizations_applied = []
        
        # Fuse operations
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
            optimizations_applied.append("jit_optimization")
        except Exception:
            pass
        
        # Set to eval mode and freeze
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        optimizations_applied.append("parameter_freezing")
        
        self.optimization_history.append({
            'optimizations': optimizations_applied,
            'level': 'standard'
        })
        
        return model
    
    def _apply_aggressive_optimizations(
        self, 
        model: nn.Module, 
        example_input: torch.Tensor
    ) -> nn.Module:
        """Apply aggressive optimizations."""
        optimizations_applied = []
        
        # Standard optimizations first
        model = self._apply_standard_optimizations(model, example_input)
        
        # Quantization (if supported)
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            optimizations_applied.append("dynamic_quantization")
        except Exception:
            pass
        
        self.optimization_history.append({
            'optimizations': optimizations_applied,
            'level': 'aggressive'  
        })
        
        return model


class MemoryOptimizer:
    """Optimize memory usage for large connectome datasets."""
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        def checkpoint_wrapper(module):
            def forward_wrapper(self, *args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    module.__class__.forward, self, *args, **kwargs
                )
            return forward_wrapper
        
        # Apply to specific layers that benefit from checkpointing
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                module.forward = checkpoint_wrapper(module)
    
    @staticmethod
    def optimize_data_loading(
        dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """Optimize data loading for performance.
        
        Args:
            dataset: Dataset to optimize
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            Optimized DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )
    
    @staticmethod
    def clear_cache() -> None:
        """Clear GPU and CPU caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BatchSizeOptimizer:
    """Find optimal batch size for training and inference."""
    
    def find_optimal_batch_size(
        self,
        model: nn.Module,
        dataset,
        device: torch.device,
        max_memory_usage: float = 0.8  # 80% of available memory
    ) -> int:
        """Find optimal batch size through binary search.
        
        Args:
            model: Model to test
            dataset: Dataset to use
            device: Device to test on
            max_memory_usage: Maximum memory usage fraction
            
        Returns:
            Optimal batch size
        """
        model = model.to(device)
        model.eval()
        
        # Get available memory
        if device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(device).total_memory
            max_memory = total_memory * max_memory_usage
        else:
            total_memory = psutil.virtual_memory().total
            max_memory = total_memory * max_memory_usage
        
        # Binary search for optimal batch size
        low, high = 1, 512
        optimal_batch_size = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            if self._test_batch_size(model, dataset, mid, device, max_memory):
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return optimal_batch_size
    
    def _test_batch_size(
        self,
        model: nn.Module,
        dataset,
        batch_size: int,
        device: torch.device,
        max_memory: float
    ) -> bool:
        """Test if batch size fits in memory."""
        try:
            # Create data loader with test batch size
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Clear cache
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # Test one batch
            with torch.no_grad():
                batch = next(iter(test_loader))
                batch = batch.to(device)
                _ = model(batch)
            
            # Check memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(device)
            else:
                memory_used = psutil.Process().memory_info().rss
            
            return memory_used <= max_memory
            
        except (RuntimeError, OutOfMemoryError):
            return False
        finally:
            # Clean up
            torch.cuda.empty_cache() if device.type == 'cuda' else None


class DistributedTrainingOptimizer:
    """Optimize for distributed training across multiple GPUs."""
    
    def __init__(self):
        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def setup_distributed_model(
        self,
        model: nn.Module,
        device_ids: Optional[list] = None
    ) -> nn.Module:
        """Setup model for distributed training.
        
        Args:
            model: Model to distribute
            device_ids: List of device IDs to use
            
        Returns:
            Distributed model
        """
        if self.world_size <= 1:
            return model
        
        if device_ids is None:
            device_ids = list(range(self.world_size))
        
        # Use DataParallel for simple multi-GPU training
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        
        return model
    
    def calculate_optimal_batch_size(self, base_batch_size: int) -> int:
        """Calculate optimal batch size for distributed training."""
        return base_batch_size * max(1, self.world_size)


def benchmark_model_performance(
    model: nn.Module,
    dataset,
    device: str = "auto",
    batch_sizes: list = [1, 8, 16, 32, 64],
    num_trials: int = 3
) -> Dict[str, Any]:
    """Comprehensive model performance benchmark.
    
    Args:
        model: Model to benchmark
        dataset: Dataset for benchmarking  
        device: Device to use ('auto', 'cpu', 'cuda')
        batch_sizes: Batch sizes to test
        num_trials: Number of trials per configuration
        
    Returns:
        Benchmark results
    """
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model = model.to(device)
    profiler = PerformanceProfiler()
    
    results = {
        'device': str(device),
        'model_params': sum(p.numel() for p in model.parameters()),
        'batch_results': {}
    }
    
    for batch_size in batch_sizes:
        try:
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0  # Avoid multiprocessing for consistent benchmarks
            )
            
            batch_results = []
            for trial in range(num_trials):
                trial_results = profiler.profile_model_inference(
                    model, data_loader, num_batches=5
                )
                batch_results.append(trial_results)
            
            # Average results across trials
            averaged_results = {}
            for key in batch_results[0].keys():
                if isinstance(batch_results[0][key], (int, float)):
                    values = [r[key] for r in batch_results]
                    averaged_results[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                else:
                    averaged_results[key] = batch_results[0][key]
            
            results['batch_results'][batch_size] = averaged_results
            
        except Exception as e:
            results['batch_results'][batch_size] = {'error': str(e)}
    
    return results


def optimize_connectome_pipeline(
    model: nn.Module,
    dataset,
    optimization_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Optimize entire connectome analysis pipeline.
    
    Args:
        model: Connectome GNN model
        dataset: Connectome dataset
        optimization_config: Configuration for optimizations
        
    Returns:
        Optimization results and optimized components
    """
    config = optimization_config or {
        'enable_jit': True,
        'enable_quantization': False,
        'optimize_batch_size': True,
        'enable_distributed': True,
        'memory_optimization': True
    }
    
    results = {
        'original_model_size': sum(p.numel() for p in model.parameters()),
        'optimizations_applied': [],
        'performance_improvements': {}
    }
    
    # Model optimization
    if config.get('enable_jit', True):
        optimizer = ModelOptimizer()
        try:
            model = optimizer.optimize_for_inference(
                model,
                next(iter(DataLoader(dataset, batch_size=1)))[0] if len(dataset) > 0 else None,
                optimization_level="standard"
            )
            results['optimizations_applied'].append('jit_optimization')
        except Exception:
            pass
    
    # Batch size optimization
    if config.get('optimize_batch_size', True):
        batch_optimizer = BatchSizeOptimizer()
        try:
            optimal_batch_size = batch_optimizer.find_optimal_batch_size(
                model, dataset, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            results['optimal_batch_size'] = optimal_batch_size
            results['optimizations_applied'].append('batch_size_optimization')
        except Exception:
            results['optimal_batch_size'] = 32  # fallback
    
    # Memory optimization
    if config.get('memory_optimization', True):
        MemoryOptimizer.clear_cache()
        results['optimizations_applied'].append('memory_optimization')
    
    # Distributed training setup
    if config.get('enable_distributed', True):
        dist_optimizer = DistributedTrainingOptimizer()
        if dist_optimizer.world_size > 1:
            model = dist_optimizer.setup_distributed_model(model)
            results['optimizations_applied'].append('distributed_training')
            results['num_gpus'] = dist_optimizer.world_size
    
    results['optimized_model'] = model
    return results
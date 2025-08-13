"""Memory optimization utilities for large-scale training and inference."""

import gc
import psutil
import os
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from contextlib import contextmanager

from ..core.utils import safe_import, Timer
from ..robust.logging_config import get_logger


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float  # Available system memory
    percent_used: float  # Percentage of system memory used
    gpu_allocated_mb: float = 0.0  # GPU memory allocated
    gpu_cached_mb: float = 0.0  # GPU memory cached


class MemoryProfiler:
    """Profiles memory usage during operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("memory_profiler")
        self.torch = safe_import('torch')
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            available_mb=system_memory.available / 1024 / 1024,
            percent_used=system_memory.percent
        )
        
        # GPU memory if available
        if self.torch and self.torch.cuda.is_available():
            stats.gpu_allocated_mb = self.torch.cuda.memory_allocated() / 1024 / 1024
            stats.gpu_cached_mb = self.torch.cuda.memory_reserved() / 1024 / 1024
        
        return stats
    
    def take_snapshot(self, label: str = None) -> MemoryStats:
        """Take a memory snapshot."""
        stats = self.get_memory_stats()
        
        snapshot = {
            'timestamp': time.time(),
            'label': label or f"snapshot_{len(self.snapshots)}",
            'stats': stats
        }
        
        self.snapshots.append(snapshot)
        
        self.logger.debug(f"Memory snapshot '{snapshot['label']}': "
                         f"RSS={stats.rss_mb:.1f}MB, "
                         f"GPU={stats.gpu_allocated_mb:.1f}MB")
        
        return stats
    
    def compare_snapshots(self, start_label: str, end_label: str) -> Dict[str, float]:
        """Compare two memory snapshots."""
        start_snapshot = None
        end_snapshot = None
        
        for snapshot in self.snapshots:
            if snapshot['label'] == start_label:
                start_snapshot = snapshot
            elif snapshot['label'] == end_label:
                end_snapshot = snapshot
        
        if not start_snapshot or not end_snapshot:
            raise ValueError("Snapshot labels not found")
        
        start_stats = start_snapshot['stats']
        end_stats = end_snapshot['stats']
        
        return {
            'rss_diff_mb': end_stats.rss_mb - start_stats.rss_mb,
            'gpu_diff_mb': end_stats.gpu_allocated_mb - start_stats.gpu_allocated_mb,
            'time_diff_s': end_snapshot['timestamp'] - start_snapshot['timestamp']
        }
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage from all snapshots."""
        if not self.snapshots:
            return {}
        
        peak_rss = max(s['stats'].rss_mb for s in self.snapshots)
        peak_gpu = max(s['stats'].gpu_allocated_mb for s in self.snapshots)
        
        return {
            'peak_rss_mb': peak_rss,
            'peak_gpu_mb': peak_gpu
        }
    
    def save_profile(self, filepath: str):
        """Save memory profile to file."""
        profile_data = {
            'snapshots': self.snapshots,
            'peak_memory': self.get_peak_memory(),
            'total_snapshots': len(self.snapshots)
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        self.logger.info(f"Memory profile saved to {filepath}")
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling an operation."""
        start_label = f"{operation_name}_start"
        end_label = f"{operation_name}_end"
        
        self.take_snapshot(start_label)
        
        try:
            yield
        finally:
            self.take_snapshot(end_label)
            
            try:
                diff = self.compare_snapshots(start_label, end_label)
                self.logger.info(f"Operation '{operation_name}' memory change: "
                               f"RSS={diff['rss_diff_mb']:+.1f}MB, "
                               f"GPU={diff['gpu_diff_mb']:+.1f}MB, "
                               f"Time={diff['time_diff_s']:.2f}s")
            except Exception as e:
                self.logger.warning(f"Failed to compute memory diff: {e}")


class MemoryOptimizer:
    """Optimizes memory usage for training and inference."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("memory_optimizer")
        self.torch = safe_import('torch')
        self.profiler = MemoryProfiler(logger)
        
        # Optimization settings
        self.auto_mixed_precision = False
        self.gradient_checkpointing = False
        self.memory_efficient_attention = False
        
    def enable_auto_mixed_precision(self, enabled: bool = True):
        """Enable automatic mixed precision training."""
        self.auto_mixed_precision = enabled
        
        if enabled and self.torch:
            self.logger.info("Enabled automatic mixed precision training")
        elif enabled:
            self.logger.warning("Cannot enable AMP: PyTorch not available")
    
    def enable_gradient_checkpointing(self, model: Any, enabled: bool = True):
        """Enable gradient checkpointing for memory efficiency."""
        if not self.torch:
            self.logger.warning("Cannot enable gradient checkpointing: PyTorch not available")
            return
        
        self.gradient_checkpointing = enabled
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            if enabled:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            
            self.logger.info(f"Gradient checkpointing {'enabled' if enabled else 'disabled'}")
        else:
            self.logger.warning("Model does not support gradient checkpointing")
    
    def optimize_model_memory(self, model: Any) -> Dict[str, Any]:
        """Apply memory optimizations to a model."""
        optimizations = {}
        
        if not self.torch:
            self.logger.warning("Cannot optimize model: PyTorch not available")
            return optimizations
        
        # Convert to half precision if beneficial
        if self.torch.cuda.is_available():
            try:
                original_memory = self.torch.cuda.memory_allocated()
                
                # Try converting to half precision
                model_half = model.half()
                
                new_memory = self.torch.cuda.memory_allocated()
                memory_saved = (original_memory - new_memory) / 1024 / 1024
                
                if memory_saved > 0:
                    optimizations['half_precision'] = {
                        'enabled': True,
                        'memory_saved_mb': memory_saved
                    }
                    self.logger.info(f"Half precision conversion saved {memory_saved:.1f}MB")
                else:
                    # Revert if no benefit
                    model = model.float()
                    optimizations['half_precision'] = {'enabled': False}
                    
            except Exception as e:
                self.logger.warning(f"Failed to apply half precision: {e}")
                optimizations['half_precision'] = {'enabled': False, 'error': str(e)}
        
        # Enable memory efficient operations
        if hasattr(self.torch.backends, 'cudnn'):
            self.torch.backends.cudnn.benchmark = True
            optimizations['cudnn_benchmark'] = True
        
        return optimizations
    
    def clear_cache(self):
        """Clear various caches to free memory."""
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch cache
        if self.torch and self.torch.cuda.is_available():
            gpu_before = self.torch.cuda.memory_allocated() / 1024 / 1024
            self.torch.cuda.empty_cache()
            gpu_after = self.torch.cuda.memory_allocated() / 1024 / 1024
            gpu_freed = gpu_before - gpu_after
            
            self.logger.info(f"Cache cleared: {collected} objects collected, "
                           f"{gpu_freed:.1f}MB GPU memory freed")
        else:
            self.logger.info(f"Cache cleared: {collected} objects collected")
    
    def monitor_memory_usage(self, threshold_mb: float = 1000) -> bool:
        """Monitor memory usage and return True if threshold exceeded."""
        stats = self.profiler.get_memory_stats()
        
        if stats.rss_mb > threshold_mb:
            self.logger.warning(f"Memory usage ({stats.rss_mb:.1f}MB) exceeds threshold ({threshold_mb}MB)")
            return True
        
        return False
    
    def auto_manage_memory(self, model: Any = None, threshold_mb: float = 1000):
        """Automatically manage memory when usage is high."""
        if self.monitor_memory_usage(threshold_mb):
            self.logger.info("High memory usage detected, applying optimizations")
            
            # Clear caches
            self.clear_cache()
            
            # Apply model optimizations if model provided
            if model:
                self.optimize_model_memory(model)
            
            # Check if memory is still high
            if self.monitor_memory_usage(threshold_mb):
                self.logger.warning("Memory usage still high after optimizations")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        stats = self.profiler.get_memory_stats()
        peak_stats = self.profiler.get_peak_memory()
        
        report = {
            'current_memory': {
                'rss_mb': stats.rss_mb,
                'vms_mb': stats.vms_mb,
                'available_mb': stats.available_mb,
                'percent_used': stats.percent_used
            },
            'peak_memory': peak_stats,
            'optimizations': {
                'auto_mixed_precision': self.auto_mixed_precision,
                'gradient_checkpointing': self.gradient_checkpointing,
                'memory_efficient_attention': self.memory_efficient_attention
            }
        }
        
        if self.torch and self.torch.cuda.is_available():
            report['gpu_memory'] = {
                'allocated_mb': stats.gpu_allocated_mb,
                'cached_mb': stats.gpu_cached_mb,
                'device_count': self.torch.cuda.device_count(),
                'device_name': self.torch.cuda.get_device_name(0) if self.torch.cuda.device_count() > 0 else None
            }
        
        return report


class GradientCheckpointing:
    """Implements gradient checkpointing for memory efficiency."""
    
    def __init__(self, model: Any, logger: Optional[logging.Logger] = None):
        self.model = model
        self.logger = logger or get_logger("gradient_checkpointing")
        self.torch = safe_import('torch')
        self.checkpointed_layers = []
        
    def apply_checkpointing(self, layer_types: List[str] = None) -> int:
        """Apply gradient checkpointing to specified layer types."""
        if not self.torch:
            self.logger.warning("Cannot apply checkpointing: PyTorch not available")
            return 0
        
        if layer_types is None:
            layer_types = ['Linear', 'Conv2d', 'TransformerLayer']
        
        checkpointed_count = 0
        
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            
            if module_type in layer_types:
                # Wrap module with checkpoint
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return self.torch.utils.checkpoint.checkpoint(
                        original_forward, *args, **kwargs
                    )
                
                module.forward = checkpointed_forward
                self.checkpointed_layers.append(name)
                checkpointed_count += 1
        
        self.logger.info(f"Applied gradient checkpointing to {checkpointed_count} layers")
        return checkpointed_count
    
    def estimate_memory_savings(self, input_size: tuple) -> Dict[str, float]:
        """Estimate memory savings from gradient checkpointing."""
        if not self.torch:
            return {}
        
        # Simple estimation based on model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        param_memory_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
        
        # Checkpointing typically saves 30-50% of activation memory
        estimated_savings_mb = param_memory_mb * 0.4
        
        return {
            'total_params': total_params,
            'param_memory_mb': param_memory_mb,
            'estimated_savings_mb': estimated_savings_mb,
            'checkpointed_layers': len(self.checkpointed_layers)
        }


class MemoryEfficientDataLoader:
    """Memory-efficient data loading with smart batching."""
    
    def __init__(self, dataset: Any, batch_size: int = 32, 
                 memory_limit_mb: float = 1000,
                 logger: Optional[logging.Logger] = None):
        self.dataset = dataset
        self.base_batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self.logger = logger or get_logger("memory_efficient_dataloader")
        
        self.profiler = MemoryProfiler(logger)
        self.current_batch_size = batch_size
        
    def adaptive_batch_size(self) -> int:
        """Adapt batch size based on memory usage."""
        stats = self.profiler.get_memory_stats()
        
        memory_usage_ratio = stats.rss_mb / self.memory_limit_mb
        
        if memory_usage_ratio > 0.9:
            # Reduce batch size if memory usage is high
            self.current_batch_size = max(1, int(self.current_batch_size * 0.8))
            self.logger.info(f"Reduced batch size to {self.current_batch_size} due to high memory usage")
        elif memory_usage_ratio < 0.6 and self.current_batch_size < self.base_batch_size:
            # Increase batch size if memory usage is low
            self.current_batch_size = min(self.base_batch_size, int(self.current_batch_size * 1.2))
            self.logger.info(f"Increased batch size to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def get_batch(self, indices: List[int]) -> Any:
        """Get a batch with memory monitoring."""
        with self.profiler.profile_operation(f"batch_load_{len(indices)}"):
            # Load batch data
            batch_data = [self.dataset[i] for i in indices]
            
            # Adapt batch size for next time
            self.adaptive_batch_size()
            
            return batch_data


def memory_efficient_training_loop(model: Any, dataloader: Any, optimizer: Any,
                                 memory_optimizer: MemoryOptimizer,
                                 max_epochs: int = 10) -> Dict[str, Any]:
    """Memory-efficient training loop with automatic optimization."""
    torch = safe_import('torch')
    if not torch:
        raise ImportError("PyTorch required for training")
    
    logger = get_logger("memory_efficient_training")
    training_stats = {
        'epochs_completed': 0,
        'memory_stats': [],
        'optimizations_applied': []
    }
    
    # Enable optimizations
    memory_optimizer.enable_auto_mixed_precision(True)
    
    for epoch in range(max_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
        
        with memory_optimizer.profiler.profile_operation(f"epoch_{epoch}"):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Auto-manage memory every 10 batches
                if batch_idx % 10 == 0:
                    memory_optimizer.auto_manage_memory(model)
                
                # Training step with mixed precision if enabled
                if memory_optimizer.auto_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        # Forward pass would go here
                        pass
                else:
                    # Regular forward pass would go here
                    pass
                
                # Backward pass would go here
                
                # Clear gradients
                optimizer.zero_grad()
        
        # Record memory stats for this epoch
        memory_stats = memory_optimizer.get_memory_report()
        training_stats['memory_stats'].append(memory_stats)
        
        training_stats['epochs_completed'] = epoch + 1
        
        logger.info(f"Completed epoch {epoch + 1}, "
                   f"Memory: {memory_stats['current_memory']['rss_mb']:.1f}MB")
    
    return training_stats
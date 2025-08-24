"""Distributed training and scaling infrastructure for Connectome-GNN-Suite."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
import os
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import pickle
import socket
from contextlib import contextmanager

from ..robust.logging_config import get_logger
from ..robust.error_handling import ConnectomeError


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    use_cuda: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


class DistributedTrainingCoordinator:
    """Coordinates distributed training across multiple processes/nodes."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.is_initialized = False
        
    def setup(self):
        """Setup distributed training environment."""
        if self.is_initialized:
            return
            
        # Set environment variables
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)
        os.environ["LOCAL_RANK"] = str(self.config.local_rank)
        
        # Initialize process group
        init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Set device
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            
        self.is_initialized = True
        self.logger.info(
            f"Distributed training initialized: rank={self.config.rank}, "
            f"world_size={self.config.world_size}"
        )
        
    def cleanup(self):
        """Clean up distributed training environment."""
        if self.is_initialized:
            destroy_process_group()
            self.is_initialized = False
            self.logger.info("Distributed training cleaned up")
            
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if not self.is_initialized:
            raise ConnectomeError("Distributed training not initialized")
            
        device_id = self.config.local_rank if self.config.use_cuda else None
        
        # Move model to appropriate device
        if device_id is not None:
            model = model.to(f"cuda:{device_id}")
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[device_id] if device_id is not None else None,
            find_unused_parameters=True
        )
        
        return model
        
    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create distributed data loader."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # Handled by sampler
            **kwargs
        )
        
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Reduce metrics across all processes."""
        if not self.is_initialized:
            return metrics
            
        reduced_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            
            if self.config.use_cuda and torch.cuda.is_available():
                tensor = tensor.cuda(self.config.local_rank)
                
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced_metrics[key] = tensor.item() / self.config.world_size
            
        return reduced_metrics
        
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast object from source rank to all ranks."""
        if not self.is_initialized:
            return obj
            
        if self.config.rank == src:
            # Serialize object
            serialized = pickle.dumps(obj)
            size_tensor = torch.tensor(len(serialized), dtype=torch.long)
        else:
            serialized = None
            size_tensor = torch.tensor(0, dtype=torch.long)
            
        if self.config.use_cuda and torch.cuda.is_available():
            size_tensor = size_tensor.cuda(self.config.local_rank)
            
        # Broadcast size
        dist.broadcast(size_tensor, src=src)
        
        # Prepare buffer
        if self.config.rank != src:
            serialized = bytearray(size_tensor.item())
            
        # Create tensor from bytes
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8)
        
        if self.config.use_cuda and torch.cuda.is_available():
            data_tensor = data_tensor.cuda(self.config.local_rank)
            
        # Broadcast data
        dist.broadcast(data_tensor, src=src)
        
        # Deserialize on non-source ranks
        if self.config.rank != src:
            obj = pickle.loads(bytes(data_tensor.cpu().numpy()))
            
        return obj
        
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        return self.config.rank == 0


class AdaptiveLoadBalancer:
    """Adaptive load balancing for distributed training."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_stats = {}
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.logger = get_logger(__name__)
        
    def submit_task(
        self,
        task_fn: Callable,
        task_args: tuple,
        task_kwargs: dict,
        priority: int = 0
    ) -> str:
        """Submit task to load balancer."""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        future = self.executor.submit(task_fn, *task_args, **task_kwargs)
        
        self.task_queue.put({
            'id': task_id,
            'future': future,
            'priority': priority,
            'submit_time': time.time()
        })
        
        return task_id
        
    def get_results(self, timeout: float = None) -> List[Dict[str, Any]]:
        """Get completed task results."""
        results = []
        
        while True:
            try:
                task = self.task_queue.get_nowait()
                
                if task['future'].done():
                    try:
                        result = task['future'].result()
                        results.append({
                            'id': task['id'],
                            'result': result,
                            'success': True,
                            'completion_time': time.time(),
                            'duration': time.time() - task['submit_time']
                        })
                    except Exception as e:
                        results.append({
                            'id': task['id'],
                            'error': str(e),
                            'success': False,
                            'completion_time': time.time(),
                            'duration': time.time() - task['submit_time']
                        })
                else:
                    # Put back unfinished task
                    self.task_queue.put(task)
                    break
                    
            except queue.Empty:
                break
                
        return results
        
    def shutdown(self):
        """Shutdown load balancer."""
        self.executor.shutdown(wait=True)


class MemoryEfficientDataLoader:
    """Memory-efficient data loading for large connectome datasets."""
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 1000  # Number of items to cache
        
        self.logger = get_logger(__name__)
        
    def __iter__(self):
        """Iterate over batches."""
        # Implement efficient iteration with caching and prefetching
        for batch_idx in range(0, len(self), self.batch_size):
            batch_data = self._load_batch(batch_idx)
            yield batch_data
            
    def _load_batch(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Load a single batch efficiently."""
        # Check cache first
        with self.cache_lock:
            if batch_idx in self.data_cache:
                return self.data_cache[batch_idx]
                
        # Load from disk
        batch_data = self._load_from_disk(batch_idx)
        
        # Add to cache
        with self.cache_lock:
            if len(self.data_cache) < self.max_cache_size:
                self.data_cache[batch_idx] = batch_data
            elif batch_idx % 10 == 0:  # Occasionally evict old entries
                # Simple LRU eviction
                oldest_key = min(self.data_cache.keys())
                del self.data_cache[oldest_key]
                self.data_cache[batch_idx] = batch_data
                
        return batch_data
        
    def _load_from_disk(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Load batch data from disk."""
        # Placeholder implementation
        # In practice, this would load actual connectome data
        return {
            'x': torch.randn(self.batch_size, 100),
            'edge_index': torch.randint(0, 100, (2, 1000)),
            'y': torch.randn(self.batch_size, 1)
        }
        
    def __len__(self):
        """Get number of batches."""
        # This would be computed from actual dataset size
        return 1000  # Placeholder


class GradientCompressionManager:
    """Manages gradient compression for efficient distributed training."""
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        quantization_bits: int = 8,
        use_error_feedback: bool = True
    ):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.use_error_feedback = use_error_feedback
        
        self.error_buffers = {}
        self.logger = get_logger(__name__)
        
    def compress_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compress gradients for efficient communication."""
        compressed = {}
        
        for name, grad in gradients.items():
            if grad is None:
                continue
                
            # Apply error feedback if enabled
            if self.use_error_feedback:
                if name in self.error_buffers:
                    grad = grad + self.error_buffers[name]
                    
            # Top-k sparsification
            compressed_grad = self._topk_sparsify(grad)
            
            # Quantization
            compressed_grad = self._quantize(compressed_grad)
            
            # Store error for feedback
            if self.use_error_feedback:
                error = grad - compressed_grad
                self.error_buffers[name] = error
                
            compressed[name] = compressed_grad
            
        return compressed
        
    def _topk_sparsify(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsification to tensor."""
        original_shape = tensor.shape
        flattened = tensor.flatten()
        
        # Select top-k elements by magnitude
        k = max(1, int(len(flattened) * self.compression_ratio))
        _, topk_indices = torch.topk(torch.abs(flattened), k)
        
        # Create sparse tensor
        sparse_tensor = torch.zeros_like(flattened)
        sparse_tensor[topk_indices] = flattened[topk_indices]
        
        return sparse_tensor.reshape(original_shape)
        
    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to reduce precision."""
        if self.quantization_bits >= 32:
            return tensor
            
        # Simple linear quantization
        max_val = tensor.abs().max()
        if max_val > 0:
            scale = (2 ** (self.quantization_bits - 1) - 1) / max_val
            quantized = torch.round(tensor * scale) / scale
            return quantized
            
        return tensor


class AutoScaler:
    """Automatic scaling based on workload and resource utilization."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        target_utilization: float = 0.8,
        scale_up_threshold: float = 0.9,
        scale_down_threshold: float = 0.5
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.utilization_history = []
        self.last_scale_time = time.time()
        self.scale_cooldown = 60  # seconds
        
        self.logger = get_logger(__name__)
        
    def should_scale(self, current_utilization: float) -> Tuple[bool, int]:
        """Determine if scaling is needed."""
        self.utilization_history.append(current_utilization)
        
        # Keep only recent history
        if len(self.utilization_history) > 10:
            self.utilization_history = self.utilization_history[-10:]
            
        # Check cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False, self.current_workers
            
        avg_utilization = sum(self.utilization_history) / len(self.utilization_history)
        
        # Scale up decision
        if (avg_utilization > self.scale_up_threshold and 
            self.current_workers < self.max_workers):
            new_workers = min(self.max_workers, self.current_workers + 1)
            self.logger.info(
                f"Scaling up: {self.current_workers} -> {new_workers} "
                f"(utilization: {avg_utilization:.2f})"
            )
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            return True, new_workers
            
        # Scale down decision
        if (avg_utilization < self.scale_down_threshold and 
            self.current_workers > self.min_workers):
            new_workers = max(self.min_workers, self.current_workers - 1)
            self.logger.info(
                f"Scaling down: {self.current_workers} -> {new_workers} "
                f"(utilization: {avg_utilization:.2f})"
            )
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            return True, new_workers
            
        return False, self.current_workers


class PerformanceProfiler:
    """Profiles performance and identifies bottlenecks."""
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
        self.memory_snapshots = []
        self.gpu_snapshots = []
        
        self.logger = get_logger(__name__)
        
    @contextmanager
    def profile_section(self, name: str):
        """Context manager for profiling code sections."""
        start_time = time.time()
        
        # Memory snapshot before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated()
        else:
            gpu_memory_before = 0
            
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Memory snapshot after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_after = torch.cuda.memory_allocated()
            else:
                gpu_memory_after = 0
                
            # Record timing
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(duration)
            
            # Record memory usage
            memory_delta = gpu_memory_after - gpu_memory_before
            self.memory_snapshots.append({
                'section': name,
                'memory_delta': memory_delta,
                'timestamp': end_time
            })
            
    def increment_counter(self, name: str, value: int = 1):
        """Increment performance counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'timing_stats': {},
            'counter_stats': self.counters.copy(),
            'memory_stats': {},
            'timestamp': time.time()
        }
        
        # Compute timing statistics
        for name, durations in self.timers.items():
            if durations:
                report['timing_stats'][name] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'mean': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'recent_avg': sum(durations[-10:]) / min(10, len(durations))
                }
                
        # Compute memory statistics
        if self.memory_snapshots:
            memory_by_section = {}
            for snapshot in self.memory_snapshots:
                section = snapshot['section']
                if section not in memory_by_section:
                    memory_by_section[section] = []
                memory_by_section[section].append(snapshot['memory_delta'])
                
            for section, deltas in memory_by_section.items():
                report['memory_stats'][section] = {
                    'total_allocations': sum(delta for delta in deltas if delta > 0),
                    'total_deallocations': sum(abs(delta) for delta in deltas if delta < 0),
                    'net_allocation': sum(deltas),
                    'peak_allocation': max(deltas) if deltas else 0
                }
                
        return report
        
    def reset_stats(self):
        """Reset all performance statistics."""
        self.timers.clear()
        self.counters.clear()
        self.memory_snapshots.clear()
        self.gpu_snapshots.clear()


# Global instances
_global_profiler = None
_global_coordinator = None

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def get_distributed_coordinator(config: DistributedConfig = None) -> DistributedTrainingCoordinator:
    """Get global distributed training coordinator."""
    global _global_coordinator
    if _global_coordinator is None:
        if config is None:
            config = DistributedConfig()
        _global_coordinator = DistributedTrainingCoordinator(config)
    return _global_coordinator


def profile_performance(name: str):
    """Decorator for profiling function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            with profiler.profile_section(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
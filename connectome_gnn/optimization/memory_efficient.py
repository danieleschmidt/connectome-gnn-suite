"""Memory-efficient training and inference optimizations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any, Optional, Callable
import math
from contextlib import contextmanager

from ..models.base import BaseConnectomeModel
from ..training import ConnectomeTrainer


class MemoryEfficientModel(nn.Module):
    """Wrapper for memory-efficient model execution."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        gradient_checkpointing: bool = True,
        activation_checkpointing: bool = True,
        memory_fraction: float = 0.8
    ):
        """Initialize memory-efficient model wrapper.
        
        Args:
            model: Base connectome model
            gradient_checkpointing: Enable gradient checkpointing
            activation_checkpointing: Enable activation checkpointing
            memory_fraction: Fraction of GPU memory to use
        """
        super().__init__()
        
        self.model = model
        self.gradient_checkpointing = gradient_checkpointing
        self.activation_checkpointing = activation_checkpointing
        self.memory_fraction = memory_fraction
        
        # Apply memory optimizations
        if gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    def _apply_gradient_checkpointing(self):
        """Apply gradient checkpointing to reduce memory usage."""
        
        # Apply to model layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Wrap in checkpoint
                original_forward = module.forward
                
                def checkpointed_forward(x):
                    return torch.utils.checkpoint.checkpoint(original_forward, x)
                
                module.forward = checkpointed_forward
    
    def forward(self, data):
        """Memory-efficient forward pass."""
        
        if self.activation_checkpointing:
            # Use activation checkpointing
            return torch.utils.checkpoint.checkpoint(self.model, data)
        else:
            return self.model(data)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class GradientCheckpointing:
    """Utility for gradient checkpointing in large models."""
    
    @staticmethod
    def apply_to_model(model: nn.Module, checkpoint_layers: Optional[List[str]] = None):
        """Apply gradient checkpointing to model layers.
        
        Args:
            model: Model to apply checkpointing to
            checkpoint_layers: Specific layer types to checkpoint
        """
        
        if checkpoint_layers is None:
            checkpoint_layers = ['Linear', 'Conv1d', 'Conv2d', 'GATConv', 'GCNConv']
        
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            
            if module_type in checkpoint_layers:
                # Store original forward
                original_forward = module.forward
                
                # Create checkpointed version
                def make_checkpointed_forward(orig_forward):
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            orig_forward, *args, **kwargs
                        )
                    return checkpointed_forward
                
                module.forward = make_checkpointed_forward(original_forward)
    
    @staticmethod
    @contextmanager
    def checkpoint_context():
        """Context manager for gradient checkpointing."""
        try:
            torch.backends.cudnn.benchmark = False  # Disable for deterministic checkpointing
            yield
        finally:
            torch.backends.cudnn.benchmark = True


class MixedPrecisionTrainer(ConnectomeTrainer):
    """Trainer with automatic mixed precision support."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        task,
        use_amp: bool = True,
        amp_opt_level: str = "O1",
        loss_scale: str = "dynamic",
        **kwargs
    ):
        """Initialize mixed precision trainer.
        
        Args:
            model: Connectome model
            task: Training task
            use_amp: Whether to use automatic mixed precision
            amp_opt_level: AMP optimization level
            loss_scale: Loss scaling strategy
            **kwargs: Additional trainer arguments
        """
        super().__init__(model, task, **kwargs)
        
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_opt_level = amp_opt_level
        
        if self.use_amp:
            # Initialize gradient scaler for mixed precision
            self.scaler = GradScaler()
            
            # Enable mixed precision
            self.model = self.model.half() if amp_opt_level == "O2" else self.model
    
    def train_epoch(self, train_loader):
        """Train epoch with mixed precision."""
        self.model.train()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision forward pass
                with autocast():
                    predictions = self.model(batch)
                    targets = self.task.prepare_targets([batch]).to(self.device)
                    targets = self.task.normalize_targets(targets)
                    loss = self.task.compute_loss(predictions, targets)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaling
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard forward/backward pass
                predictions = self.model(batch)
                targets = self.task.prepare_targets([batch]).to(self.device)
                targets = self.task.normalize_targets(targets)
                loss = self.task.compute_loss(predictions, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            all_predictions.append(predictions.detach().cpu().float())
            all_targets.append(targets.detach().cpu().float())
        
        # Compute epoch metrics
        epoch_loss /= len(train_loader)
        
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)
        else:
            metrics = {}
        
        return epoch_loss, metrics


class AdaptiveMemoryManager:
    """Adaptive memory management for varying graph sizes."""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        memory_threshold: float = 0.9
    ):
        """Initialize adaptive memory manager.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            memory_threshold: Memory threshold for adaptation
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.memory_threshold = memory_threshold
        
        self.oom_count = 0
        self.success_count = 0
    
    def adapt_batch_size(self, current_memory_usage: float) -> int:
        """Adapt batch size based on memory usage.
        
        Args:
            current_memory_usage: Current GPU memory usage fraction
            
        Returns:
            Adapted batch size
        """
        
        if current_memory_usage > self.memory_threshold:
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            self.oom_count += 1
        elif current_memory_usage < self.memory_threshold * 0.7:
            # Increase batch size cautiously
            self.current_batch_size = min(
                self.initial_batch_size,
                int(self.current_batch_size * 1.2)
            )
            self.success_count += 1
        
        return self.current_batch_size
    
    def handle_oom_exception(self):
        """Handle out-of-memory exception."""
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size aggressively
        self.current_batch_size = max(
            self.min_batch_size,
            self.current_batch_size // 4
        )
        
        self.oom_count += 1
        
        return self.current_batch_size


class ProgressiveResizing:
    """Progressive resizing for connectome data augmentation."""
    
    def __init__(
        self,
        start_size: int = 50,
        end_size: int = 200,
        resize_epochs: List[int] = None
    ):
        """Initialize progressive resizing.
        
        Args:
            start_size: Starting graph size (number of nodes)
            end_size: Final graph size
            resize_epochs: Epochs at which to resize
        """
        self.start_size = start_size
        self.end_size = end_size
        self.resize_epochs = resize_epochs or [10, 20, 30, 40]
        
        self.current_size = start_size
    
    def get_current_size(self, epoch: int) -> int:
        """Get current graph size for given epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current graph size
        """
        
        # Determine current size based on epoch
        size_increment = (self.end_size - self.start_size) / len(self.resize_epochs)
        
        current_level = 0
        for resize_epoch in self.resize_epochs:
            if epoch >= resize_epoch:
                current_level += 1
            else:
                break
        
        self.current_size = min(
            self.end_size,
            self.start_size + int(current_level * size_increment)
        )
        
        return self.current_size
    
    def resize_graph(self, data, target_size: int):
        """Resize graph to target size.
        
        Args:
            data: Input graph data
            target_size: Target number of nodes
            
        Returns:
            Resized graph data
        """
        
        current_size = data.x.size(0)
        
        if current_size == target_size:
            return data
        elif current_size > target_size:
            # Downsample nodes
            indices = torch.randperm(current_size)[:target_size]
            indices = torch.sort(indices)[0]
            
            # Create mapping for edge indices
            old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
            
            # Update node features
            resized_data = data.clone()
            resized_data.x = data.x[indices]
            
            # Update edge indices
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                # Filter edges that involve selected nodes
                edge_mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
                
                if edge_mask.any():
                    filtered_edges = data.edge_index[:, edge_mask]
                    
                    # Remap edge indices
                    new_edge_index = torch.zeros_like(filtered_edges)
                    for i in range(filtered_edges.size(1)):
                        new_edge_index[0, i] = old_to_new[filtered_edges[0, i].item()]
                        new_edge_index[1, i] = old_to_new[filtered_edges[1, i].item()]
                    
                    resized_data.edge_index = new_edge_index
                    
                    # Update edge attributes
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        resized_data.edge_attr = data.edge_attr[edge_mask]
                else:
                    # No valid edges after filtering
                    resized_data.edge_index = torch.empty((2, 0), dtype=torch.long)
                    if hasattr(data, 'edge_attr'):
                        resized_data.edge_attr = torch.empty((0, data.edge_attr.size(1)))
            
            return resized_data
        
        else:
            # Upsample nodes (pad with zeros or duplicate)
            resized_data = data.clone()
            
            # Pad node features
            padding_size = target_size - current_size
            padding = torch.zeros(padding_size, data.x.size(1), dtype=data.x.dtype, device=data.x.device)
            resized_data.x = torch.cat([data.x, padding], dim=0)
            
            return resized_data


class MemoryEfficientDataLoader:
    """Memory-efficient data loader for large connectome datasets."""
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        adaptive_batching: bool = True,
        memory_limit_gb: float = 8.0,
        preload_factor: int = 2
    ):
        """Initialize memory-efficient data loader.
        
        Args:
            dataset: Connectome dataset
            batch_size: Base batch size
            adaptive_batching: Enable adaptive batch sizing
            memory_limit_gb: Memory limit in GB
            preload_factor: Number of batches to preload
        """
        self.dataset = dataset
        self.base_batch_size = batch_size
        self.adaptive_batching = adaptive_batching
        self.memory_limit = memory_limit_gb * 1024**3  # Convert to bytes
        self.preload_factor = preload_factor
        
        # Adaptive memory manager
        if adaptive_batching:
            self.memory_manager = AdaptiveMemoryManager(batch_size)
        
        # Precompute data sizes
        self.data_sizes = self._compute_data_sizes()
    
    def _compute_data_sizes(self) -> List[int]:
        """Compute memory size for each data sample."""
        
        sizes = []
        
        for data in self.dataset:
            # Estimate memory usage
            size = 0
            
            # Node features
            size += data.x.numel() * data.x.element_size()
            
            # Edge indices
            if hasattr(data, 'edge_index'):
                size += data.edge_index.numel() * data.edge_index.element_size()
            
            # Edge attributes
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                size += data.edge_attr.numel() * data.edge_attr.element_size()
            
            sizes.append(size)
        
        return sizes
    
    def create_batches(self) -> List[List[int]]:
        """Create memory-aware batches."""
        
        batches = []
        current_batch = []
        current_size = 0
        
        # Target batch memory size
        target_batch_size = self.memory_limit // (self.preload_factor * 2)
        
        for idx, data_size in enumerate(self.data_sizes):
            
            # Check if adding this sample exceeds memory limit
            if current_size + data_size > target_batch_size and current_batch:
                # Finalize current batch
                batches.append(current_batch)
                current_batch = [idx]
                current_size = data_size
            else:
                # Add to current batch
                current_batch.append(idx)
                current_size += data_size
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def __iter__(self):
        """Iterate over memory-efficient batches."""
        
        batches = self.create_batches()
        
        for batch_indices in batches:
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            yield batch_data
    
    def __len__(self):
        """Number of batches."""
        return len(self.create_batches())


class IncrementalAggregation:
    """Incremental aggregation for large graph processing."""
    
    def __init__(self, aggregation_function: str = "mean"):
        """Initialize incremental aggregation.
        
        Args:
            aggregation_function: Aggregation function (mean, sum, max)
        """
        self.aggregation_function = aggregation_function
        self.reset()
    
    def reset(self):
        """Reset aggregation state."""
        self.accumulated_values = None
        self.count = 0
        self.max_values = None
    
    def update(self, values: torch.Tensor):
        """Update aggregation with new values.
        
        Args:
            values: New values to aggregate
        """
        
        if self.accumulated_values is None:
            # Initialize
            self.accumulated_values = values.clone()
            self.max_values = values.clone()
            self.count = 1
        else:
            # Update aggregation
            if self.aggregation_function == "mean" or self.aggregation_function == "sum":
                self.accumulated_values += values
            elif self.aggregation_function == "max":
                self.max_values = torch.max(self.max_values, values)
            
            self.count += 1
    
    def get_result(self) -> torch.Tensor:
        """Get final aggregation result.
        
        Returns:
            Aggregated values
        """
        
        if self.accumulated_values is None:
            return torch.tensor(0.0)
        
        if self.aggregation_function == "mean":
            return self.accumulated_values / self.count
        elif self.aggregation_function == "sum":
            return self.accumulated_values
        elif self.aggregation_function == "max":
            return self.max_values
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregation_function}")


class MemoryProfiler:
    """Memory profiler for connectome GNN training."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.memory_log = []
        self.peak_memory = 0
        self.start_memory = 0
    
    def start_profiling(self):
        """Start memory profiling."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        
        self.memory_log = []
        self.peak_memory = 0
    
    def log_memory(self, tag: str = ""):
        """Log current memory usage.
        
        Args:
            tag: Optional tag for this measurement
        """
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            self.memory_log.append({
                'tag': tag,
                'current_memory_gb': current_memory / 1024**3,
                'peak_memory_gb': peak_memory / 1024**3,
                'memory_since_start_gb': (current_memory - self.start_memory) / 1024**3
            })
            
            self.peak_memory = max(self.peak_memory, peak_memory)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary.
        
        Returns:
            Memory usage summary
        """
        
        return {
            'peak_memory_gb': self.peak_memory / 1024**3,
            'start_memory_gb': self.start_memory / 1024**3,
            'memory_log': self.memory_log,
            'total_measurements': len(self.memory_log)
        }
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage over time.
        
        Args:
            save_path: Optional path to save plot
        """
        
        if not self.memory_log:
            print("No memory data to plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        measurements = list(range(len(self.memory_log)))
        current_memory = [log['current_memory_gb'] for log in self.memory_log]
        peak_memory = [log['peak_memory_gb'] for log in self.memory_log]
        
        # Plot
        ax.plot(measurements, current_memory, label='Current Memory', marker='o')
        ax.plot(measurements, peak_memory, label='Peak Memory', marker='s')
        
        ax.set_xlabel('Measurement')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('GPU Memory Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add tags as annotations
        for i, log in enumerate(self.memory_log):
            if log['tag']:
                ax.annotate(log['tag'], (i, current_memory[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
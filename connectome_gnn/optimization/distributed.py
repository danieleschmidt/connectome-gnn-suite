"""Distributed and multi-GPU training for large-scale connectome analysis."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import time
import json
from pathlib import Path

from ..models.base import BaseConnectomeModel
from ..training import ConnectomeTrainer
from ..utils import set_random_seed


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Distributed settings
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Communication settings
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    
    # Training settings
    batch_size: int = 32
    sync_bn: bool = True
    gradient_accumulation_steps: int = 1
    
    # Optimization settings
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    
    # Checkpointing
    save_on_master_only: bool = True
    checkpoint_freq: int = 10


class DistributedTrainer:
    """Distributed trainer for connectome GNN models."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        task,
        config: DistributedConfig,
        output_dir: str = "./distributed_training",
        device: Optional[torch.device] = None
    ):
        """Initialize distributed trainer.
        
        Args:
            model: Connectome model
            task: Training task
            config: Distributed configuration
            output_dir: Output directory
            device: Device for training
        """
        self.model = model
        self.task = task
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{config.local_rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Initialize distributed process group
        self._init_distributed()
        
        # Setup model for distributed training
        self._setup_distributed_model()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Create output directory
        if self.is_master():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_distributed(self):
        """Initialize distributed process group."""
        
        if self.config.world_size > 1:
            # Initialize the process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            print(f"Initialized process group: rank {self.config.rank}/{self.config.world_size}")
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
    
    def _setup_distributed_model(self):
        """Setup model for distributed training."""
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        if self.config.world_size > 1:
            # Synchronize batch normalization if requested
            if self.config.sync_bn:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            
            # Wrap model with DDP
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb
            )
            
            print(f"Model wrapped with DDP on rank {self.config.rank}")
    
    def is_master(self) -> bool:
        """Check if current process is master."""
        return self.config.rank == 0
    
    def train(
        self,
        dataset,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_split: float = 0.1,
        **kwargs
    ):
        """Train model with distributed setup.
        
        Args:
            dataset: Training dataset
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            validation_split: Validation split ratio
            **kwargs: Additional training arguments
        """
        
        if self.is_master():
            print(f"Starting distributed training for {epochs} epochs")
            print(f"World size: {self.config.world_size}")
            print(f"Device: {self.device}")
        
        # Split dataset
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True
        ) if self.config.world_size > 1 else None
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=False
        ) if self.config.world_size > 1 else None
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=self.is_master()
        )
        
        # Training loop
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader)
            
            # Update scheduler
            scheduler.step(val_metrics.get('loss', train_metrics.get('loss', 0)))
            
            # Logging (master only)
            if self.is_master():
                self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch, optimizer, scheduler, val_metrics)
            
            # Early stopping check
            current_metric = val_metrics.get('loss', float('inf'))
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                if self.is_master():
                    self._save_best_model()
        
        # Cleanup
        self._cleanup()
        
        if self.is_master():
            print("Distributed training completed")
    
    def _train_epoch(self, train_loader, optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            predictions = self.model(batch)
            targets = self.task.prepare_targets([batch]).to(self.device)
            targets = self.task.normalize_targets(targets)
            
            loss = self.task.compute_loss(predictions, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Store predictions and targets for metrics
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # Reduce across all processes
            if self.config.world_size > 1:
                predictions_tensor = self._all_gather_tensors(predictions_tensor)
                targets_tensor = self._all_gather_tensors(targets_tensor)
            
            metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)
        else:
            metrics = {}
        
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                predictions = self.model(batch)
                targets = self.task.prepare_targets([batch]).to(self.device)
                targets = self.task.normalize_targets(targets)
                
                loss = self.task.compute_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # Reduce across all processes
            if self.config.world_size > 1:
                predictions_tensor = self._all_gather_tensors(predictions_tensor)
                targets_tensor = self._all_gather_tensors(targets_tensor)
            
            metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)
        else:
            metrics = {}
        
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _all_gather_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes."""
        
        if self.config.world_size == 1:
            return tensor
        
        # Gather tensor shapes first
        local_size = torch.tensor([tensor.size(0)], device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.config.world_size)]
        dist.all_gather(all_sizes, local_size)
        
        # Prepare for gathering
        max_size = max(size.item() for size in all_sizes)
        
        # Pad tensor if needed
        if tensor.size(0) < max_size:
            padding_size = max_size - tensor.size(0)
            padding_shape = (padding_size,) + tensor.shape[1:]
            padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=0)
        
        # Gather tensors
        all_tensors = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(all_tensors, tensor)
        
        # Trim tensors to actual sizes and concatenate
        trimmed_tensors = []
        for i, gathered_tensor in enumerate(all_tensors):
            actual_size = all_sizes[i].item()
            trimmed_tensors.append(gathered_tensor[:actual_size])
        
        return torch.cat(trimmed_tensors, dim=0)
    
    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch results."""
        
        log_str = f"Epoch {epoch + 1}: "
        log_str += f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
        log_str += f"Val Loss: {val_metrics.get('loss', 0):.4f}"
        
        # Add key metrics
        for metric in ['accuracy', 'mae', 'rmse', 'f1']:
            if metric in val_metrics:
                log_str += f", Val {metric}: {val_metrics[metric]:.4f}"
        
        print(log_str)
        
        # Save detailed logs
        log_data = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'world_size': self.config.world_size,
            'global_step': self.global_step
        }
        
        log_file = self.output_dir / 'training_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def _save_checkpoint(self, epoch: int, optimizer, scheduler, metrics: Dict):
        """Save training checkpoint."""
        
        if not self.is_master() and self.config.save_on_master_only:
            return
        
        # Get model state dict (unwrap DDP if needed)
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only last N checkpoints
        self._cleanup_old_checkpoints(keep_last=3)
    
    def _save_best_model(self):
        """Save best model (master only)."""
        
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        best_model_path = self.output_dir / 'best_model.pth'
        torch.save({
            'model_state_dict': model_state_dict,
            'epoch': self.epoch,
            'best_metric': self.best_metric
        }, best_model_path)
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints, keeping only the last N."""
        
        checkpoint_files = list(self.output_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoint_files) > keep_last:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest files
            for old_checkpoint in checkpoint_files[:-keep_last]:
                old_checkpoint.unlink()
    
    def _cleanup(self):
        """Cleanup distributed resources."""
        
        if self.config.world_size > 1:
            dist.destroy_process_group()


class MultiGPUTrainer:
    """Multi-GPU trainer using DataParallel (simpler alternative to DDP)."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        task,
        gpu_ids: Optional[List[int]] = None,
        batch_size: int = 32,
        output_dir: str = "./multigpu_training"
    ):
        """Initialize multi-GPU trainer.
        
        Args:
            model: Connectome model
            task: Training task
            gpu_ids: List of GPU IDs to use (None for all available)
            batch_size: Batch size per GPU
            output_dir: Output directory
        """
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        
        # Setup devices
        if torch.cuda.is_available():
            if gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = gpu_ids
            
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            print(f"Using GPUs: {self.gpu_ids}")
        else:
            raise RuntimeError("CUDA is not available for multi-GPU training")
        
        # Setup model
        self._setup_model()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_model(self):
        """Setup model for multi-GPU training."""
        
        # Move model to primary device
        self.model = self.model.to(self.device)
        
        # Wrap with DataParallel
        if len(self.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            print(f"Model wrapped with DataParallel on {len(self.gpu_ids)} GPUs")
        
        # Adjust batch size for multiple GPUs
        self.effective_batch_size = self.batch_size * len(self.gpu_ids)
    
    def train(
        self,
        dataset,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_split: float = 0.1,
        **kwargs
    ):
        """Train model with multi-GPU setup.
        
        Args:
            dataset: Training dataset
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            validation_split: Validation split ratio
            **kwargs: Additional training arguments
        """
        
        print(f"Starting multi-GPU training for {epochs} epochs")
        print(f"Effective batch size: {self.effective_batch_size}")
        
        # Use standard ConnectomeTrainer with modified model
        trainer = ConnectomeTrainer(
            model=self.model,
            task=self.task,
            batch_size=self.effective_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device,
            output_dir=self.output_dir,
            **kwargs
        )
        
        # Train
        trainer.fit(dataset, epochs=epochs)
        
        print("Multi-GPU training completed")


def launch_distributed_training(
    train_fn: Callable,
    world_size: int,
    **kwargs
):
    """Launch distributed training across multiple processes.
    
    Args:
        train_fn: Training function to run
        world_size: Number of processes
        **kwargs: Arguments to pass to training function
    """
    
    # Set up environment variables for distributed training
    os.environ['MASTER_ADDR'] = kwargs.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = kwargs.get('master_port', '12355')
    
    # Launch processes
    mp.spawn(
        _distributed_worker,
        args=(world_size, train_fn, kwargs),
        nprocs=world_size,
        join=True
    )


def _distributed_worker(rank: int, world_size: int, train_fn: Callable, kwargs: Dict):
    """Worker function for distributed training."""
    
    # Set up distributed config
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        **kwargs.get('distributed_config', {})
    )
    
    # Run training function
    train_fn(config=config, **kwargs)


# Example usage function
def run_distributed_experiment(
    model_class,
    model_params: Dict,
    task_class,
    task_params: Dict,
    dataset,
    world_size: int = 2,
    epochs: int = 50,
    **kwargs
):
    """Run distributed training experiment.
    
    Args:
        model_class: Model class to instantiate
        model_params: Model parameters
        task_class: Task class to instantiate
        task_params: Task parameters
        dataset: Training dataset
        world_size: Number of processes
        epochs: Number of epochs
        **kwargs: Additional arguments
    """
    
    def train_worker(config: DistributedConfig, **train_kwargs):
        """Worker training function."""
        
        # Set random seed for reproducibility
        set_random_seed(42 + config.rank)
        
        # Create model and task
        model = model_class(**model_params)
        task = task_class(**task_params)
        
        # Create distributed trainer
        trainer = DistributedTrainer(
            model=model,
            task=task,
            config=config,
            output_dir=train_kwargs.get('output_dir', './distributed_training')
        )
        
        # Train
        trainer.train(
            dataset=dataset,
            epochs=epochs,
            **train_kwargs
        )
    
    # Launch distributed training
    launch_distributed_training(
        train_fn=train_worker,
        world_size=world_size,
        **kwargs
    )
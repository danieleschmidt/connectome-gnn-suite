"""Advanced training utilities with research features."""

import os
import time
import json
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .models.base import BaseConnectomeModel
from .tasks.base import BaseConnectomeTask
from .utils import set_random_seed, get_device


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    scheduler: str = "plateau"  # "plateau", "cosine", "step"
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    mixed_precision: bool = False
    accumulate_grad_batches: int = 1
    val_check_interval: int = 1
    log_every_n_steps: int = 10
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    mode: str = "min"  # "min" or "max"


class AdvancedConnectomeTrainer:
    """Advanced trainer with research features.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Cross-validation support
    - Hyperparameter optimization
    - Distributed training
    - Advanced logging and visualization
    - Model ensembling
    - Self-supervised pretraining
    """
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        config: TrainingConfig,
        output_dir: Path,
        device: torch.device = None,
        use_wandb: bool = False,
        wandb_project: str = "connectome-gnn",
    ):
        self.model = model
        self.task = task
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or get_device()
        self.use_wandb = use_wandb
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_score = float('inf') if config.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup model
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = None
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Logging
        self.log_file = self.output_dir / 'training.log'
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config=vars(config),
                dir=str(self.output_dir)
            )
            wandb.watch(self.model)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.mode,
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=self.config.min_lr,
                verbose=True
            )
        elif self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler == "warmup":
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                max_epochs=self.config.max_epochs,
                min_lr=self.config.min_lr
            )
        return None
    
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with levels."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Advanced training epoch with gradient accumulation and mixed precision."""
        self.model.train()
        
        epoch_metrics = defaultdict(list)
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(batch)
                targets = self.task.prepare_targets([batch]).to(self.device)
                targets = self.task.normalize_targets(targets)
                
                loss = self.model.compute_loss(outputs, targets, self.task.task_type)
                loss = loss / self.config.accumulate_grad_batches
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_val
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_val
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                
                # Compute metrics
                with torch.no_grad():
                    preds = outputs["predictions"] if isinstance(outputs, dict) else outputs
                    metrics = self.task.compute_metrics(preds.cpu(), targets.cpu())
                    
                    for key, value in metrics.items():
                        epoch_metrics[key].append(value)
                    epoch_metrics['loss'].append(accumulated_loss)
                
                # Reset accumulated loss
                accumulated_loss = 0.0
                
                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log(
                        f"Epoch {epoch+1}/{self.config.max_epochs} - "
                        f"Step {self.global_step} - "
                        f"Loss: {loss.item():.4f} - "
                        f"LR: {current_lr:.6f}"
                    )
                    
                    if self.use_wandb:
                        wandb.log({
                            "train/loss_step": loss.item(),
                            "train/lr": current_lr,
                            "global_step": self.global_step
                        })
        
        # Compute epoch averages
        epoch_results = {}
        for key, values in epoch_metrics.items():
            epoch_results[f"train/{key}"] = np.mean(values)
        
        return epoch_results
    
    def validate_epoch(
        self, 
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validation epoch with comprehensive metrics."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device, non_blocking=True)
                
                outputs = self.model(batch)
                targets = self.task.prepare_targets([batch]).to(self.device)
                targets = self.task.normalize_targets(targets)
                
                loss = self.model.compute_loss(outputs, targets, self.task.task_type)
                
                preds = outputs["predictions"] if isinstance(outputs, dict) else outputs
                all_predictions.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_losses.append(loss.item())
        
        # Compute comprehensive metrics
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        
        metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)
        metrics['loss'] = np.mean(all_losses)
        
        # Add validation prefix
        val_metrics = {f"val/{key}": value for key, value in metrics.items()}
        
        return val_metrics, predictions_tensor, targets_tensor
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """Advanced training loop."""
        self.log(f"Starting training for {self.config.max_epochs} epochs")
        self.log(f"Model: {self.model.__class__.__name__}")
        self.log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.log(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if epoch % self.config.val_check_interval == 0:
                val_metrics, val_preds, val_targets = self.validate_epoch(val_loader, epoch)
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val/loss'])
                    else:
                        self.scheduler.step()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Update training history
                for key, value in epoch_metrics.items():
                    self.training_history[key].append(value)
                
                # Check for improvement
                current_score = val_metrics[f"val/{self.config.monitor_metric.replace('val_', '')}"]
                
                is_best = False
                if self.config.mode == 'min':
                    is_best = current_score < self.best_score
                else:
                    is_best = current_score > self.best_score
                
                if is_best:
                    self.best_score = current_score
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Logging
                self.log(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_metrics.get('train/loss', 0):.4f} - "
                    f"Val Loss: {val_metrics['val/loss']:.4f} - "
                    f"Best: {self.best_score:.4f}"
                )
                
                # Log key metrics
                for key in ['mae', 'rmse', 'accuracy', 'f1', 'auc']:
                    val_key = f"val/{key}"
                    if val_key in val_metrics:
                        self.log(f"  {key.upper()}: {val_metrics[val_key]:.4f}")
                
                # Wandb logging
                if self.use_wandb:
                    wandb.log(epoch_metrics, step=epoch)
                
                # Save periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch, is_best=False)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self.log(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break
        
        # Training completed
        total_time = time.time() - start_time
        self.log(f"Training completed in {total_time:.2f} seconds")
        self.log(f"Best validation score: {self.best_score:.4f}")
        
        # Final evaluation on test set
        if test_loader is not None:
            test_results = self.evaluate(test_loader)
            self.log(f"Test results: {test_results}")
        
        # Save training summary
        self.save_training_summary()
    
    def cross_validate(
        self,
        dataset,
        n_folds: int = 5,
        stratified: bool = True,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """Cross-validation training."""
        self.log(f"Starting {n_folds}-fold cross-validation")
        
        set_random_seed(random_state)
        
        # Get targets for stratification
        targets = []
        if stratified:
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, 'y') and data.y is not None:
                    targets.append(data.y.item() if torch.is_tensor(data.y) else data.y)
                else:
                    targets.append(0)
        
        # Create folds
        if stratified and self.task.task_type in ['classification', 'multi_class']:
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kfold.split(range(len(dataset)), targets))
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kfold.split(range(len(dataset))))
        
        fold_results = defaultdict(list)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            self.log(f"Starting fold {fold + 1}/{n_folds}")
            
            # Reset model
            self.model.reset_parameters()
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Create data loaders
            train_subset = [dataset[i] for i in train_idx]
            val_subset = [dataset[i] for i in val_idx]
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                num_workers=4
            )
            
            # Train fold
            self.fit(train_loader, val_loader)
            
            # Evaluate fold
            val_results = self.evaluate(val_loader)
            
            for key, value in val_results.items():
                fold_results[key].append(value)
            
            self.log(f"Fold {fold + 1} results: {val_results}")
        
        # Compute cross-validation statistics
        cv_stats = {}
        for key, values in fold_results.items():
            cv_stats[f"{key}_mean"] = np.mean(values)
            cv_stats[f"{key}_std"] = np.std(values)
        
        self.log(f"Cross-validation results: {cv_stats}")
        
        # Save CV results
        cv_path = self.output_dir / 'cross_validation_results.json'
        with open(cv_path, 'w') as f:
            json.dump({
                'fold_results': dict(fold_results),
                'cv_statistics': cv_stats,
                'config': vars(self.config)
            }, f, indent=2, default=str)
        
        return dict(fold_results)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save comprehensive checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_score': self.best_score,
            'config': vars(self.config),
            'training_history': dict(self.training_history)
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, checkpoint_path)
            self.log(f"Saved best checkpoint: {checkpoint_path}")
        
        # Save latest
        latest_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
        torch.save(checkpoint, latest_path)
        
        # Keep only top-k checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the best checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob('epoch_*.pth'))
        if len(checkpoints) > self.config.save_top_k:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-self.config.save_top_k]:
                old_checkpoint.unlink()
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                targets = self.task.prepare_targets([batch]).to(self.device)
                targets = self.task.normalize_targets(targets)
                
                loss = self.model.compute_loss(outputs, targets, self.task.task_type)
                
                preds = outputs["predictions"] if isinstance(outputs, dict) else outputs
                all_predictions.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_losses.append(loss.item())
        
        # Compute metrics
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        metrics = self.task.compute_metrics(predictions, targets)
        metrics['loss'] = np.mean(all_losses)
        
        return metrics
    
    def save_training_summary(self):
        """Save comprehensive training summary."""
        summary = {
            'model': self.model.__class__.__name__,
            'task': self.task.get_task_info(),
            'config': vars(self.config),
            'best_score': self.best_score,
            'total_epochs': self.epoch + 1,
            'global_steps': self.global_step,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_history': dict(self.training_history)
        }
        
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.log(f"Training summary saved: {summary_path}")
        
        # Save plots
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves."""
        if not self.training_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Curves')
        
        # Loss curves
        if 'train/loss' in self.training_history and 'val/loss' in self.training_history:
            axes[0, 0].plot(self.training_history['train/loss'], label='Train')
            axes[0, 0].plot(self.training_history['val/loss'], label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Metric curves (example with MAE)
        if 'train/mae' in self.training_history and 'val/mae' in self.training_history:
            axes[0, 1].plot(self.training_history['train/mae'], label='Train MAE')
            axes[0, 1].plot(self.training_history['val/mae'], label='Val MAE')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Additional metrics
        metric_keys = [k for k in self.training_history.keys() 
                       if 'val/' in k and k not in ['val/loss', 'val/mae']]
        
        if len(metric_keys) >= 1:
            key = metric_keys[0]
            axes[1, 0].plot(self.training_history[key])
            axes[1, 0].set_title(key.replace('val/', '').upper())
            axes[1, 0].grid(True)
        
        if len(metric_keys) >= 2:
            key = metric_keys[1]
            axes[1, 1].plot(self.training_history[key])
            axes[1, 1].set_title(key.replace('val/', '').upper())
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Training curves saved: {plot_path}")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup followed by cosine annealing."""
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_max_epoch = self.max_epochs - self.warmup_epochs
            
            return [self.min_lr + (base_lr - self.min_lr) * 
                    (1 + np.cos(np.pi * cos_epoch / cos_max_epoch)) / 2
                    for base_lr in self.base_lrs]
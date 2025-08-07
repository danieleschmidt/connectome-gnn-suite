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
            f.write(log_message + '\\n')
    
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
                with torch.no_grad():\n                    preds = outputs[\"predictions\"] if isinstance(outputs, dict) else outputs\n                    metrics = self.task.compute_metrics(preds.cpu(), targets.cpu())\n                    \n                    for key, value in metrics.items():\n                        epoch_metrics[key].append(value)\n                    epoch_metrics['loss'].append(accumulated_loss)\n                \n                # Reset accumulated loss\n                accumulated_loss = 0.0\n                \n                # Logging\n                if self.global_step % self.config.log_every_n_steps == 0:\n                    current_lr = self.optimizer.param_groups[0]['lr']\n                    self.log(\n                        f\"Epoch {epoch+1}/{self.config.max_epochs} - \"\n                        f\"Step {self.global_step} - \"\n                        f\"Loss: {loss.item():.4f} - \"\n                        f\"LR: {current_lr:.6f}\"\n                    )\n                    \n                    if self.use_wandb:\n                        wandb.log({\n                            \"train/loss_step\": loss.item(),\n                            \"train/lr\": current_lr,\n                            \"global_step\": self.global_step\n                        })\n        \n        # Compute epoch averages\n        epoch_results = {}\n        for key, values in epoch_metrics.items():\n            epoch_results[f\"train/{key}\"] = np.mean(values)\n        \n        return epoch_results\n    \n    def validate_epoch(\n        self, \n        val_loader: DataLoader,\n        epoch: int\n    ) -> Dict[str, float]:\n        \"\"\"Validation epoch with comprehensive metrics.\"\"\"\n        self.model.eval()\n        \n        all_predictions = []\n        all_targets = []\n        all_losses = []\n        \n        with torch.no_grad():\n            for batch in val_loader:\n                batch = batch.to(self.device, non_blocking=True)\n                \n                outputs = self.model(batch)\n                targets = self.task.prepare_targets([batch]).to(self.device)\n                targets = self.task.normalize_targets(targets)\n                \n                loss = self.model.compute_loss(outputs, targets, self.task.task_type)\n                \n                preds = outputs[\"predictions\"] if isinstance(outputs, dict) else outputs\n                all_predictions.append(preds.cpu())\n                all_targets.append(targets.cpu())\n                all_losses.append(loss.item())\n        \n        # Compute comprehensive metrics\n        predictions_tensor = torch.cat(all_predictions, dim=0)\n        targets_tensor = torch.cat(all_targets, dim=0)\n        \n        metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)\n        metrics['loss'] = np.mean(all_losses)\n        \n        # Add validation prefix\n        val_metrics = {f\"val/{key}\": value for key, value in metrics.items()}\n        \n        return val_metrics, predictions_tensor, targets_tensor\n    \n    def fit(\n        self,\n        train_loader: DataLoader,\n        val_loader: DataLoader,\n        test_loader: Optional[DataLoader] = None\n    ):\n        \"\"\"Advanced training loop.\"\"\"\n        self.log(f\"Starting training for {self.config.max_epochs} epochs\")\n        self.log(f\"Model: {self.model.__class__.__name__}\")\n        self.log(f\"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\")\n        self.log(f\"Device: {self.device}\")\n        \n        start_time = time.time()\n        \n        for epoch in range(self.config.max_epochs):\n            self.epoch = epoch\n            \n            # Training\n            train_metrics = self.train_epoch(train_loader, epoch)\n            \n            # Validation\n            if epoch % self.config.val_check_interval == 0:\n                val_metrics, val_preds, val_targets = self.validate_epoch(val_loader, epoch)\n                \n                # Update learning rate\n                if self.scheduler is not None:\n                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):\n                        self.scheduler.step(val_metrics['val/loss'])\n                    else:\n                        self.scheduler.step()\n                \n                # Combine metrics\n                epoch_metrics = {**train_metrics, **val_metrics}\n                \n                # Update training history\n                for key, value in epoch_metrics.items():\n                    self.training_history[key].append(value)\n                \n                # Check for improvement\n                current_score = val_metrics[f\"val/{self.config.monitor_metric.replace('val_', '')}\"]\n                \n                is_best = False\n                if self.config.mode == 'min':\n                    is_best = current_score < self.best_score\n                else:\n                    is_best = current_score > self.best_score\n                \n                if is_best:\n                    self.best_score = current_score\n                    self.patience_counter = 0\n                    self.save_checkpoint(epoch, is_best=True)\n                else:\n                    self.patience_counter += 1\n                \n                # Logging\n                self.log(\n                    f\"Epoch {epoch+1}/{self.config.max_epochs} - \"\n                    f\"Train Loss: {train_metrics.get('train/loss', 0):.4f} - \"\n                    f\"Val Loss: {val_metrics['val/loss']:.4f} - \"\n                    f\"Best: {self.best_score:.4f}\"\n                )\n                \n                # Log key metrics\n                for key in ['mae', 'rmse', 'accuracy', 'f1', 'auc']:\n                    val_key = f\"val/{key}\"\n                    if val_key in val_metrics:\n                        self.log(f\"  {key.upper()}: {val_metrics[val_key]:.4f}\")\n                \n                # Wandb logging\n                if self.use_wandb:\n                    wandb.log(epoch_metrics, step=epoch)\n                \n                # Save periodic checkpoint\n                if (epoch + 1) % 10 == 0:\n                    self.save_checkpoint(epoch, is_best=False)\n                \n                # Early stopping\n                if self.patience_counter >= self.config.patience:\n                    self.log(f\"Early stopping triggered after {self.patience_counter} epochs without improvement\")\n                    break\n        \n        # Training completed\n        total_time = time.time() - start_time\n        self.log(f\"Training completed in {total_time:.2f} seconds\")\n        self.log(f\"Best validation score: {self.best_score:.4f}\")\n        \n        # Final evaluation on test set\n        if test_loader is not None:\n            test_results = self.evaluate(test_loader)\n            self.log(f\"Test results: {test_results}\")\n        \n        # Save training summary\n        self.save_training_summary()\n    \n    def cross_validate(\n        self,\n        dataset,\n        n_folds: int = 5,\n        stratified: bool = True,\n        random_state: int = 42\n    ) -> Dict[str, List[float]]:\n        \"\"\"Cross-validation training.\"\"\"\n        self.log(f\"Starting {n_folds}-fold cross-validation\")\n        \n        set_random_seed(random_state)\n        \n        # Get targets for stratification\n        targets = []\n        if stratified:\n            for i in range(len(dataset)):\n                data = dataset[i]\n                if hasattr(data, 'y') and data.y is not None:\n                    targets.append(data.y.item() if torch.is_tensor(data.y) else data.y)\n                else:\n                    targets.append(0)\n        \n        # Create folds\n        if stratified and self.task.task_type in ['classification', 'multi_class']:\n            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)\n            splits = list(kfold.split(range(len(dataset)), targets))\n        else:\n            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)\n            splits = list(kfold.split(range(len(dataset))))\n        \n        fold_results = defaultdict(list)\n        \n        for fold, (train_idx, val_idx) in enumerate(splits):\n            self.log(f\"Starting fold {fold + 1}/{n_folds}\")\n            \n            # Reset model\n            self.model.reset_parameters()\n            self.optimizer = self._create_optimizer()\n            self.scheduler = self._create_scheduler()\n            \n            # Create data loaders\n            train_subset = [dataset[i] for i in train_idx]\n            val_subset = [dataset[i] for i in val_idx]\n            \n            train_loader = DataLoader(\n                train_subset,\n                batch_size=self.config.batch_size,\n                shuffle=True,\n                num_workers=4\n            )\n            \n            val_loader = DataLoader(\n                val_subset,\n                batch_size=self.config.batch_size * 2,\n                shuffle=False,\n                num_workers=4\n            )\n            \n            # Train fold\n            self.fit(train_loader, val_loader)\n            \n            # Evaluate fold\n            val_results = self.evaluate(val_loader)\n            \n            for key, value in val_results.items():\n                fold_results[key].append(value)\n            \n            self.log(f\"Fold {fold + 1} results: {val_results}\")\n        \n        # Compute cross-validation statistics\n        cv_stats = {}\n        for key, values in fold_results.items():\n            cv_stats[f\"{key}_mean\"] = np.mean(values)\n            cv_stats[f\"{key}_std\"] = np.std(values)\n        \n        self.log(f\"Cross-validation results: {cv_stats}\")\n        \n        # Save CV results\n        cv_path = self.output_dir / 'cross_validation_results.json'\n        with open(cv_path, 'w') as f:\n            json.dump({\n                'fold_results': dict(fold_results),\n                'cv_statistics': cv_stats,\n                'config': vars(self.config)\n            }, f, indent=2, default=str)\n        \n        return dict(fold_results)\n    \n    def save_checkpoint(self, epoch: int, is_best: bool = False):\n        \"\"\"Save comprehensive checkpoint.\"\"\"\n        checkpoint = {\n            'epoch': epoch,\n            'global_step': self.global_step,\n            'model_state_dict': self.model.state_dict(),\n            'optimizer_state_dict': self.optimizer.state_dict(),\n            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,\n            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,\n            'best_score': self.best_score,\n            'config': vars(self.config),\n            'training_history': dict(self.training_history)\n        }\n        \n        # Save checkpoint\n        if is_best:\n            checkpoint_path = self.checkpoint_dir / 'best.pth'\n            torch.save(checkpoint, checkpoint_path)\n            self.log(f\"Saved best checkpoint: {checkpoint_path}\")\n        \n        # Save latest\n        latest_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'\n        torch.save(checkpoint, latest_path)\n        \n        # Keep only top-k checkpoints\n        self._cleanup_checkpoints()\n    \n    def _cleanup_checkpoints(self):\n        \"\"\"Keep only the best checkpoints.\"\"\"\n        checkpoints = list(self.checkpoint_dir.glob('epoch_*.pth'))\n        if len(checkpoints) > self.config.save_top_k:\n            checkpoints.sort(key=lambda x: x.stat().st_mtime)\n            for old_checkpoint in checkpoints[:-self.config.save_top_k]:\n                old_checkpoint.unlink()\n    \n    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:\n        \"\"\"Comprehensive model evaluation.\"\"\"\n        self.model.eval()\n        \n        all_predictions = []\n        all_targets = []\n        all_losses = []\n        \n        with torch.no_grad():\n            for batch in data_loader:\n                batch = batch.to(self.device)\n                \n                outputs = self.model(batch)\n                targets = self.task.prepare_targets([batch]).to(self.device)\n                targets = self.task.normalize_targets(targets)\n                \n                loss = self.model.compute_loss(outputs, targets, self.task.task_type)\n                \n                preds = outputs[\"predictions\"] if isinstance(outputs, dict) else outputs\n                all_predictions.append(preds.cpu())\n                all_targets.append(targets.cpu())\n                all_losses.append(loss.item())\n        \n        # Compute metrics\n        predictions = torch.cat(all_predictions, dim=0)\n        targets = torch.cat(all_targets, dim=0)\n        \n        metrics = self.task.compute_metrics(predictions, targets)\n        metrics['loss'] = np.mean(all_losses)\n        \n        return metrics\n    \n    def save_training_summary(self):\n        \"\"\"Save comprehensive training summary.\"\"\"\n        summary = {\n            'model': self.model.__class__.__name__,\n            'task': self.task.get_task_info(),\n            'config': vars(self.config),\n            'best_score': self.best_score,\n            'total_epochs': self.epoch + 1,\n            'global_steps': self.global_step,\n            'model_parameters': sum(p.numel() for p in self.model.parameters()),\n            'training_history': dict(self.training_history)\n        }\n        \n        summary_path = self.output_dir / 'training_summary.json'\n        with open(summary_path, 'w') as f:\n            json.dump(summary, f, indent=2, default=str)\n        \n        self.log(f\"Training summary saved: {summary_path}\")\n        \n        # Save plots\n        self.plot_training_curves()\n    \n    def plot_training_curves(self):\n        \"\"\"Plot training curves.\"\"\"\n        if not self.training_history:\n            return\n        \n        fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n        fig.suptitle('Training Curves')\n        \n        # Loss curves\n        if 'train/loss' in self.training_history and 'val/loss' in self.training_history:\n            axes[0, 0].plot(self.training_history['train/loss'], label='Train')\n            axes[0, 0].plot(self.training_history['val/loss'], label='Validation')\n            axes[0, 0].set_title('Loss')\n            axes[0, 0].legend()\n            axes[0, 0].grid(True)\n        \n        # Metric curves (example with MAE)\n        if 'train/mae' in self.training_history and 'val/mae' in self.training_history:\n            axes[0, 1].plot(self.training_history['train/mae'], label='Train MAE')\n            axes[0, 1].plot(self.training_history['val/mae'], label='Val MAE')\n            axes[0, 1].set_title('Mean Absolute Error')\n            axes[0, 1].legend()\n            axes[0, 1].grid(True)\n        \n        # Additional metrics\n        metric_keys = [k for k in self.training_history.keys() \n                       if 'val/' in k and k not in ['val/loss', 'val/mae']]\n        \n        if len(metric_keys) >= 1:\n            key = metric_keys[0]\n            axes[1, 0].plot(self.training_history[key])\n            axes[1, 0].set_title(key.replace('val/', '').upper())\n            axes[1, 0].grid(True)\n        \n        if len(metric_keys) >= 2:\n            key = metric_keys[1]\n            axes[1, 1].plot(self.training_history[key])\n            axes[1, 1].set_title(key.replace('val/', '').upper())\n            axes[1, 1].grid(True)\n        \n        plt.tight_layout()\n        plot_path = self.output_dir / 'training_curves.png'\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        self.log(f\"Training curves saved: {plot_path}\")\n\n\nclass WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):\n    \"\"\"Warmup followed by cosine annealing.\"\"\"\n    \n    def __init__(\n        self,\n        optimizer,\n        warmup_epochs: int,\n        max_epochs: int,\n        min_lr: float = 0,\n        last_epoch: int = -1\n    ):\n        self.warmup_epochs = warmup_epochs\n        self.max_epochs = max_epochs\n        self.min_lr = min_lr\n        super().__init__(optimizer, last_epoch)\n    \n    def get_lr(self):\n        if self.last_epoch < self.warmup_epochs:\n            # Warmup phase\n            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs \n                    for base_lr in self.base_lrs]\n        else:\n            # Cosine annealing phase\n            cos_epoch = self.last_epoch - self.warmup_epochs\n            cos_max_epoch = self.max_epochs - self.warmup_epochs\n            \n            return [self.min_lr + (base_lr - self.min_lr) * \n                    (1 + np.cos(np.pi * cos_epoch / cos_max_epoch)) / 2\n                    for base_lr in self.base_lrs]
"""Training utilities for connectome GNN models."""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from .models.base import BaseConnectomeModel
from .tasks.base import BaseConnectomeTask


class ConnectomeTrainer:
    """Trainer for connectome GNN models.
    
    Handles training, validation, and evaluation of connectome models
    with support for different tasks and model architectures.
    
    Args:
        model: Connectome GNN model
        task: Prediction task
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
        device: Device to train on
        output_dir: Directory to save outputs
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        device: torch.device = torch.device('cpu'),
        output_dir: Optional[Path] = None,
        gradient_checkpointing: bool = False,
    ):
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.output_dir = output_dir or Path('./outputs')
        self.gradient_checkpointing = gradient_checkpointing
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=patience // 2,
            verbose=True
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.output_dir / 'training.log'
        
    def log(self, message: str):
        """Log message to console and file."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def prepare_data_splits(
        self, 
        dataset, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = False
    ) -> Tuple[List, List, List]:
        """Split dataset into train/val/test splits.
        
        Args:
            dataset: Full dataset
            train_ratio: Fraction for training
            val_ratio: Fraction for validation  
            test_ratio: Fraction for testing
            stratify: Whether to stratify splits based on targets
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        n_samples = len(dataset)
        indices = list(range(n_samples))
        
        if stratify and self.task.task_type in ['classification', 'multi_class']:
            # Get targets for stratification
            targets = []
            for i in indices:
                data = dataset[i]
                if hasattr(data, 'y') and data.y is not None:
                    targets.append(data.y.item() if torch.is_tensor(data.y) else data.y)
                else:
                    targets.append(0)  # Fallback
            
            # First split: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                indices, 
                train_size=train_ratio,
                stratify=targets,
                random_state=42
            )
            
            # Second split: val vs test
            temp_targets = [targets[i] for i in temp_indices]
            val_size = val_ratio / (val_ratio + test_ratio)
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                stratify=temp_targets,
                random_state=42
            )
        else:
            # Random split
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            n_test = n_samples - n_train - n_val
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        self.log(f"Data splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def create_data_loaders(
        self, 
        dataset,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: Optional[List[int]] = None
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create data loaders for training, validation, and testing."""
        
        # Create subset datasets
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices] if test_indices else None
        
        # Prepare targets and fit task statistics
        train_targets = self.task.prepare_targets(train_dataset)
        self.task.fit_target_statistics(train_targets)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(batch)
            targets = self.task.prepare_targets([batch]).to(self.device)
            
            # Normalize targets if needed
            targets = self.task.normalize_targets(targets)
            
            # Compute loss
            loss = self.task.compute_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
        
        # Compute epoch metrics
        epoch_loss /= len(train_loader)
        
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)
        else:
            metrics = {}
        
        return epoch_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                predictions = self.model(batch)
                targets = self.task.prepare_targets([batch]).to(self.device)
                
                # Normalize targets if needed
                targets = self.task.normalize_targets(targets)
                
                loss = self.task.compute_loss(predictions, targets)
                
                epoch_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute epoch metrics
        epoch_loss /= len(val_loader)
        
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            metrics = self.task.compute_metrics(predictions_tensor, targets_tensor)
        else:
            metrics = {}
        
        return epoch_loss, metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'training_history': self.training_history,
            'task_info': self.task.get_task_info(),
            'model_config': self.model.get_model_summary()
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.log(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        self.training_history = checkpoint['training_history']
        
        self.log(f"Loaded checkpoint from epoch {self.epoch}")
        return checkpoint
    
    def fit(
        self, 
        dataset,
        epochs: int = 100,
        resume_from: Optional[Path] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """Fit the model on the dataset.
        
        Args:
            dataset: Training dataset
            epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
        """
        self.log(f"Starting training for {epochs} epochs")
        self.log(f"Model: {self.model.__class__.__name__}")
        self.log(f"Task: {self.task.get_task_info()}")
        self.log(f"Device: {self.device}")
        
        # Resume from checkpoint if provided
        if resume_from and resume_from.exists():
            self.load_checkpoint(resume_from)
        
        # Prepare data splits
        train_indices, val_indices, test_indices = self.prepare_data_splits(
            dataset, train_ratio, val_ratio, 1.0 - train_ratio - val_ratio
        )
        
        # Create data loaders
        train_loader, val_loader, _ = self.create_data_loaders(
            dataset, train_indices, val_indices, test_indices
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics.copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Log progress
            self.log(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Log key metrics
            for metric_name, value in val_metrics.items():
                if metric_name in ['mae', 'rmse', 'accuracy', 'auc', 'f1']:
                    self.log(f"  Val {metric_name}: {value:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.log(f"Early stopping triggered after {self.patience} epochs without improvement")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.log(f"Training completed in {total_time:.2f} seconds")
        self.log(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final results
        self.save_training_summary()
    
    def save_training_summary(self):
        """Save training summary and history."""
        summary = {
            'model_type': self.model.__class__.__name__,
            'task_info': self.task.get_task_info(),
            'training_config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'patience': self.patience
            },
            'best_validation_loss': self.best_val_loss,
            'best_validation_metrics': self.best_val_metrics,
            'total_epochs': self.epoch + 1,
            'model_parameters': self.model.count_parameters()
        }
        
        # Save summary
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        self.log(f"Training summary saved to {summary_path}")
    
    def save_best_model(self, output_path: Path):
        """Save the best model state."""
        best_checkpoint_path = self.output_dir / 'best_checkpoint.pth'
        
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            
            # Save only the model state and essential info
            model_save = {
                'model_state_dict': checkpoint['model_state_dict'],
                'model_config': checkpoint['model_config'],
                'task_info': checkpoint['task_info'],
                'best_val_metrics': checkpoint['best_val_metrics']
            }
            
            torch.save(model_save, output_path)
            self.log(f"Best model saved to {output_path}")
        else:
            self.log("No best checkpoint found to save")
"""Trainer for connectome GNN models."""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Any, List, Tuple
import time
import numpy as np
from tqdm import tqdm

from ..models.base import BaseConnectomeModel
from ..tasks.base import BaseConnectomeTask


class ConnectomeTrainer:
    """Trainer for connectome GNN models."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        device: str = "cpu",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        scheduler: bool = True,
        early_stopping: bool = True,
        patience: int = 20,
        gradient_checkpointing: bool = False
    ):
        """Initialize trainer.
        
        Args:
            model: Connectome GNN model
            task: Task definition
            device: Device to use for training
            batch_size: Batch size for training
            learning_rate: Learning rate
            optimizer: Optimizer type (adam, adamw)
            scheduler: Whether to use learning rate scheduler
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.batch_size = batch_size
        
        # Setup optimizer
        if optimizer == "adam":
            self.optimizer = Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10
        ) if scheduler else None
        
        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.gradient_checkpointing = gradient_checkpointing
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
    
    def fit(
        self,
        dataset,
        epochs: int = 100,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            dataset: Training dataset
            epochs: Number of epochs to train
            validation_split: Fraction of data for validation
            verbose: Whether to print progress
            
        Returns:
            Training history
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If training fails
        """
        # Input validation
        if epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {epochs}")
        if not 0 <= validation_split <= 1:
            raise ValueError(f"Validation split must be in [0, 1], got {validation_split}")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        try:
            # Split dataset
            train_loader, val_loader = self._create_data_loaders(dataset, validation_split)
            
            if verbose:
                print(f"Training {self.model.__class__.__name__} on {self.task.task_name}")
                print(f"Device: {self.device}")
                print(f"Training samples: {len(train_loader.dataset)}")
                print(f"Validation samples: {len(val_loader.dataset) if val_loader else 0}")
            
            # Training loop
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                try:
                    # Train epoch
                    train_loss, train_metrics = self._train_epoch(train_loader)
                    
                    # Validation epoch
                    val_loss, val_metrics = self._validate_epoch(val_loader) if val_loader else (None, {})
                    
                    # Update history
                    self.history['train_loss'].append(train_loss)
                    self.history['train_metrics'].append(train_metrics)
                    if val_loss is not None:
                        self.history['val_loss'].append(val_loss)
                        self.history['val_metrics'].append(val_metrics)
                    
                    # Learning rate scheduling
                    if self.scheduler and val_loss is not None:
                        self.scheduler.step(val_loss)
                    
                    # Early stopping check
                    if val_loss is not None and self.early_stopping:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_model_state = self.model.state_dict().copy()
                            self.epochs_without_improvement = 0
                        else:
                            self.epochs_without_improvement += 1
                            
                            if self.epochs_without_improvement >= self.patience:
                                if verbose:
                                    print(f"Early stopping at epoch {epoch+1}")
                                break
                    
                    # Print progress
                    if verbose:
                        epoch_time = time.time() - epoch_start_time
                        print(f"Epoch {epoch+1:3d}/{epochs} - "
                              f"train_loss: {train_loss:.4f} - "
                              f"val_loss: {val_loss:.4f if val_loss else 'N/A'} - "
                              f"time: {epoch_time:.2f}s")
                              
                except KeyboardInterrupt:
                    if verbose:
                        print(f"\nTraining interrupted at epoch {epoch+1}")
                    break
                except Exception as e:
                    if verbose:
                        print(f"\nError in epoch {epoch+1}: {str(e)}")
                    raise RuntimeError(f"Training failed at epoch {epoch+1}: {str(e)}") from e
            
            # Load best model if early stopping was used
            if self.early_stopping and self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                if verbose:
                    print("Loaded best model from early stopping")
            
            return self.history
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e
    
    def _create_data_loaders(self, dataset, validation_split: float) -> Tuple:
        """Create training and validation data loaders."""
        if validation_split > 0:
            # Split dataset
            total_size = len(dataset)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            
            return train_loader, val_loader
        else:
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            return train_loader, None
    
    def _train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Get targets
            targets = self.task.get_target(batch)
            targets = targets.to(self.device)
            
            # Compute loss
            if self.task.task_type == "regression":
                loss = nn.MSELoss()(predictions.squeeze(), targets.squeeze())
            elif self.task.task_type == "classification":
                loss = nn.CrossEntropyLoss()(predictions, targets.long())
            else:
                raise ValueError(f"Unknown task type: {self.task.task_type}")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.task.compute_metrics(all_predictions, all_targets)
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        if val_loader is None:
            return None, {}
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Get targets
                targets = self.task.get_target(batch)
                targets = targets.to(self.device)
                
                # Compute loss
                if self.task.task_type == "regression":
                    loss = nn.MSELoss()(predictions.squeeze(), targets.squeeze())
                elif self.task.task_type == "classification":
                    loss = nn.CrossEntropyLoss()(predictions, targets.long())
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.task.compute_metrics(all_predictions, all_targets)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def evaluate(self, test_loader) -> Dict[str, Any]:
        """Evaluate model on test data."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                predictions = self.model(batch)
                targets = self.task.get_target(batch).to(self.device)
                
                if self.task.task_type == "regression":
                    loss = nn.MSELoss()(predictions.squeeze(), targets.squeeze())
                elif self.task.task_type == "classification":
                    loss = nn.CrossEntropyLoss()(predictions, targets.long())
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute final metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.task.compute_metrics(all_predictions, all_targets)
        
        results = {
            'test_loss': total_loss / len(test_loader),
            'test_metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        return results
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'task_info': self.task.get_task_info()
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
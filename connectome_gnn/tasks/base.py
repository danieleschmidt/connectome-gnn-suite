"""Base task definitions for connectome analysis."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
import numpy as np


class BaseConnectomeTask(ABC):
    """Abstract base class for connectome analysis tasks.
    
    Defines the interface for different types of connectome tasks including
    node-level, edge-level, and graph-level predictions.
    
    Args:
        task_type: Type of task ("regression", "classification", "multi_class")
        target_name: Name of the target variable
        normalize: Whether to normalize targets
        class_weights: Optional class weights for imbalanced datasets
    """
    
    def __init__(
        self,
        task_type: str = "regression",
        target_name: str = "target",
        normalize: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.task_type = task_type.lower()
        self.target_name = target_name
        self.normalize = normalize
        self.class_weights = class_weights
        
        # Validation
        if self.task_type not in ["regression", "classification", "multi_class"]:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Statistics for normalization
        self.target_mean = None
        self.target_std = None
        self.target_min = None
        self.target_max = None
        
        # Class information
        self.num_classes = 1 if task_type == "regression" else 2
        self.class_names = None
        
    @abstractmethod
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Prepare target values from raw data.
        
        Args:
            data_list: List of data samples
            
        Returns:
            Tensor of target values
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute task-specific evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    def fit_target_statistics(self, targets: torch.Tensor):
        """Fit normalization statistics on training targets.
        
        Args:
            targets: Training targets
        """
        if self.normalize and self.task_type == "regression":
            # For regression tasks
            self.target_mean = torch.mean(targets)
            self.target_std = torch.std(targets)
            self.target_min = torch.min(targets)
            self.target_max = torch.max(targets)
        elif self.task_type in ["classification", "multi_class"]:
            # For classification tasks
            if targets.dim() == 1:
                self.num_classes = len(torch.unique(targets))
            else:
                self.num_classes = targets.size(1)
    
    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize target values.
        
        Args:
            targets: Raw targets
            
        Returns:
            Normalized targets
        """
        if not self.normalize or self.task_type != "regression":
            return targets
        
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Target statistics not fitted. Call fit_target_statistics first.")
        
        return (targets - self.target_mean) / (self.target_std + 1e-8)
    
    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            
        Returns:
            Denormalized predictions
        """
        if not self.normalize or self.task_type != "regression":
            return predictions
        
        if self.target_mean is None or self.target_std is None:
            return predictions
        
        return predictions * self.target_std + self.target_mean
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the task.
        
        Returns:
            Dictionary containing task information
        """
        info = {
            'task_type': self.task_type,
            'target_name': self.target_name,
            'normalize': self.normalize,
            'num_classes': self.num_classes,
        }
        
        if self.task_type == "regression" and self.target_mean is not None:
            info.update({
                'target_mean': self.target_mean.item() if torch.is_tensor(self.target_mean) else self.target_mean,
                'target_std': self.target_std.item() if torch.is_tensor(self.target_std) else self.target_std,
                'target_range': (
                    self.target_min.item() if torch.is_tensor(self.target_min) else self.target_min,
                    self.target_max.item() if torch.is_tensor(self.target_max) else self.target_max
                )
            })
        
        if self.class_names is not None:
            info['class_names'] = self.class_names
        
        return info


class RegressionTask(BaseConnectomeTask):
    """Base class for regression tasks."""
    
    def __init__(self, target_name: str = "target", normalize: bool = True, **kwargs):
        super().__init__(
            task_type="regression",
            target_name=target_name,
            normalize=normalize,
            **kwargs
        )
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss for regression."""
        return nn.functional.mse_loss(predictions.squeeze(), targets.float())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute regression metrics."""
        # Convert to numpy for sklearn metrics
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Denormalize if needed
        if self.normalize:
            pred_tensor = self.denormalize_predictions(predictions.squeeze())
            target_tensor = self.denormalize_predictions(targets)
            pred_np = pred_tensor.detach().cpu().numpy().flatten()
            target_np = target_tensor.detach().cpu().numpy().flatten()
        
        mae = mean_absolute_error(target_np, pred_np)
        mse = mean_squared_error(target_np, pred_np)
        rmse = np.sqrt(mse)
        
        # Compute correlation
        correlation = np.corrcoef(pred_np, target_np)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Compute R²
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': mae,
            'mse': mse,  
            'rmse': rmse,
            'correlation': correlation,
            'r2': r2
        }


class ClassificationTask(BaseConnectomeTask):
    """Base class for binary classification tasks."""
    
    def __init__(
        self, 
        target_name: str = "target",
        class_names: Optional[List[str]] = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(
            task_type="classification",
            target_name=target_name,
            normalize=False,
            class_weights=class_weights,
            **kwargs
        )
        self.class_names = class_names or ["negative", "positive"]
        self.num_classes = 2
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss."""
        loss_fn = nn.BCEWithLogitsLoss(weight=self.class_weights)
        return loss_fn(predictions.squeeze(), targets.float())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute classification metrics."""
        # Convert predictions to probabilities
        probs = torch.sigmoid(predictions.squeeze()).detach().cpu().numpy()
        pred_classes = (probs > 0.5).astype(int)
        target_np = targets.detach().cpu().numpy().astype(int)
        
        accuracy = accuracy_score(target_np, pred_classes)
        
        try:
            auc = roc_auc_score(target_np, probs)
        except ValueError:
            # Handle case where only one class is present
            auc = 0.5
        
        # Calculate precision, recall, F1
        tp = np.sum((pred_classes == 1) & (target_np == 1))
        fp = np.sum((pred_classes == 1) & (target_np == 0))
        fn = np.sum((pred_classes == 0) & (target_np == 1))
        tn = np.sum((pred_classes == 0) & (target_np == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }


class MultiClassTask(BaseConnectomeTask):
    """Base class for multi-class classification tasks."""
    
    def __init__(
        self,
        target_name: str = "target",
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(
            task_type="multi_class",
            target_name=target_name,
            normalize=False,
            class_weights=class_weights,
            **kwargs
        )
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for multi-class."""
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fn(predictions, targets.long())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute multi-class metrics."""
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy().astype(int)
        
        accuracy = accuracy_score(target_np, pred_classes)
        
        # Per-class metrics
        metrics = {'accuracy': accuracy}
        
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            
            # Binary metrics for each class
            binary_targets = (target_np == i).astype(int)
            binary_preds = (pred_classes == i).astype(int)
            
            tp = np.sum((binary_preds == 1) & (binary_targets == 1))
            fp = np.sum((binary_preds == 1) & (binary_targets == 0))
            fn = np.sum((binary_preds == 0) & (binary_targets == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
        
        # Macro averages
        precisions = [metrics[f'{name}_precision'] for name in self.class_names]
        recalls = [metrics[f'{name}_recall'] for name in self.class_names]
        f1s = [metrics[f'{name}_f1'] for name in self.class_names]
        
        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['macro_f1'] = np.mean(f1s)
        
        return metrics


class TaskFactory:
    """Factory for creating connectome tasks."""
    
    @staticmethod
    def create_task(task_config: Dict[str, Any]) -> BaseConnectomeTask:
        """Create a task from configuration.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            Configured task instance
        """
        task_type = task_config.get('type', 'regression').lower()
        
        if task_type == 'regression':
            return RegressionTask(**task_config)
        elif task_type == 'classification':
            return ClassificationTask(**task_config)
        elif task_type == 'multi_class':
            return MultiClassTask(**task_config)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def get_available_tasks() -> List[str]:
        """Get list of available task types."""
        return ['regression', 'classification', 'multi_class']


def compute_task_specific_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convenience function to compute loss for different task types.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task
        class_weights: Optional class weights
        
    Returns:
        Loss value
    """
    if task_type == "regression":
        return nn.functional.mse_loss(predictions.squeeze(), targets.float())
    elif task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)
        return loss_fn(predictions.squeeze(), targets.float())
    elif task_type == "multi_class":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        return loss_fn(predictions, targets.long())
    else:
        raise ValueError(f"Unknown task type: {task_type}")
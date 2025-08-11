"""Base task class for connectome analysis."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
)


class BaseConnectomeTask(ABC):
    """Base class for connectome analysis tasks."""
    
    def __init__(
        self,
        task_name: str,
        task_type: str,
        target_key: str,
        normalize: bool = True
    ):
        self.task_name = task_name
        self.task_type = task_type
        self.target_key = target_key
        self.normalize = normalize
        
        self.target_mean = 0.0
        self.target_std = 1.0
        self.num_classes = 1
        
    @abstractmethod
    def get_target(self, data: Data) -> torch.Tensor:
        """Extract target variable from data."""
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute task-specific evaluation metrics."""
        pass
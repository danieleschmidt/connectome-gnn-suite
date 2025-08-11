"""Graph-level tasks for connectome analysis."""

import torch
from torch_geometric.data import Data
from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

from .base import BaseConnectomeTask


class CognitiveScorePrediction(BaseConnectomeTask):
    """Predict cognitive scores from connectome data."""
    
    def __init__(self, target: str = "cognitive_total", normalize: bool = True):
        super().__init__(f"cognitive_prediction_{target}", "regression", target, normalize)
    
    def get_target(self, data: Data) -> torch.Tensor:
        if hasattr(data, self.target_key):
            return getattr(data, self.target_key)
        elif hasattr(data, 'y_cognitive'):
            return data.y_cognitive
        else:
            raise ValueError(f"No cognitive target found")
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        return {
            'mse': float(mean_squared_error(target_np, pred_np)),
            'mae': float(mean_absolute_error(target_np, pred_np))
        }


class AgeRegression(BaseConnectomeTask):
    """Predict age from brain connectivity."""
    
    def __init__(self, normalize: bool = True):
        super().__init__("age_regression", "regression", "age", normalize)
    
    def get_target(self, data: Data) -> torch.Tensor:
        if hasattr(data, 'age'):
            return data.age
        elif hasattr(data, 'y_age'):
            return data.y_age
        else:
            raise ValueError("No age target found")
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        return {
            'mae': float(mean_absolute_error(target_np, pred_np)),
            'mse': float(mean_squared_error(target_np, pred_np))
        }


class SubjectClassification(BaseConnectomeTask):
    """Classify subjects based on demographics."""
    
    def __init__(self, target: str = "sex", num_classes: int = 2):
        super().__init__(f"{target}_classification", "classification", target, False)
        self.num_classes = num_classes
    
    def get_target(self, data: Data) -> torch.Tensor:
        if hasattr(data, self.target_key):
            return getattr(data, self.target_key)
        else:
            raise ValueError(f"No {self.target_key} target found")
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=-1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        pred_np = pred_classes.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy().astype(int)
        
        return {
            'accuracy': float(accuracy_score(target_np, pred_np)),
            'f1_score': float(f1_score(target_np, pred_np, average='macro'))
        }


class ConnectomeTaskSuite:
    """Collection of standard connectome analysis tasks."""
    
    def __init__(self):
        self.tasks = {
            'age_regression': AgeRegression(),
            'sex_classification': SubjectClassification(target='sex', num_classes=2),
            'cognitive_prediction': CognitiveScorePrediction()
        }
    
    def get_task(self, task_name: str) -> BaseConnectomeTask:
        return self.tasks[task_name]
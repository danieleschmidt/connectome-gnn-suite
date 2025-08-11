"""Node-level tasks for connectome analysis."""

import torch
from torch_geometric.data import Data
from typing import Dict
from .base import BaseConnectomeTask


class NodeLevelTask(BaseConnectomeTask):
    """Base class for node-level tasks."""
    
    def __init__(self, task_name: str = "node_classification"):
        super().__init__(task_name, "classification", "node_labels", False)
    
    def get_target(self, data: Data) -> torch.Tensor:
        if hasattr(data, 'node_labels'):
            return data.node_labels
        else:
            raise ValueError("No node-level targets found")
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=-1)
        else:
            pred_classes = predictions.round().long()
        
        accuracy = (pred_classes == targets).float().mean()
        return {'accuracy': float(accuracy)}
"""Edge-level tasks for connectome analysis."""

import torch
from torch_geometric.data import Data
from typing import Dict
from .base import BaseConnectomeTask


class EdgeLevelTask(BaseConnectomeTask):
    """Base class for edge-level tasks."""
    
    def __init__(self, task_name: str = "edge_prediction"):
        super().__init__(task_name, "regression", "edge_weights", True)
    
    def get_target(self, data: Data) -> torch.Tensor:
        if hasattr(data, 'edge_weights'):
            return data.edge_weights
        elif hasattr(data, 'edge_attr'):
            return data.edge_attr.squeeze()
        else:
            raise ValueError("No edge-level targets found")
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        mse = torch.mean((predictions - targets) ** 2)
        mae = torch.mean(torch.abs(predictions - targets))
        return {'mse': float(mse), 'mae': float(mae)}
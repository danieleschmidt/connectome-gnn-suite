"""Edge-level tasks for connection predictions."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score

from .base import BaseConnectomeTask, RegressionTask, ClassificationTask


class EdgeLevelTask(BaseConnectomeTask):
    """Base class for edge-level (connection) prediction tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Extract edge-level targets from data samples.
        
        Args:
            data_list: List of connectome data objects
            
        Returns:
            Tensor of targets [total_edges]
        """
        all_targets = []
        
        for data in data_list:
            if hasattr(data, 'edge_y') and data.edge_y is not None:
                # Direct edge targets available
                all_targets.append(data.edge_y)
            elif hasattr(data, 'edge_attr') and data.edge_attr is not None:
                # Use edge attributes as targets
                if hasattr(data, 'target_edge_indices'):
                    target_attrs = data.edge_attr[:, data.target_edge_indices]
                    all_targets.append(target_attrs)
                else:
                    # Use first column of edge attributes
                    all_targets.append(data.edge_attr[:, 0])
            else:
                raise ValueError("No edge-level target information available")
        
        return torch.cat(all_targets, dim=0)


class ConnectionPrediction(EdgeLevelTask, RegressionTask):
    """Predict connection strengths between brain regions.
    
    Predicts the strength or weight of connections in brain networks
    based on structural, functional, or other connectivity measures.
    
    Args:
        connection_type: Type of connection to predict ('structural', 'functional', 'effective')
        normalize: Whether to normalize connection strengths
        edge_threshold: Minimum connection strength to consider
    """
    
    def __init__(
        self,
        connection_type: str = "structural",
        normalize: bool = True,
        edge_threshold: float = 0.0,
        **kwargs
    ):
        super().__init__(
            target_name=f"{connection_type}_strength",
            normalize=normalize,
            **kwargs
        )
        
        self.connection_type = connection_type
        self.edge_threshold = edge_threshold
        
        # Connection type characteristics
        self.connection_info = {
            'structural': {
                'range': (0.0, 1.0),
                'description': 'Structural connectivity strength (DTI-based)',
                'unit': 'normalized'
            },
            'functional': {
                'range': (-1.0, 1.0),
                'description': 'Functional connectivity strength (correlation)',
                'unit': 'correlation'
            },
            'effective': {
                'range': (-2.0, 2.0),
                'description': 'Effective connectivity strength (causal)',
                'unit': 'causal strength'
            }
        }
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Prepare connection strength targets with thresholding."""
        targets = super().prepare_targets(data_list)
        
        # Apply edge threshold if specified
        if self.edge_threshold > 0:
            targets = torch.where(torch.abs(targets) >= self.edge_threshold, targets, torch.zeros_like(targets))
        
        return targets
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the connection type being predicted."""
        conn_info = self.connection_info.get(self.connection_type, {
            'range': (0, 1),
            'description': f'Connection strength: {self.connection_type}',
            'unit': 'unknown'
        })
        
        return {
            'connection_type': self.connection_type,
            'description': conn_info['description'],
            'typical_range': conn_info['range'],
            'unit': conn_info['unit'],
            'edge_threshold': self.edge_threshold,
            'normalize': self.normalize
        }


class LinkPrediction(EdgeLevelTask, ClassificationTask):
    """Binary prediction of link existence between brain regions.
    
    Predicts whether a connection exists between two brain regions
    based on network topology and node features.
    
    Args:
        threshold: Threshold for determining link existence
        negative_sampling_ratio: Ratio of negative samples to positive
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        negative_sampling_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__(
            target_name="link_existence",
            class_names=["no_connection", "connection"],
            **kwargs
        )
        
        self.threshold = threshold
        self.negative_sampling_ratio = negative_sampling_ratio
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Prepare binary link existence targets."""
        all_targets = []
        
        for data in data_list:
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                # Convert connection strengths to binary based on threshold
                edge_strengths = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr
                binary_targets = (torch.abs(edge_strengths) >= self.threshold).float()
                all_targets.append(binary_targets)
            else:
                # If no edge attributes, assume all edges exist (binary = 1)
                num_edges = data.edge_index.size(1)
                binary_targets = torch.ones(num_edges)
                all_targets.append(binary_targets)
        
        return torch.cat(all_targets, dim=0)
    
    def generate_negative_samples(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate negative samples for link prediction.
        
        Args:
            data: Graph data object
            
        Returns:
            Tuple of (negative_edge_index, negative_targets)
        """
        num_nodes = data.x.size(0)
        num_positive = data.edge_index.size(1)
        num_negative = int(num_positive * self.negative_sampling_ratio)
        
        # Create adjacency matrix to avoid sampling existing edges
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        adj[data.edge_index[0], data.edge_index[1]] = True
        
        # Sample negative edges
        negative_edges = []
        attempts = 0
        max_attempts = num_negative * 10
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            # Avoid self-loops and existing edges
            if src != dst and not adj[src, dst]:
                negative_edges.append([src, dst])
                adj[src, dst] = True  # Mark as sampled
            
            attempts += 1
        
        if len(negative_edges) == 0:
            # Fallback: create minimal negative samples
            negative_edge_index = torch.zeros((2, 0), dtype=torch.long)
            negative_targets = torch.zeros(0)
        else:
            negative_edge_index = torch.tensor(negative_edges, dtype=torch.long).t()
            negative_targets = torch.zeros(len(negative_edges))
        
        return negative_edge_index, negative_targets


class ConnectionStrengthRegression(ConnectionPrediction):
    """Regression task for predicting specific connection strengths."""
    
    def __init__(self, connection_type: str = "functional", **kwargs):
        super().__init__(connection_type=connection_type, **kwargs)
    
    def compute_connection_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute connection-specific regression metrics."""
        metrics = self.compute_metrics(predictions, targets)
        
        # Connection-specific analysis
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Denormalize if needed
        if self.normalize:
            pred_tensor = self.denormalize_predictions(predictions.squeeze())
            target_tensor = self.denormalize_predictions(targets)
            pred_np = pred_tensor.detach().cpu().numpy().flatten()
            target_np = target_tensor.detach().cpu().numpy().flatten()
        
        # Connection strength specific metrics
        strong_connections = np.abs(target_np) > 0.5
        if strong_connections.sum() > 0:
            strong_mae = np.mean(np.abs(pred_np[strong_connections] - target_np[strong_connections]))
            metrics['strong_connections_mae'] = strong_mae
        
        weak_connections = np.abs(target_np) <= 0.1
        if weak_connections.sum() > 0:
            weak_mae = np.mean(np.abs(pred_np[weak_connections] - target_np[weak_connections]))
            metrics['weak_connections_mae'] = weak_mae
        
        # Correlation analysis for different connection strengths
        if self.connection_type == "functional":
            positive_corr = target_np > 0.1
            negative_corr = target_np < -0.1
            
            if positive_corr.sum() > 1:
                pos_corr = np.corrcoef(pred_np[positive_corr], target_np[positive_corr])[0, 1]
                metrics['positive_connections_correlation'] = pos_corr if not np.isnan(pos_corr) else 0.0
            
            if negative_corr.sum() > 1:
                neg_corr = np.corrcoef(pred_np[negative_corr], target_np[negative_corr])[0, 1]
                metrics['negative_connections_correlation'] = neg_corr if not np.isnan(neg_corr) else 0.0
        
        return metrics


class MissingLinkPrediction(LinkPrediction):
    """Predict missing links in incomplete connectome data."""
    
    def __init__(self, mask_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
    
    def create_missing_link_task(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create missing link prediction task by masking existing edges.
        
        Args:
            data: Complete graph data
            
        Returns:
            Tuple of (masked_edge_index, missing_edges, missing_targets)
        """
        num_edges = data.edge_index.size(1)
        num_missing = int(num_edges * self.mask_ratio)
        
        # Randomly select edges to mask
        edge_indices = torch.randperm(num_edges)
        missing_indices = edge_indices[:num_missing]
        keep_indices = edge_indices[num_missing:]
        
        # Create masked graph
        masked_edge_index = data.edge_index[:, keep_indices]
        
        # Missing edges and their targets
        missing_edges = data.edge_index[:, missing_indices]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            missing_targets = (torch.abs(data.edge_attr[missing_indices, 0]) >= self.threshold).float()
        else:
            missing_targets = torch.ones(num_missing)  # All missing edges are positive
        
        return masked_edge_index, missing_edges, missing_targets


class DynamicLinkPrediction(EdgeLevelTask, ClassificationTask):
    """Predict dynamic changes in brain connectivity over time."""
    
    def __init__(
        self,
        change_threshold: float = 0.1,
        time_window: int = 10,
        **kwargs
    ):
        super().__init__(
            target_name="connectivity_change",
            class_names=["stable", "changed"],
            **kwargs
        )
        
        self.change_threshold = change_threshold
        self.time_window = time_window
    
    def detect_connectivity_changes(
        self, 
        connectivity_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Detect significant changes in connectivity over time.
        
        Args:
            connectivity_sequence: Time series of connectivity [time_steps, num_edges]
            
        Returns:
            Binary tensor indicating significant changes [num_edges]
        """
        if connectivity_sequence.size(0) < 2:
            return torch.zeros(connectivity_sequence.size(1))
        
        # Compute changes between consecutive time points
        changes = torch.diff(connectivity_sequence, dim=0)
        
        # Detect significant changes
        max_changes = torch.max(torch.abs(changes), dim=0)[0]
        significant_changes = (max_changes >= self.change_threshold).float()
        
        return significant_changes


class EdgeTaskSuite:
    """Suite of standardized edge-level connectome tasks."""
    
    def __init__(self):
        self.available_tasks = {
            'connection_prediction': {
                'structural_strength': lambda: ConnectionStrengthRegression(connection_type="structural"),
                'functional_strength': lambda: ConnectionStrengthRegression(connection_type="functional"),
                'effective_strength': lambda: ConnectionStrengthRegression(connection_type="effective"),
            },
            'link_prediction': {
                'binary_links': LinkPrediction,
                'missing_links': MissingLinkPrediction,
                'dynamic_changes': DynamicLinkPrediction,
            },
            'connectivity_analysis': {
                'threshold_prediction': lambda: ConnectionPrediction(edge_threshold=0.1),
                'correlation_prediction': lambda: ConnectionPrediction(connection_type="functional"),
            }
        }
    
    def get_task(self, category: str, task_name: str, **kwargs) -> BaseConnectomeTask:
        """Get a specific edge-level task."""
        if category not in self.available_tasks:
            raise ValueError(f"Unknown category: {category}")
        
        if task_name not in self.available_tasks[category]:
            raise ValueError(f"Unknown task: {task_name}")
        
        task_class = self.available_tasks[category][task_name]
        
        if callable(task_class):
            return task_class(**kwargs)
        else:
            return task_class(**kwargs)
    
    def list_tasks(self) -> Dict[str, List[str]]:
        """List all available edge-level tasks."""
        return {category: list(tasks.keys()) for category, tasks in self.available_tasks.items()}
    
    def create_link_prediction_benchmark(self, mask_ratios: List[float] = None) -> List[BaseConnectomeTask]:
        """Create benchmark suite for link prediction tasks."""
        if mask_ratios is None:
            mask_ratios = [0.05, 0.1, 0.2]
        
        benchmark_tasks = []
        for ratio in mask_ratios:
            task = MissingLinkPrediction(mask_ratio=ratio)
            benchmark_tasks.append(task)
        
        return benchmark_tasks
"""Node-level tasks for brain region predictions."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any

from .base import BaseConnectomeTask, RegressionTask, ClassificationTask


class NodeLevelTask(BaseConnectomeTask):
    """Base class for node-level (brain region) prediction tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Extract node-level targets from data samples.
        
        Args:
            data_list: List of connectome data objects
            
        Returns:
            Tensor of targets [total_nodes]
        """
        all_targets = []
        
        for data in data_list:
            if hasattr(data, 'node_y') and data.node_y is not None:
                # Direct node targets available
                all_targets.append(data.node_y)
            elif hasattr(data, 'x') and data.x is not None:
                # Extract from node features if target is embedded
                if hasattr(data, 'target_node_indices'):
                    target_features = data.x[:, data.target_node_indices]
                    all_targets.append(target_features)
                else:
                    raise ValueError("No node-level targets or target indices specified")
            else:
                raise ValueError("No node-level target information available")
        
        return torch.cat(all_targets, dim=0)


class BrainRegionPrediction(NodeLevelTask, RegressionTask):
    """Predict properties of individual brain regions.
    
    Examples include regional volume, cortical thickness, surface area,
    metabolic activity, or other region-specific measurements.
    
    Args:
        target_property: Name of regional property to predict
        atlas: Brain atlas used for region definition
        normalize: Whether to normalize target values
    """
    
    def __init__(
        self,
        target_property: str = "cortical_thickness",
        atlas: str = "AAL",
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(
            target_name=target_property,
            normalize=normalize,
            **kwargs
        )
        
        self.target_property = target_property
        self.atlas = atlas
        
        # Common regional properties and their characteristics
        self.regional_properties = {
            'cortical_thickness': {
                'range': (1.0, 5.0),
                'unit': 'mm',
                'description': 'Cortical thickness measurement'
            },
            'volume': {
                'range': (100, 50000),
                'unit': 'mm³',
                'description': 'Regional brain volume'
            },
            'surface_area': {
                'range': (10, 5000),
                'unit': 'mm²',
                'description': 'Cortical surface area'
            },
            'myelin_content': {
                'range': (0.5, 2.0),
                'unit': 'T1w/T2w ratio',
                'description': 'Myelin content proxy'
            },
            'metabolism': {
                'range': (0.5, 2.5),
                'unit': 'normalized',
                'description': 'Metabolic activity measure'
            },
            'connectivity_strength': {
                'range': (0.0, 1.0),
                'unit': 'normalized',
                'description': 'Average connectivity strength'
            }
        }
    
    def get_property_info(self) -> Dict[str, Any]:
        """Get information about the regional property being predicted."""
        prop_info = self.regional_properties.get(self.target_property, {
            'range': (0, 1),
            'unit': 'unknown',
            'description': f'Regional property: {self.target_property}'
        })
        
        return {
            'property': self.target_property,
            'atlas': self.atlas,
            'description': prop_info['description'],
            'typical_range': prop_info['range'],
            'unit': prop_info['unit'],
            'task_type': 'regression',
            'normalize': self.normalize
        }


class RegionClassification(NodeLevelTask, ClassificationTask):
    """Binary classification of brain regions.
    
    Examples include lesion detection, abnormal region identification,
    or functional network assignment.
    
    Args:
        target: Name of classification target
        positive_class: Label for positive class (e.g., 'lesion', 'abnormal')
        negative_class: Label for negative class (e.g., 'healthy', 'normal')
    """
    
    def __init__(
        self,
        target: str = "lesion_presence",
        positive_class: str = "lesion",
        negative_class: str = "healthy",
        **kwargs
    ):
        super().__init__(
            target_name=target,
            class_names=[negative_class, positive_class],
            **kwargs
        )
        
        self.positive_class = positive_class
        self.negative_class = negative_class


class CorticalThicknessPrediction(BrainRegionPrediction):
    """Predict cortical thickness from connectivity patterns."""
    
    def __init__(self, atlas: str = "AAL", **kwargs):
        super().__init__(
            target_property="cortical_thickness",
            atlas=atlas,
            **kwargs
        )
    
    def compute_thickness_statistics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute cortical thickness-specific statistics."""
        # Standard regression metrics
        metrics = self.compute_metrics(predictions, targets)
        
        # Thickness-specific analysis
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Denormalize if needed
        if self.normalize:
            pred_tensor = self.denormalize_predictions(predictions.squeeze())
            target_tensor = self.denormalize_predictions(targets)
            pred_np = pred_tensor.detach().cpu().numpy().flatten()
            target_np = target_tensor.detach().cpu().numpy().flatten()
        
        # Thickness-specific metrics
        thickness_diff = pred_np - target_np
        metrics.update({
            'mean_thickness_diff': float(np.mean(thickness_diff)),
            'std_thickness_diff': float(np.std(thickness_diff)),
            'max_thickness_error': float(np.max(np.abs(thickness_diff))),
            'regions_within_0.1mm': float(np.mean(np.abs(thickness_diff) < 0.1)),
            'regions_within_0.2mm': float(np.mean(np.abs(thickness_diff) < 0.2)),
        })
        
        return metrics


class VolumetricPrediction(BrainRegionPrediction):
    """Predict regional brain volumes from connectivity."""
    
    def __init__(self, atlas: str = "AAL", **kwargs):
        super().__init__(
            target_property="volume",
            atlas=atlas,
            **kwargs
        )
    
    def compute_volume_statistics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute volume-specific statistics."""
        metrics = self.compute_metrics(predictions, targets)
        
        # Volume-specific analysis
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        if self.normalize:
            pred_tensor = self.denormalize_predictions(predictions.squeeze())
            target_tensor = self.denormalize_predictions(targets)
            pred_np = pred_tensor.detach().cpu().numpy().flatten()
            target_np = target_tensor.detach().cpu().numpy().flatten()
        
        # Volume-specific metrics
        volume_ratio = pred_np / (target_np + 1e-8)
        metrics.update({
            'mean_volume_ratio': float(np.mean(volume_ratio)),
            'std_volume_ratio': float(np.std(volume_ratio)),
            'regions_within_10_percent': float(np.mean(np.abs(volume_ratio - 1) < 0.1)),
            'regions_within_20_percent': float(np.mean(np.abs(volume_ratio - 1) < 0.2)),
        })
        
        return metrics


class LesionDetection(RegionClassification):
    """Detect lesions in brain regions from connectivity patterns."""
    
    def __init__(self, **kwargs):
        super().__init__(
            target="lesion_presence",
            positive_class="lesion",
            negative_class="healthy",
            **kwargs
        )
    
    def compute_detection_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute lesion detection-specific metrics."""
        metrics = self.compute_metrics(predictions, targets)
        
        # Lesion detection specific metrics
        probs = torch.sigmoid(predictions.squeeze()).detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy().astype(int)
        
        # Different threshold analysis
        thresholds = [0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            pred_classes = (probs > thresh).astype(int)
            accuracy = np.mean(pred_classes == target_np)
            
            # Sensitivity and specificity
            tp = np.sum((pred_classes == 1) & (target_np == 1))
            tn = np.sum((pred_classes == 0) & (target_np == 0))
            fp = np.sum((pred_classes == 1) & (target_np == 0))
            fn = np.sum((pred_classes == 0) & (target_np == 1))
            
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            
            metrics[f'accuracy_thresh_{thresh}'] = accuracy
            metrics[f'sensitivity_thresh_{thresh}'] = sensitivity
            metrics[f'specificity_thresh_{thresh}'] = specificity
        
        return metrics


class FunctionalNetworkAssignment(NodeLevelTask, ClassificationTask):
    """Assign brain regions to functional networks.
    
    Predicts which functional network each brain region belongs to
    based on connectivity patterns.
    """
    
    def __init__(
        self,
        networks: List[str] = None,
        atlas: str = "AAL",
        **kwargs
    ):
        if networks is None:
            networks = [
                "visual", "somatomotor", "dorsal_attention",
                "ventral_attention", "limbic", "frontoparietal", "default"
            ]
        
        # Convert to multi-class task
        super().__init__(
            target_name="functional_network",
            class_names=networks,
            **kwargs
        )
        
        self.networks = networks
        self.atlas = atlas
        self.num_classes = len(networks)
        self.task_type = "multi_class"
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class cross-entropy loss."""
        return torch.nn.functional.cross_entropy(predictions, targets.long())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute network assignment metrics."""
        pred_classes = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy().astype(int)
        
        # Overall accuracy
        accuracy = np.mean(pred_classes == target_np)
        metrics = {'accuracy': accuracy}
        
        # Per-network metrics
        for i, network in enumerate(self.networks):
            network_mask = (target_np == i)
            if network_mask.sum() > 0:
                network_accuracy = np.mean(pred_classes[network_mask] == target_np[network_mask])
                metrics[f'{network}_accuracy'] = network_accuracy
        
        return metrics


class NodeTaskSuite:
    """Suite of standardized node-level connectome tasks."""
    
    def __init__(self):
        self.available_tasks = {
            'structural_prediction': {
                'cortical_thickness': CorticalThicknessPrediction,
                'volume': VolumetricPrediction,
                'surface_area': lambda: BrainRegionPrediction(target_property="surface_area"),
            },
            'pathology_detection': {
                'lesion_detection': LesionDetection,
                'abnormality_detection': lambda: RegionClassification(target="abnormal"),
            },
            'functional_assignment': {
                'network_assignment': FunctionalNetworkAssignment,
            }
        }
    
    def get_task(self, category: str, task_name: str, **kwargs) -> BaseConnectomeTask:
        """Get a specific node-level task."""
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
        """List all available node-level tasks."""
        return {category: list(tasks.keys()) for category, tasks in self.available_tasks.items()}
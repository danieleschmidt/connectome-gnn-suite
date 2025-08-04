"""Graph-level tasks for subject-level predictions."""

import torch
from typing import Dict, List, Optional, Any
import pandas as pd

from .base import BaseConnectomeTask, RegressionTask, ClassificationTask, MultiClassTask


class GraphLevelTask(BaseConnectomeTask):
    """Base class for graph-level (subject-level) prediction tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Extract graph-level targets from data samples.
        
        Args:
            data_list: List of connectome data objects
            
        Returns:
            Tensor of targets [num_graphs]
        """
        targets = []
        
        for data in data_list:
            if hasattr(data, 'y') and data.y is not None:
                # Direct target available
                targets.append(data.y)
            elif hasattr(data, 'demographics') and isinstance(data.demographics, dict):
                # Extract from demographics
                if self.target_name in data.demographics:
                    targets.append(data.demographics[self.target_name])
                else:
                    raise ValueError(f"Target '{self.target_name}' not found in demographics")
            else:
                raise ValueError(f"No target information available in data")
        
        return torch.tensor(targets, dtype=torch.float32)


class CognitiveScorePrediction(GraphLevelTask, RegressionTask):
    """Predict cognitive scores from brain connectivity.
    
    Common cognitive measures include fluid intelligence, working memory,
    processing speed, and other neuropsychological assessments.
    
    Args:
        target: Name of cognitive measure to predict
        normalize: Whether to normalize target scores
    """
    
    def __init__(
        self,
        target: str = "fluid_intelligence",
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(
            target_name=target,
            normalize=normalize,
            **kwargs
        )
        
        # Common cognitive measures and their typical ranges
        self.cognitive_measures = {
            'fluid_intelligence': {'range': (70, 130), 'description': 'Fluid intelligence score'},
            'working_memory': {'range': (0, 100), 'description': 'Working memory capacity'},
            'processing_speed': {'range': (0, 100), 'description': 'Processing speed index'},
            'verbal_comprehension': {'range': (70, 130), 'description': 'Verbal comprehension index'},
            'perceptual_reasoning': {'range': (70, 130), 'description': 'Perceptual reasoning index'},
            'executive_function': {'range': (0, 100), 'description': 'Executive function composite'},
            'attention': {'range': (0, 100), 'description': 'Attention composite score'},
            'memory': {'range': (0, 100), 'description': 'Memory composite score'}
        }
    
    def get_cognitive_info(self) -> Dict[str, Any]:
        """Get information about the cognitive measure being predicted."""
        measure_info = self.cognitive_measures.get(self.target_name, {
            'range': (0, 100),
            'description': f'Cognitive measure: {self.target_name}'
        })
        
        return {
            'measure': self.target_name,
            'description': measure_info['description'],
            'typical_range': measure_info['range'],
            'task_type': 'regression',
            'normalize': self.normalize
        }


class SubjectClassification(GraphLevelTask, ClassificationTask):
    """Binary classification of subjects based on connectome patterns.
    
    Examples include disease diagnosis, cognitive state classification,
    or demographic group prediction.
    
    Args:
        target: Name of classification target
        positive_class: Label for positive class
        negative_class: Label for negative class
        class_weights: Optional class weights for imbalanced data
    """
    
    def __init__(
        self,
        target: str = "diagnosis",
        positive_class: str = "patient",
        negative_class: str = "control",
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(
            target_name=target,
            class_names=[negative_class, positive_class],
            class_weights=class_weights,
            **kwargs
        )
        
        self.positive_class = positive_class
        self.negative_class = negative_class
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Prepare binary classification targets."""
        targets = []
        
        for data in data_list:
            target_value = None
            
            if hasattr(data, 'y') and data.y is not None:
                target_value = data.y
            elif hasattr(data, 'demographics') and isinstance(data.demographics, dict):
                if self.target_name in data.demographics:
                    target_value = data.demographics[self.target_name]
            
            if target_value is None:
                raise ValueError(f"Target '{self.target_name}' not found")
            
            # Convert to binary (0/1)
            if isinstance(target_value, str):
                binary_target = 1 if target_value.lower() == self.positive_class.lower() else 0
            else:
                binary_target = int(target_value)
            
            targets.append(binary_target)
        
        return torch.tensor(targets, dtype=torch.float32)


class AgeRegression(GraphLevelTask, RegressionTask):
    """Predict age from brain connectivity patterns.
    
    Age prediction is commonly used to study brain aging and compute
    'brain age gap' - the difference between predicted and chronological age.
    """
    
    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(
            target_name="age",
            normalize=normalize,
            **kwargs
        )
    
    def compute_brain_age_gap(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute brain age gap (predicted - chronological age).
        
        Args:
            predictions: Predicted ages
            targets: Chronological ages
            
        Returns:
            Brain age gap values
        """
        # Denormalize if needed
        if self.normalize:
            pred_ages = self.denormalize_predictions(predictions.squeeze())
            chron_ages = self.denormalize_predictions(targets)
        else:
            pred_ages = predictions.squeeze()
            chron_ages = targets
        
        return pred_ages - chron_ages
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute age regression metrics including brain age gap statistics."""
        # Standard regression metrics
        metrics = super().compute_metrics(predictions, targets)
        
        # Brain age gap analysis
        brain_age_gap = self.compute_brain_age_gap(predictions, targets)
        gap_np = brain_age_gap.detach().cpu().numpy()
        
        metrics.update({
            'brain_age_gap_mean': float(gap_np.mean()),
            'brain_age_gap_std': float(gap_np.std()),
            'brain_age_gap_abs_mean': float(np.abs(gap_np).mean()),
        })
        
        return metrics


class SexClassification(GraphLevelTask, ClassificationTask):
    """Classify biological sex from brain connectivity."""
    
    def __init__(self, **kwargs):
        super().__init__(
            target_name="sex",
            positive_class="M",
            negative_class="F",
            class_names=["Female", "Male"],
            **kwargs
        )


class ClinicalDiagnosis(GraphLevelTask, MultiClassTask):
    """Multi-class diagnosis prediction from connectome data.
    
    Args:
        conditions: List of clinical conditions to classify
        control_label: Label for healthy controls
    """
    
    def __init__(
        self,
        conditions: List[str] = ["autism", "schizophrenia", "depression"],
        control_label: str = "control",
        **kwargs
    ):
        all_classes = [control_label] + conditions
        super().__init__(
            target_name="diagnosis",
            num_classes=len(all_classes),
            class_names=all_classes,
            **kwargs
        )
        
        self.conditions = conditions
        self.control_label = control_label
        self.label_to_index = {label: i for i, label in enumerate(all_classes)}
    
    def prepare_targets(self, data_list: List[Any]) -> torch.Tensor:
        """Prepare multi-class diagnosis targets."""
        targets = []
        
        for data in data_list:
            target_value = None
            
            if hasattr(data, 'y') and data.y is not None:
                target_value = data.y
            elif hasattr(data, 'demographics') and isinstance(data.demographics, dict):
                if self.target_name in data.demographics:
                    target_value = data.demographics[self.target_name]
            
            if target_value is None:
                raise ValueError(f"Target '{self.target_name}' not found")
            
            # Convert to class index
            if isinstance(target_value, str):
                class_index = self.label_to_index.get(target_value.lower(), 0)
            else:
                class_index = int(target_value)
            
            targets.append(class_index)
        
        return torch.tensor(targets, dtype=torch.long)


class ConnectomeTaskSuite:
    """Suite of standardized connectome prediction tasks.
    
    Provides easy access to common neuroimaging prediction tasks
    with standardized evaluation protocols.
    """
    
    def __init__(self):
        self.available_tasks = {
            'cognitive_prediction': {
                'fluid_intelligence': CognitiveScorePrediction,
                'working_memory': lambda: CognitiveScorePrediction(target="working_memory"),
                'processing_speed': lambda: CognitiveScorePrediction(target="processing_speed"),
            },
            'demographic_prediction': {
                'age_regression': AgeRegression,
                'sex_classification': SexClassification,
            },
            'clinical_diagnosis': {
                'autism_diagnosis': lambda: SubjectClassification(target="autism", positive_class="autism"),
                'adhd_diagnosis': lambda: SubjectClassification(target="adhd", positive_class="adhd"),
                'multi_diagnosis': ClinicalDiagnosis,
            }
        }
    
    def get_task(self, category: str, task_name: str, **kwargs) -> BaseConnectomeTask:
        """Get a specific task from the suite.
        
        Args:
            category: Task category ('cognitive_prediction', 'demographic_prediction', etc.)
            task_name: Specific task name
            **kwargs: Additional task parameters
            
        Returns:
            Configured task instance
        """
        if category not in self.available_tasks:
            raise ValueError(f"Unknown category: {category}")
        
        if task_name not in self.available_tasks[category]:
            raise ValueError(f"Unknown task: {task_name} in category {category}")
        
        task_class = self.available_tasks[category][task_name]
        
        if callable(task_class):
            if kwargs:
                # If it's a lambda, we need to create a new one with kwargs
                return task_class(**kwargs) if hasattr(task_class, '__name__') else task_class()
            else:
                return task_class()
        else:
            return task_class(**kwargs)
    
    def list_tasks(self) -> Dict[str, List[str]]:
        """List all available tasks by category."""
        return {category: list(tasks.keys()) for category, tasks in self.available_tasks.items()}
    
    def create_benchmark_suite(self) -> List[BaseConnectomeTask]:
        """Create a standard benchmark suite for connectome analysis."""
        benchmark_tasks = [
            self.get_task('cognitive_prediction', 'fluid_intelligence'),
            self.get_task('demographic_prediction', 'age_regression'),
            self.get_task('demographic_prediction', 'sex_classification'),
        ]
        
        return benchmark_tasks
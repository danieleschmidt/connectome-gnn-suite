"""Test suite for connectome tasks and evaluation."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from connectome_gnn.tasks.base import (
    BaseConnectomeTask, RegressionTask, ClassificationTask, MultiClassTask
)
from connectome_gnn.tasks.graph_level import (
    CognitiveScorePrediction, AgeRegression, SexClassification,
    SubjectClassification, ClinicalDiagnosis, ConnectomeTaskSuite
)
from connectome_gnn.tasks.node_level import (
    NodeRegression, NodeClassification, BrainRegionPrediction
)
from connectome_gnn.tasks.edge_level import (
    EdgePrediction, ConnectivityStrengthPrediction
)


class MockData:
    """Mock data class for testing."""
    
    def __init__(self, target_value=None, demographics=None):
        self.y = target_value
        self.demographics = demographics or {}


class TestBaseConnectomeTask:
    """Test base task functionality."""
    
    def test_task_initialization(self):
        """Test task initialization."""
        task = RegressionTask(target_name="age", normalize=True)
        
        assert task.task_type == "regression"
        assert task.target_name == "age"
        assert task.normalize == True
        assert task.num_classes == 1
    
    def test_invalid_task_type(self):
        """Test initialization with invalid task type."""
        with pytest.raises(ValueError):
            BaseConnectomeTask(task_type="invalid")
    
    def test_target_statistics_fitting(self):
        """Test fitting target statistics."""
        task = RegressionTask()
        
        # Sample targets
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        task.fit_target_statistics(targets)
        
        assert task.target_mean is not None
        assert task.target_std is not None
        assert abs(task.target_mean - 3.0) < 1e-6
    
    def test_target_normalization(self):
        """Test target normalization."""
        task = RegressionTask(normalize=True)
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        task.fit_target_statistics(targets)
        
        normalized = task.normalize_targets(targets)
        
        # Should have approximately zero mean and unit variance
        assert abs(torch.mean(normalized)) < 1e-6
        assert abs(torch.std(normalized) - 1.0) < 1e-6
    
    def test_denormalization(self):
        """Test denormalization of predictions."""
        task = RegressionTask(normalize=True)
        
        targets = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        task.fit_target_statistics(targets)
        
        # Normalize then denormalize
        normalized = task.normalize_targets(targets)
        denormalized = task.denormalize_predictions(normalized)
        
        # Should recover original values
        assert torch.allclose(targets, denormalized, rtol=1e-5)


class TestRegressionTask:
    """Test regression task implementation."""
    
    def test_loss_computation(self):
        """Test MSE loss computation."""
        task = RegressionTask()
        
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])
        
        loss = task.compute_loss(predictions, targets)
        
        assert loss.item() > 0
        assert loss.item() < 1.0  # Should be small for close predictions
    
    def test_regression_metrics(self):
        """Test regression metrics computation."""
        task = RegressionTask()
        
        # Perfect predictions
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = task.compute_metrics(predictions, targets)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'correlation' in metrics
        assert 'r2' in metrics
        
        # Perfect predictions should have near-zero error
        assert metrics['mae'] < 1e-6
        assert metrics['mse'] < 1e-6
        assert abs(metrics['correlation'] - 1.0) < 1e-6
        assert abs(metrics['r2'] - 1.0) < 1e-6
    
    def test_regression_with_noise(self):
        """Test regression metrics with noisy predictions."""
        task = RegressionTask()
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = targets + torch.randn_like(targets) * 0.1  # Add noise
        
        metrics = task.compute_metrics(predictions, targets)
        
        # Should have some error but good correlation
        assert metrics['mae'] > 0
        assert metrics['correlation'] > 0.8  # Should still be highly correlated


class TestClassificationTask:
    """Test binary classification task."""
    
    def test_classification_loss(self):
        """Test binary cross-entropy loss."""
        task = ClassificationTask()
        
        # Logits (before sigmoid)
        predictions = torch.tensor([2.0, -1.0, 3.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        
        loss = task.compute_loss(predictions, targets)
        
        assert loss.item() > 0
    
    def test_classification_metrics(self):
        """Test classification metrics."""
        task = ClassificationTask()
        
        # Perfect predictions (high confidence)
        predictions = torch.tensor([5.0, -5.0, 5.0, -5.0])  # Logits
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        metrics = task.compute_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Perfect predictions
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_imbalanced_classification(self):
        """Test classification with imbalanced data."""
        task = ClassificationTask()
        
        # Imbalanced: mostly negative class
        predictions = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0])
        targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        
        metrics = task.compute_metrics(predictions, targets)
        
        # Should handle imbalanced data
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1


class TestMultiClassTask:
    """Test multi-class classification task."""
    
    def test_multiclass_initialization(self):
        """Test multi-class task initialization."""
        task = MultiClassTask(
            num_classes=3,
            class_names=["class_0", "class_1", "class_2"]
        )
        
        assert task.num_classes == 3
        assert len(task.class_names) == 3
    
    def test_multiclass_loss(self):
        """Test cross-entropy loss."""
        task = MultiClassTask(num_classes=3)
        
        # Logits for 3 classes
        predictions = torch.tensor([
            [2.0, 1.0, 0.5],  # Class 0
            [0.5, 2.0, 1.0],  # Class 1
            [1.0, 0.5, 2.0]   # Class 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        loss = task.compute_loss(predictions, targets)
        
        assert loss.item() > 0
    
    def test_multiclass_metrics(self):
        """Test multi-class metrics."""
        task = MultiClassTask(
            num_classes=3,
            class_names=["A", "B", "C"]
        )
        
        # Perfect predictions
        predictions = torch.tensor([
            [5.0, 0.0, 0.0],  # Class A
            [0.0, 5.0, 0.0],  # Class B
            [0.0, 0.0, 5.0]   # Class C
        ])
        targets = torch.tensor([0, 1, 2])
        
        metrics = task.compute_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'A_precision' in metrics
        assert 'B_recall' in metrics
        assert 'C_f1' in metrics
        assert 'macro_precision' in metrics
        assert 'macro_recall' in metrics
        assert 'macro_f1' in metrics
        
        # Perfect predictions
        assert metrics['accuracy'] == 1.0
        assert metrics['macro_precision'] == 1.0


class TestGraphLevelTasks:
    """Test graph-level prediction tasks."""
    
    def test_cognitive_score_prediction(self):
        """Test cognitive score prediction task."""
        task = CognitiveScorePrediction(target="fluid_intelligence")
        
        assert task.target_name == "fluid_intelligence"
        assert task.task_type == "regression"
        
        # Test cognitive info
        info = task.get_cognitive_info()
        assert 'measure' in info
        assert 'description' in info
        assert 'typical_range' in info
    
    def test_age_regression(self):
        """Test age regression task."""
        task = AgeRegression()
        
        assert task.target_name == "age"
        assert task.task_type == "regression"
        
        # Test brain age gap computation
        predictions = torch.tensor([25.0, 30.0, 35.0])
        targets = torch.tensor([24.0, 32.0, 33.0])
        
        brain_age_gap = task.compute_brain_age_gap(predictions, targets)
        expected_gap = torch.tensor([1.0, -2.0, 2.0])
        
        assert torch.allclose(brain_age_gap, expected_gap)
    
    def test_sex_classification(self):
        """Test sex classification task."""
        task = SexClassification()
        
        assert task.target_name == "sex"
        assert task.task_type == "classification"
        assert "Male" in task.class_names
        assert "Female" in task.class_names
    
    def test_clinical_diagnosis(self):
        """Test clinical diagnosis task."""
        conditions = ["autism", "adhd", "depression"]
        task = ClinicalDiagnosis(conditions=conditions)
        
        assert task.task_type == "multi_class"
        assert task.num_classes == 4  # 3 conditions + control
        assert "control" in task.class_names
        assert "autism" in task.class_names
    
    def test_target_preparation(self):
        """Test target preparation from data."""
        task = AgeRegression()
        
        # Create mock data with demographics
        data_list = [
            MockData(demographics={"age": 25, "sex": "M"}),
            MockData(demographics={"age": 30, "sex": "F"}),
            MockData(demographics={"age": 35, "sex": "M"})
        ]
        
        targets = task.prepare_targets(data_list)
        expected = torch.tensor([25.0, 30.0, 35.0])
        
        assert torch.equal(targets, expected)
    
    def test_target_preparation_from_y(self):
        """Test target preparation from direct y values."""
        task = RegressionTask(target_name="score")
        
        data_list = [
            MockData(target_value=1.5),
            MockData(target_value=2.3),
            MockData(target_value=3.1)
        ]
        
        targets = task.prepare_targets(data_list)
        expected = torch.tensor([1.5, 2.3, 3.1])
        
        assert torch.allclose(targets, expected)


class TestNodeLevelTasks:
    """Test node-level prediction tasks."""
    
    def test_brain_region_prediction(self):
        """Test brain region prediction task."""
        task = BrainRegionPrediction(
            region_mapping={"frontal": 0, "parietal": 1, "temporal": 2}
        )
        
        assert task.task_type == "multi_class"
        assert task.num_classes == 3
        
        # Test region encoding
        regions = ["frontal", "parietal", "temporal", "frontal"]
        encoded = task.encode_regions(regions)
        expected = torch.tensor([0, 1, 2, 0])
        
        assert torch.equal(encoded, expected)
    
    def test_node_regression(self):
        """Test node-level regression."""
        task = NodeRegression(target_name="activation")
        
        assert task.task_type == "regression"
        
        # Create mock node data
        num_nodes = 90
        predictions = torch.randn(num_nodes, 1)
        targets = torch.randn(num_nodes)
        
        loss = task.compute_loss(predictions, targets)
        metrics = task.compute_metrics(predictions, targets)
        
        assert loss.item() > 0
        assert 'mae' in metrics
        assert 'correlation' in metrics


class TestEdgeLevelTasks:
    """Test edge-level prediction tasks."""
    
    def test_edge_prediction(self):
        """Test edge prediction task."""
        task = EdgePrediction()
        
        assert task.task_type == "classification"
        
        # Test edge predictions
        num_edges = 100
        predictions = torch.randn(num_edges, 1)  # Logits
        targets = torch.randint(0, 2, (num_edges,)).float()
        
        loss = task.compute_loss(predictions, targets)
        metrics = task.compute_metrics(predictions, targets)
        
        assert loss.item() > 0
        assert 'accuracy' in metrics
        assert 'auc' in metrics
    
    def test_connectivity_strength_prediction(self):
        """Test connectivity strength prediction."""
        task = ConnectivityStrengthPrediction()
        
        assert task.task_type == "regression"
        
        # Test strength predictions
        num_edges = 200
        predictions = torch.randn(num_edges, 1)
        targets = torch.abs(torch.randn(num_edges))  # Positive strengths
        
        loss = task.compute_loss(predictions, targets)
        metrics = task.compute_metrics(predictions, targets)
        
        assert loss.item() > 0
        assert 'mae' in metrics


class TestConnectomeTaskSuite:
    """Test task suite functionality."""
    
    def test_task_suite_initialization(self):
        """Test task suite initialization."""
        suite = ConnectomeTaskSuite()
        
        assert 'cognitive_prediction' in suite.available_tasks
        assert 'demographic_prediction' in suite.available_tasks
        assert 'clinical_diagnosis' in suite.available_tasks
    
    def test_task_retrieval(self):
        """Test task retrieval from suite."""
        suite = ConnectomeTaskSuite()
        
        # Get cognitive task
        task = suite.get_task('cognitive_prediction', 'fluid_intelligence')
        assert isinstance(task, CognitiveScorePrediction)
        assert task.target_name == "fluid_intelligence"
        
        # Get demographic task
        task = suite.get_task('demographic_prediction', 'age_regression')
        assert isinstance(task, AgeRegression)
        
        # Get classification task
        task = suite.get_task('demographic_prediction', 'sex_classification')
        assert isinstance(task, SexClassification)
    
    def test_invalid_task_retrieval(self):
        """Test invalid task retrieval."""
        suite = ConnectomeTaskSuite()
        
        with pytest.raises(ValueError):
            suite.get_task('invalid_category', 'task_name')
        
        with pytest.raises(ValueError):
            suite.get_task('cognitive_prediction', 'invalid_task')
    
    def test_task_listing(self):
        """Test task listing."""
        suite = ConnectomeTaskSuite()
        
        tasks = suite.list_tasks()
        
        assert 'cognitive_prediction' in tasks
        assert 'fluid_intelligence' in tasks['cognitive_prediction']
        assert 'age_regression' in tasks['demographic_prediction']
    
    def test_benchmark_suite_creation(self):
        """Test benchmark suite creation."""
        suite = ConnectomeTaskSuite()
        
        benchmark_tasks = suite.create_benchmark_suite()
        
        assert len(benchmark_tasks) > 0
        assert any(isinstance(task, CognitiveScorePrediction) for task in benchmark_tasks)
        assert any(isinstance(task, AgeRegression) for task in benchmark_tasks)
        assert any(isinstance(task, SexClassification) for task in benchmark_tasks)


class TestTaskMetrics:
    """Test task-specific metrics computation."""
    
    def test_regression_metric_ranges(self):
        """Test that regression metrics are in expected ranges."""
        task = RegressionTask()
        
        # Random predictions
        predictions = torch.randn(100, 1)
        targets = torch.randn(100)
        
        metrics = task.compute_metrics(predictions, targets)
        
        # MAE and RMSE should be positive
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        
        # Correlation should be between -1 and 1
        assert -1 <= metrics['correlation'] <= 1
        
        # R² should be reasonable (can be negative for bad predictions)
        assert metrics['r2'] <= 1
    
    def test_classification_metric_ranges(self):
        """Test that classification metrics are in expected ranges."""
        task = ClassificationTask()
        
        # Random predictions
        predictions = torch.randn(100, 1)
        targets = torch.randint(0, 2, (100,)).float()
        
        metrics = task.compute_metrics(predictions, targets)
        
        # All metrics should be between 0 and 1
        for metric_name in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
            assert 0 <= metrics[metric_name] <= 1
        
        # Counts should be non-negative integers
        for count_name in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
            assert metrics[count_name] >= 0
            assert isinstance(metrics[count_name], (int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__])
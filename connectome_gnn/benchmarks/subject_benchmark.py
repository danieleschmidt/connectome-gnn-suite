"""Subject-level prediction benchmarks."""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
import json
import time

from ..models.base import BaseConnectomeModel
from ..tasks.base import BaseConnectomeTask
from ..advanced_training import AdvancedConnectomeTrainer, TrainingConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    metrics: List[str] = None
    save_predictions: bool = True
    save_models: bool = False
    verbose: bool = True


class SubjectBenchmark:
    """Comprehensive benchmarking suite for subject-level predictions.
    
    Supports multiple tasks:
    - Age prediction
    - Sex classification
    - Cognitive scores (7 different measures)
    - Clinical diagnosis (when available)
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        output_dir: Path = None
    ):
        self.config = config or BenchmarkConfig()
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default metrics per task type
        self.default_metrics = {
            'regression': ['mae', 'rmse', 'r2', 'pearson_r', 'spearman_r'],
            'binary_classification': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap'],
            'multiclass_classification': ['accuracy', 'precision', 'recall', 'f1', 'macro_f1'],
        }
        
        # Results storage
        self.results = {}
        self.detailed_results = {}
    
    def evaluate(
        self,
        model: BaseConnectomeModel,
        dataset,
        task: BaseConnectomeTask,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate model on dataset with comprehensive metrics.
        
        Args:
            model: Trained connectome model
            dataset: Dataset to evaluate on
            task: Prediction task
            metrics: List of metrics to compute
            
        Returns:
            Dictionary containing evaluation results
        """
        if metrics is None:
            metrics = self.default_metrics.get(task.task_type, ['mae', 'accuracy'])
        
        print(f"Starting evaluation with {len(dataset)} samples")
        print(f"Task: {task.get_task_info()}")
        print(f"Metrics: {metrics}")
        
        start_time = time.time()
        
        # Cross-validation evaluation
        cv_results = self._cross_validate(
            model, dataset, task, metrics
        )
        
        # Holdout evaluation
        holdout_results = self._holdout_evaluate(
            model, dataset, task, metrics
        )
        
        # Combine results
        evaluation_results = {
            'cross_validation': cv_results,
            'holdout': holdout_results,
            'task_info': task.get_task_info(),
            'model_info': {
                'name': model.__class__.__name__,
                'parameters': sum(p.numel() for p in model.parameters()),
            },
            'dataset_info': {
                'size': len(dataset),
                'features': dataset[0].x.shape[1] if hasattr(dataset[0], 'x') else 'unknown'
            },
            'evaluation_time': time.time() - start_time,
            'config': self.config.__dict__
        }
        
        # Save results
        self._save_results(evaluation_results, task.task_name)
        
        # Generate report
        self._generate_report(evaluation_results, task.task_name)
        
        return evaluation_results
    
    def _cross_validate(
        self,
        model: BaseConnectomeModel,
        dataset,
        task: BaseConnectomeTask,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        print(f"Running {self.config.n_folds}-fold cross-validation...")
        
        # Prepare stratification labels if needed
        if self.config.stratify and task.task_type in ['binary_classification', 'multiclass_classification']:
            y_labels = []
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, 'y') and data.y is not None:
                    if torch.is_tensor(data.y):
                        y_labels.append(int(data.y.item()))
                    else:
                        y_labels.append(int(data.y))
                else:
                    y_labels.append(0)  # Default label
            
            kfold = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            splits = list(kfold.split(range(len(dataset)), y_labels))
        else:
            kfold = KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            splits = list(kfold.split(range(len(dataset))))
        
        fold_results = []
        all_predictions = []
        all_targets = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            if self.config.verbose:
                print(f"  Fold {fold + 1}/{self.config.n_folds}")
            
            # Create fold datasets
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            # Clone and train model for this fold
            fold_model = type(model)(**model.get_config() if hasattr(model, 'get_config') else {})
            fold_model.load_state_dict(model.state_dict())
            
            # Quick training on fold (simplified)
            fold_trainer = self._create_trainer(fold_model, task)
            
            # Create data loaders
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
            
            # Train for a few epochs (simplified for benchmarking)
            fold_trainer.fit(train_loader, val_loader)
            
            # Evaluate fold
            fold_results_dict = fold_trainer.evaluate(val_loader)
            fold_results.append(fold_results_dict)
            
            # Store predictions for later analysis
            predictions, targets = self._get_predictions_targets(fold_model, val_data, task)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
        
        # Compute cross-validation statistics
        cv_stats = self._compute_cv_statistics(fold_results, metrics)
        
        # Compute overall metrics on all predictions
        if all_predictions and all_targets:
            overall_metrics = self._compute_metrics(
                np.array(all_predictions),
                np.array(all_targets),
                task.task_type,
                metrics
            )
            cv_stats['overall_metrics'] = overall_metrics
        
        return {
            'fold_results': fold_results,
            'cv_statistics': cv_stats,
            'all_predictions': all_predictions if self.config.save_predictions else None,
            'all_targets': all_targets if self.config.save_predictions else None
        }
    
    def _holdout_evaluate(
        self,
        model: BaseConnectomeModel,
        dataset,
        task: BaseConnectomeTask,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform holdout evaluation."""
        if self.config.verbose:
            print("Running holdout evaluation...")
        
        # Split dataset
        n_samples = len(dataset)
        n_test = int(n_samples * self.config.test_size)
        n_train = n_samples - n_test
        
        # Random split
        np.random.seed(self.config.random_state)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # Create datasets
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]
        
        # Train model
        trainer = self._create_trainer(model, task)
        
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        
        # Train model
        trainer.fit(train_loader, test_loader)
        
        # Evaluate
        test_results = trainer.evaluate(test_loader)
        
        # Get predictions for detailed analysis
        predictions, targets = self._get_predictions_targets(model, test_data, task)
        
        # Compute additional metrics
        detailed_metrics = self._compute_metrics(
            np.array(predictions),
            np.array(targets),
            task.task_type,
            metrics
        )
        
        return {
            'test_metrics': test_results,
            'detailed_metrics': detailed_metrics,
            'predictions': predictions if self.config.save_predictions else None,
            'targets': targets if self.config.save_predictions else None,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
    
    def _create_trainer(self, model: BaseConnectomeModel, task: BaseConnectomeTask):
        """Create trainer for benchmarking."""
        config = TrainingConfig(
            batch_size=8,
            learning_rate=1e-3,
            max_epochs=20,  # Reduced for benchmarking
            patience=5,
            val_check_interval=1
        )
        
        return AdvancedConnectomeTrainer(
            model=model,
            task=task,
            config=config,
            output_dir=self.output_dir / "training_temp",
            use_wandb=False
        )
    
    def _get_predictions_targets(self, model, dataset, task):
        """Get model predictions and targets."""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in dataset:
                # Get prediction
                output = model(data)
                if isinstance(output, dict):
                    pred = output['predictions']
                else:
                    pred = output
                
                # Get target
                target = task.prepare_targets([data])
                target = task.normalize_targets(target)
                
                predictions.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        return predictions, targets
    
    def _compute_cv_statistics(self, fold_results, metrics):
        """Compute cross-validation statistics."""
        cv_stats = {}
        
        # Collect metric values across folds
        for metric in metrics:
            values = []
            for fold_result in fold_results:
                if metric in fold_result:
                    values.append(fold_result[metric])
            
            if values:
                cv_stats[f"{metric}_mean"] = np.mean(values)
                cv_stats[f"{metric}_std"] = np.std(values)
                cv_stats[f"{metric}_values"] = values
        
        return cv_stats
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        task_type: str,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        results = {}
        
        try:
            if task_type == 'regression':
                if 'mae' in metrics:
                    results['mae'] = mean_absolute_error(targets, predictions)
                if 'rmse' in metrics:
                    results['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
                if 'r2' in metrics:
                    results['r2'] = r2_score(targets, predictions)
                if 'pearson_r' in metrics:
                    corr, _ = pearsonr(targets.flatten(), predictions.flatten())
                    results['pearson_r'] = corr if not np.isnan(corr) else 0.0
                if 'spearman_r' in metrics:
                    corr, _ = spearmanr(targets.flatten(), predictions.flatten())
                    results['spearman_r'] = corr if not np.isnan(corr) else 0.0
            
            elif task_type in ['binary_classification', 'multiclass_classification']:
                # Convert predictions to class labels if needed
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    pred_classes = np.argmax(predictions, axis=1)
                    pred_probs = predictions
                else:
                    pred_classes = (predictions > 0.5).astype(int)
                    pred_probs = predictions
                
                if targets.ndim > 1:
                    target_classes = np.argmax(targets, axis=1)
                else:
                    target_classes = targets.astype(int)
                
                if 'accuracy' in metrics:
                    results['accuracy'] = accuracy_score(target_classes, pred_classes)
                if 'precision' in metrics:
                    results['precision'] = precision_score(
                        target_classes, pred_classes, average='weighted', zero_division=0
                    )
                if 'recall' in metrics:
                    results['recall'] = recall_score(
                        target_classes, pred_classes, average='weighted', zero_division=0
                    )
                if 'f1' in metrics:
                    results['f1'] = f1_score(
                        target_classes, pred_classes, average='weighted', zero_division=0
                    )
                if 'macro_f1' in metrics:
                    results['macro_f1'] = f1_score(
                        target_classes, pred_classes, average='macro', zero_division=0
                    )
                
                # AUC metrics (only for binary or with probabilities)
                if task_type == 'binary_classification' and pred_probs.ndim == 1:
                    if 'auc' in metrics:
                        try:
                            results['auc'] = roc_auc_score(target_classes, pred_probs)
                        except ValueError:
                            results['auc'] = 0.5  # Random performance
                    if 'ap' in metrics:
                        try:
                            results['ap'] = average_precision_score(target_classes, pred_probs)
                        except ValueError:
                            results['ap'] = 0.5
        
        except Exception as e:
            print(f"Error computing metrics: {e}")
            # Return default values
            for metric in metrics:
                results[metric] = 0.0
        
        return results
    
    def _save_results(self, results: Dict[str, Any], task_name: str):
        """Save benchmark results to file."""
        results_file = self.output_dir / f"{task_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        if self.config.verbose:
            print(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    def _generate_report(self, results: Dict[str, Any], task_name: str):
        """Generate comprehensive benchmark report."""
        if self.config.verbose:
            print("\n=== BENCHMARK REPORT ===")
            print(f"Task: {task_name}")
            print(f"Model: {results['model_info']['name']}")
            print(f"Dataset Size: {results['dataset_info']['size']}")
            print(f"Model Parameters: {results['model_info']['parameters']:,}")
            
            # Cross-validation results
            if 'cross_validation' in results:
                print("\n--- Cross-Validation Results ---")
                cv_stats = results['cross_validation']['cv_statistics']
                for key, value in cv_stats.items():
                    if key.endswith('_mean'):
                        metric = key.replace('_mean', '')
                        std_key = f"{metric}_std"
                        if std_key in cv_stats:
                            print(f"{metric.upper()}: {value:.4f} ± {cv_stats[std_key]:.4f}")
                        else:
                            print(f"{metric.upper()}: {value:.4f}")
            
            # Holdout results
            if 'holdout' in results:
                print("\n--- Holdout Results ---")
                holdout_metrics = results['holdout']['detailed_metrics']
                for metric, value in holdout_metrics.items():
                    print(f"{metric.upper()}: {value:.4f}")
            
            print(f"\nEvaluation Time: {results['evaluation_time']:.2f} seconds")
        
        # Generate plots
        self._generate_plots(results, task_name)
    
    def _generate_plots(self, results: Dict[str, Any], task_name: str):
        """Generate visualization plots for results."""
        try:
            # Cross-validation metrics plot
            if 'cross_validation' in results and 'cv_statistics' in results['cross_validation']:
                self._plot_cv_results(results['cross_validation']['cv_statistics'], task_name)
            
            # Prediction plots if available
            if 'holdout' in results and results['holdout'].get('predictions'):
                self._plot_predictions(
                    results['holdout']['predictions'],
                    results['holdout']['targets'],
                    task_name,
                    results['task_info']['task_type']
                )
        
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def _plot_cv_results(self, cv_stats: Dict[str, Any], task_name: str):
        """Plot cross-validation results."""
        # Find metrics with values across folds
        metrics_data = {}
        for key, value in cv_stats.items():
            if key.endswith('_values') and isinstance(value, list):
                metric_name = key.replace('_values', '')
                metrics_data[metric_name] = value
        
        if not metrics_data:
            return
        
        # Create plot
        n_metrics = len(metrics_data)
        fig, axes = plt.subplots(1, min(n_metrics, 4), figsize=(4*min(n_metrics, 4), 4))
        
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 4:
            axes = axes[:4]
            metrics_data = dict(list(metrics_data.items())[:4])
        
        for idx, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            # Box plot of cross-validation results
            ax.boxplot(values)
            ax.set_title(f'{metric.upper()}\
Cross-Validation')
            ax.set_ylabel('Score')
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            ax.legend()
        
        plt.suptitle(f'Cross-Validation Results: {task_name}')
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{task_name}_cv_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.config.verbose:
            print(f"Cross-validation plot saved to {plot_path}")
    
    def _plot_predictions(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        task_name: str,
        task_type: str
    ):
        """Plot predictions vs targets."""
        pred_array = np.concatenate(predictions) if isinstance(predictions[0], np.ndarray) else np.array(predictions)
        target_array = np.concatenate(targets) if isinstance(targets[0], np.ndarray) else np.array(targets)
        
        if task_type == 'regression':
            # Scatter plot for regression
            plt.figure(figsize=(8, 6))
            plt.scatter(target_array, pred_array, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(target_array.min(), pred_array.min())
            max_val = max(target_array.max(), pred_array.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
            
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Predictions vs True Values: {task_name}')
            plt.legend()
            
            # Add correlation info
            corr = np.corrcoef(target_array.flatten(), pred_array.flatten())[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # Classification: confusion matrix
            if pred_array.ndim > 1:
                pred_classes = np.argmax(pred_array, axis=1)
            else:
                pred_classes = (pred_array > 0.5).astype(int)
            
            if target_array.ndim > 1:
                target_classes = np.argmax(target_array, axis=1)
            else:
                target_classes = target_array.astype(int)
            
            cm = confusion_matrix(target_classes, pred_classes)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix: {task_name}')
        
        plt.tight_layout()
        plot_path = self.output_dir / f'{task_name}_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.config.verbose:
            print(f"Predictions plot saved to {plot_path}")


# Available benchmark tasks
class AgePredictionBenchmark(SubjectBenchmark):
    """Age prediction benchmark."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_name = "age_prediction"
        self.metrics = ['mae', 'rmse', 'r2', 'pearson_r']


class SexClassificationBenchmark(SubjectBenchmark):
    """Sex classification benchmark."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_name = "sex_classification"
        self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']


class CognitiveScoreBenchmark(SubjectBenchmark):
    """Cognitive score prediction benchmark."""
    
    def __init__(self, cognitive_measure: str = "fluid_intelligence", **kwargs):
        super().__init__(**kwargs)
        self.task_name = f"cognitive_{cognitive_measure}"
        self.cognitive_measure = cognitive_measure
        self.metrics = ['mae', 'rmse', 'r2', 'pearson_r', 'spearman_r']